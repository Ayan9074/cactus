#include <arm_sve.h>
#include <cstdint>
#ifndef __ARM_FEATURE_SME2
#error "kernel_sme2.cpp must be compiled with SME2 enabled (e.g. -march=armv9.2-a+sme2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <arm_sme.h>
#include <arm_neon.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#if defined(__clang__)
#define CACTUS_UNROLL4 _Pragma("clang loop unroll_count(4)")
#else
#define CACTUS_UNROLL4
#endif

static inline void cactus_pack_a_f16_row_block(
    const __fp16* __restrict a,
    __fp16* __restrict a_packed,
    size_t K,
    size_t rb,
    size_t row0,
    size_t active_r,
    size_t tile_rows,
    size_t tile_pairs,
    size_t k_pairs
) {
    const size_t block_stride = k_pairs * tile_pairs;
    const bool even_k = ((K & 1u) == 0);
    for (size_t kp = 0; kp < k_pairs; ++kp) {
        const size_t k0 = kp * 2;
        const size_t k1 = k0 + 1;
        __fp16* dst = a_packed + rb * block_stride + kp * tile_pairs;
        const __fp16* src_col = a + row0 * K + k0;

        if (even_k) {
            CACTUS_UNROLL4
            for (size_t r = 0; r < active_r; ++r) {
                dst[2 * r] = src_col[0];
                dst[2 * r + 1] = src_col[1];
                src_col += K;
            }
        } else {
            const bool has_k1 = (k1 < K);
            CACTUS_UNROLL4
            for (size_t r = 0; r < active_r; ++r) {
                dst[2 * r] = src_col[0];
                dst[2 * r + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                src_col += K;
            }
        }
        CACTUS_UNROLL4
        for (size_t r = active_r; r < tile_rows; ++r) {
            dst[2 * r] = static_cast<__fp16>(0);
            dst[2 * r + 1] = static_cast<__fp16>(0);
        }
    }
}

static void cactus_pack_b_f16_from_bt(
    const __fp16* __restrict b_transposed,
    __fp16* __restrict b_packed,
    size_t K,
    size_t N,
    size_t tile_cols,
    size_t tile_pairs
) {
    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_cols - 1) / tile_cols;
    const size_t full_col_blocks = N / tile_cols;
    const size_t cb4_tiles = (full_col_blocks / 4) * 4;
    const size_t cb2_tiles = ((full_col_blocks - cb4_tiles) / 2) * 2;
    const size_t cb1_tiles = col_blocks - cb4_tiles - cb2_tiles;
    const size_t cb4_groups = cb4_tiles / 4;
    const size_t cb2_groups = cb2_tiles / 2;

    const size_t off_cb4 = 0;
    const size_t off_cb2 = off_cb4 + cb4_groups * k_pairs * 4 * tile_pairs;
    const size_t off_cb1 = off_cb2 + cb2_groups * k_pairs * 2 * tile_pairs;

    const bool even_k = ((K & 1u) == 0);

    CactusThreading::parallel_for(cb4_groups * k_pairs, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t g4 = idx / k_pairs;
                const size_t kp = idx % k_pairs;
                const size_t col0 = g4 * 4 * tile_cols;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;
                __fp16* dst = b_packed + off_cb4 + (g4 * k_pairs + kp) * (4 * tile_pairs);

                for (size_t t = 0; t < 4; ++t) {
                    __fp16* dst_t = dst + t * tile_pairs;
                    const size_t col_t = col0 + t * tile_cols;
                    const __fp16* src_col = b_transposed + col_t * K + k0;
                    if (even_k) {
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = src_col[1];
                            src_col += K;
                        }
                    } else {
                        const bool has_k1 = (k1 < K);
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                            src_col += K;
                        }
                    }
                }
            }
        });

    CactusThreading::parallel_for(cb2_groups * k_pairs, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t g2 = idx / k_pairs;
                const size_t kp = idx % k_pairs;
                const size_t col0 = cb4_tiles * tile_cols + g2 * 2 * tile_cols;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;
                __fp16* dst = b_packed + off_cb2 + (g2 * k_pairs + kp) * (2 * tile_pairs);

                for (size_t t = 0; t < 2; ++t) {
                    __fp16* dst_t = dst + t * tile_pairs;
                    const size_t col_t = col0 + t * tile_cols;
                    const __fp16* src_col = b_transposed + col_t * K + k0;
                    if (even_k) {
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = src_col[1];
                            src_col += K;
                        }
                    } else {
                        const bool has_k1 = (k1 < K);
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                            src_col += K;
                        }
                    }
                }
            }
        });

    CactusThreading::parallel_for(cb1_tiles * k_pairs, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t g1 = idx / k_pairs;
                const size_t kp = idx % k_pairs;
                const size_t cb = cb4_tiles + cb2_tiles + g1;
                const size_t col0 = cb * tile_cols;
                const size_t active_c = (col0 < N) ? std::min(tile_cols, N - col0) : 0;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;
                __fp16* dst = b_packed + off_cb1 + (g1 * k_pairs + kp) * tile_pairs;
                const __fp16* src_col = b_transposed + col0 * K + k0;
                if (even_k) {
                    CACTUS_UNROLL4
                    for (size_t c = 0; c < active_c; ++c) {
                        dst[2 * c] = src_col[0];
                        dst[2 * c + 1] = src_col[1];
                        src_col += K;
                    }
                } else {
                    const bool has_k1 = (k1 < K);
                    CACTUS_UNROLL4
                    for (size_t c = 0; c < active_c; ++c) {
                        dst[2 * c] = src_col[0];
                        dst[2 * c + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                        src_col += K;
                    }
                }
                CACTUS_UNROLL4
                for (size_t c = active_c; c < tile_cols; ++c) {
                    dst[2 * c] = static_cast<__fp16>(0);
                    dst[2 * c + 1] = static_cast<__fp16>(0);
                }
            }
        });
}

static void cactus_matmul_f16_sme2_worker(
    const __fp16* a_packed,
    const __fp16* b_packed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row,
    size_t tile_rows,
    size_t tile_pairs
) __arm_streaming __arm_inout("za") {
    if (start_row >= end_row) return;
    (void)M;

    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;
    const size_t full_col_blocks = N / tile_rows;
    const size_t cb4_tiles = (full_col_blocks / 4) * 4;
    const size_t cb2_tiles = ((full_col_blocks - cb4_tiles) / 2) * 2;
    const size_t cb4_groups = cb4_tiles / 4;
    const size_t cb2_groups = cb2_tiles / 2;
    const size_t cb1_off = cb4_groups * k_pairs * 4 * tile_pairs + cb2_groups * k_pairs * 2 * tile_pairs;
    const size_t a_row_block_stride = k_pairs * tile_pairs;

    const svcount_t pNh_full_c = svptrue_c16();
    const svbool_t pNh_full = svptrue_b16();
    const svbool_t pN32_full = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(tile_rows));
    const svcount_t pOut16_full_c = svwhilelt_c16(static_cast<uint64_t>(0), static_cast<uint64_t>(tile_rows), 2);
    const svfloat32_t z0_f32 = svdup_n_f32(0.0f);
#define CACTUS_STORE_FULL_TILE_ROW(dst_idx, out32_vec)                                   \
    do {                                                                                  \
        svfloat16_t out16_local = svcvt_f16_f32_z(pN32_full, (out32_vec));               \
        out16_local = svuzp1_f16(out16_local, out16_local);                              \
        svst1_f16_x2(pOut16_full_c, &c[(dst_idx)], svcreate2(out16_local, out16_local)); \
    } while (0)

    for (size_t row = start_row; row < end_row; row += tile_rows) {
        const size_t rb = row / tile_rows;
        const size_t active_r = std::min(tile_rows, end_row - row);
        const bool full_r = (active_r == tile_rows);
        const svbool_t pMh = full_r
            ? svptrue_b16()
            : svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_r * 2));

        size_t cb = 0;

        for (; cb < cb4_tiles; cb += 4) {
            svzero_za();
            const size_t g4 = cb / 4;
            const __fp16* b_g4_base = b_packed + g4 * k_pairs * 4 * tile_pairs;

            size_t kp = 0;
            for (; kp + 3 < k_pairs; kp += 4) {
                const __fp16* a_ptr0 = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr0 = b_g4_base + kp * (4 * tile_pairs);
                const __fp16* a_ptr1 = a_ptr0 + tile_pairs;
                const __fp16* b_ptr1 = b_ptr0 + 4 * tile_pairs;
                const __fp16* a_ptr2 = a_ptr1 + tile_pairs;
                const __fp16* b_ptr2 = b_ptr1 + 4 * tile_pairs;
                const __fp16* a_ptr3 = a_ptr2 + tile_pairs;
                const __fp16* b_ptr3 = b_ptr2 + 4 * tile_pairs;

                const svfloat16_t zA0 = svld1(pMh, a_ptr0);
                const svfloat16x4_t zB40 = svld1_f16_x4(pNh_full_c, b_ptr0);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA0, svget4(zB40, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA0, svget4(zB40, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA0, svget4(zB40, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA0, svget4(zB40, 3));

                const svfloat16_t zA1 = svld1(pMh, a_ptr1);
                const svfloat16x4_t zB41 = svld1_f16_x4(pNh_full_c, b_ptr1);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA1, svget4(zB41, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA1, svget4(zB41, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA1, svget4(zB41, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA1, svget4(zB41, 3));

                const svfloat16_t zA2 = svld1(pMh, a_ptr2);
                const svfloat16x4_t zB42 = svld1_f16_x4(pNh_full_c, b_ptr2);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA2, svget4(zB42, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA2, svget4(zB42, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA2, svget4(zB42, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA2, svget4(zB42, 3));

                const svfloat16_t zA3 = svld1(pMh, a_ptr3);
                const svfloat16x4_t zB43 = svld1_f16_x4(pNh_full_c, b_ptr3);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA3, svget4(zB43, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA3, svget4(zB43, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA3, svget4(zB43, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA3, svget4(zB43, 3));
            }

            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_g4_base + kp * (4 * tile_pairs);

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16x4_t zB4 = svld1_f16_x4(pNh_full_c, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA, svget4(zB4, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA, svget4(zB4, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA, svget4(zB4, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA, svget4(zB4, 3));
            }

            const size_t col = cb * tile_rows;
            if (full_r) {
                for (size_t trow = 0; trow < tile_rows; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT2 = svread_hor_za32_f32_vg4(2, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT3 = svread_hor_za32_f32_vg4(3, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 2 * tile_rows, svget4(zT2, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 3 * tile_rows, svget4(zT3, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 2 * tile_rows, svget4(zT2, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 3 * tile_rows, svget4(zT3, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 2 * tile_rows, svget4(zT2, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 3 * tile_rows, svget4(zT3, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 2 * tile_rows, svget4(zT2, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 3 * tile_rows, svget4(zT3, 3));
                }
            } else {
                size_t trow = 0;
                for (; trow + 3 < active_r; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT2 = svread_hor_za32_f32_vg4(2, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT3 = svread_hor_za32_f32_vg4(3, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 2 * tile_rows, svget4(zT2, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 3 * tile_rows, svget4(zT3, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 2 * tile_rows, svget4(zT2, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 3 * tile_rows, svget4(zT3, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 2 * tile_rows, svget4(zT2, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 3 * tile_rows, svget4(zT3, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 2 * tile_rows, svget4(zT2, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 3 * tile_rows, svget4(zT3, 3));
                }
                for (; trow + 1 < active_r; trow += 2) {
                    const svfloat32x2_t zT0 = svread_hor_za32_f32_vg2(0, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT1 = svread_hor_za32_f32_vg2(1, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT2 = svread_hor_za32_f32_vg2(2, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT3 = svread_hor_za32_f32_vg2(3, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget2(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget2(zT1, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 2 * tile_rows, svget2(zT2, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 3 * tile_rows, svget2(zT3, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget2(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget2(zT1, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 2 * tile_rows, svget2(zT2, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 3 * tile_rows, svget2(zT3, 1));
                }
                for (; trow < active_r; ++trow) {
                    const size_t dst = (row + trow) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst, svread_hor_za32_f32_m(z0_f32, pN32_full, 0, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 1, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + 2 * tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 2, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + 3 * tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 3, static_cast<uint32_t>(trow)));
                }
            }
        }

        for (; cb < cb4_tiles + cb2_tiles; cb += 2) {
            svzero_za();
            const size_t g2 = (cb - cb4_tiles) / 2;
            const __fp16* b_g2_base = b_packed + cb4_groups * k_pairs * 4 * tile_pairs + g2 * k_pairs * 2 * tile_pairs;

            size_t kp = 0;
            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_g2_base + kp * (2 * tile_pairs);

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16x2_t zB2 = svld1_f16_x2(pNh_full_c, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA, svget2(zB2, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA, svget2(zB2, 1));
            }

            const size_t col = cb * tile_rows;
            if (full_r) {
                for (size_t trow = 0; trow < tile_rows; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                }
            } else {
                size_t trow = 0;
                for (; trow + 3 < active_r; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                }
                for (; trow + 1 < active_r; trow += 2) {
                    const svfloat32x2_t zT0 = svread_hor_za32_f32_vg2(0, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT1 = svread_hor_za32_f32_vg2(1, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget2(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget2(zT1, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget2(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget2(zT1, 1));
                }
                for (; trow < active_r; ++trow) {
                    const size_t dst = (row + trow) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst, svread_hor_za32_f32_m(z0_f32, pN32_full, 0, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 1, static_cast<uint32_t>(trow)));
                }
            }
        }

        for (; cb < col_blocks; ++cb) {
            const size_t col = cb * tile_rows;
            const size_t active_c = std::min(tile_rows, N - col);
            const svbool_t pNh = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c * 2));
            const svbool_t pN32 = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));
            const svbool_t pOut16 = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));
            const size_t g1 = cb - cb4_tiles - cb2_tiles;
            const __fp16* b_g1_base = b_packed + cb1_off + g1 * k_pairs * tile_pairs;

            svzero_za();

            size_t kp = 0;
            for (; kp + 1 < k_pairs; kp += 2) {
                const __fp16* a_ptr0 = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr0 = b_g1_base + kp * tile_pairs;
                const __fp16* a_ptr1 = a_ptr0 + tile_pairs;
                const __fp16* b_ptr1 = b_ptr0 + tile_pairs;

                const svfloat16_t zA0 = svld1(pMh, a_ptr0);
                const svfloat16_t zB0 = svld1(pNh, b_ptr0);
                svmopa_za32_f16_m(0, pMh, pNh, zA0, zB0);

                const svfloat16_t zA1 = svld1(pMh, a_ptr1);
                const svfloat16_t zB1 = svld1(pNh, b_ptr1);
                svmopa_za32_f16_m(0, pMh, pNh, zA1, zB1);
            }

            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_g1_base + kp * tile_pairs;

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16_t zB = svld1(pNh, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh, zA, zB);
            }

            for (size_t trow = 0; trow < active_r; ++trow) {
                svfloat32_t out32 = svread_hor_za32_f32_m(z0_f32, pN32, 0, static_cast<uint32_t>(trow));
                svfloat16_t out16 = svcvt_f16_f32_z(pN32, out32);
                out16 = svuzp1_f16(out16, out16);
                svst1(pOut16, &c[(row + trow) * N + col], out16);
            }
        }
    }
#undef CACTUS_STORE_FULL_TILE_ROW
}

__arm_new("za") __arm_locally_streaming
static void cactus_matmul_f16_sme2_thread_entry(
    const __fp16* a,
    __fp16* a_packed,
    const __fp16* b_packed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t row_block_size,
    size_t start_block,
    size_t end_block
) {
    const size_t tile_rows = svcntsw();
    const size_t tile_pairs = svcnth();
    const size_t k_pairs = (K + 1) / 2;

    for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
        const size_t start_row = block_idx * row_block_size;
        const size_t end_row = std::min(start_row + row_block_size, M);

        for (size_t row = start_row; row < end_row; row += tile_rows) {
            const size_t rb = row / tile_rows;
            const size_t active_r = std::min(tile_rows, end_row - row);
            cactus_pack_a_f16_row_block(
                a,
                a_packed,
                K,
                rb,
                row,
                active_r,
                tile_rows,
                tile_pairs,
                k_pairs
            );
        }

        cactus_matmul_f16_sme2_worker(
            a_packed,
            b_packed,
            c,
            M,
            K,
            N,
            start_row,
            end_row,
            tile_rows,
            tile_pairs
        );
    }
}

__arm_new("za") __arm_locally_streaming
void cactus_matmul_f16_sme2_caller(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {
    const size_t tile_rows = svcntsw();
    const size_t tile_pairs = svcnth();
    constexpr size_t SME2_TILES_PER_THREAD = 3;

    const size_t row_blocks = (M + tile_rows - 1) / tile_rows;
    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;

    std::unique_ptr<__fp16[]> a_packed(new __fp16[row_blocks * k_pairs * tile_pairs]);
    std::unique_ptr<__fp16[]> b_packed(new __fp16[k_pairs * col_blocks * tile_pairs]);
    __fp16* a_packed_ptr = a_packed.get();
    const __fp16* b_packed_ptr = b_packed.get();

    cactus_pack_b_f16_from_bt(b_transposed, b_packed.get(), K, N, tile_rows, tile_pairs);

    const size_t row_block_size = SME2_TILES_PER_THREAD * tile_rows;
    const size_t num_row_blocks = (M + row_block_size - 1) / row_block_size;

    auto& pool = CactusThreading::get_thread_pool();
    const size_t num_workers = std::min(pool.num_workers(), num_row_blocks);
    if (num_workers <= 1) {
        cactus_matmul_f16_sme2_thread_entry(
            a,
            a_packed_ptr,
            b_packed_ptr,
            c,
            M,
            K,
            N,
            row_block_size,
            0,
            num_row_blocks
        );
        return;
    }

    std::atomic<size_t> next_block{0};
    pool.enqueue_n_threads(num_workers, num_workers, [&](size_t, size_t) {
        while (true) {
            const size_t block_idx = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block_idx >= num_row_blocks) break;
            cactus_matmul_f16_sme2_thread_entry(
                a,
                a_packed_ptr,
                b_packed_ptr,
                c,
                M,
                K,
                N,
                row_block_size,
                block_idx,
                block_idx + 1
            );
        }
    });
    pool.wait_all();
}

namespace {
constexpr size_t CACTUS_ATTN_H64_HEAD_DIM = 64;
constexpr size_t CACTUS_ATTN_H64_Q_TILE_DEFAULT = 4;
constexpr size_t CACTUS_ATTN_H64_Q_TILE_MAX = 8;
constexpr size_t CACTUS_ATTN_H64_KV_TILE = 32;

struct CactusAttentionH64Sme2Workspace {
    alignas(64) __fp16 q_tile[CACTUS_ATTN_H64_Q_TILE_MAX * CACTUS_ATTN_H64_HEAD_DIM];
    alignas(64) __fp16 k_tile[CACTUS_ATTN_H64_KV_TILE * CACTUS_ATTN_H64_HEAD_DIM];

    alignas(64) float scores[CACTUS_ATTN_H64_Q_TILE_MAX * CACTUS_ATTN_H64_KV_TILE];
    alignas(64) float out_accum[CACTUS_ATTN_H64_Q_TILE_MAX * CACTUS_ATTN_H64_HEAD_DIM];
    float running_max[CACTUS_ATTN_H64_Q_TILE_MAX];
    float running_sum[CACTUS_ATTN_H64_Q_TILE_MAX];

    size_t packed_tile_rows = 0;
    size_t packed_tile_pairs = 0;
    std::vector<__fp16> score_a_packed;
    std::vector<__fp16> score_b_packed;
};

static thread_local CactusAttentionH64Sme2Workspace g_attention_h64_sme2_workspace;

static inline size_t cactus_div_up_size(size_t x, size_t y) {
    return (x + y - 1) / y;
}

struct CactusSme2TileShape {
    size_t tile_rows;
    size_t tile_pairs;
};

static size_t cactus_attention_h64_sme2_q_tile() {
    static const size_t q_tile = []() {
        const char* env = std::getenv("CACTUS_ATTENTION_SME2_Q_TILE");
        if (!env || env[0] == '\0') return CACTUS_ATTN_H64_Q_TILE_DEFAULT;
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end == env || (end && *end != '\0')) return CACTUS_ATTN_H64_Q_TILE_DEFAULT;
        return (value >= 8) ? static_cast<size_t>(8) : CACTUS_ATTN_H64_Q_TILE_DEFAULT;
    }();
    return q_tile;
}

__arm_new("za") __arm_locally_streaming
static CactusSme2TileShape __attribute__((noinline)) cactus_attention_h64_sme2_tile_shape() {
    CactusSme2TileShape shape{};
    shape.tile_rows = svcntsw();
    shape.tile_pairs = svcnth();
    return shape;
}

static inline void cactus_attention_h64_sme2_prepare_buffers(
    CactusAttentionH64Sme2Workspace& ws,
    size_t tile_rows,
    size_t tile_pairs
) {
    if (ws.packed_tile_rows == tile_rows && ws.packed_tile_pairs == tile_pairs) return;
    ws.packed_tile_rows = tile_rows;
    ws.packed_tile_pairs = tile_pairs;

    const size_t score_k_pairs = (CACTUS_ATTN_H64_HEAD_DIM + 1) / 2;
    const size_t score_col_blocks = cactus_div_up_size(CACTUS_ATTN_H64_KV_TILE, tile_rows);
    ws.score_a_packed.resize(score_k_pairs * tile_pairs);
    ws.score_b_packed.resize(score_col_blocks * score_k_pairs * tile_pairs);
}

static inline void cactus_attention_pack_a_pairs_row_major(
    const __fp16* a,
    __fp16* a_packed,
    size_t rows,
    size_t row_stride,
    size_t k,
    size_t tile_rows,
    size_t tile_pairs
) {
    const size_t k_pairs = (k + 1) / 2;
    for (size_t kp = 0; kp < k_pairs; ++kp) {
        const size_t k0 = kp * 2;
        const size_t k1 = k0 + 1;
        __fp16* dst = a_packed + kp * tile_pairs;
        for (size_t r = 0; r < rows; ++r) {
            const __fp16* row_ptr = a + r * row_stride;
            dst[2 * r] = row_ptr[k0];
            dst[2 * r + 1] = (k1 < k) ? row_ptr[k1] : static_cast<__fp16>(0);
        }
        for (size_t r = rows; r < tile_rows; ++r) {
            dst[2 * r] = static_cast<__fp16>(0);
            dst[2 * r + 1] = static_cast<__fp16>(0);
        }
    }
}

static inline void cactus_attention_pack_b_pairs_from_bt(
    const __fp16* bt,
    __fp16* b_packed,
    size_t k,
    size_t cols,
    size_t tile_rows,
    size_t tile_pairs
) {
    const size_t k_pairs = (k + 1) / 2;
    const size_t col_blocks = cactus_div_up_size(cols, tile_rows);
    for (size_t cb = 0; cb < col_blocks; ++cb) {
        const size_t col0 = cb * tile_rows;
        const size_t active_c = std::min(tile_rows, cols - col0);
        for (size_t kp = 0; kp < k_pairs; ++kp) {
            const size_t k0 = kp * 2;
            const size_t k1 = k0 + 1;
            __fp16* dst = b_packed + (cb * k_pairs + kp) * tile_pairs;
            for (size_t c = 0; c < active_c; ++c) {
                const __fp16* src = bt + (col0 + c) * k + k0;
                dst[2 * c] = src[0];
                dst[2 * c + 1] = (k1 < k) ? src[1] : static_cast<__fp16>(0);
            }
            for (size_t c = active_c; c < tile_rows; ++c) {
                dst[2 * c] = static_cast<__fp16>(0);
                dst[2 * c + 1] = static_cast<__fp16>(0);
            }
        }
    }
}

static void __attribute__((noinline)) cactus_attention_mopa_packed_to_f32(
    const __fp16* a_packed,
    const __fp16* b_packed,
    float* out,
    size_t rows,
    size_t k,
    size_t cols,
    size_t out_stride,
    size_t tile_rows,
    size_t tile_pairs
) __arm_streaming __arm_inout("za") {
    if (rows == 0 || cols == 0 || k == 0) return;

    const size_t k_pairs = (k + 1) / 2;
    const size_t col_blocks = cactus_div_up_size(cols, tile_rows);
    const svfloat32_t z0_f32 = svdup_n_f32(0.0f);
    const svbool_t pMh = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(rows * 2));

    for (size_t cb = 0; cb < col_blocks; ++cb) {
        const size_t col0 = cb * tile_rows;
        const size_t active_c = std::min(tile_rows, cols - col0);
        const svbool_t pNh = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c * 2));
        const svbool_t pN32 = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));

        svzero_za();
        for (size_t kp = 0; kp < k_pairs; ++kp) {
            const svfloat16_t zA = svld1(pMh, a_packed + kp * tile_pairs);
            const svfloat16_t zB = svld1(pNh, b_packed + (cb * k_pairs + kp) * tile_pairs);
            svmopa_za32_f16_m(0, pMh, pNh, zA, zB);
        }

        for (size_t r = 0; r < rows; ++r) {
            svfloat32_t row_out = svread_hor_za32_f32_m(
                z0_f32,
                pN32,
                0,
                static_cast<uint32_t>(r)
            );
            svst1(pN32, out + r * out_stride + col0, row_out);
        }
    }
}

__arm_new("za") __arm_locally_streaming
static void __attribute__((noinline)) cactus_attention_score_tile_sme2_fp16(
    CactusAttentionH64Sme2Workspace& ws,
    size_t rows,
    size_t kv_block,
    size_t tile_rows,
    size_t tile_pairs
) {
    cactus_attention_pack_a_pairs_row_major(
        ws.q_tile,
        ws.score_a_packed.data(),
        rows,
        CACTUS_ATTN_H64_HEAD_DIM,
        CACTUS_ATTN_H64_HEAD_DIM,
        tile_rows,
        tile_pairs
    );
    cactus_attention_pack_b_pairs_from_bt(
        ws.k_tile,
        ws.score_b_packed.data(),
        CACTUS_ATTN_H64_HEAD_DIM,
        kv_block,
        tile_rows,
        tile_pairs
    );
    cactus_attention_mopa_packed_to_f32(
        ws.score_a_packed.data(),
        ws.score_b_packed.data(),
        ws.scores,
        rows,
        CACTUS_ATTN_H64_HEAD_DIM,
        kv_block,
        CACTUS_ATTN_H64_KV_TILE,
        tile_rows,
        tile_pairs
    );
}
} // namespace

void cactus_attention_f16_h64_sme2_caller(
    const __fp16* queries,
    const __fp16* keys,
    const __fp16* values,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t kv_seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    float scale,
    size_t position_offset,
    bool is_causal
) {
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t q_batch_stride = seq_len * num_q_heads * CACTUS_ATTN_H64_HEAD_DIM;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads * CACTUS_ATTN_H64_HEAD_DIM;
    const size_t o_batch_stride = q_batch_stride;
    const size_t q_seq_stride = num_q_heads * CACTUS_ATTN_H64_HEAD_DIM;
    const size_t kv_seq_stride = num_kv_heads * CACTUS_ATTN_H64_HEAD_DIM;
    const size_t o_seq_stride = q_seq_stride;

    if (batch_size == 0 || seq_len == 0 || num_q_heads == 0 || num_kv_heads == 0 || group_size == 0) return;
    if (kv_seq_len == 0) {
        std::memset(output, 0, batch_size * o_batch_stride * sizeof(__fp16));
        return;
    }

    const CactusSme2TileShape tile_shape = cactus_attention_h64_sme2_tile_shape();
    const size_t tile_rows = tile_shape.tile_rows;
    const size_t tile_pairs = tile_shape.tile_pairs;
    if (tile_rows == 0 || tile_pairs == 0) return;
    const size_t q_tile = std::min(std::min(cactus_attention_h64_sme2_q_tile(), tile_rows), CACTUS_ATTN_H64_Q_TILE_MAX);
    if (q_tile == 0) return;
    const size_t q_tiles = cactus_div_up_size(seq_len, q_tile);

    CactusThreading::parallel_for(batch_size * num_q_heads * q_tiles, CactusThreading::Thresholds::ATTENTION,
        [&](size_t start, size_t end) {
            CactusAttentionH64Sme2Workspace& ws = g_attention_h64_sme2_workspace;
            cactus_attention_h64_sme2_prepare_buffers(ws, tile_rows, tile_pairs);

            for (size_t work = start; work < end; ++work) {
                const size_t batch = work / (num_q_heads * q_tiles);
                const size_t rem = work % (num_q_heads * q_tiles);
                const size_t q_head = rem / q_tiles;
                const size_t tile_idx = rem % q_tiles;
                const size_t kv_head = q_head / group_size;
                const size_t q0 = tile_idx * q_tile;
                const size_t q_rows = std::min(q_tile, seq_len - q0);

                size_t max_kv_end = 0;
                size_t row_kv_end[CACTUS_ATTN_H64_Q_TILE_MAX] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (size_t r = 0; r < q_rows; ++r) {
                    const size_t q_pos = q0 + r;
                    const size_t abs_q = position_offset + q_pos;
                    const size_t kv_end = is_causal ? std::min(kv_seq_len, abs_q + 1) : kv_seq_len;
                    row_kv_end[r] = kv_end;
                    max_kv_end = std::max(max_kv_end, kv_end);

                    const __fp16* q_src = queries + batch * q_batch_stride + q_pos * q_seq_stride + q_head * CACTUS_ATTN_H64_HEAD_DIM;
                    std::memcpy(ws.q_tile + r * CACTUS_ATTN_H64_HEAD_DIM, q_src, CACTUS_ATTN_H64_HEAD_DIM * sizeof(__fp16));
                }

                for (size_t r = 0; r < q_rows; ++r) {
                    ws.running_max[r] = NEG_INF;
                    ws.running_sum[r] = 0.0f;
                    float* out_row = ws.out_accum + r * CACTUS_ATTN_H64_HEAD_DIM;
                    std::memset(out_row, 0, CACTUS_ATTN_H64_HEAD_DIM * sizeof(float));
                }

                for (size_t kv0 = 0; kv0 < max_kv_end; kv0 += CACTUS_ATTN_H64_KV_TILE) {
                    const size_t kv1 = std::min(kv0 + CACTUS_ATTN_H64_KV_TILE, max_kv_end);
                    const size_t block_len = kv1 - kv0;

                    for (size_t i = 0; i < block_len; ++i) {
                        const size_t kv_pos = kv0 + i;
                        const __fp16* k_src = keys + batch * kv_batch_stride + kv_pos * kv_seq_stride + kv_head * CACTUS_ATTN_H64_HEAD_DIM;
                        std::memcpy(ws.k_tile + i * CACTUS_ATTN_H64_HEAD_DIM, k_src, CACTUS_ATTN_H64_HEAD_DIM * sizeof(__fp16));
                    }

                    cactus_attention_score_tile_sme2_fp16(ws, q_rows, block_len, tile_rows, tile_pairs);

                    for (size_t r = 0; r < q_rows; ++r) {
                        const size_t valid_len = (row_kv_end[r] > kv0)
                            ? std::min(block_len, row_kv_end[r] - kv0)
                            : 0;

                        if (valid_len == 0) {
                            std::memset(ws.scores + r * CACTUS_ATTN_H64_KV_TILE, 0, CACTUS_ATTN_H64_KV_TILE * sizeof(float));
                            continue;
                        }

                        float* score_row = ws.scores + r * CACTUS_ATTN_H64_KV_TILE;
                        float block_max = score_row[0] * scale;
                        for (size_t i = 1; i < valid_len; ++i) {
                            block_max = std::max(block_max, score_row[i] * scale);
                        }

                        const float prev_max = ws.running_max[r];
                        const float new_max = std::max(prev_max, block_max);
                        const float scale_old = (prev_max == NEG_INF) ? 0.0f : std::exp(prev_max - new_max);

                        if (scale_old != 1.0f) {
                            ws.running_sum[r] *= scale_old;
                            float* out_row = ws.out_accum + r * CACTUS_ATTN_H64_HEAD_DIM;
                            for (size_t d = 0; d < CACTUS_ATTN_H64_HEAD_DIM; ++d) {
                                out_row[d] *= scale_old;
                            }
                        }
                        ws.running_max[r] = new_max;

                        float block_sum = 0.0f;
                        for (size_t i = 0; i < valid_len; ++i) {
                            const float p = std::exp(score_row[i] * scale - new_max);
                            score_row[i] = p;
                            block_sum += p;
                        }
                        for (size_t i = valid_len; i < CACTUS_ATTN_H64_KV_TILE; ++i) {
                            score_row[i] = 0.0f;
                        }

                        ws.running_sum[r] += block_sum;
                    }

                    for (size_t i = 0; i < block_len; ++i) {
                        const size_t kv_pos = kv0 + i;
                        const __fp16* v_src = values + batch * kv_batch_stride + kv_pos * kv_seq_stride + kv_head * CACTUS_ATTN_H64_HEAD_DIM;
                        for (size_t d = 0; d < CACTUS_ATTN_H64_HEAD_DIM; d += 8) {
                            const float16x8_t vv = vld1q_f16(v_src + d);
                            const float32x4_t v_low = vcvt_f32_f16(vget_low_f16(vv));
                            const float32x4_t v_high = vcvt_f32_f16(vget_high_f16(vv));
                            for (size_t r = 0; r < q_rows; ++r) {
                                const float weight = ws.scores[r * CACTUS_ATTN_H64_KV_TILE + i];
                                if (weight == 0.0f) continue;
                                const float32x4_t wv = vdupq_n_f32(weight);
                                float* out_row = ws.out_accum + r * CACTUS_ATTN_H64_HEAD_DIM;
                                float32x4_t acc_lo = vld1q_f32(out_row + d);
                                float32x4_t acc_hi = vld1q_f32(out_row + d + 4);
                                acc_lo = vfmaq_f32(acc_lo, v_low, wv);
                                acc_hi = vfmaq_f32(acc_hi, v_high, wv);
                                vst1q_f32(out_row + d, acc_lo);
                                vst1q_f32(out_row + d + 4, acc_hi);
                            }
                        }
                    }
                }

                for (size_t r = 0; r < q_rows; ++r) {
                    const size_t q_pos = q0 + r;
                    __fp16* dst = output + batch * o_batch_stride + q_pos * o_seq_stride + q_head * CACTUS_ATTN_H64_HEAD_DIM;
                    const float sum = ws.running_sum[r];
                    if (sum <= 0.0f) {
                        std::memset(dst, 0, CACTUS_ATTN_H64_HEAD_DIM * sizeof(__fp16));
                        continue;
                    }
                    const float inv_sum = 1.0f / sum;
                    const float* out_row = ws.out_accum + r * CACTUS_ATTN_H64_HEAD_DIM;
                    for (size_t d = 0; d < CACTUS_ATTN_H64_HEAD_DIM; ++d) {
                        dst[d] = static_cast<__fp16>(out_row[d] * inv_sum);
                    }
                }
            }
        });
}

#undef CACTUS_UNROLL4
