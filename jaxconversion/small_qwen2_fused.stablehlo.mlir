module @jit_small_qwen2_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x5xi32>, %arg1: tensor<1x5xi32>, %arg2: tensor<151665x128xf16>, %arg3: tensor<128xf16>, %arg4: tensor<151665x128xf16>, %arg5: tensor<128xf16>, %arg6: tensor<128x128xf16>, %arg7: tensor<128xf16>, %arg8: tensor<64x128xf16>, %arg9: tensor<64xf16>, %arg10: tensor<64x128xf16>, %arg11: tensor<64xf16>, %arg12: tensor<128x128xf16>, %arg13: tensor<128xf16>, %arg14: tensor<128xf16>, %arg15: tensor<32x128xf16>, %arg16: tensor<32xf16>, %arg17: tensor<32x128xf16>, %arg18: tensor<32xf16>, %arg19: tensor<128x32xf16>, %arg20: tensor<128xf16>, %arg21: tensor<128xf16>, %arg22: tensor<128x128xf16>, %arg23: tensor<128xf16>, %arg24: tensor<64x128xf16>, %arg25: tensor<64xf16>, %arg26: tensor<64x128xf16>, %arg27: tensor<64xf16>, %arg28: tensor<128x128xf16>, %arg29: tensor<128xf16>, %arg30: tensor<128xf16>, %arg31: tensor<32x128xf16>, %arg32: tensor<32xf16>, %arg33: tensor<32x128xf16>, %arg34: tensor<32xf16>, %arg35: tensor<128x32xf16>, %arg36: tensor<128xf16>) -> (tensor<1x5x151665xf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x5xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<1x5xi32>, tensor<1x5xi32>) -> tensor<1x5xi1>
    %c_0 = stablehlo.constant dense<151665> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x5xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x5xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<1x5xi1>, tensor<1x5xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x5xi32>) -> tensor<1x5x1xi32>
    %6 = "stablehlo.gather"(%arg2, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<151665x128xf16>, tensor<1x5x1xi32>) -> tensor<1x5x128xf16>
    %7 = stablehlo.convert %6 : (tensor<1x5x128xf16>) -> tensor<1x5x128xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x5x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x5x128xf32>, tensor<f32>) -> tensor<1x5xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_1 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %12 = stablehlo.divide %10, %11 : tensor<1x5x1xf32>
    %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x5x1xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x5x1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x128xf32>
    %17 = stablehlo.multiply %7, %16 : tensor<1x5x128xf32>
    %18 = stablehlo.convert %17 : (tensor<1x5x128xf32>) -> tensor<1x5x128xf16>
    %19 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %21 = stablehlo.multiply %18, %20 : tensor<1x5x128xf16>
    %22 = stablehlo.dot_general %21, %arg6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<128x128xf16>) -> tensor<1x5x128xf16>
    %23 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %25 = stablehlo.add %22, %24 : tensor<1x5x128xf16>
    %26 = stablehlo.dot_general %21, %arg8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<64x128xf16>) -> tensor<1x5x64xf16>
    %27 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x5x64xf16>
    %29 = stablehlo.add %26, %28 : tensor<1x5x64xf16>
    %30 = stablehlo.dot_general %21, %arg10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<64x128xf16>) -> tensor<1x5x64xf16>
    %31 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x5x64xf16>
    %33 = stablehlo.add %30, %32 : tensor<1x5x64xf16>
    %34 = stablehlo.reshape %25 : (tensor<1x5x128xf16>) -> tensor<1x5x4x32xf16>
    %35 = stablehlo.transpose %34, dims = [0, 2, 1, 3] : (tensor<1x5x4x32xf16>) -> tensor<1x4x5x32xf16>
    %36 = stablehlo.reshape %29 : (tensor<1x5x64xf16>) -> tensor<1x5x2x32xf16>
    %37 = stablehlo.transpose %36, dims = [0, 2, 1, 3] : (tensor<1x5x2x32xf16>) -> tensor<1x2x5x32xf16>
    %38 = stablehlo.reshape %33 : (tensor<1x5x64xf16>) -> tensor<1x5x2x32xf16>
    %39 = stablehlo.transpose %38, dims = [0, 2, 1, 3] : (tensor<1x5x2x32xf16>) -> tensor<1x2x5x32xf16>
    %40 = stablehlo.iota dim = 0 : tensor<16xf32>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %42 = stablehlo.multiply %41, %40 : tensor<16xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %44 = stablehlo.add %43, %42 : tensor<16xf32>
    %cst_5 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %46 = stablehlo.divide %44, %45 : tensor<16xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %48 = stablehlo.power %47, %46 : tensor<16xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %50 = stablehlo.divide %49, %48 : tensor<16xf32>
    %51 = stablehlo.convert %arg1 : (tensor<1x5xi32>) -> tensor<1x5xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %53 = stablehlo.broadcast_in_dim %50, dims = [2] : (tensor<16xf32>) -> tensor<1x1x16xf32>
    %54 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x16xf32>
    %55 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<1x1x16xf32>) -> tensor<1x5x16xf32>
    %56 = stablehlo.multiply %54, %55 : tensor<1x5x16xf32>
    %57 = stablehlo.concatenate %56, %56, dim = 2 : (tensor<1x5x16xf32>, tensor<1x5x16xf32>) -> tensor<1x5x32xf32>
    %58 = stablehlo.cosine %57 : tensor<1x5x32xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 2, 3] : (tensor<1x5x32xf32>) -> tensor<1x1x5x32xf32>
    %60 = stablehlo.convert %59 : (tensor<1x1x5x32xf32>) -> tensor<1x1x5x32xf16>
    %61 = stablehlo.sine %57 : tensor<1x5x32xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 2, 3] : (tensor<1x5x32xf32>) -> tensor<1x1x5x32xf32>
    %63 = stablehlo.convert %62 : (tensor<1x1x5x32xf32>) -> tensor<1x1x5x32xf16>
    %64 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %65 = stablehlo.multiply %35, %64 : tensor<1x4x5x32xf16>
    %66 = stablehlo.slice %35 [0:1, 0:4, 0:5, 0:16] : (tensor<1x4x5x32xf16>) -> tensor<1x4x5x16xf16>
    %67 = stablehlo.slice %35 [0:1, 0:4, 0:5, 16:32] : (tensor<1x4x5x32xf16>) -> tensor<1x4x5x16xf16>
    %68 = stablehlo.negate %67 : tensor<1x4x5x16xf16>
    %69 = stablehlo.concatenate %68, %66, dim = 3 : (tensor<1x4x5x16xf16>, tensor<1x4x5x16xf16>) -> tensor<1x4x5x32xf16>
    %70 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %71 = stablehlo.multiply %69, %70 : tensor<1x4x5x32xf16>
    %72 = stablehlo.add %65, %71 : tensor<1x4x5x32xf16>
    %73 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x2x5x32xf16>
    %74 = stablehlo.multiply %37, %73 : tensor<1x2x5x32xf16>
    %75 = stablehlo.slice %37 [0:1, 0:2, 0:5, 0:16] : (tensor<1x2x5x32xf16>) -> tensor<1x2x5x16xf16>
    %76 = stablehlo.slice %37 [0:1, 0:2, 0:5, 16:32] : (tensor<1x2x5x32xf16>) -> tensor<1x2x5x16xf16>
    %77 = stablehlo.negate %76 : tensor<1x2x5x16xf16>
    %78 = stablehlo.concatenate %77, %75, dim = 3 : (tensor<1x2x5x16xf16>, tensor<1x2x5x16xf16>) -> tensor<1x2x5x32xf16>
    %79 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x2x5x32xf16>
    %80 = stablehlo.multiply %78, %79 : tensor<1x2x5x32xf16>
    %81 = stablehlo.add %74, %80 : tensor<1x2x5x32xf16>
    %82 = stablehlo.slice %81 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %83 = stablehlo.slice %81 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %84 = stablehlo.slice %81 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %85 = stablehlo.slice %81 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %86 = stablehlo.concatenate %82, %83, %84, %85, dim = 1 : (tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %87 = stablehlo.slice %39 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %88 = stablehlo.slice %39 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %89 = stablehlo.slice %39 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %90 = stablehlo.slice %39 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %91 = stablehlo.concatenate %87, %88, %89, %90, dim = 1 : (tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %92 = stablehlo.dot_general %72, %86, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x5x32xf16>, tensor<1x4x5x32xf16>) -> tensor<1x4x5x5xf16>
    %cst_8 = stablehlo.constant dense<1.767580e-01> : tensor<f16>
    %93 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f16>) -> tensor<1x4x5x5xf16>
    %94 = stablehlo.multiply %92, %93 : tensor<1x4x5x5xf16>
    %c_9 = stablehlo.constant dense<true> : tensor<i1>
    %95 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i1>) -> tensor<5x5xi1>
    %96 = call @tril(%95) : (tensor<5x5xi1>) -> tensor<5x5xi1>
    %97 = stablehlo.broadcast_in_dim %96, dims = [2, 3] : (tensor<5x5xi1>) -> tensor<1x1x5x5xi1>
    %cst_10 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %98 = call @_where(%97, %94, %cst_10) : (tensor<1x1x5x5xi1>, tensor<1x4x5x5xf16>, tensor<f16>) -> tensor<1x4x5x5xf16>
    %cst_11 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %99 = stablehlo.reduce(%98 init: %cst_11) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x5x5xf16>, tensor<f16>) -> tensor<1x4x5xf16>
    %cst_12 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %100 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f16>) -> tensor<1x4x5xf16>
    %101 = stablehlo.maximum %100, %99 : tensor<1x4x5xf16>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2] : (tensor<1x4x5xf16>) -> tensor<1x4x5x1xf16>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x4x5x1xf16>) -> tensor<1x4x5x5xf16>
    %104 = stablehlo.subtract %98, %103 : tensor<1x4x5x5xf16>
    %105 = stablehlo.exponential %104 : tensor<1x4x5x5xf16>
    %106 = stablehlo.convert %105 : (tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %107 = stablehlo.reduce(%106 init: %cst_13) applies stablehlo.add across dimensions = [3] : (tensor<1x4x5x5xf32>, tensor<f32>) -> tensor<1x4x5xf32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0, 1, 2] : (tensor<1x4x5xf32>) -> tensor<1x4x5x1xf32>
    %109 = stablehlo.convert %108 : (tensor<1x4x5x1xf32>) -> tensor<1x4x5x1xf16>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2, 3] : (tensor<1x4x5x1xf16>) -> tensor<1x4x5x5xf16>
    %111 = stablehlo.divide %105, %110 : tensor<1x4x5x5xf16>
    %112 = stablehlo.dot_general %111, %91, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x5x5xf16>, tensor<1x4x5x32xf16>) -> tensor<1x4x5x32xf16>
    %113 = stablehlo.transpose %112, dims = [0, 2, 1, 3] : (tensor<1x4x5x32xf16>) -> tensor<1x5x4x32xf16>
    %114 = stablehlo.reshape %113 : (tensor<1x5x4x32xf16>) -> tensor<1x5x128xf16>
    %115 = stablehlo.dot_general %114, %arg12, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<128x128xf16>) -> tensor<1x5x128xf16>
    %116 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %118 = stablehlo.add %115, %117 : tensor<1x5x128xf16>
    %119 = stablehlo.add %6, %118 : tensor<1x5x128xf16>
    %120 = stablehlo.convert %119 : (tensor<1x5x128xf16>) -> tensor<1x5x128xf32>
    %121 = stablehlo.multiply %120, %120 : tensor<1x5x128xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %122 = stablehlo.reduce(%121 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<1x5x128xf32>, tensor<f32>) -> tensor<1x5xf32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_15 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %124 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %125 = stablehlo.divide %123, %124 : tensor<1x5x1xf32>
    %cst_16 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %126 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %127 = stablehlo.add %125, %126 : tensor<1x5x1xf32>
    %128 = stablehlo.rsqrt %127 : tensor<1x5x1xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x128xf32>
    %130 = stablehlo.multiply %120, %129 : tensor<1x5x128xf32>
    %131 = stablehlo.convert %130 : (tensor<1x5x128xf32>) -> tensor<1x5x128xf16>
    %132 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %134 = stablehlo.multiply %131, %133 : tensor<1x5x128xf16>
    %135 = stablehlo.dot_general %134, %arg15, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<32x128xf16>) -> tensor<1x5x32xf16>
    %136 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %137 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2] : (tensor<1x1x32xf16>) -> tensor<1x5x32xf16>
    %138 = stablehlo.add %135, %137 : tensor<1x5x32xf16>
    %139 = stablehlo.dot_general %134, %arg17, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<32x128xf16>) -> tensor<1x5x32xf16>
    %140 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %141 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x1x32xf16>) -> tensor<1x5x32xf16>
    %142 = stablehlo.add %139, %141 : tensor<1x5x32xf16>
    %143 = stablehlo.negate %138 : tensor<1x5x32xf16>
    %144 = stablehlo.exponential %143 : tensor<1x5x32xf16>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %145 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f16>) -> tensor<1x5x32xf16>
    %146 = stablehlo.add %145, %144 : tensor<1x5x32xf16>
    %cst_18 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %147 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f16>) -> tensor<1x5x32xf16>
    %148 = stablehlo.divide %147, %146 : tensor<1x5x32xf16>
    %149 = stablehlo.multiply %138, %148 : tensor<1x5x32xf16>
    %150 = stablehlo.multiply %149, %142 : tensor<1x5x32xf16>
    %151 = stablehlo.dot_general %150, %arg19, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x32xf16>, tensor<128x32xf16>) -> tensor<1x5x128xf16>
    %152 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %154 = stablehlo.add %151, %153 : tensor<1x5x128xf16>
    %155 = stablehlo.add %119, %154 : tensor<1x5x128xf16>
    %156 = stablehlo.convert %155 : (tensor<1x5x128xf16>) -> tensor<1x5x128xf32>
    %157 = stablehlo.multiply %156, %156 : tensor<1x5x128xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %158 = stablehlo.reduce(%157 init: %cst_19) applies stablehlo.add across dimensions = [2] : (tensor<1x5x128xf32>, tensor<f32>) -> tensor<1x5xf32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_20 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %160 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %161 = stablehlo.divide %159, %160 : tensor<1x5x1xf32>
    %cst_21 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %162 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %163 = stablehlo.add %161, %162 : tensor<1x5x1xf32>
    %164 = stablehlo.rsqrt %163 : tensor<1x5x1xf32>
    %165 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x128xf32>
    %166 = stablehlo.multiply %156, %165 : tensor<1x5x128xf32>
    %167 = stablehlo.convert %166 : (tensor<1x5x128xf32>) -> tensor<1x5x128xf16>
    %168 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %170 = stablehlo.multiply %167, %169 : tensor<1x5x128xf16>
    %171 = stablehlo.dot_general %170, %arg22, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<128x128xf16>) -> tensor<1x5x128xf16>
    %172 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %174 = stablehlo.add %171, %173 : tensor<1x5x128xf16>
    %175 = stablehlo.dot_general %170, %arg24, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<64x128xf16>) -> tensor<1x5x64xf16>
    %176 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %177 = stablehlo.broadcast_in_dim %176, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x5x64xf16>
    %178 = stablehlo.add %175, %177 : tensor<1x5x64xf16>
    %179 = stablehlo.dot_general %170, %arg26, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<64x128xf16>) -> tensor<1x5x64xf16>
    %180 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x5x64xf16>
    %182 = stablehlo.add %179, %181 : tensor<1x5x64xf16>
    %183 = stablehlo.reshape %174 : (tensor<1x5x128xf16>) -> tensor<1x5x4x32xf16>
    %184 = stablehlo.transpose %183, dims = [0, 2, 1, 3] : (tensor<1x5x4x32xf16>) -> tensor<1x4x5x32xf16>
    %185 = stablehlo.reshape %178 : (tensor<1x5x64xf16>) -> tensor<1x5x2x32xf16>
    %186 = stablehlo.transpose %185, dims = [0, 2, 1, 3] : (tensor<1x5x2x32xf16>) -> tensor<1x2x5x32xf16>
    %187 = stablehlo.reshape %182 : (tensor<1x5x64xf16>) -> tensor<1x5x2x32xf16>
    %188 = stablehlo.transpose %187, dims = [0, 2, 1, 3] : (tensor<1x5x2x32xf16>) -> tensor<1x2x5x32xf16>
    %189 = stablehlo.iota dim = 0 : tensor<16xf32>
    %cst_22 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %191 = stablehlo.multiply %190, %189 : tensor<16xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %192 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %193 = stablehlo.add %192, %191 : tensor<16xf32>
    %cst_24 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %194 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %195 = stablehlo.divide %193, %194 : tensor<16xf32>
    %cst_25 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %196 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %197 = stablehlo.power %196, %195 : tensor<16xf32>
    %cst_26 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %199 = stablehlo.divide %198, %197 : tensor<16xf32>
    %200 = stablehlo.convert %arg1 : (tensor<1x5xi32>) -> tensor<1x5xf32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %202 = stablehlo.broadcast_in_dim %199, dims = [2] : (tensor<16xf32>) -> tensor<1x1x16xf32>
    %203 = stablehlo.broadcast_in_dim %201, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x16xf32>
    %204 = stablehlo.broadcast_in_dim %202, dims = [0, 1, 2] : (tensor<1x1x16xf32>) -> tensor<1x5x16xf32>
    %205 = stablehlo.multiply %203, %204 : tensor<1x5x16xf32>
    %206 = stablehlo.concatenate %205, %205, dim = 2 : (tensor<1x5x16xf32>, tensor<1x5x16xf32>) -> tensor<1x5x32xf32>
    %207 = stablehlo.cosine %206 : tensor<1x5x32xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 2, 3] : (tensor<1x5x32xf32>) -> tensor<1x1x5x32xf32>
    %209 = stablehlo.convert %208 : (tensor<1x1x5x32xf32>) -> tensor<1x1x5x32xf16>
    %210 = stablehlo.sine %206 : tensor<1x5x32xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 2, 3] : (tensor<1x5x32xf32>) -> tensor<1x1x5x32xf32>
    %212 = stablehlo.convert %211 : (tensor<1x1x5x32xf32>) -> tensor<1x1x5x32xf16>
    %213 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %214 = stablehlo.multiply %184, %213 : tensor<1x4x5x32xf16>
    %215 = stablehlo.slice %184 [0:1, 0:4, 0:5, 0:16] : (tensor<1x4x5x32xf16>) -> tensor<1x4x5x16xf16>
    %216 = stablehlo.slice %184 [0:1, 0:4, 0:5, 16:32] : (tensor<1x4x5x32xf16>) -> tensor<1x4x5x16xf16>
    %217 = stablehlo.negate %216 : tensor<1x4x5x16xf16>
    %218 = stablehlo.concatenate %217, %215, dim = 3 : (tensor<1x4x5x16xf16>, tensor<1x4x5x16xf16>) -> tensor<1x4x5x32xf16>
    %219 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %220 = stablehlo.multiply %218, %219 : tensor<1x4x5x32xf16>
    %221 = stablehlo.add %214, %220 : tensor<1x4x5x32xf16>
    %222 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x2x5x32xf16>
    %223 = stablehlo.multiply %186, %222 : tensor<1x2x5x32xf16>
    %224 = stablehlo.slice %186 [0:1, 0:2, 0:5, 0:16] : (tensor<1x2x5x32xf16>) -> tensor<1x2x5x16xf16>
    %225 = stablehlo.slice %186 [0:1, 0:2, 0:5, 16:32] : (tensor<1x2x5x32xf16>) -> tensor<1x2x5x16xf16>
    %226 = stablehlo.negate %225 : tensor<1x2x5x16xf16>
    %227 = stablehlo.concatenate %226, %224, dim = 3 : (tensor<1x2x5x16xf16>, tensor<1x2x5x16xf16>) -> tensor<1x2x5x32xf16>
    %228 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2, 3] : (tensor<1x1x5x32xf16>) -> tensor<1x2x5x32xf16>
    %229 = stablehlo.multiply %227, %228 : tensor<1x2x5x32xf16>
    %230 = stablehlo.add %223, %229 : tensor<1x2x5x32xf16>
    %231 = stablehlo.slice %230 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %232 = stablehlo.slice %230 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %233 = stablehlo.slice %230 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %234 = stablehlo.slice %230 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %235 = stablehlo.concatenate %231, %232, %233, %234, dim = 1 : (tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %236 = stablehlo.slice %188 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %237 = stablehlo.slice %188 [0:1, 0:1, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %238 = stablehlo.slice %188 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %239 = stablehlo.slice %188 [0:1, 1:2, 0:5, 0:32] : (tensor<1x2x5x32xf16>) -> tensor<1x1x5x32xf16>
    %240 = stablehlo.concatenate %236, %237, %238, %239, dim = 1 : (tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>, tensor<1x1x5x32xf16>) -> tensor<1x4x5x32xf16>
    %241 = stablehlo.dot_general %221, %235, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x5x32xf16>, tensor<1x4x5x32xf16>) -> tensor<1x4x5x5xf16>
    %cst_27 = stablehlo.constant dense<1.767580e-01> : tensor<f16>
    %242 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f16>) -> tensor<1x4x5x5xf16>
    %243 = stablehlo.multiply %241, %242 : tensor<1x4x5x5xf16>
    %c_28 = stablehlo.constant dense<true> : tensor<i1>
    %244 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i1>) -> tensor<5x5xi1>
    %245 = call @tril(%244) : (tensor<5x5xi1>) -> tensor<5x5xi1>
    %246 = stablehlo.broadcast_in_dim %245, dims = [2, 3] : (tensor<5x5xi1>) -> tensor<1x1x5x5xi1>
    %cst_29 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %247 = call @_where(%246, %243, %cst_29) : (tensor<1x1x5x5xi1>, tensor<1x4x5x5xf16>, tensor<f16>) -> tensor<1x4x5x5xf16>
    %cst_30 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %248 = stablehlo.reduce(%247 init: %cst_30) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x5x5xf16>, tensor<f16>) -> tensor<1x4x5xf16>
    %cst_31 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %249 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f16>) -> tensor<1x4x5xf16>
    %250 = stablehlo.maximum %249, %248 : tensor<1x4x5xf16>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2] : (tensor<1x4x5xf16>) -> tensor<1x4x5x1xf16>
    %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1, 2, 3] : (tensor<1x4x5x1xf16>) -> tensor<1x4x5x5xf16>
    %253 = stablehlo.subtract %247, %252 : tensor<1x4x5x5xf16>
    %254 = stablehlo.exponential %253 : tensor<1x4x5x5xf16>
    %255 = stablehlo.convert %254 : (tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %256 = stablehlo.reduce(%255 init: %cst_32) applies stablehlo.add across dimensions = [3] : (tensor<1x4x5x5xf32>, tensor<f32>) -> tensor<1x4x5xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<1x4x5xf32>) -> tensor<1x4x5x1xf32>
    %258 = stablehlo.convert %257 : (tensor<1x4x5x1xf32>) -> tensor<1x4x5x1xf16>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 2, 3] : (tensor<1x4x5x1xf16>) -> tensor<1x4x5x5xf16>
    %260 = stablehlo.divide %254, %259 : tensor<1x4x5x5xf16>
    %261 = stablehlo.dot_general %260, %240, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x5x5xf16>, tensor<1x4x5x32xf16>) -> tensor<1x4x5x32xf16>
    %262 = stablehlo.transpose %261, dims = [0, 2, 1, 3] : (tensor<1x4x5x32xf16>) -> tensor<1x5x4x32xf16>
    %263 = stablehlo.reshape %262 : (tensor<1x5x4x32xf16>) -> tensor<1x5x128xf16>
    %264 = stablehlo.dot_general %263, %arg28, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<128x128xf16>) -> tensor<1x5x128xf16>
    %265 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %267 = stablehlo.add %264, %266 : tensor<1x5x128xf16>
    %268 = stablehlo.add %155, %267 : tensor<1x5x128xf16>
    %269 = stablehlo.convert %268 : (tensor<1x5x128xf16>) -> tensor<1x5x128xf32>
    %270 = stablehlo.multiply %269, %269 : tensor<1x5x128xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %271 = stablehlo.reduce(%270 init: %cst_33) applies stablehlo.add across dimensions = [2] : (tensor<1x5x128xf32>, tensor<f32>) -> tensor<1x5xf32>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_34 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %273 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %274 = stablehlo.divide %272, %273 : tensor<1x5x1xf32>
    %cst_35 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %275 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %276 = stablehlo.add %274, %275 : tensor<1x5x1xf32>
    %277 = stablehlo.rsqrt %276 : tensor<1x5x1xf32>
    %278 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x128xf32>
    %279 = stablehlo.multiply %269, %278 : tensor<1x5x128xf32>
    %280 = stablehlo.convert %279 : (tensor<1x5x128xf32>) -> tensor<1x5x128xf16>
    %281 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %282 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %283 = stablehlo.multiply %280, %282 : tensor<1x5x128xf16>
    %284 = stablehlo.dot_general %283, %arg31, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<32x128xf16>) -> tensor<1x5x32xf16>
    %285 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2] : (tensor<1x1x32xf16>) -> tensor<1x5x32xf16>
    %287 = stablehlo.add %284, %286 : tensor<1x5x32xf16>
    %288 = stablehlo.dot_general %283, %arg33, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<32x128xf16>) -> tensor<1x5x32xf16>
    %289 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2] : (tensor<1x1x32xf16>) -> tensor<1x5x32xf16>
    %291 = stablehlo.add %288, %290 : tensor<1x5x32xf16>
    %292 = stablehlo.negate %287 : tensor<1x5x32xf16>
    %293 = stablehlo.exponential %292 : tensor<1x5x32xf16>
    %cst_36 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %294 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f16>) -> tensor<1x5x32xf16>
    %295 = stablehlo.add %294, %293 : tensor<1x5x32xf16>
    %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %296 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f16>) -> tensor<1x5x32xf16>
    %297 = stablehlo.divide %296, %295 : tensor<1x5x32xf16>
    %298 = stablehlo.multiply %287, %297 : tensor<1x5x32xf16>
    %299 = stablehlo.multiply %298, %291 : tensor<1x5x32xf16>
    %300 = stablehlo.dot_general %299, %arg35, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x32xf16>, tensor<128x32xf16>) -> tensor<1x5x128xf16>
    %301 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %303 = stablehlo.add %300, %302 : tensor<1x5x128xf16>
    %304 = stablehlo.add %268, %303 : tensor<1x5x128xf16>
    %305 = stablehlo.convert %304 : (tensor<1x5x128xf16>) -> tensor<1x5x128xf32>
    %306 = stablehlo.multiply %305, %305 : tensor<1x5x128xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %307 = stablehlo.reduce(%306 init: %cst_38) applies stablehlo.add across dimensions = [2] : (tensor<1x5x128xf32>, tensor<f32>) -> tensor<1x5xf32>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_39 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %309 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %310 = stablehlo.divide %308, %309 : tensor<1x5x1xf32>
    %cst_40 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %311 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %312 = stablehlo.add %310, %311 : tensor<1x5x1xf32>
    %313 = stablehlo.rsqrt %312 : tensor<1x5x1xf32>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x128xf32>
    %315 = stablehlo.multiply %305, %314 : tensor<1x5x128xf32>
    %316 = stablehlo.convert %315 : (tensor<1x5x128xf32>) -> tensor<1x5x128xf16>
    %317 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2] : (tensor<1x1x128xf16>) -> tensor<1x5x128xf16>
    %319 = stablehlo.multiply %316, %318 : tensor<1x5x128xf16>
    %320 = stablehlo.dot_general %319, %arg4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x128xf16>, tensor<151665x128xf16>) -> tensor<1x5x151665xf16>
    return %320 : tensor<1x5x151665xf16>
  }
  func.func private @tril(%arg0: tensor<5x5xi1>) -> tensor<5x5xi1> {
    %0 = stablehlo.iota dim = 0 : tensor<5x5xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<5x5xi32>
    %2 = stablehlo.add %0, %1 : tensor<5x5xi32>
    %3 = stablehlo.iota dim = 1 : tensor<5x5xi32>
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<5x5xi32>, tensor<5x5xi32>) -> tensor<5x5xi1>
    %c_0 = stablehlo.constant dense<false> : tensor<i1>
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i1>) -> tensor<5x5xi1>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<5x5xi1>, tensor<5x5xi1>
    return %6 : tensor<5x5xi1>
  }
  func.func private @_where(%arg0: tensor<1x1x5x5xi1>, %arg1: tensor<1x4x5x5xf16>, %arg2: tensor<f16>) -> tensor<1x4x5x5xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x5x5xi1>) -> tensor<1x4x5x5xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<1x4x5x5xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<1x4x5x5xi1>, tensor<1x4x5x5xf16>
    return %2 : tensor<1x4x5x5xf16>
  }
}
