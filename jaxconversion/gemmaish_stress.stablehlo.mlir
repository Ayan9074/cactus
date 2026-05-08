module @jit_gemmaish_prefill attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8xi32>, %arg1: tensor<1x8xf16>, %arg2: tensor<1x8x8xf16>, %arg3: tensor<320x64xf16>, %arg4: tensor<64xf16>, %arg5: tensor<4x16x64xf16>, %arg6: tensor<2x2x64x16xf16>, %arg7: tensor<4x64x16xf16>, %arg8: tensor<2x64x160xf16>, %arg9: tensor<160x64xf16>, %arg10: tensor<64xf16>, %arg11: tensor<64xf16>, %arg12: tensor<64xf16>, %arg13: tensor<64xf16>, %arg14: tensor<4x16x64xf16>, %arg15: tensor<2x2x64x16xf16>, %arg16: tensor<4x64x16xf16>, %arg17: tensor<2x64x160xf16>, %arg18: tensor<160x64xf16>, %arg19: tensor<64xf16>, %arg20: tensor<64xf16>, %arg21: tensor<64xf16>, %arg22: tensor<64xf16>, %arg23: tensor<4x16x64xf16>, %arg24: tensor<2x2x64x16xf16>, %arg25: tensor<4x64x16xf16>, %arg26: tensor<2x64x160xf16>, %arg27: tensor<160x64xf16>, %arg28: tensor<64xf16>, %arg29: tensor<64xf16>, %arg30: tensor<64xf16>, %arg31: tensor<64xf16>, %arg32: tensor<4x16x64xf16>, %arg33: tensor<2x2x64x16xf16>, %arg34: tensor<4x64x16xf16>, %arg35: tensor<2x64x160xf16>, %arg36: tensor<160x64xf16>, %arg37: tensor<64xf16>, %arg38: tensor<64xf16>, %arg39: tensor<64xf16>, %arg40: tensor<64xf16>) -> (tensor<1x8x320xf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %c_0 = stablehlo.constant dense<320> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x8xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<1x8xi1>, tensor<1x8xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %6 = "stablehlo.gather"(%arg3, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<320x64xf16>, tensor<1x8x1xi32>) -> tensor<1x8x64xf16>
    %cst = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %8 = stablehlo.multiply %6, %7 : tensor<1x8x64xf16>
    %9 = stablehlo.convert %8 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<1x8x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%10 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_2 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %14 = stablehlo.divide %12, %13 : tensor<1x8x1xf32>
    %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x8x1xf32>
    %17 = stablehlo.rsqrt %16 : tensor<1x8x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %19 = stablehlo.multiply %9, %18 : tensor<1x8x64xf32>
    %20 = stablehlo.convert %19 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %21 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %23 = stablehlo.multiply %20, %22 : tensor<1x8x64xf16>
    %24 = stablehlo.dot_general %23, %arg7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x64xf16>, tensor<4x64x16xf16>) -> tensor<1x8x4x16xf16>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1, 3] : (tensor<1x8x4x16xf16>) -> tensor<1x4x8x16xf16>
    %26 = stablehlo.dot_general %arg6, %23, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x64x16xf16>, tensor<1x8x64xf16>) -> tensor<2x2x16x1x8xf16>
    %27 = stablehlo.transpose %26, dims = [3, 0, 4, 1, 2] : (tensor<2x2x16x1x8xf16>) -> tensor<1x2x8x2x16xf16>
    %28 = stablehlo.slice %27 [0:1, 0:1, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %29 = stablehlo.reshape %28 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %30 = stablehlo.transpose %29, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %31 = stablehlo.slice %27 [0:1, 1:2, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %32 = stablehlo.reshape %31 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %33 = stablehlo.transpose %32, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %34 = stablehlo.iota dim = 0 : tensor<16xf32>
    %cst_4 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %36 = stablehlo.divide %34, %35 : tensor<16xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %38 = stablehlo.power %37, %36 : tensor<16xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %40 = stablehlo.divide %39, %38 : tensor<16xf32>
    %41 = stablehlo.convert %arg1 : (tensor<1x8xf16>) -> tensor<1x8xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 2] : (tensor<1x8xf32>) -> tensor<1x1x8x1xf32>
    %43 = stablehlo.broadcast_in_dim %40, dims = [3] : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3] : (tensor<1x1x8x1xf32>) -> tensor<1x1x8x16xf32>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2, 3] : (tensor<1x1x1x16xf32>) -> tensor<1x1x8x16xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x1x8x16xf32>
    %47 = stablehlo.cosine %46 : tensor<1x1x8x16xf32>
    %48 = stablehlo.convert %47 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %49 = stablehlo.sine %46 : tensor<1x1x8x16xf32>
    %50 = stablehlo.convert %49 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %51 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %52 = stablehlo.multiply %25, %51 : tensor<1x4x8x16xf16>
    %53 = stablehlo.slice %25 [0:1, 0:4, 0:8, 0:8] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %54 = stablehlo.slice %25 [0:1, 0:4, 0:8, 8:16] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %55 = stablehlo.negate %54 : tensor<1x4x8x8xf16>
    %56 = stablehlo.concatenate %55, %53, dim = 3 : (tensor<1x4x8x8xf16>, tensor<1x4x8x8xf16>) -> tensor<1x4x8x16xf16>
    %57 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %58 = stablehlo.multiply %56, %57 : tensor<1x4x8x16xf16>
    %59 = stablehlo.add %52, %58 : tensor<1x4x8x16xf16>
    %60 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %61 = stablehlo.multiply %30, %60 : tensor<1x2x8x16xf16>
    %62 = stablehlo.slice %30 [0:1, 0:2, 0:8, 0:8] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %63 = stablehlo.slice %30 [0:1, 0:2, 0:8, 8:16] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %64 = stablehlo.negate %63 : tensor<1x2x8x8xf16>
    %65 = stablehlo.concatenate %64, %62, dim = 3 : (tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>) -> tensor<1x2x8x16xf16>
    %66 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %67 = stablehlo.multiply %65, %66 : tensor<1x2x8x16xf16>
    %68 = stablehlo.add %61, %67 : tensor<1x2x8x16xf16>
    %69 = stablehlo.slice %68 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %70 = stablehlo.slice %68 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %71 = stablehlo.slice %68 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %72 = stablehlo.slice %68 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %73 = stablehlo.concatenate %69, %70, %71, %72, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %74 = stablehlo.slice %33 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %75 = stablehlo.slice %33 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %76 = stablehlo.slice %33 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %77 = stablehlo.slice %33 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %78 = stablehlo.concatenate %74, %75, %76, %77, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %79 = stablehlo.dot_general %59, %73, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x16xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %cst_7 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %80 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f16>) -> tensor<1x4x8x8xf16>
    %81 = stablehlo.multiply %79, %80 : tensor<1x4x8x8xf16>
    %82 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x8xf16>) -> tensor<1x8x1xf16>
    %83 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<1x8xf16>) -> tensor<1x1x8xf16>
    %84 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<1x8x1xf16>) -> tensor<1x8x8xf16>
    %85 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2] : (tensor<1x1x8xf16>) -> tensor<1x8x8xf16>
    %86 = stablehlo.compare GE, %84, %85, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %87 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f16>) -> tensor<1x8x8xf16>
    %88 = stablehlo.compare GT, %arg2, %87, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %89 = stablehlo.and %86, %88 : tensor<1x8x8xi1>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0, 2, 3] : (tensor<1x8x8xi1>) -> tensor<1x1x8x8xi1>
    %cst_9 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %91 = call @_where(%90, %81, %cst_9) : (tensor<1x1x8x8xi1>, tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8x8xf16>
    %cst_10 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %92 = stablehlo.reduce(%91 init: %cst_10) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8xf16>
    %cst_11 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %93 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f16>) -> tensor<1x4x8xf16>
    %94 = stablehlo.maximum %93, %92 : tensor<1x4x8xf16>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2] : (tensor<1x4x8xf16>) -> tensor<1x4x8x1xf16>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %97 = stablehlo.subtract %91, %96 : tensor<1x4x8x8xf16>
    %98 = stablehlo.exponential %97 : tensor<1x4x8x8xf16>
    %99 = stablehlo.convert %98 : (tensor<1x4x8x8xf16>) -> tensor<1x4x8x8xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %100 = stablehlo.reduce(%99 init: %cst_12) applies stablehlo.add across dimensions = [3] : (tensor<1x4x8x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2] : (tensor<1x4x8xf32>) -> tensor<1x4x8x1xf32>
    %102 = stablehlo.convert %101 : (tensor<1x4x8x1xf32>) -> tensor<1x4x8x1xf16>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %104 = stablehlo.divide %98, %103 : tensor<1x4x8x8xf16>
    %105 = stablehlo.dot_general %104, %78, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x8xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x16xf16>
    %106 = stablehlo.transpose %105, dims = [0, 2, 1, 3] : (tensor<1x4x8x16xf16>) -> tensor<1x8x4x16xf16>
    %107 = stablehlo.dot_general %106, %arg5, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x16xf16>, tensor<4x16x64xf16>) -> tensor<1x8x64xf16>
    %108 = stablehlo.add %8, %107 : tensor<1x8x64xf16>
    %109 = stablehlo.convert %108 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %110 = stablehlo.multiply %109, %109 : tensor<1x8x64xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %111 = stablehlo.reduce(%110 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_14 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %113 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %114 = stablehlo.divide %112, %113 : tensor<1x8x1xf32>
    %cst_15 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %115 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %116 = stablehlo.add %114, %115 : tensor<1x8x1xf32>
    %117 = stablehlo.rsqrt %116 : tensor<1x8x1xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %119 = stablehlo.multiply %109, %118 : tensor<1x8x64xf32>
    %120 = stablehlo.convert %119 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %121 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %123 = stablehlo.multiply %120, %122 : tensor<1x8x64xf16>
    %cst_16 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %124 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %125 = stablehlo.multiply %124, %123 : tensor<1x8x64xf16>
    %126 = stablehlo.add %108, %125 : tensor<1x8x64xf16>
    %127 = stablehlo.convert %126 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %128 = stablehlo.multiply %127, %127 : tensor<1x8x64xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %129 = stablehlo.reduce(%128 init: %cst_17) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_18 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %131 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %132 = stablehlo.divide %130, %131 : tensor<1x8x1xf32>
    %cst_19 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %133 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %134 = stablehlo.add %132, %133 : tensor<1x8x1xf32>
    %135 = stablehlo.rsqrt %134 : tensor<1x8x1xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %137 = stablehlo.multiply %127, %136 : tensor<1x8x64xf32>
    %138 = stablehlo.convert %137 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %139 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %140 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %141 = stablehlo.multiply %138, %140 : tensor<1x8x64xf16>
    %142 = stablehlo.dot_general %arg8, %141, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x64x160xf16>, tensor<1x8x64xf16>) -> tensor<2x160x1x8xf16>
    %143 = stablehlo.transpose %142, dims = [2, 0, 3, 1] : (tensor<2x160x1x8xf16>) -> tensor<1x2x8x160xf16>
    %144 = stablehlo.slice %143 [0:1, 0:1, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %145 = stablehlo.reshape %144 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %146 = stablehlo.slice %143 [0:1, 1:2, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %147 = stablehlo.reshape %146 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %148 = stablehlo.negate %145 : tensor<1x8x160xf16>
    %149 = stablehlo.exponential %148 : tensor<1x8x160xf16>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %150 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %151 = stablehlo.add %150, %149 : tensor<1x8x160xf16>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %152 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %153 = stablehlo.divide %152, %151 : tensor<1x8x160xf16>
    %154 = stablehlo.multiply %145, %153 : tensor<1x8x160xf16>
    %155 = stablehlo.multiply %154, %147 : tensor<1x8x160xf16>
    %156 = stablehlo.dot_general %155, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x160xf16>, tensor<160x64xf16>) -> tensor<1x8x64xf16>
    %157 = stablehlo.add %126, %156 : tensor<1x8x64xf16>
    %158 = stablehlo.convert %157 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %159 = stablehlo.multiply %158, %158 : tensor<1x8x64xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %160 = stablehlo.reduce(%159 init: %cst_22) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_23 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %162 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %163 = stablehlo.divide %161, %162 : tensor<1x8x1xf32>
    %cst_24 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %164 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %165 = stablehlo.add %163, %164 : tensor<1x8x1xf32>
    %166 = stablehlo.rsqrt %165 : tensor<1x8x1xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %168 = stablehlo.multiply %158, %167 : tensor<1x8x64xf32>
    %169 = stablehlo.convert %168 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %170 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %172 = stablehlo.multiply %169, %171 : tensor<1x8x64xf16>
    %cst_25 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %173 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %174 = stablehlo.multiply %173, %172 : tensor<1x8x64xf16>
    %175 = stablehlo.add %157, %174 : tensor<1x8x64xf16>
    %cst_26 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %176 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %177 = stablehlo.compare GT, %175, %176, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_27 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %178 = call @_where_62(%177, %cst_27, %175) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %cst_28 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %179 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %180 = stablehlo.compare LT, %178, %179, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_29 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %181 = call @_where_62(%180, %cst_29, %178) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %182 = stablehlo.convert %181 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %183 = stablehlo.multiply %182, %182 : tensor<1x8x64xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %184 = stablehlo.reduce(%183 init: %cst_30) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_31 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %186 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %187 = stablehlo.divide %185, %186 : tensor<1x8x1xf32>
    %cst_32 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %188 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %189 = stablehlo.add %187, %188 : tensor<1x8x1xf32>
    %190 = stablehlo.rsqrt %189 : tensor<1x8x1xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %192 = stablehlo.multiply %182, %191 : tensor<1x8x64xf32>
    %193 = stablehlo.convert %192 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %194 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %195 = stablehlo.broadcast_in_dim %194, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %196 = stablehlo.multiply %193, %195 : tensor<1x8x64xf16>
    %197 = stablehlo.dot_general %196, %arg16, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x64xf16>, tensor<4x64x16xf16>) -> tensor<1x8x4x16xf16>
    %198 = stablehlo.transpose %197, dims = [0, 2, 1, 3] : (tensor<1x8x4x16xf16>) -> tensor<1x4x8x16xf16>
    %199 = stablehlo.dot_general %arg15, %196, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x64x16xf16>, tensor<1x8x64xf16>) -> tensor<2x2x16x1x8xf16>
    %200 = stablehlo.transpose %199, dims = [3, 0, 4, 1, 2] : (tensor<2x2x16x1x8xf16>) -> tensor<1x2x8x2x16xf16>
    %201 = stablehlo.slice %200 [0:1, 0:1, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %202 = stablehlo.reshape %201 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %203 = stablehlo.transpose %202, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %204 = stablehlo.slice %200 [0:1, 1:2, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %205 = stablehlo.reshape %204 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %206 = stablehlo.transpose %205, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %207 = stablehlo.iota dim = 0 : tensor<16xf32>
    %cst_33 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %208 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %209 = stablehlo.divide %207, %208 : tensor<16xf32>
    %cst_34 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %210 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %211 = stablehlo.power %210, %209 : tensor<16xf32>
    %cst_35 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %212 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %213 = stablehlo.divide %212, %211 : tensor<16xf32>
    %214 = stablehlo.convert %arg1 : (tensor<1x8xf16>) -> tensor<1x8xf32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [0, 2] : (tensor<1x8xf32>) -> tensor<1x1x8x1xf32>
    %216 = stablehlo.broadcast_in_dim %213, dims = [3] : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %217 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2, 3] : (tensor<1x1x8x1xf32>) -> tensor<1x1x8x16xf32>
    %218 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2, 3] : (tensor<1x1x1x16xf32>) -> tensor<1x1x8x16xf32>
    %219 = stablehlo.multiply %217, %218 : tensor<1x1x8x16xf32>
    %220 = stablehlo.cosine %219 : tensor<1x1x8x16xf32>
    %221 = stablehlo.convert %220 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %222 = stablehlo.sine %219 : tensor<1x1x8x16xf32>
    %223 = stablehlo.convert %222 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %224 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %225 = stablehlo.multiply %198, %224 : tensor<1x4x8x16xf16>
    %226 = stablehlo.slice %198 [0:1, 0:4, 0:8, 0:8] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %227 = stablehlo.slice %198 [0:1, 0:4, 0:8, 8:16] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %228 = stablehlo.negate %227 : tensor<1x4x8x8xf16>
    %229 = stablehlo.concatenate %228, %226, dim = 3 : (tensor<1x4x8x8xf16>, tensor<1x4x8x8xf16>) -> tensor<1x4x8x16xf16>
    %230 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %231 = stablehlo.multiply %229, %230 : tensor<1x4x8x16xf16>
    %232 = stablehlo.add %225, %231 : tensor<1x4x8x16xf16>
    %233 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %234 = stablehlo.multiply %203, %233 : tensor<1x2x8x16xf16>
    %235 = stablehlo.slice %203 [0:1, 0:2, 0:8, 0:8] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %236 = stablehlo.slice %203 [0:1, 0:2, 0:8, 8:16] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %237 = stablehlo.negate %236 : tensor<1x2x8x8xf16>
    %238 = stablehlo.concatenate %237, %235, dim = 3 : (tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>) -> tensor<1x2x8x16xf16>
    %239 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %240 = stablehlo.multiply %238, %239 : tensor<1x2x8x16xf16>
    %241 = stablehlo.add %234, %240 : tensor<1x2x8x16xf16>
    %242 = stablehlo.slice %241 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %243 = stablehlo.slice %241 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %244 = stablehlo.slice %241 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %245 = stablehlo.slice %241 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %246 = stablehlo.concatenate %242, %243, %244, %245, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %247 = stablehlo.slice %206 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %248 = stablehlo.slice %206 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %249 = stablehlo.slice %206 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %250 = stablehlo.slice %206 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %251 = stablehlo.concatenate %247, %248, %249, %250, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %252 = stablehlo.dot_general %232, %246, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x16xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %cst_36 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %253 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f16>) -> tensor<1x4x8x8xf16>
    %254 = stablehlo.multiply %252, %253 : tensor<1x4x8x8xf16>
    %255 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x8xf16>) -> tensor<1x8x1xf16>
    %256 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<1x8xf16>) -> tensor<1x1x8xf16>
    %257 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<1x8x1xf16>) -> tensor<1x8x8xf16>
    %258 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<1x1x8xf16>) -> tensor<1x8x8xf16>
    %259 = stablehlo.compare GE, %257, %258, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %260 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f16>) -> tensor<1x8x8xf16>
    %261 = stablehlo.compare GT, %arg2, %260, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %262 = stablehlo.and %259, %261 : tensor<1x8x8xi1>
    %263 = stablehlo.broadcast_in_dim %262, dims = [0, 2, 3] : (tensor<1x8x8xi1>) -> tensor<1x1x8x8xi1>
    %cst_38 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %264 = call @_where(%263, %254, %cst_38) : (tensor<1x1x8x8xi1>, tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8x8xf16>
    %cst_39 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %265 = stablehlo.reduce(%264 init: %cst_39) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8xf16>
    %cst_40 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %266 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f16>) -> tensor<1x4x8xf16>
    %267 = stablehlo.maximum %266, %265 : tensor<1x4x8xf16>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2] : (tensor<1x4x8xf16>) -> tensor<1x4x8x1xf16>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %270 = stablehlo.subtract %264, %269 : tensor<1x4x8x8xf16>
    %271 = stablehlo.exponential %270 : tensor<1x4x8x8xf16>
    %272 = stablehlo.convert %271 : (tensor<1x4x8x8xf16>) -> tensor<1x4x8x8xf32>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %273 = stablehlo.reduce(%272 init: %cst_41) applies stablehlo.add across dimensions = [3] : (tensor<1x4x8x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2] : (tensor<1x4x8xf32>) -> tensor<1x4x8x1xf32>
    %275 = stablehlo.convert %274 : (tensor<1x4x8x1xf32>) -> tensor<1x4x8x1xf16>
    %276 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %277 = stablehlo.divide %271, %276 : tensor<1x4x8x8xf16>
    %278 = stablehlo.dot_general %277, %251, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x8xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x16xf16>
    %279 = stablehlo.transpose %278, dims = [0, 2, 1, 3] : (tensor<1x4x8x16xf16>) -> tensor<1x8x4x16xf16>
    %280 = stablehlo.dot_general %279, %arg14, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x16xf16>, tensor<4x16x64xf16>) -> tensor<1x8x64xf16>
    %281 = stablehlo.add %181, %280 : tensor<1x8x64xf16>
    %282 = stablehlo.convert %281 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %283 = stablehlo.multiply %282, %282 : tensor<1x8x64xf32>
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %284 = stablehlo.reduce(%283 init: %cst_42) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %285 = stablehlo.broadcast_in_dim %284, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_43 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %286 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %287 = stablehlo.divide %285, %286 : tensor<1x8x1xf32>
    %cst_44 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %288 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %289 = stablehlo.add %287, %288 : tensor<1x8x1xf32>
    %290 = stablehlo.rsqrt %289 : tensor<1x8x1xf32>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %292 = stablehlo.multiply %282, %291 : tensor<1x8x64xf32>
    %293 = stablehlo.convert %292 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %294 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %296 = stablehlo.multiply %293, %295 : tensor<1x8x64xf16>
    %cst_45 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %297 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %298 = stablehlo.multiply %297, %296 : tensor<1x8x64xf16>
    %299 = stablehlo.add %281, %298 : tensor<1x8x64xf16>
    %300 = stablehlo.convert %299 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %301 = stablehlo.multiply %300, %300 : tensor<1x8x64xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %302 = stablehlo.reduce(%301 init: %cst_46) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_47 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %304 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %305 = stablehlo.divide %303, %304 : tensor<1x8x1xf32>
    %cst_48 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %306 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %307 = stablehlo.add %305, %306 : tensor<1x8x1xf32>
    %308 = stablehlo.rsqrt %307 : tensor<1x8x1xf32>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %310 = stablehlo.multiply %300, %309 : tensor<1x8x64xf32>
    %311 = stablehlo.convert %310 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %312 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %314 = stablehlo.multiply %311, %313 : tensor<1x8x64xf16>
    %315 = stablehlo.dot_general %arg17, %314, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x64x160xf16>, tensor<1x8x64xf16>) -> tensor<2x160x1x8xf16>
    %316 = stablehlo.transpose %315, dims = [2, 0, 3, 1] : (tensor<2x160x1x8xf16>) -> tensor<1x2x8x160xf16>
    %317 = stablehlo.slice %316 [0:1, 0:1, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %318 = stablehlo.reshape %317 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %319 = stablehlo.slice %316 [0:1, 1:2, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %320 = stablehlo.reshape %319 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %321 = stablehlo.negate %318 : tensor<1x8x160xf16>
    %322 = stablehlo.exponential %321 : tensor<1x8x160xf16>
    %cst_49 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %323 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %324 = stablehlo.add %323, %322 : tensor<1x8x160xf16>
    %cst_50 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %325 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %326 = stablehlo.divide %325, %324 : tensor<1x8x160xf16>
    %327 = stablehlo.multiply %318, %326 : tensor<1x8x160xf16>
    %328 = stablehlo.multiply %327, %320 : tensor<1x8x160xf16>
    %329 = stablehlo.dot_general %328, %arg18, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x160xf16>, tensor<160x64xf16>) -> tensor<1x8x64xf16>
    %330 = stablehlo.add %299, %329 : tensor<1x8x64xf16>
    %331 = stablehlo.convert %330 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %332 = stablehlo.multiply %331, %331 : tensor<1x8x64xf32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %333 = stablehlo.reduce(%332 init: %cst_51) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_52 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %335 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %336 = stablehlo.divide %334, %335 : tensor<1x8x1xf32>
    %cst_53 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %337 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %338 = stablehlo.add %336, %337 : tensor<1x8x1xf32>
    %339 = stablehlo.rsqrt %338 : tensor<1x8x1xf32>
    %340 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %341 = stablehlo.multiply %331, %340 : tensor<1x8x64xf32>
    %342 = stablehlo.convert %341 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %343 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %345 = stablehlo.multiply %342, %344 : tensor<1x8x64xf16>
    %cst_54 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %346 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %347 = stablehlo.multiply %346, %345 : tensor<1x8x64xf16>
    %348 = stablehlo.add %330, %347 : tensor<1x8x64xf16>
    %cst_55 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %349 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %350 = stablehlo.compare GT, %348, %349, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_56 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %351 = call @_where_62(%350, %cst_56, %348) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %cst_57 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %352 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %353 = stablehlo.compare LT, %351, %352, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_58 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %354 = call @_where_62(%353, %cst_58, %351) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %355 = stablehlo.convert %354 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %356 = stablehlo.multiply %355, %355 : tensor<1x8x64xf32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %357 = stablehlo.reduce(%356 init: %cst_59) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %358 = stablehlo.broadcast_in_dim %357, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_60 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %359 = stablehlo.broadcast_in_dim %cst_60, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %360 = stablehlo.divide %358, %359 : tensor<1x8x1xf32>
    %cst_61 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %361 = stablehlo.broadcast_in_dim %cst_61, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %362 = stablehlo.add %360, %361 : tensor<1x8x1xf32>
    %363 = stablehlo.rsqrt %362 : tensor<1x8x1xf32>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %365 = stablehlo.multiply %355, %364 : tensor<1x8x64xf32>
    %366 = stablehlo.convert %365 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %367 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %369 = stablehlo.multiply %366, %368 : tensor<1x8x64xf16>
    %370 = stablehlo.dot_general %369, %arg25, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x64xf16>, tensor<4x64x16xf16>) -> tensor<1x8x4x16xf16>
    %371 = stablehlo.transpose %370, dims = [0, 2, 1, 3] : (tensor<1x8x4x16xf16>) -> tensor<1x4x8x16xf16>
    %372 = stablehlo.dot_general %arg24, %369, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x64x16xf16>, tensor<1x8x64xf16>) -> tensor<2x2x16x1x8xf16>
    %373 = stablehlo.transpose %372, dims = [3, 0, 4, 1, 2] : (tensor<2x2x16x1x8xf16>) -> tensor<1x2x8x2x16xf16>
    %374 = stablehlo.slice %373 [0:1, 0:1, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %375 = stablehlo.reshape %374 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %376 = stablehlo.transpose %375, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %377 = stablehlo.slice %373 [0:1, 1:2, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %378 = stablehlo.reshape %377 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %379 = stablehlo.transpose %378, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %380 = stablehlo.iota dim = 0 : tensor<16xf32>
    %cst_62 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %381 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %382 = stablehlo.divide %380, %381 : tensor<16xf32>
    %cst_63 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %383 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %384 = stablehlo.power %383, %382 : tensor<16xf32>
    %cst_64 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %385 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %386 = stablehlo.divide %385, %384 : tensor<16xf32>
    %387 = stablehlo.convert %arg1 : (tensor<1x8xf16>) -> tensor<1x8xf32>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 2] : (tensor<1x8xf32>) -> tensor<1x1x8x1xf32>
    %389 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %390 = stablehlo.broadcast_in_dim %388, dims = [0, 1, 2, 3] : (tensor<1x1x8x1xf32>) -> tensor<1x1x8x16xf32>
    %391 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<1x1x1x16xf32>) -> tensor<1x1x8x16xf32>
    %392 = stablehlo.multiply %390, %391 : tensor<1x1x8x16xf32>
    %393 = stablehlo.cosine %392 : tensor<1x1x8x16xf32>
    %394 = stablehlo.convert %393 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %395 = stablehlo.sine %392 : tensor<1x1x8x16xf32>
    %396 = stablehlo.convert %395 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %397 = stablehlo.broadcast_in_dim %394, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %398 = stablehlo.multiply %371, %397 : tensor<1x4x8x16xf16>
    %399 = stablehlo.slice %371 [0:1, 0:4, 0:8, 0:8] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %400 = stablehlo.slice %371 [0:1, 0:4, 0:8, 8:16] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %401 = stablehlo.negate %400 : tensor<1x4x8x8xf16>
    %402 = stablehlo.concatenate %401, %399, dim = 3 : (tensor<1x4x8x8xf16>, tensor<1x4x8x8xf16>) -> tensor<1x4x8x16xf16>
    %403 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %404 = stablehlo.multiply %402, %403 : tensor<1x4x8x16xf16>
    %405 = stablehlo.add %398, %404 : tensor<1x4x8x16xf16>
    %406 = stablehlo.broadcast_in_dim %394, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %407 = stablehlo.multiply %376, %406 : tensor<1x2x8x16xf16>
    %408 = stablehlo.slice %376 [0:1, 0:2, 0:8, 0:8] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %409 = stablehlo.slice %376 [0:1, 0:2, 0:8, 8:16] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %410 = stablehlo.negate %409 : tensor<1x2x8x8xf16>
    %411 = stablehlo.concatenate %410, %408, dim = 3 : (tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>) -> tensor<1x2x8x16xf16>
    %412 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %413 = stablehlo.multiply %411, %412 : tensor<1x2x8x16xf16>
    %414 = stablehlo.add %407, %413 : tensor<1x2x8x16xf16>
    %415 = stablehlo.slice %414 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %416 = stablehlo.slice %414 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %417 = stablehlo.slice %414 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %418 = stablehlo.slice %414 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %419 = stablehlo.concatenate %415, %416, %417, %418, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %420 = stablehlo.slice %379 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %421 = stablehlo.slice %379 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %422 = stablehlo.slice %379 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %423 = stablehlo.slice %379 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %424 = stablehlo.concatenate %420, %421, %422, %423, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %425 = stablehlo.dot_general %405, %419, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x16xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %cst_65 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %426 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f16>) -> tensor<1x4x8x8xf16>
    %427 = stablehlo.multiply %425, %426 : tensor<1x4x8x8xf16>
    %428 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x8xf16>) -> tensor<1x8x1xf16>
    %429 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<1x8xf16>) -> tensor<1x1x8xf16>
    %430 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2] : (tensor<1x8x1xf16>) -> tensor<1x8x8xf16>
    %431 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2] : (tensor<1x1x8xf16>) -> tensor<1x8x8xf16>
    %432 = stablehlo.compare GE, %430, %431, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %433 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f16>) -> tensor<1x8x8xf16>
    %434 = stablehlo.compare GT, %arg2, %433, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %435 = stablehlo.and %432, %434 : tensor<1x8x8xi1>
    %436 = stablehlo.broadcast_in_dim %435, dims = [0, 2, 3] : (tensor<1x8x8xi1>) -> tensor<1x1x8x8xi1>
    %cst_67 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %437 = call @_where(%436, %427, %cst_67) : (tensor<1x1x8x8xi1>, tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8x8xf16>
    %cst_68 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %438 = stablehlo.reduce(%437 init: %cst_68) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8xf16>
    %cst_69 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %439 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f16>) -> tensor<1x4x8xf16>
    %440 = stablehlo.maximum %439, %438 : tensor<1x4x8xf16>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2] : (tensor<1x4x8xf16>) -> tensor<1x4x8x1xf16>
    %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %443 = stablehlo.subtract %437, %442 : tensor<1x4x8x8xf16>
    %444 = stablehlo.exponential %443 : tensor<1x4x8x8xf16>
    %445 = stablehlo.convert %444 : (tensor<1x4x8x8xf16>) -> tensor<1x4x8x8xf32>
    %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %446 = stablehlo.reduce(%445 init: %cst_70) applies stablehlo.add across dimensions = [3] : (tensor<1x4x8x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<1x4x8xf32>) -> tensor<1x4x8x1xf32>
    %448 = stablehlo.convert %447 : (tensor<1x4x8x1xf32>) -> tensor<1x4x8x1xf16>
    %449 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %450 = stablehlo.divide %444, %449 : tensor<1x4x8x8xf16>
    %451 = stablehlo.dot_general %450, %424, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x8xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x16xf16>
    %452 = stablehlo.transpose %451, dims = [0, 2, 1, 3] : (tensor<1x4x8x16xf16>) -> tensor<1x8x4x16xf16>
    %453 = stablehlo.dot_general %452, %arg23, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x16xf16>, tensor<4x16x64xf16>) -> tensor<1x8x64xf16>
    %454 = stablehlo.add %354, %453 : tensor<1x8x64xf16>
    %455 = stablehlo.convert %454 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %456 = stablehlo.multiply %455, %455 : tensor<1x8x64xf32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %457 = stablehlo.reduce(%456 init: %cst_71) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_72 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %459 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %460 = stablehlo.divide %458, %459 : tensor<1x8x1xf32>
    %cst_73 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %461 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %462 = stablehlo.add %460, %461 : tensor<1x8x1xf32>
    %463 = stablehlo.rsqrt %462 : tensor<1x8x1xf32>
    %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %465 = stablehlo.multiply %455, %464 : tensor<1x8x64xf32>
    %466 = stablehlo.convert %465 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %467 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %468 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %469 = stablehlo.multiply %466, %468 : tensor<1x8x64xf16>
    %cst_74 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %470 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %471 = stablehlo.multiply %470, %469 : tensor<1x8x64xf16>
    %472 = stablehlo.add %454, %471 : tensor<1x8x64xf16>
    %473 = stablehlo.convert %472 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %474 = stablehlo.multiply %473, %473 : tensor<1x8x64xf32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %475 = stablehlo.reduce(%474 init: %cst_75) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %476 = stablehlo.broadcast_in_dim %475, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_76 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %477 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %478 = stablehlo.divide %476, %477 : tensor<1x8x1xf32>
    %cst_77 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %479 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %480 = stablehlo.add %478, %479 : tensor<1x8x1xf32>
    %481 = stablehlo.rsqrt %480 : tensor<1x8x1xf32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %483 = stablehlo.multiply %473, %482 : tensor<1x8x64xf32>
    %484 = stablehlo.convert %483 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %485 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %486 = stablehlo.broadcast_in_dim %485, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %487 = stablehlo.multiply %484, %486 : tensor<1x8x64xf16>
    %488 = stablehlo.dot_general %arg26, %487, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x64x160xf16>, tensor<1x8x64xf16>) -> tensor<2x160x1x8xf16>
    %489 = stablehlo.transpose %488, dims = [2, 0, 3, 1] : (tensor<2x160x1x8xf16>) -> tensor<1x2x8x160xf16>
    %490 = stablehlo.slice %489 [0:1, 0:1, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %491 = stablehlo.reshape %490 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %492 = stablehlo.slice %489 [0:1, 1:2, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %493 = stablehlo.reshape %492 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %494 = stablehlo.negate %491 : tensor<1x8x160xf16>
    %495 = stablehlo.exponential %494 : tensor<1x8x160xf16>
    %cst_78 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %496 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %497 = stablehlo.add %496, %495 : tensor<1x8x160xf16>
    %cst_79 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %498 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %499 = stablehlo.divide %498, %497 : tensor<1x8x160xf16>
    %500 = stablehlo.multiply %491, %499 : tensor<1x8x160xf16>
    %501 = stablehlo.multiply %500, %493 : tensor<1x8x160xf16>
    %502 = stablehlo.dot_general %501, %arg27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x160xf16>, tensor<160x64xf16>) -> tensor<1x8x64xf16>
    %503 = stablehlo.add %472, %502 : tensor<1x8x64xf16>
    %504 = stablehlo.convert %503 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %505 = stablehlo.multiply %504, %504 : tensor<1x8x64xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %506 = stablehlo.reduce(%505 init: %cst_80) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %507 = stablehlo.broadcast_in_dim %506, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_81 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %508 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %509 = stablehlo.divide %507, %508 : tensor<1x8x1xf32>
    %cst_82 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %510 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %511 = stablehlo.add %509, %510 : tensor<1x8x1xf32>
    %512 = stablehlo.rsqrt %511 : tensor<1x8x1xf32>
    %513 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %514 = stablehlo.multiply %504, %513 : tensor<1x8x64xf32>
    %515 = stablehlo.convert %514 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %516 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %518 = stablehlo.multiply %515, %517 : tensor<1x8x64xf16>
    %cst_83 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %519 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %520 = stablehlo.multiply %519, %518 : tensor<1x8x64xf16>
    %521 = stablehlo.add %503, %520 : tensor<1x8x64xf16>
    %cst_84 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %522 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %523 = stablehlo.compare GT, %521, %522, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_85 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %524 = call @_where_62(%523, %cst_85, %521) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %cst_86 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %525 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %526 = stablehlo.compare LT, %524, %525, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_87 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %527 = call @_where_62(%526, %cst_87, %524) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %528 = stablehlo.convert %527 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %529 = stablehlo.multiply %528, %528 : tensor<1x8x64xf32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %530 = stablehlo.reduce(%529 init: %cst_88) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %531 = stablehlo.broadcast_in_dim %530, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_89 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %532 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %533 = stablehlo.divide %531, %532 : tensor<1x8x1xf32>
    %cst_90 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %534 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %535 = stablehlo.add %533, %534 : tensor<1x8x1xf32>
    %536 = stablehlo.rsqrt %535 : tensor<1x8x1xf32>
    %537 = stablehlo.broadcast_in_dim %536, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %538 = stablehlo.multiply %528, %537 : tensor<1x8x64xf32>
    %539 = stablehlo.convert %538 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %540 = stablehlo.broadcast_in_dim %arg37, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %542 = stablehlo.multiply %539, %541 : tensor<1x8x64xf16>
    %543 = stablehlo.dot_general %542, %arg34, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x64xf16>, tensor<4x64x16xf16>) -> tensor<1x8x4x16xf16>
    %544 = stablehlo.transpose %543, dims = [0, 2, 1, 3] : (tensor<1x8x4x16xf16>) -> tensor<1x4x8x16xf16>
    %545 = stablehlo.dot_general %arg33, %542, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x64x16xf16>, tensor<1x8x64xf16>) -> tensor<2x2x16x1x8xf16>
    %546 = stablehlo.transpose %545, dims = [3, 0, 4, 1, 2] : (tensor<2x2x16x1x8xf16>) -> tensor<1x2x8x2x16xf16>
    %547 = stablehlo.slice %546 [0:1, 0:1, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %548 = stablehlo.reshape %547 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %549 = stablehlo.transpose %548, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %550 = stablehlo.slice %546 [0:1, 1:2, 0:8, 0:2, 0:16] : (tensor<1x2x8x2x16xf16>) -> tensor<1x1x8x2x16xf16>
    %551 = stablehlo.reshape %550 : (tensor<1x1x8x2x16xf16>) -> tensor<1x8x2x16xf16>
    %552 = stablehlo.transpose %551, dims = [0, 2, 1, 3] : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %553 = stablehlo.iota dim = 0 : tensor<16xf32>
    %cst_91 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %554 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %555 = stablehlo.divide %553, %554 : tensor<16xf32>
    %cst_92 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %556 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %557 = stablehlo.power %556, %555 : tensor<16xf32>
    %cst_93 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %558 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %559 = stablehlo.divide %558, %557 : tensor<16xf32>
    %560 = stablehlo.convert %arg1 : (tensor<1x8xf16>) -> tensor<1x8xf32>
    %561 = stablehlo.broadcast_in_dim %560, dims = [0, 2] : (tensor<1x8xf32>) -> tensor<1x1x8x1xf32>
    %562 = stablehlo.broadcast_in_dim %559, dims = [3] : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %563 = stablehlo.broadcast_in_dim %561, dims = [0, 1, 2, 3] : (tensor<1x1x8x1xf32>) -> tensor<1x1x8x16xf32>
    %564 = stablehlo.broadcast_in_dim %562, dims = [0, 1, 2, 3] : (tensor<1x1x1x16xf32>) -> tensor<1x1x8x16xf32>
    %565 = stablehlo.multiply %563, %564 : tensor<1x1x8x16xf32>
    %566 = stablehlo.cosine %565 : tensor<1x1x8x16xf32>
    %567 = stablehlo.convert %566 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %568 = stablehlo.sine %565 : tensor<1x1x8x16xf32>
    %569 = stablehlo.convert %568 : (tensor<1x1x8x16xf32>) -> tensor<1x1x8x16xf16>
    %570 = stablehlo.broadcast_in_dim %567, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %571 = stablehlo.multiply %544, %570 : tensor<1x4x8x16xf16>
    %572 = stablehlo.slice %544 [0:1, 0:4, 0:8, 0:8] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %573 = stablehlo.slice %544 [0:1, 0:4, 0:8, 8:16] : (tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %574 = stablehlo.negate %573 : tensor<1x4x8x8xf16>
    %575 = stablehlo.concatenate %574, %572, dim = 3 : (tensor<1x4x8x8xf16>, tensor<1x4x8x8xf16>) -> tensor<1x4x8x16xf16>
    %576 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %577 = stablehlo.multiply %575, %576 : tensor<1x4x8x16xf16>
    %578 = stablehlo.add %571, %577 : tensor<1x4x8x16xf16>
    %579 = stablehlo.broadcast_in_dim %567, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %580 = stablehlo.multiply %549, %579 : tensor<1x2x8x16xf16>
    %581 = stablehlo.slice %549 [0:1, 0:2, 0:8, 0:8] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %582 = stablehlo.slice %549 [0:1, 0:2, 0:8, 8:16] : (tensor<1x2x8x16xf16>) -> tensor<1x2x8x8xf16>
    %583 = stablehlo.negate %582 : tensor<1x2x8x8xf16>
    %584 = stablehlo.concatenate %583, %581, dim = 3 : (tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>) -> tensor<1x2x8x16xf16>
    %585 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2, 3] : (tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf16>
    %586 = stablehlo.multiply %584, %585 : tensor<1x2x8x16xf16>
    %587 = stablehlo.add %580, %586 : tensor<1x2x8x16xf16>
    %588 = stablehlo.slice %587 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %589 = stablehlo.slice %587 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %590 = stablehlo.slice %587 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %591 = stablehlo.slice %587 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %592 = stablehlo.concatenate %588, %589, %590, %591, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %593 = stablehlo.slice %552 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %594 = stablehlo.slice %552 [0:1, 0:1, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %595 = stablehlo.slice %552 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %596 = stablehlo.slice %552 [0:1, 1:2, 0:8, 0:16] : (tensor<1x2x8x16xf16>) -> tensor<1x1x8x16xf16>
    %597 = stablehlo.concatenate %593, %594, %595, %596, dim = 1 : (tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x4x8x16xf16>
    %598 = stablehlo.dot_general %578, %592, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x16xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x8xf16>
    %cst_94 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %599 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f16>) -> tensor<1x4x8x8xf16>
    %600 = stablehlo.multiply %598, %599 : tensor<1x4x8x8xf16>
    %601 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x8xf16>) -> tensor<1x8x1xf16>
    %602 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<1x8xf16>) -> tensor<1x1x8xf16>
    %603 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2] : (tensor<1x8x1xf16>) -> tensor<1x8x8xf16>
    %604 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2] : (tensor<1x1x8xf16>) -> tensor<1x8x8xf16>
    %605 = stablehlo.compare GE, %603, %604, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %606 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f16>) -> tensor<1x8x8xf16>
    %607 = stablehlo.compare GT, %arg2, %606, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %608 = stablehlo.and %605, %607 : tensor<1x8x8xi1>
    %609 = stablehlo.broadcast_in_dim %608, dims = [0, 2, 3] : (tensor<1x8x8xi1>) -> tensor<1x1x8x8xi1>
    %cst_96 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %610 = call @_where(%609, %600, %cst_96) : (tensor<1x1x8x8xi1>, tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8x8xf16>
    %cst_97 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %611 = stablehlo.reduce(%610 init: %cst_97) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x8x8xf16>, tensor<f16>) -> tensor<1x4x8xf16>
    %cst_98 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %612 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<f16>) -> tensor<1x4x8xf16>
    %613 = stablehlo.maximum %612, %611 : tensor<1x4x8xf16>
    %614 = stablehlo.broadcast_in_dim %613, dims = [0, 1, 2] : (tensor<1x4x8xf16>) -> tensor<1x4x8x1xf16>
    %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %616 = stablehlo.subtract %610, %615 : tensor<1x4x8x8xf16>
    %617 = stablehlo.exponential %616 : tensor<1x4x8x8xf16>
    %618 = stablehlo.convert %617 : (tensor<1x4x8x8xf16>) -> tensor<1x4x8x8xf32>
    %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %619 = stablehlo.reduce(%618 init: %cst_99) applies stablehlo.add across dimensions = [3] : (tensor<1x4x8x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
    %620 = stablehlo.broadcast_in_dim %619, dims = [0, 1, 2] : (tensor<1x4x8xf32>) -> tensor<1x4x8x1xf32>
    %621 = stablehlo.convert %620 : (tensor<1x4x8x1xf32>) -> tensor<1x4x8x1xf16>
    %622 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf16>) -> tensor<1x4x8x8xf16>
    %623 = stablehlo.divide %617, %622 : tensor<1x4x8x8xf16>
    %624 = stablehlo.dot_general %623, %597, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x8xf16>, tensor<1x4x8x16xf16>) -> tensor<1x4x8x16xf16>
    %625 = stablehlo.transpose %624, dims = [0, 2, 1, 3] : (tensor<1x4x8x16xf16>) -> tensor<1x8x4x16xf16>
    %626 = stablehlo.dot_general %625, %arg32, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x16xf16>, tensor<4x16x64xf16>) -> tensor<1x8x64xf16>
    %627 = stablehlo.add %527, %626 : tensor<1x8x64xf16>
    %628 = stablehlo.convert %627 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %629 = stablehlo.multiply %628, %628 : tensor<1x8x64xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %630 = stablehlo.reduce(%629 init: %cst_100) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_101 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %632 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %633 = stablehlo.divide %631, %632 : tensor<1x8x1xf32>
    %cst_102 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %634 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %635 = stablehlo.add %633, %634 : tensor<1x8x1xf32>
    %636 = stablehlo.rsqrt %635 : tensor<1x8x1xf32>
    %637 = stablehlo.broadcast_in_dim %636, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %638 = stablehlo.multiply %628, %637 : tensor<1x8x64xf32>
    %639 = stablehlo.convert %638 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %640 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %641 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %642 = stablehlo.multiply %639, %641 : tensor<1x8x64xf16>
    %cst_103 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %643 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %644 = stablehlo.multiply %643, %642 : tensor<1x8x64xf16>
    %645 = stablehlo.add %627, %644 : tensor<1x8x64xf16>
    %646 = stablehlo.convert %645 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %647 = stablehlo.multiply %646, %646 : tensor<1x8x64xf32>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %648 = stablehlo.reduce(%647 init: %cst_104) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %649 = stablehlo.broadcast_in_dim %648, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_105 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %650 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %651 = stablehlo.divide %649, %650 : tensor<1x8x1xf32>
    %cst_106 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %652 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %653 = stablehlo.add %651, %652 : tensor<1x8x1xf32>
    %654 = stablehlo.rsqrt %653 : tensor<1x8x1xf32>
    %655 = stablehlo.broadcast_in_dim %654, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %656 = stablehlo.multiply %646, %655 : tensor<1x8x64xf32>
    %657 = stablehlo.convert %656 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %658 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %659 = stablehlo.broadcast_in_dim %658, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %660 = stablehlo.multiply %657, %659 : tensor<1x8x64xf16>
    %661 = stablehlo.dot_general %arg35, %660, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x64x160xf16>, tensor<1x8x64xf16>) -> tensor<2x160x1x8xf16>
    %662 = stablehlo.transpose %661, dims = [2, 0, 3, 1] : (tensor<2x160x1x8xf16>) -> tensor<1x2x8x160xf16>
    %663 = stablehlo.slice %662 [0:1, 0:1, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %664 = stablehlo.reshape %663 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %665 = stablehlo.slice %662 [0:1, 1:2, 0:8, 0:160] : (tensor<1x2x8x160xf16>) -> tensor<1x1x8x160xf16>
    %666 = stablehlo.reshape %665 : (tensor<1x1x8x160xf16>) -> tensor<1x8x160xf16>
    %667 = stablehlo.negate %664 : tensor<1x8x160xf16>
    %668 = stablehlo.exponential %667 : tensor<1x8x160xf16>
    %cst_107 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %669 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %670 = stablehlo.add %669, %668 : tensor<1x8x160xf16>
    %cst_108 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %671 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f16>) -> tensor<1x8x160xf16>
    %672 = stablehlo.divide %671, %670 : tensor<1x8x160xf16>
    %673 = stablehlo.multiply %664, %672 : tensor<1x8x160xf16>
    %674 = stablehlo.multiply %673, %666 : tensor<1x8x160xf16>
    %675 = stablehlo.dot_general %674, %arg36, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x160xf16>, tensor<160x64xf16>) -> tensor<1x8x64xf16>
    %676 = stablehlo.add %645, %675 : tensor<1x8x64xf16>
    %677 = stablehlo.convert %676 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %678 = stablehlo.multiply %677, %677 : tensor<1x8x64xf32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %679 = stablehlo.reduce(%678 init: %cst_109) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_110 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %681 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %682 = stablehlo.divide %680, %681 : tensor<1x8x1xf32>
    %cst_111 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %683 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %684 = stablehlo.add %682, %683 : tensor<1x8x1xf32>
    %685 = stablehlo.rsqrt %684 : tensor<1x8x1xf32>
    %686 = stablehlo.broadcast_in_dim %685, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %687 = stablehlo.multiply %677, %686 : tensor<1x8x64xf32>
    %688 = stablehlo.convert %687 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %689 = stablehlo.broadcast_in_dim %arg40, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %691 = stablehlo.multiply %688, %690 : tensor<1x8x64xf16>
    %cst_112 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %692 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %693 = stablehlo.multiply %692, %691 : tensor<1x8x64xf16>
    %694 = stablehlo.add %676, %693 : tensor<1x8x64xf16>
    %cst_113 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %695 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %696 = stablehlo.compare GT, %694, %695, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_114 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %697 = call @_where_62(%696, %cst_114, %694) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %cst_115 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %698 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %699 = stablehlo.compare LT, %697, %698, FLOAT : (tensor<1x8x64xf16>, tensor<1x8x64xf16>) -> tensor<1x8x64xi1>
    %cst_116 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %700 = call @_where_62(%699, %cst_116, %697) : (tensor<1x8x64xi1>, tensor<f16>, tensor<1x8x64xf16>) -> tensor<1x8x64xf16>
    %701 = stablehlo.convert %700 : (tensor<1x8x64xf16>) -> tensor<1x8x64xf32>
    %702 = stablehlo.multiply %701, %701 : tensor<1x8x64xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %703 = stablehlo.reduce(%702 init: %cst_117) applies stablehlo.add across dimensions = [2] : (tensor<1x8x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %704 = stablehlo.broadcast_in_dim %703, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_118 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %705 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %706 = stablehlo.divide %704, %705 : tensor<1x8x1xf32>
    %cst_119 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %707 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %708 = stablehlo.add %706, %707 : tensor<1x8x1xf32>
    %709 = stablehlo.rsqrt %708 : tensor<1x8x1xf32>
    %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
    %711 = stablehlo.multiply %701, %710 : tensor<1x8x64xf32>
    %712 = stablehlo.convert %711 : (tensor<1x8x64xf32>) -> tensor<1x8x64xf16>
    %713 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %714 = stablehlo.broadcast_in_dim %713, dims = [0, 1, 2] : (tensor<1x1x64xf16>) -> tensor<1x8x64xf16>
    %715 = stablehlo.multiply %712, %714 : tensor<1x8x64xf16>
    %716 = stablehlo.dot_general %715, %arg3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x64xf16>, tensor<320x64xf16>) -> tensor<1x8x320xf16>
    %cst_120 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %717 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f16>) -> tensor<1x8x320xf16>
    %718 = stablehlo.divide %716, %717 : tensor<1x8x320xf16>
    %719 = stablehlo.tanh %718 : tensor<1x8x320xf16>
    %cst_121 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %720 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f16>) -> tensor<1x8x320xf16>
    %721 = stablehlo.multiply %719, %720 : tensor<1x8x320xf16>
    return %721 : tensor<1x8x320xf16>
  }
  func.func private @_where(%arg0: tensor<1x1x8x8xi1>, %arg1: tensor<1x4x8x8xf16>, %arg2: tensor<f16>) -> tensor<1x4x8x8xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x4x8x8xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<1x4x8x8xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<1x4x8x8xi1>, tensor<1x4x8x8xf16>
    return %2 : tensor<1x4x8x8xf16>
  }
  func.func private @_where_62(%arg0: tensor<1x8x64xi1>, %arg1: tensor<f16>, %arg2: tensor<1x8x64xf16>) -> tensor<1x8x64xf16> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f16>) -> tensor<1x8x64xf16>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<1x8x64xi1>, tensor<1x8x64xf16>
    return %1 : tensor<1x8x64xf16>
  }
}
