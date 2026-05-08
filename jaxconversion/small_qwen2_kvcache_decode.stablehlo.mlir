module @jit_decode_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<2x1x2x5x32xf16>, %arg3: tensor<2x1x2x5x32xf16>, %arg4: tensor<151665x128xf16>, %arg5: tensor<128xf16>, %arg6: tensor<151665x128xf16>, %arg7: tensor<128xf16>, %arg8: tensor<128x128xf16>, %arg9: tensor<128xf16>, %arg10: tensor<64x128xf16>, %arg11: tensor<64xf16>, %arg12: tensor<64x128xf16>, %arg13: tensor<64xf16>, %arg14: tensor<128x128xf16>, %arg15: tensor<128xf16>, %arg16: tensor<128xf16>, %arg17: tensor<32x128xf16>, %arg18: tensor<32xf16>, %arg19: tensor<32x128xf16>, %arg20: tensor<32xf16>, %arg21: tensor<128x32xf16>, %arg22: tensor<128xf16>, %arg23: tensor<128xf16>, %arg24: tensor<128x128xf16>, %arg25: tensor<128xf16>, %arg26: tensor<64x128xf16>, %arg27: tensor<64xf16>, %arg28: tensor<64x128xf16>, %arg29: tensor<64xf16>, %arg30: tensor<128x128xf16>, %arg31: tensor<128xf16>, %arg32: tensor<128xf16>, %arg33: tensor<32x128xf16>, %arg34: tensor<32xf16>, %arg35: tensor<32x128xf16>, %arg36: tensor<32xf16>, %arg37: tensor<128x32xf16>, %arg38: tensor<128xf16>) -> (tensor<1x1x151665xf16> {jax.result_info = "result[0]"}, tensor<2x1x2x6x32xf16> {jax.result_info = "result[1]"}, tensor<2x1x2x6x32xf16> {jax.result_info = "result[2]"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %c_0 = stablehlo.constant dense<151665> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x1xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<1x1xi1>, tensor<1x1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %6 = "stablehlo.gather"(%arg4, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<151665x128xf16>, tensor<1x1x1xi32>) -> tensor<1x1x128xf16>
    %7 = stablehlo.slice %arg2 [0:1, 0:1, 0:2, 0:5, 0:32] : (tensor<2x1x2x5x32xf16>) -> tensor<1x1x2x5x32xf16>
    %8 = stablehlo.reshape %7 : (tensor<1x1x2x5x32xf16>) -> tensor<1x2x5x32xf16>
    %9 = stablehlo.slice %arg3 [0:1, 0:1, 0:2, 0:5, 0:32] : (tensor<2x1x2x5x32xf16>) -> tensor<1x1x2x5x32xf16>
    %10 = stablehlo.reshape %9 : (tensor<1x1x2x5x32xf16>) -> tensor<1x2x5x32xf16>
    %11 = stablehlo.convert %6 : (tensor<1x1x128xf16>) -> tensor<1x1x128xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x1x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %cst_1 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %16 = stablehlo.divide %14, %15 : tensor<1x1x1xf32>
    %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x1x1xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x128xf32>
    %21 = stablehlo.multiply %11, %20 : tensor<1x1x128xf32>
    %22 = stablehlo.convert %21 : (tensor<1x1x128xf32>) -> tensor<1x1x128xf16>
    %23 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %24 = stablehlo.multiply %22, %23 : tensor<1x1x128xf16>
    %25 = stablehlo.dot_general %24, %arg8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<128x128xf16>) -> tensor<1x1x128xf16>
    %26 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %27 = stablehlo.add %25, %26 : tensor<1x1x128xf16>
    %28 = stablehlo.dot_general %24, %arg10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<64x128xf16>) -> tensor<1x1x64xf16>
    %29 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %30 = stablehlo.add %28, %29 : tensor<1x1x64xf16>
    %31 = stablehlo.dot_general %24, %arg12, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<64x128xf16>) -> tensor<1x1x64xf16>
    %32 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %33 = stablehlo.add %31, %32 : tensor<1x1x64xf16>
    %34 = stablehlo.reshape %27 : (tensor<1x1x128xf16>) -> tensor<1x1x4x32xf16>
    %35 = stablehlo.transpose %34, dims = [0, 2, 1, 3] : (tensor<1x1x4x32xf16>) -> tensor<1x4x1x32xf16>
    %36 = stablehlo.reshape %30 : (tensor<1x1x64xf16>) -> tensor<1x1x2x32xf16>
    %37 = stablehlo.transpose %36, dims = [0, 2, 1, 3] : (tensor<1x1x2x32xf16>) -> tensor<1x2x1x32xf16>
    %38 = stablehlo.reshape %33 : (tensor<1x1x64xf16>) -> tensor<1x1x2x32xf16>
    %39 = stablehlo.transpose %38, dims = [0, 2, 1, 3] : (tensor<1x1x2x32xf16>) -> tensor<1x2x1x32xf16>
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
    %51 = stablehlo.convert %arg1 : (tensor<1x1xi32>) -> tensor<1x1xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %53 = stablehlo.broadcast_in_dim %50, dims = [2] : (tensor<16xf32>) -> tensor<1x1x16xf32>
    %54 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x16xf32>
    %55 = stablehlo.multiply %54, %53 : tensor<1x1x16xf32>
    %56 = stablehlo.concatenate %55, %55, dim = 2 : (tensor<1x1x16xf32>, tensor<1x1x16xf32>) -> tensor<1x1x32xf32>
    %57 = stablehlo.cosine %56 : tensor<1x1x32xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 2, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %59 = stablehlo.convert %58 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf16>
    %60 = stablehlo.sine %56 : tensor<1x1x32xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0, 2, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %62 = stablehlo.convert %61 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf16>
    %63 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x4x1x32xf16>
    %64 = stablehlo.multiply %35, %63 : tensor<1x4x1x32xf16>
    %65 = stablehlo.slice %35 [0:1, 0:4, 0:1, 0:16] : (tensor<1x4x1x32xf16>) -> tensor<1x4x1x16xf16>
    %66 = stablehlo.slice %35 [0:1, 0:4, 0:1, 16:32] : (tensor<1x4x1x32xf16>) -> tensor<1x4x1x16xf16>
    %67 = stablehlo.negate %66 : tensor<1x4x1x16xf16>
    %68 = stablehlo.concatenate %67, %65, dim = 3 : (tensor<1x4x1x16xf16>, tensor<1x4x1x16xf16>) -> tensor<1x4x1x32xf16>
    %69 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x4x1x32xf16>
    %70 = stablehlo.multiply %68, %69 : tensor<1x4x1x32xf16>
    %71 = stablehlo.add %64, %70 : tensor<1x4x1x32xf16>
    %72 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x2x1x32xf16>
    %73 = stablehlo.multiply %37, %72 : tensor<1x2x1x32xf16>
    %74 = stablehlo.slice %37 [0:1, 0:2, 0:1, 0:16] : (tensor<1x2x1x32xf16>) -> tensor<1x2x1x16xf16>
    %75 = stablehlo.slice %37 [0:1, 0:2, 0:1, 16:32] : (tensor<1x2x1x32xf16>) -> tensor<1x2x1x16xf16>
    %76 = stablehlo.negate %75 : tensor<1x2x1x16xf16>
    %77 = stablehlo.concatenate %76, %74, dim = 3 : (tensor<1x2x1x16xf16>, tensor<1x2x1x16xf16>) -> tensor<1x2x1x32xf16>
    %78 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x2x1x32xf16>
    %79 = stablehlo.multiply %77, %78 : tensor<1x2x1x32xf16>
    %80 = stablehlo.add %73, %79 : tensor<1x2x1x32xf16>
    %81 = stablehlo.concatenate %8, %80, dim = 2 : (tensor<1x2x5x32xf16>, tensor<1x2x1x32xf16>) -> tensor<1x2x6x32xf16>
    %82 = stablehlo.concatenate %10, %39, dim = 2 : (tensor<1x2x5x32xf16>, tensor<1x2x1x32xf16>) -> tensor<1x2x6x32xf16>
    %83 = stablehlo.slice %81 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %84 = stablehlo.slice %81 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %85 = stablehlo.slice %81 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %86 = stablehlo.slice %81 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %87 = stablehlo.concatenate %83, %84, %85, %86, dim = 1 : (tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>) -> tensor<1x4x6x32xf16>
    %88 = stablehlo.slice %82 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %89 = stablehlo.slice %82 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %90 = stablehlo.slice %82 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %91 = stablehlo.slice %82 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %92 = stablehlo.concatenate %88, %89, %90, %91, dim = 1 : (tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>) -> tensor<1x4x6x32xf16>
    %93 = stablehlo.dot_general %71, %87, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x1x32xf16>, tensor<1x4x6x32xf16>) -> tensor<1x4x1x6xf16>
    %cst_8 = stablehlo.constant dense<1.767580e-01> : tensor<f16>
    %94 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f16>) -> tensor<1x4x1x6xf16>
    %95 = stablehlo.multiply %93, %94 : tensor<1x4x1x6xf16>
    %cst_9 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %96 = stablehlo.reduce(%95 init: %cst_9) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x1x6xf16>, tensor<f16>) -> tensor<1x4x1xf16>
    %cst_10 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %97 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f16>) -> tensor<1x4x1xf16>
    %98 = stablehlo.maximum %97, %96 : tensor<1x4x1xf16>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2] : (tensor<1x4x1xf16>) -> tensor<1x4x1x1xf16>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf16>) -> tensor<1x4x1x6xf16>
    %101 = stablehlo.subtract %95, %100 : tensor<1x4x1x6xf16>
    %102 = stablehlo.exponential %101 : tensor<1x4x1x6xf16>
    %103 = stablehlo.convert %102 : (tensor<1x4x1x6xf16>) -> tensor<1x4x1x6xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %104 = stablehlo.reduce(%103 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x4x1x6xf32>, tensor<f32>) -> tensor<1x4x1xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x1x1xf32>
    %106 = stablehlo.convert %105 : (tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf16>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf16>) -> tensor<1x4x1x6xf16>
    %108 = stablehlo.divide %102, %107 : tensor<1x4x1x6xf16>
    %109 = stablehlo.dot_general %108, %92, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x1x6xf16>, tensor<1x4x6x32xf16>) -> tensor<1x4x1x32xf16>
    %110 = stablehlo.transpose %109, dims = [0, 2, 1, 3] : (tensor<1x4x1x32xf16>) -> tensor<1x1x4x32xf16>
    %111 = stablehlo.reshape %110 : (tensor<1x1x4x32xf16>) -> tensor<1x1x128xf16>
    %112 = stablehlo.dot_general %111, %arg14, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<128x128xf16>) -> tensor<1x1x128xf16>
    %113 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %114 = stablehlo.add %112, %113 : tensor<1x1x128xf16>
    %115 = stablehlo.add %6, %114 : tensor<1x1x128xf16>
    %116 = stablehlo.convert %115 : (tensor<1x1x128xf16>) -> tensor<1x1x128xf32>
    %117 = stablehlo.multiply %116, %116 : tensor<1x1x128xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %118 = stablehlo.reduce(%117 init: %cst_12) applies stablehlo.add across dimensions = [2] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %cst_13 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %120 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %121 = stablehlo.divide %119, %120 : tensor<1x1x1xf32>
    %cst_14 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %122 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %123 = stablehlo.add %121, %122 : tensor<1x1x1xf32>
    %124 = stablehlo.rsqrt %123 : tensor<1x1x1xf32>
    %125 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x128xf32>
    %126 = stablehlo.multiply %116, %125 : tensor<1x1x128xf32>
    %127 = stablehlo.convert %126 : (tensor<1x1x128xf32>) -> tensor<1x1x128xf16>
    %128 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %129 = stablehlo.multiply %127, %128 : tensor<1x1x128xf16>
    %130 = stablehlo.dot_general %129, %arg17, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<32x128xf16>) -> tensor<1x1x32xf16>
    %131 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %132 = stablehlo.add %130, %131 : tensor<1x1x32xf16>
    %133 = stablehlo.dot_general %129, %arg19, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<32x128xf16>) -> tensor<1x1x32xf16>
    %134 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %135 = stablehlo.add %133, %134 : tensor<1x1x32xf16>
    %136 = stablehlo.negate %132 : tensor<1x1x32xf16>
    %137 = stablehlo.exponential %136 : tensor<1x1x32xf16>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %138 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f16>) -> tensor<1x1x32xf16>
    %139 = stablehlo.add %138, %137 : tensor<1x1x32xf16>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %140 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f16>) -> tensor<1x1x32xf16>
    %141 = stablehlo.divide %140, %139 : tensor<1x1x32xf16>
    %142 = stablehlo.multiply %132, %141 : tensor<1x1x32xf16>
    %143 = stablehlo.multiply %142, %135 : tensor<1x1x32xf16>
    %144 = stablehlo.dot_general %143, %arg21, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xf16>, tensor<128x32xf16>) -> tensor<1x1x128xf16>
    %145 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %146 = stablehlo.add %144, %145 : tensor<1x1x128xf16>
    %147 = stablehlo.add %115, %146 : tensor<1x1x128xf16>
    %148 = stablehlo.slice %arg2 [1:2, 0:1, 0:2, 0:5, 0:32] : (tensor<2x1x2x5x32xf16>) -> tensor<1x1x2x5x32xf16>
    %149 = stablehlo.reshape %148 : (tensor<1x1x2x5x32xf16>) -> tensor<1x2x5x32xf16>
    %150 = stablehlo.slice %arg3 [1:2, 0:1, 0:2, 0:5, 0:32] : (tensor<2x1x2x5x32xf16>) -> tensor<1x1x2x5x32xf16>
    %151 = stablehlo.reshape %150 : (tensor<1x1x2x5x32xf16>) -> tensor<1x2x5x32xf16>
    %152 = stablehlo.convert %147 : (tensor<1x1x128xf16>) -> tensor<1x1x128xf32>
    %153 = stablehlo.multiply %152, %152 : tensor<1x1x128xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %154 = stablehlo.reduce(%153 init: %cst_17) applies stablehlo.add across dimensions = [2] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1xf32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %cst_18 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %156 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %157 = stablehlo.divide %155, %156 : tensor<1x1x1xf32>
    %cst_19 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %158 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %159 = stablehlo.add %157, %158 : tensor<1x1x1xf32>
    %160 = stablehlo.rsqrt %159 : tensor<1x1x1xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x128xf32>
    %162 = stablehlo.multiply %152, %161 : tensor<1x1x128xf32>
    %163 = stablehlo.convert %162 : (tensor<1x1x128xf32>) -> tensor<1x1x128xf16>
    %164 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %165 = stablehlo.multiply %163, %164 : tensor<1x1x128xf16>
    %166 = stablehlo.dot_general %165, %arg24, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<128x128xf16>) -> tensor<1x1x128xf16>
    %167 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %168 = stablehlo.add %166, %167 : tensor<1x1x128xf16>
    %169 = stablehlo.dot_general %165, %arg26, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<64x128xf16>) -> tensor<1x1x64xf16>
    %170 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %171 = stablehlo.add %169, %170 : tensor<1x1x64xf16>
    %172 = stablehlo.dot_general %165, %arg28, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<64x128xf16>) -> tensor<1x1x64xf16>
    %173 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<64xf16>) -> tensor<1x1x64xf16>
    %174 = stablehlo.add %172, %173 : tensor<1x1x64xf16>
    %175 = stablehlo.reshape %168 : (tensor<1x1x128xf16>) -> tensor<1x1x4x32xf16>
    %176 = stablehlo.transpose %175, dims = [0, 2, 1, 3] : (tensor<1x1x4x32xf16>) -> tensor<1x4x1x32xf16>
    %177 = stablehlo.reshape %171 : (tensor<1x1x64xf16>) -> tensor<1x1x2x32xf16>
    %178 = stablehlo.transpose %177, dims = [0, 2, 1, 3] : (tensor<1x1x2x32xf16>) -> tensor<1x2x1x32xf16>
    %179 = stablehlo.reshape %174 : (tensor<1x1x64xf16>) -> tensor<1x1x2x32xf16>
    %180 = stablehlo.transpose %179, dims = [0, 2, 1, 3] : (tensor<1x1x2x32xf16>) -> tensor<1x2x1x32xf16>
    %181 = stablehlo.iota dim = 0 : tensor<16xf32>
    %cst_20 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %182 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %183 = stablehlo.multiply %182, %181 : tensor<16xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %184 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %185 = stablehlo.add %184, %183 : tensor<16xf32>
    %cst_22 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %186 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %187 = stablehlo.divide %185, %186 : tensor<16xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %188 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %189 = stablehlo.power %188, %187 : tensor<16xf32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %191 = stablehlo.divide %190, %189 : tensor<16xf32>
    %192 = stablehlo.convert %arg1 : (tensor<1x1xi32>) -> tensor<1x1xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %194 = stablehlo.broadcast_in_dim %191, dims = [2] : (tensor<16xf32>) -> tensor<1x1x16xf32>
    %195 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x16xf32>
    %196 = stablehlo.multiply %195, %194 : tensor<1x1x16xf32>
    %197 = stablehlo.concatenate %196, %196, dim = 2 : (tensor<1x1x16xf32>, tensor<1x1x16xf32>) -> tensor<1x1x32xf32>
    %198 = stablehlo.cosine %197 : tensor<1x1x32xf32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0, 2, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %200 = stablehlo.convert %199 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf16>
    %201 = stablehlo.sine %197 : tensor<1x1x32xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0, 2, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %203 = stablehlo.convert %202 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf16>
    %204 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x4x1x32xf16>
    %205 = stablehlo.multiply %176, %204 : tensor<1x4x1x32xf16>
    %206 = stablehlo.slice %176 [0:1, 0:4, 0:1, 0:16] : (tensor<1x4x1x32xf16>) -> tensor<1x4x1x16xf16>
    %207 = stablehlo.slice %176 [0:1, 0:4, 0:1, 16:32] : (tensor<1x4x1x32xf16>) -> tensor<1x4x1x16xf16>
    %208 = stablehlo.negate %207 : tensor<1x4x1x16xf16>
    %209 = stablehlo.concatenate %208, %206, dim = 3 : (tensor<1x4x1x16xf16>, tensor<1x4x1x16xf16>) -> tensor<1x4x1x32xf16>
    %210 = stablehlo.broadcast_in_dim %203, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x4x1x32xf16>
    %211 = stablehlo.multiply %209, %210 : tensor<1x4x1x32xf16>
    %212 = stablehlo.add %205, %211 : tensor<1x4x1x32xf16>
    %213 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x2x1x32xf16>
    %214 = stablehlo.multiply %178, %213 : tensor<1x2x1x32xf16>
    %215 = stablehlo.slice %178 [0:1, 0:2, 0:1, 0:16] : (tensor<1x2x1x32xf16>) -> tensor<1x2x1x16xf16>
    %216 = stablehlo.slice %178 [0:1, 0:2, 0:1, 16:32] : (tensor<1x2x1x32xf16>) -> tensor<1x2x1x16xf16>
    %217 = stablehlo.negate %216 : tensor<1x2x1x16xf16>
    %218 = stablehlo.concatenate %217, %215, dim = 3 : (tensor<1x2x1x16xf16>, tensor<1x2x1x16xf16>) -> tensor<1x2x1x32xf16>
    %219 = stablehlo.broadcast_in_dim %203, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf16>) -> tensor<1x2x1x32xf16>
    %220 = stablehlo.multiply %218, %219 : tensor<1x2x1x32xf16>
    %221 = stablehlo.add %214, %220 : tensor<1x2x1x32xf16>
    %222 = stablehlo.concatenate %149, %221, dim = 2 : (tensor<1x2x5x32xf16>, tensor<1x2x1x32xf16>) -> tensor<1x2x6x32xf16>
    %223 = stablehlo.concatenate %151, %180, dim = 2 : (tensor<1x2x5x32xf16>, tensor<1x2x1x32xf16>) -> tensor<1x2x6x32xf16>
    %224 = stablehlo.slice %222 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %225 = stablehlo.slice %222 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %226 = stablehlo.slice %222 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %227 = stablehlo.slice %222 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %228 = stablehlo.concatenate %224, %225, %226, %227, dim = 1 : (tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>) -> tensor<1x4x6x32xf16>
    %229 = stablehlo.slice %223 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %230 = stablehlo.slice %223 [0:1, 0:1, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %231 = stablehlo.slice %223 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %232 = stablehlo.slice %223 [0:1, 1:2, 0:6, 0:32] : (tensor<1x2x6x32xf16>) -> tensor<1x1x6x32xf16>
    %233 = stablehlo.concatenate %229, %230, %231, %232, dim = 1 : (tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>, tensor<1x1x6x32xf16>) -> tensor<1x4x6x32xf16>
    %234 = stablehlo.dot_general %212, %228, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x1x32xf16>, tensor<1x4x6x32xf16>) -> tensor<1x4x1x6xf16>
    %cst_25 = stablehlo.constant dense<1.767580e-01> : tensor<f16>
    %235 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f16>) -> tensor<1x4x1x6xf16>
    %236 = stablehlo.multiply %234, %235 : tensor<1x4x1x6xf16>
    %cst_26 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %237 = stablehlo.reduce(%236 init: %cst_26) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x1x6xf16>, tensor<f16>) -> tensor<1x4x1xf16>
    %cst_27 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %238 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f16>) -> tensor<1x4x1xf16>
    %239 = stablehlo.maximum %238, %237 : tensor<1x4x1xf16>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2] : (tensor<1x4x1xf16>) -> tensor<1x4x1x1xf16>
    %241 = stablehlo.broadcast_in_dim %240, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf16>) -> tensor<1x4x1x6xf16>
    %242 = stablehlo.subtract %236, %241 : tensor<1x4x1x6xf16>
    %243 = stablehlo.exponential %242 : tensor<1x4x1x6xf16>
    %244 = stablehlo.convert %243 : (tensor<1x4x1x6xf16>) -> tensor<1x4x1x6xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %245 = stablehlo.reduce(%244 init: %cst_28) applies stablehlo.add across dimensions = [3] : (tensor<1x4x1x6xf32>, tensor<f32>) -> tensor<1x4x1xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x1x1xf32>
    %247 = stablehlo.convert %246 : (tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf16>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf16>) -> tensor<1x4x1x6xf16>
    %249 = stablehlo.divide %243, %248 : tensor<1x4x1x6xf16>
    %250 = stablehlo.dot_general %249, %233, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4x1x6xf16>, tensor<1x4x6x32xf16>) -> tensor<1x4x1x32xf16>
    %251 = stablehlo.transpose %250, dims = [0, 2, 1, 3] : (tensor<1x4x1x32xf16>) -> tensor<1x1x4x32xf16>
    %252 = stablehlo.reshape %251 : (tensor<1x1x4x32xf16>) -> tensor<1x1x128xf16>
    %253 = stablehlo.dot_general %252, %arg30, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<128x128xf16>) -> tensor<1x1x128xf16>
    %254 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %255 = stablehlo.add %253, %254 : tensor<1x1x128xf16>
    %256 = stablehlo.add %147, %255 : tensor<1x1x128xf16>
    %257 = stablehlo.convert %256 : (tensor<1x1x128xf16>) -> tensor<1x1x128xf32>
    %258 = stablehlo.multiply %257, %257 : tensor<1x1x128xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %259 = stablehlo.reduce(%258 init: %cst_29) applies stablehlo.add across dimensions = [2] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1xf32>
    %260 = stablehlo.broadcast_in_dim %259, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %cst_30 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %261 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %262 = stablehlo.divide %260, %261 : tensor<1x1x1xf32>
    %cst_31 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %263 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %264 = stablehlo.add %262, %263 : tensor<1x1x1xf32>
    %265 = stablehlo.rsqrt %264 : tensor<1x1x1xf32>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x128xf32>
    %267 = stablehlo.multiply %257, %266 : tensor<1x1x128xf32>
    %268 = stablehlo.convert %267 : (tensor<1x1x128xf32>) -> tensor<1x1x128xf16>
    %269 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %270 = stablehlo.multiply %268, %269 : tensor<1x1x128xf16>
    %271 = stablehlo.dot_general %270, %arg33, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<32x128xf16>) -> tensor<1x1x32xf16>
    %272 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %273 = stablehlo.add %271, %272 : tensor<1x1x32xf16>
    %274 = stablehlo.dot_general %270, %arg35, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<32x128xf16>) -> tensor<1x1x32xf16>
    %275 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<32xf16>) -> tensor<1x1x32xf16>
    %276 = stablehlo.add %274, %275 : tensor<1x1x32xf16>
    %277 = stablehlo.negate %273 : tensor<1x1x32xf16>
    %278 = stablehlo.exponential %277 : tensor<1x1x32xf16>
    %cst_32 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %279 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f16>) -> tensor<1x1x32xf16>
    %280 = stablehlo.add %279, %278 : tensor<1x1x32xf16>
    %cst_33 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %281 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f16>) -> tensor<1x1x32xf16>
    %282 = stablehlo.divide %281, %280 : tensor<1x1x32xf16>
    %283 = stablehlo.multiply %273, %282 : tensor<1x1x32xf16>
    %284 = stablehlo.multiply %283, %276 : tensor<1x1x32xf16>
    %285 = stablehlo.dot_general %284, %arg37, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xf16>, tensor<128x32xf16>) -> tensor<1x1x128xf16>
    %286 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %287 = stablehlo.add %285, %286 : tensor<1x1x128xf16>
    %288 = stablehlo.add %256, %287 : tensor<1x1x128xf16>
    %289 = stablehlo.convert %288 : (tensor<1x1x128xf16>) -> tensor<1x1x128xf32>
    %290 = stablehlo.multiply %289, %289 : tensor<1x1x128xf32>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %291 = stablehlo.reduce(%290 init: %cst_34) applies stablehlo.add across dimensions = [2] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1xf32>
    %292 = stablehlo.broadcast_in_dim %291, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %cst_35 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %293 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %294 = stablehlo.divide %292, %293 : tensor<1x1x1xf32>
    %cst_36 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %295 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %296 = stablehlo.add %294, %295 : tensor<1x1x1xf32>
    %297 = stablehlo.rsqrt %296 : tensor<1x1x1xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x128xf32>
    %299 = stablehlo.multiply %289, %298 : tensor<1x1x128xf32>
    %300 = stablehlo.convert %299 : (tensor<1x1x128xf32>) -> tensor<1x1x128xf16>
    %301 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<128xf16>) -> tensor<1x1x128xf16>
    %302 = stablehlo.multiply %300, %301 : tensor<1x1x128xf16>
    %303 = stablehlo.dot_general %302, %arg6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1x128xf16>, tensor<151665x128xf16>) -> tensor<1x1x151665xf16>
    %304 = stablehlo.broadcast_in_dim %81, dims = [1, 2, 3, 4] : (tensor<1x2x6x32xf16>) -> tensor<1x1x2x6x32xf16>
    %305 = stablehlo.broadcast_in_dim %222, dims = [1, 2, 3, 4] : (tensor<1x2x6x32xf16>) -> tensor<1x1x2x6x32xf16>
    %306 = stablehlo.concatenate %304, %305, dim = 0 : (tensor<1x1x2x6x32xf16>, tensor<1x1x2x6x32xf16>) -> tensor<2x1x2x6x32xf16>
    %307 = stablehlo.broadcast_in_dim %82, dims = [1, 2, 3, 4] : (tensor<1x2x6x32xf16>) -> tensor<1x1x2x6x32xf16>
    %308 = stablehlo.broadcast_in_dim %223, dims = [1, 2, 3, 4] : (tensor<1x2x6x32xf16>) -> tensor<1x1x2x6x32xf16>
    %309 = stablehlo.concatenate %307, %308, dim = 0 : (tensor<1x1x2x6x32xf16>, tensor<1x1x2x6x32xf16>) -> tensor<2x1x2x6x32xf16>
    return %303, %306, %309 : tensor<1x1x151665xf16>, tensor<2x1x2x6x32xf16>, tensor<2x1x2x6x32xf16>
  }
}
