module @jit_gemmaish_prefill attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x16xi32>, %arg1: tensor<2x16xf16>, %arg2: tensor<2x16x16xf16>, %arg3: tensor<1024x512xf16>, %arg4: tensor<512xf16>, %arg5: tensor<4x128x512xf16>, %arg6: tensor<2x2x512x128xf16>, %arg7: tensor<4x512x128xf16>, %arg8: tensor<2x512x1024xf16>, %arg9: tensor<1024x512xf16>, %arg10: tensor<512xf16>, %arg11: tensor<512xf16>, %arg12: tensor<512xf16>, %arg13: tensor<512xf16>, %arg14: tensor<4x128xf16>, %arg15: tensor<2x128xf16>, %arg16: tensor<4x128x512xf16>, %arg17: tensor<2x2x512x128xf16>, %arg18: tensor<4x512x128xf16>, %arg19: tensor<2x512x1024xf16>, %arg20: tensor<1024x512xf16>, %arg21: tensor<512xf16>, %arg22: tensor<512xf16>, %arg23: tensor<512xf16>, %arg24: tensor<512xf16>, %arg25: tensor<4x128xf16>, %arg26: tensor<2x128xf16>, %arg27: tensor<4x128x512xf16>, %arg28: tensor<2x2x512x128xf16>, %arg29: tensor<4x512x128xf16>, %arg30: tensor<2x512x1024xf16>, %arg31: tensor<1024x512xf16>, %arg32: tensor<512xf16>, %arg33: tensor<512xf16>, %arg34: tensor<512xf16>, %arg35: tensor<512xf16>, %arg36: tensor<4x128xf16>, %arg37: tensor<2x128xf16>, %arg38: tensor<4x128x512xf16>, %arg39: tensor<2x2x512x128xf16>, %arg40: tensor<4x512x128xf16>, %arg41: tensor<2x512x1024xf16>, %arg42: tensor<1024x512xf16>, %arg43: tensor<512xf16>, %arg44: tensor<512xf16>, %arg45: tensor<512xf16>, %arg46: tensor<512xf16>, %arg47: tensor<4x128xf16>, %arg48: tensor<2x128xf16>) -> (tensor<2x16x1024xf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %c_0 = stablehlo.constant dense<1024> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<2x16xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<2x16xi1>, tensor<2x16xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x1xi32>
    %6 = "stablehlo.gather"(%arg3, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<1024x512xf16>, tensor<2x16x1xi32>) -> tensor<2x16x512xf16>
    %cst = stablehlo.constant dense<2.262500e+01> : tensor<f16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %8 = stablehlo.multiply %6, %7 : tensor<2x16x512xf16>
    %9 = stablehlo.convert %8 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<2x16x512xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%10 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_2 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %14 = stablehlo.divide %12, %13 : tensor<2x16x1xf32>
    %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %16 = stablehlo.add %14, %15 : tensor<2x16x1xf32>
    %17 = stablehlo.rsqrt %16 : tensor<2x16x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %19 = stablehlo.multiply %9, %18 : tensor<2x16x512xf32>
    %20 = stablehlo.convert %19 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %21 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %23 = stablehlo.multiply %20, %22 : tensor<2x16x512xf16>
    %24 = stablehlo.dot_general %23, %arg7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16x512xf16>, tensor<4x512x128xf16>) -> tensor<2x16x4x128xf16>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1, 3] : (tensor<2x16x4x128xf16>) -> tensor<2x4x16x128xf16>
    %26 = stablehlo.dot_general %arg6, %23, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x512x128xf16>, tensor<2x16x512xf16>) -> tensor<2x2x128x2x16xf16>
    %27 = stablehlo.transpose %26, dims = [3, 0, 4, 1, 2] : (tensor<2x2x128x2x16xf16>) -> tensor<2x2x16x2x128xf16>
    %28 = stablehlo.slice %27 [0:2, 0:1, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %29 = stablehlo.reshape %28 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %30 = stablehlo.transpose %29, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %31 = stablehlo.slice %27 [0:2, 1:2, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %32 = stablehlo.reshape %31 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %33 = stablehlo.transpose %32, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %34 = stablehlo.convert %25 : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf32>
    %35 = stablehlo.multiply %34, %34 : tensor<2x4x16x128xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.reduce(%35 init: %cst_4) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x128xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %cst_5 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %39 = stablehlo.divide %37, %38 : tensor<2x4x16x1xf32>
    %cst_6 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %41 = stablehlo.add %39, %40 : tensor<2x4x16x1xf32>
    %42 = stablehlo.rsqrt %41 : tensor<2x4x16x1xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x128xf32>
    %44 = stablehlo.multiply %34, %43 : tensor<2x4x16x128xf32>
    %45 = stablehlo.convert %44 : (tensor<2x4x16x128xf32>) -> tensor<2x4x16x128xf16>
    %46 = stablehlo.broadcast_in_dim %arg14, dims = [1, 3] : (tensor<4x128xf16>) -> tensor<1x4x1x128xf16>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2, 3] : (tensor<1x4x1x128xf16>) -> tensor<2x4x16x128xf16>
    %48 = stablehlo.multiply %45, %47 : tensor<2x4x16x128xf16>
    %49 = stablehlo.convert %30 : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x128xf32>
    %50 = stablehlo.multiply %49, %49 : tensor<2x2x16x128xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51 = stablehlo.reduce(%50 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<2x2x16x128xf32>, tensor<f32>) -> tensor<2x2x16xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<2x2x16xf32>) -> tensor<2x2x16x1xf32>
    %cst_8 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %54 = stablehlo.divide %52, %53 : tensor<2x2x16x1xf32>
    %cst_9 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %56 = stablehlo.add %54, %55 : tensor<2x2x16x1xf32>
    %57 = stablehlo.rsqrt %56 : tensor<2x2x16x1xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<2x2x16x1xf32>) -> tensor<2x2x16x128xf32>
    %59 = stablehlo.multiply %49, %58 : tensor<2x2x16x128xf32>
    %60 = stablehlo.convert %59 : (tensor<2x2x16x128xf32>) -> tensor<2x2x16x128xf16>
    %61 = stablehlo.broadcast_in_dim %arg15, dims = [1, 3] : (tensor<2x128xf16>) -> tensor<1x2x1x128xf16>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x2x1x128xf16>) -> tensor<2x2x16x128xf16>
    %63 = stablehlo.multiply %60, %62 : tensor<2x2x16x128xf16>
    %64 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_10 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %66 = stablehlo.divide %64, %65 : tensor<128xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %68 = stablehlo.power %67, %66 : tensor<128xf32>
    %cst_12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %70 = stablehlo.divide %69, %68 : tensor<128xf32>
    %71 = stablehlo.convert %arg1 : (tensor<2x16xf16>) -> tensor<2x16xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 2] : (tensor<2x16xf32>) -> tensor<2x1x16x1xf32>
    %73 = stablehlo.broadcast_in_dim %70, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %74 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<2x1x16x1xf32>) -> tensor<2x1x16x128xf32>
    %75 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x16x128xf32>
    %76 = stablehlo.multiply %74, %75 : tensor<2x1x16x128xf32>
    %77 = stablehlo.cosine %76 : tensor<2x1x16x128xf32>
    %78 = stablehlo.convert %77 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %79 = stablehlo.sine %76 : tensor<2x1x16x128xf32>
    %80 = stablehlo.convert %79 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %81 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %82 = stablehlo.multiply %48, %81 : tensor<2x4x16x128xf16>
    %83 = stablehlo.slice %48 [0:2, 0:4, 0:16, 0:64] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %84 = stablehlo.slice %48 [0:2, 0:4, 0:16, 64:128] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %85 = stablehlo.negate %84 : tensor<2x4x16x64xf16>
    %86 = stablehlo.concatenate %85, %83, dim = 3 : (tensor<2x4x16x64xf16>, tensor<2x4x16x64xf16>) -> tensor<2x4x16x128xf16>
    %87 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %88 = stablehlo.multiply %86, %87 : tensor<2x4x16x128xf16>
    %89 = stablehlo.add %82, %88 : tensor<2x4x16x128xf16>
    %90 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %91 = stablehlo.multiply %63, %90 : tensor<2x2x16x128xf16>
    %92 = stablehlo.slice %63 [0:2, 0:2, 0:16, 0:64] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %93 = stablehlo.slice %63 [0:2, 0:2, 0:16, 64:128] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %94 = stablehlo.negate %93 : tensor<2x2x16x64xf16>
    %95 = stablehlo.concatenate %94, %92, dim = 3 : (tensor<2x2x16x64xf16>, tensor<2x2x16x64xf16>) -> tensor<2x2x16x128xf16>
    %96 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %97 = stablehlo.multiply %95, %96 : tensor<2x2x16x128xf16>
    %98 = stablehlo.add %91, %97 : tensor<2x2x16x128xf16>
    %99 = stablehlo.slice %98 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %100 = stablehlo.slice %98 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %101 = stablehlo.slice %98 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %102 = stablehlo.slice %98 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %103 = stablehlo.concatenate %99, %100, %101, %102, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %104 = stablehlo.slice %33 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %105 = stablehlo.slice %33 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %106 = stablehlo.slice %33 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %107 = stablehlo.slice %33 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %108 = stablehlo.concatenate %104, %105, %106, %107, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %109 = stablehlo.dot_general %89, %103, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x128xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x16xf16>
    %cst_13 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %110 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %111 = stablehlo.multiply %109, %110 : tensor<2x4x16x16xf16>
    %cst_14 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %112 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %113 = stablehlo.divide %111, %112 : tensor<2x4x16x16xf16>
    %114 = stablehlo.tanh %113 : tensor<2x4x16x16xf16>
    %cst_15 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %115 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %116 = stablehlo.multiply %114, %115 : tensor<2x4x16x16xf16>
    %117 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x16xf16>) -> tensor<2x16x1xf16>
    %118 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x16xf16>) -> tensor<2x1x16xf16>
    %119 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<2x16x1xf16>) -> tensor<2x16x16xf16>
    %120 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2] : (tensor<2x1x16xf16>) -> tensor<2x16x16xf16>
    %121 = stablehlo.compare GE, %119, %120, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %122 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f16>) -> tensor<2x16x16xf16>
    %123 = stablehlo.compare GT, %arg2, %122, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %124 = stablehlo.and %121, %123 : tensor<2x16x16xi1>
    %125 = stablehlo.broadcast_in_dim %124, dims = [0, 2, 3] : (tensor<2x16x16xi1>) -> tensor<2x1x16x16xi1>
    %cst_17 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %126 = call @_where(%125, %116, %cst_17) : (tensor<2x1x16x16xi1>, tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16x16xf16>
    %cst_18 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %127 = stablehlo.reduce(%126 init: %cst_18) applies stablehlo.maximum across dimensions = [3] : (tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16xf16>
    %cst_19 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %128 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f16>) -> tensor<2x4x16xf16>
    %129 = stablehlo.maximum %128, %127 : tensor<2x4x16xf16>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2] : (tensor<2x4x16xf16>) -> tensor<2x4x16x1xf16>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %132 = stablehlo.subtract %126, %131 : tensor<2x4x16x16xf16>
    %133 = stablehlo.exponential %132 : tensor<2x4x16x16xf16>
    %134 = stablehlo.convert %133 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %135 = stablehlo.reduce(%134 init: %cst_20) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %137 = stablehlo.convert %136 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %138 = stablehlo.broadcast_in_dim %137, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %139 = stablehlo.divide %133, %138 : tensor<2x4x16x16xf16>
    %140 = stablehlo.convert %139 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %141 = stablehlo.reduce(%140 init: %cst_21) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %143 = stablehlo.convert %142 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %145 = stablehlo.divide %139, %144 : tensor<2x4x16x16xf16>
    %146 = stablehlo.dot_general %145, %108, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x16xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf16>
    %147 = stablehlo.transpose %146, dims = [0, 2, 1, 3] : (tensor<2x4x16x128xf16>) -> tensor<2x16x4x128xf16>
    %148 = stablehlo.dot_general %147, %arg5, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x4x128xf16>, tensor<4x128x512xf16>) -> tensor<2x16x512xf16>
    %149 = stablehlo.add %8, %148 : tensor<2x16x512xf16>
    %150 = stablehlo.convert %149 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %151 = stablehlo.multiply %150, %150 : tensor<2x16x512xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %152 = stablehlo.reduce(%151 init: %cst_22) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_23 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %154 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %155 = stablehlo.divide %153, %154 : tensor<2x16x1xf32>
    %cst_24 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %156 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %157 = stablehlo.add %155, %156 : tensor<2x16x1xf32>
    %158 = stablehlo.rsqrt %157 : tensor<2x16x1xf32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %160 = stablehlo.multiply %150, %159 : tensor<2x16x512xf32>
    %161 = stablehlo.convert %160 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %162 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %163 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %164 = stablehlo.multiply %161, %163 : tensor<2x16x512xf16>
    %cst_25 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %165 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %166 = stablehlo.multiply %165, %164 : tensor<2x16x512xf16>
    %167 = stablehlo.add %149, %166 : tensor<2x16x512xf16>
    %168 = stablehlo.convert %167 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %169 = stablehlo.multiply %168, %168 : tensor<2x16x512xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %170 = stablehlo.reduce(%169 init: %cst_26) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_27 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %172 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %173 = stablehlo.divide %171, %172 : tensor<2x16x1xf32>
    %cst_28 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %174 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %175 = stablehlo.add %173, %174 : tensor<2x16x1xf32>
    %176 = stablehlo.rsqrt %175 : tensor<2x16x1xf32>
    %177 = stablehlo.broadcast_in_dim %176, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %178 = stablehlo.multiply %168, %177 : tensor<2x16x512xf32>
    %179 = stablehlo.convert %178 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %180 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %182 = stablehlo.multiply %179, %181 : tensor<2x16x512xf16>
    %183 = stablehlo.dot_general %arg8, %182, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x512x1024xf16>, tensor<2x16x512xf16>) -> tensor<2x1024x2x16xf16>
    %184 = stablehlo.transpose %183, dims = [2, 0, 3, 1] : (tensor<2x1024x2x16xf16>) -> tensor<2x2x16x1024xf16>
    %185 = stablehlo.slice %184 [0:2, 0:1, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %186 = stablehlo.reshape %185 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %187 = stablehlo.slice %184 [0:2, 1:2, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %188 = stablehlo.reshape %187 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %189 = stablehlo.negate %186 : tensor<2x16x1024xf16>
    %190 = stablehlo.exponential %189 : tensor<2x16x1024xf16>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %191 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %192 = stablehlo.add %191, %190 : tensor<2x16x1024xf16>
    %cst_30 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %193 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %194 = stablehlo.divide %193, %192 : tensor<2x16x1024xf16>
    %195 = stablehlo.multiply %186, %194 : tensor<2x16x1024xf16>
    %196 = stablehlo.multiply %195, %188 : tensor<2x16x1024xf16>
    %197 = stablehlo.dot_general %196, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x1024xf16>, tensor<1024x512xf16>) -> tensor<2x16x512xf16>
    %198 = stablehlo.add %167, %197 : tensor<2x16x512xf16>
    %199 = stablehlo.convert %198 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %200 = stablehlo.multiply %199, %199 : tensor<2x16x512xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %201 = stablehlo.reduce(%200 init: %cst_31) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_32 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %203 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %204 = stablehlo.divide %202, %203 : tensor<2x16x1xf32>
    %cst_33 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %205 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %206 = stablehlo.add %204, %205 : tensor<2x16x1xf32>
    %207 = stablehlo.rsqrt %206 : tensor<2x16x1xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %209 = stablehlo.multiply %199, %208 : tensor<2x16x512xf32>
    %210 = stablehlo.convert %209 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %211 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %213 = stablehlo.multiply %210, %212 : tensor<2x16x512xf16>
    %cst_34 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %214 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %215 = stablehlo.multiply %214, %213 : tensor<2x16x512xf16>
    %216 = stablehlo.add %198, %215 : tensor<2x16x512xf16>
    %217 = stablehlo.slice %arg2 [0:2, 0:16, 0:1] : (tensor<2x16x16xf16>) -> tensor<2x16x1xf16>
    %218 = stablehlo.reshape %217 : (tensor<2x16x1xf16>) -> tensor<2x16xf16>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %219 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f16>) -> tensor<2x16xf16>
    %220 = stablehlo.compare GT, %218, %219, FLOAT : (tensor<2x16xf16>, tensor<2x16xf16>) -> tensor<2x16xi1>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1] : (tensor<2x16xi1>) -> tensor<2x16x1xi1>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %222 = call @_where_87(%221, %216, %cst_36) : (tensor<2x16x1xi1>, tensor<2x16x512xf16>, tensor<f16>) -> tensor<2x16x512xf16>
    %cst_37 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %223 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %224 = stablehlo.compare GT, %222, %223, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_38 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %225 = call @_where_92(%224, %cst_38, %222) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %cst_39 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %226 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %227 = stablehlo.compare LT, %225, %226, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_40 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %228 = call @_where_92(%227, %cst_40, %225) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %229 = stablehlo.convert %228 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %230 = stablehlo.multiply %229, %229 : tensor<2x16x512xf32>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %231 = stablehlo.reduce(%230 init: %cst_41) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %232 = stablehlo.broadcast_in_dim %231, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_42 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %233 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %234 = stablehlo.divide %232, %233 : tensor<2x16x1xf32>
    %cst_43 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %235 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %236 = stablehlo.add %234, %235 : tensor<2x16x1xf32>
    %237 = stablehlo.rsqrt %236 : tensor<2x16x1xf32>
    %238 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %239 = stablehlo.multiply %229, %238 : tensor<2x16x512xf32>
    %240 = stablehlo.convert %239 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %241 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %243 = stablehlo.multiply %240, %242 : tensor<2x16x512xf16>
    %244 = stablehlo.dot_general %243, %arg18, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16x512xf16>, tensor<4x512x128xf16>) -> tensor<2x16x4x128xf16>
    %245 = stablehlo.transpose %244, dims = [0, 2, 1, 3] : (tensor<2x16x4x128xf16>) -> tensor<2x4x16x128xf16>
    %246 = stablehlo.dot_general %arg17, %243, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x512x128xf16>, tensor<2x16x512xf16>) -> tensor<2x2x128x2x16xf16>
    %247 = stablehlo.transpose %246, dims = [3, 0, 4, 1, 2] : (tensor<2x2x128x2x16xf16>) -> tensor<2x2x16x2x128xf16>
    %248 = stablehlo.slice %247 [0:2, 0:1, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %249 = stablehlo.reshape %248 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %250 = stablehlo.transpose %249, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %251 = stablehlo.slice %247 [0:2, 1:2, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %252 = stablehlo.reshape %251 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %253 = stablehlo.transpose %252, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %254 = stablehlo.convert %245 : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf32>
    %255 = stablehlo.multiply %254, %254 : tensor<2x4x16x128xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %256 = stablehlo.reduce(%255 init: %cst_44) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x128xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %cst_45 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %258 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %259 = stablehlo.divide %257, %258 : tensor<2x4x16x1xf32>
    %cst_46 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %260 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %261 = stablehlo.add %259, %260 : tensor<2x4x16x1xf32>
    %262 = stablehlo.rsqrt %261 : tensor<2x4x16x1xf32>
    %263 = stablehlo.broadcast_in_dim %262, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x128xf32>
    %264 = stablehlo.multiply %254, %263 : tensor<2x4x16x128xf32>
    %265 = stablehlo.convert %264 : (tensor<2x4x16x128xf32>) -> tensor<2x4x16x128xf16>
    %266 = stablehlo.broadcast_in_dim %arg25, dims = [1, 3] : (tensor<4x128xf16>) -> tensor<1x4x1x128xf16>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2, 3] : (tensor<1x4x1x128xf16>) -> tensor<2x4x16x128xf16>
    %268 = stablehlo.multiply %265, %267 : tensor<2x4x16x128xf16>
    %269 = stablehlo.convert %250 : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x128xf32>
    %270 = stablehlo.multiply %269, %269 : tensor<2x2x16x128xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %271 = stablehlo.reduce(%270 init: %cst_47) applies stablehlo.add across dimensions = [3] : (tensor<2x2x16x128xf32>, tensor<f32>) -> tensor<2x2x16xf32>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2] : (tensor<2x2x16xf32>) -> tensor<2x2x16x1xf32>
    %cst_48 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %273 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %274 = stablehlo.divide %272, %273 : tensor<2x2x16x1xf32>
    %cst_49 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %275 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %276 = stablehlo.add %274, %275 : tensor<2x2x16x1xf32>
    %277 = stablehlo.rsqrt %276 : tensor<2x2x16x1xf32>
    %278 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<2x2x16x1xf32>) -> tensor<2x2x16x128xf32>
    %279 = stablehlo.multiply %269, %278 : tensor<2x2x16x128xf32>
    %280 = stablehlo.convert %279 : (tensor<2x2x16x128xf32>) -> tensor<2x2x16x128xf16>
    %281 = stablehlo.broadcast_in_dim %arg26, dims = [1, 3] : (tensor<2x128xf16>) -> tensor<1x2x1x128xf16>
    %282 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2, 3] : (tensor<1x2x1x128xf16>) -> tensor<2x2x16x128xf16>
    %283 = stablehlo.multiply %280, %282 : tensor<2x2x16x128xf16>
    %284 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_50 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %285 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %286 = stablehlo.divide %284, %285 : tensor<128xf32>
    %cst_51 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %287 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %288 = stablehlo.power %287, %286 : tensor<128xf32>
    %cst_52 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %289 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %290 = stablehlo.divide %289, %288 : tensor<128xf32>
    %291 = stablehlo.convert %arg1 : (tensor<2x16xf16>) -> tensor<2x16xf32>
    %292 = stablehlo.broadcast_in_dim %291, dims = [0, 2] : (tensor<2x16xf32>) -> tensor<2x1x16x1xf32>
    %293 = stablehlo.broadcast_in_dim %290, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %294 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2, 3] : (tensor<2x1x16x1xf32>) -> tensor<2x1x16x128xf32>
    %295 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x16x128xf32>
    %296 = stablehlo.multiply %294, %295 : tensor<2x1x16x128xf32>
    %297 = stablehlo.cosine %296 : tensor<2x1x16x128xf32>
    %298 = stablehlo.convert %297 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %299 = stablehlo.sine %296 : tensor<2x1x16x128xf32>
    %300 = stablehlo.convert %299 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %301 = stablehlo.broadcast_in_dim %298, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %302 = stablehlo.multiply %268, %301 : tensor<2x4x16x128xf16>
    %303 = stablehlo.slice %268 [0:2, 0:4, 0:16, 0:64] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %304 = stablehlo.slice %268 [0:2, 0:4, 0:16, 64:128] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %305 = stablehlo.negate %304 : tensor<2x4x16x64xf16>
    %306 = stablehlo.concatenate %305, %303, dim = 3 : (tensor<2x4x16x64xf16>, tensor<2x4x16x64xf16>) -> tensor<2x4x16x128xf16>
    %307 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %308 = stablehlo.multiply %306, %307 : tensor<2x4x16x128xf16>
    %309 = stablehlo.add %302, %308 : tensor<2x4x16x128xf16>
    %310 = stablehlo.broadcast_in_dim %298, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %311 = stablehlo.multiply %283, %310 : tensor<2x2x16x128xf16>
    %312 = stablehlo.slice %283 [0:2, 0:2, 0:16, 0:64] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %313 = stablehlo.slice %283 [0:2, 0:2, 0:16, 64:128] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %314 = stablehlo.negate %313 : tensor<2x2x16x64xf16>
    %315 = stablehlo.concatenate %314, %312, dim = 3 : (tensor<2x2x16x64xf16>, tensor<2x2x16x64xf16>) -> tensor<2x2x16x128xf16>
    %316 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %317 = stablehlo.multiply %315, %316 : tensor<2x2x16x128xf16>
    %318 = stablehlo.add %311, %317 : tensor<2x2x16x128xf16>
    %319 = stablehlo.slice %318 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %320 = stablehlo.slice %318 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %321 = stablehlo.slice %318 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %322 = stablehlo.slice %318 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %323 = stablehlo.concatenate %319, %320, %321, %322, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %324 = stablehlo.slice %253 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %325 = stablehlo.slice %253 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %326 = stablehlo.slice %253 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %327 = stablehlo.slice %253 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %328 = stablehlo.concatenate %324, %325, %326, %327, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %329 = stablehlo.dot_general %309, %323, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x128xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x16xf16>
    %cst_53 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %330 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %331 = stablehlo.multiply %329, %330 : tensor<2x4x16x16xf16>
    %cst_54 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %332 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %333 = stablehlo.divide %331, %332 : tensor<2x4x16x16xf16>
    %334 = stablehlo.tanh %333 : tensor<2x4x16x16xf16>
    %cst_55 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %335 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %336 = stablehlo.multiply %334, %335 : tensor<2x4x16x16xf16>
    %337 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x16xf16>) -> tensor<2x16x1xf16>
    %338 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x16xf16>) -> tensor<2x1x16xf16>
    %339 = stablehlo.broadcast_in_dim %337, dims = [0, 1, 2] : (tensor<2x16x1xf16>) -> tensor<2x16x16xf16>
    %340 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2] : (tensor<2x1x16xf16>) -> tensor<2x16x16xf16>
    %341 = stablehlo.compare GE, %339, %340, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %342 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f16>) -> tensor<2x16x16xf16>
    %343 = stablehlo.compare GT, %arg2, %342, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %344 = stablehlo.and %341, %343 : tensor<2x16x16xi1>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 2, 3] : (tensor<2x16x16xi1>) -> tensor<2x1x16x16xi1>
    %cst_57 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %346 = call @_where(%345, %336, %cst_57) : (tensor<2x1x16x16xi1>, tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16x16xf16>
    %cst_58 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %347 = stablehlo.reduce(%346 init: %cst_58) applies stablehlo.maximum across dimensions = [3] : (tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16xf16>
    %cst_59 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %348 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f16>) -> tensor<2x4x16xf16>
    %349 = stablehlo.maximum %348, %347 : tensor<2x4x16xf16>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2] : (tensor<2x4x16xf16>) -> tensor<2x4x16x1xf16>
    %351 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %352 = stablehlo.subtract %346, %351 : tensor<2x4x16x16xf16>
    %353 = stablehlo.exponential %352 : tensor<2x4x16x16xf16>
    %354 = stablehlo.convert %353 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %355 = stablehlo.reduce(%354 init: %cst_60) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %357 = stablehlo.convert %356 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %358 = stablehlo.broadcast_in_dim %357, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %359 = stablehlo.divide %353, %358 : tensor<2x4x16x16xf16>
    %360 = stablehlo.convert %359 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %361 = stablehlo.reduce(%360 init: %cst_61) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %363 = stablehlo.convert %362 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %365 = stablehlo.divide %359, %364 : tensor<2x4x16x16xf16>
    %366 = stablehlo.dot_general %365, %328, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x16xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf16>
    %367 = stablehlo.transpose %366, dims = [0, 2, 1, 3] : (tensor<2x4x16x128xf16>) -> tensor<2x16x4x128xf16>
    %368 = stablehlo.dot_general %367, %arg16, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x4x128xf16>, tensor<4x128x512xf16>) -> tensor<2x16x512xf16>
    %369 = stablehlo.add %228, %368 : tensor<2x16x512xf16>
    %370 = stablehlo.convert %369 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %371 = stablehlo.multiply %370, %370 : tensor<2x16x512xf32>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %372 = stablehlo.reduce(%371 init: %cst_62) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %373 = stablehlo.broadcast_in_dim %372, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_63 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %374 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %375 = stablehlo.divide %373, %374 : tensor<2x16x1xf32>
    %cst_64 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %376 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %377 = stablehlo.add %375, %376 : tensor<2x16x1xf32>
    %378 = stablehlo.rsqrt %377 : tensor<2x16x1xf32>
    %379 = stablehlo.broadcast_in_dim %378, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %380 = stablehlo.multiply %370, %379 : tensor<2x16x512xf32>
    %381 = stablehlo.convert %380 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %382 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %383 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %384 = stablehlo.multiply %381, %383 : tensor<2x16x512xf16>
    %cst_65 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %385 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %386 = stablehlo.multiply %385, %384 : tensor<2x16x512xf16>
    %387 = stablehlo.add %369, %386 : tensor<2x16x512xf16>
    %388 = stablehlo.convert %387 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %389 = stablehlo.multiply %388, %388 : tensor<2x16x512xf32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %390 = stablehlo.reduce(%389 init: %cst_66) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %391 = stablehlo.broadcast_in_dim %390, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_67 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %392 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %393 = stablehlo.divide %391, %392 : tensor<2x16x1xf32>
    %cst_68 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %394 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %395 = stablehlo.add %393, %394 : tensor<2x16x1xf32>
    %396 = stablehlo.rsqrt %395 : tensor<2x16x1xf32>
    %397 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %398 = stablehlo.multiply %388, %397 : tensor<2x16x512xf32>
    %399 = stablehlo.convert %398 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %400 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %401 = stablehlo.broadcast_in_dim %400, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %402 = stablehlo.multiply %399, %401 : tensor<2x16x512xf16>
    %403 = stablehlo.dot_general %arg19, %402, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x512x1024xf16>, tensor<2x16x512xf16>) -> tensor<2x1024x2x16xf16>
    %404 = stablehlo.transpose %403, dims = [2, 0, 3, 1] : (tensor<2x1024x2x16xf16>) -> tensor<2x2x16x1024xf16>
    %405 = stablehlo.slice %404 [0:2, 0:1, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %406 = stablehlo.reshape %405 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %407 = stablehlo.slice %404 [0:2, 1:2, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %408 = stablehlo.reshape %407 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %409 = stablehlo.negate %406 : tensor<2x16x1024xf16>
    %410 = stablehlo.exponential %409 : tensor<2x16x1024xf16>
    %cst_69 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %411 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %412 = stablehlo.add %411, %410 : tensor<2x16x1024xf16>
    %cst_70 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %413 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %414 = stablehlo.divide %413, %412 : tensor<2x16x1024xf16>
    %415 = stablehlo.multiply %406, %414 : tensor<2x16x1024xf16>
    %416 = stablehlo.multiply %415, %408 : tensor<2x16x1024xf16>
    %417 = stablehlo.dot_general %416, %arg20, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x1024xf16>, tensor<1024x512xf16>) -> tensor<2x16x512xf16>
    %418 = stablehlo.add %387, %417 : tensor<2x16x512xf16>
    %419 = stablehlo.convert %418 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %420 = stablehlo.multiply %419, %419 : tensor<2x16x512xf32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %421 = stablehlo.reduce(%420 init: %cst_71) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_72 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %423 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %424 = stablehlo.divide %422, %423 : tensor<2x16x1xf32>
    %cst_73 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %425 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %426 = stablehlo.add %424, %425 : tensor<2x16x1xf32>
    %427 = stablehlo.rsqrt %426 : tensor<2x16x1xf32>
    %428 = stablehlo.broadcast_in_dim %427, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %429 = stablehlo.multiply %419, %428 : tensor<2x16x512xf32>
    %430 = stablehlo.convert %429 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %431 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %432 = stablehlo.broadcast_in_dim %431, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %433 = stablehlo.multiply %430, %432 : tensor<2x16x512xf16>
    %cst_74 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %434 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %435 = stablehlo.multiply %434, %433 : tensor<2x16x512xf16>
    %436 = stablehlo.add %418, %435 : tensor<2x16x512xf16>
    %437 = stablehlo.slice %arg2 [0:2, 0:16, 0:1] : (tensor<2x16x16xf16>) -> tensor<2x16x1xf16>
    %438 = stablehlo.reshape %437 : (tensor<2x16x1xf16>) -> tensor<2x16xf16>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %439 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f16>) -> tensor<2x16xf16>
    %440 = stablehlo.compare GT, %438, %439, FLOAT : (tensor<2x16xf16>, tensor<2x16xf16>) -> tensor<2x16xi1>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1] : (tensor<2x16xi1>) -> tensor<2x16x1xi1>
    %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %442 = call @_where_87(%441, %436, %cst_76) : (tensor<2x16x1xi1>, tensor<2x16x512xf16>, tensor<f16>) -> tensor<2x16x512xf16>
    %cst_77 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %443 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %444 = stablehlo.compare GT, %442, %443, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_78 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %445 = call @_where_92(%444, %cst_78, %442) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %cst_79 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %446 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %447 = stablehlo.compare LT, %445, %446, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_80 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %448 = call @_where_92(%447, %cst_80, %445) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %449 = stablehlo.convert %448 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %450 = stablehlo.multiply %449, %449 : tensor<2x16x512xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %451 = stablehlo.reduce(%450 init: %cst_81) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %452 = stablehlo.broadcast_in_dim %451, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_82 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %453 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %454 = stablehlo.divide %452, %453 : tensor<2x16x1xf32>
    %cst_83 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %455 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %456 = stablehlo.add %454, %455 : tensor<2x16x1xf32>
    %457 = stablehlo.rsqrt %456 : tensor<2x16x1xf32>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %459 = stablehlo.multiply %449, %458 : tensor<2x16x512xf32>
    %460 = stablehlo.convert %459 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %461 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %463 = stablehlo.multiply %460, %462 : tensor<2x16x512xf16>
    %464 = stablehlo.dot_general %463, %arg29, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16x512xf16>, tensor<4x512x128xf16>) -> tensor<2x16x4x128xf16>
    %465 = stablehlo.transpose %464, dims = [0, 2, 1, 3] : (tensor<2x16x4x128xf16>) -> tensor<2x4x16x128xf16>
    %466 = stablehlo.dot_general %arg28, %463, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x512x128xf16>, tensor<2x16x512xf16>) -> tensor<2x2x128x2x16xf16>
    %467 = stablehlo.transpose %466, dims = [3, 0, 4, 1, 2] : (tensor<2x2x128x2x16xf16>) -> tensor<2x2x16x2x128xf16>
    %468 = stablehlo.slice %467 [0:2, 0:1, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %469 = stablehlo.reshape %468 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %470 = stablehlo.transpose %469, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %471 = stablehlo.slice %467 [0:2, 1:2, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %472 = stablehlo.reshape %471 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %473 = stablehlo.transpose %472, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %474 = stablehlo.convert %465 : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf32>
    %475 = stablehlo.multiply %474, %474 : tensor<2x4x16x128xf32>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %476 = stablehlo.reduce(%475 init: %cst_84) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x128xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %cst_85 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %478 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %479 = stablehlo.divide %477, %478 : tensor<2x4x16x1xf32>
    %cst_86 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %480 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %481 = stablehlo.add %479, %480 : tensor<2x4x16x1xf32>
    %482 = stablehlo.rsqrt %481 : tensor<2x4x16x1xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x128xf32>
    %484 = stablehlo.multiply %474, %483 : tensor<2x4x16x128xf32>
    %485 = stablehlo.convert %484 : (tensor<2x4x16x128xf32>) -> tensor<2x4x16x128xf16>
    %486 = stablehlo.broadcast_in_dim %arg36, dims = [1, 3] : (tensor<4x128xf16>) -> tensor<1x4x1x128xf16>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2, 3] : (tensor<1x4x1x128xf16>) -> tensor<2x4x16x128xf16>
    %488 = stablehlo.multiply %485, %487 : tensor<2x4x16x128xf16>
    %489 = stablehlo.convert %470 : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x128xf32>
    %490 = stablehlo.multiply %489, %489 : tensor<2x2x16x128xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %491 = stablehlo.reduce(%490 init: %cst_87) applies stablehlo.add across dimensions = [3] : (tensor<2x2x16x128xf32>, tensor<f32>) -> tensor<2x2x16xf32>
    %492 = stablehlo.broadcast_in_dim %491, dims = [0, 1, 2] : (tensor<2x2x16xf32>) -> tensor<2x2x16x1xf32>
    %cst_88 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %493 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %494 = stablehlo.divide %492, %493 : tensor<2x2x16x1xf32>
    %cst_89 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %495 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %496 = stablehlo.add %494, %495 : tensor<2x2x16x1xf32>
    %497 = stablehlo.rsqrt %496 : tensor<2x2x16x1xf32>
    %498 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2, 3] : (tensor<2x2x16x1xf32>) -> tensor<2x2x16x128xf32>
    %499 = stablehlo.multiply %489, %498 : tensor<2x2x16x128xf32>
    %500 = stablehlo.convert %499 : (tensor<2x2x16x128xf32>) -> tensor<2x2x16x128xf16>
    %501 = stablehlo.broadcast_in_dim %arg37, dims = [1, 3] : (tensor<2x128xf16>) -> tensor<1x2x1x128xf16>
    %502 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2, 3] : (tensor<1x2x1x128xf16>) -> tensor<2x2x16x128xf16>
    %503 = stablehlo.multiply %500, %502 : tensor<2x2x16x128xf16>
    %504 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_90 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %505 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %506 = stablehlo.divide %504, %505 : tensor<128xf32>
    %cst_91 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %507 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %508 = stablehlo.power %507, %506 : tensor<128xf32>
    %cst_92 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %509 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %510 = stablehlo.divide %509, %508 : tensor<128xf32>
    %511 = stablehlo.convert %arg1 : (tensor<2x16xf16>) -> tensor<2x16xf32>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0, 2] : (tensor<2x16xf32>) -> tensor<2x1x16x1xf32>
    %513 = stablehlo.broadcast_in_dim %510, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %514 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2, 3] : (tensor<2x1x16x1xf32>) -> tensor<2x1x16x128xf32>
    %515 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x16x128xf32>
    %516 = stablehlo.multiply %514, %515 : tensor<2x1x16x128xf32>
    %517 = stablehlo.cosine %516 : tensor<2x1x16x128xf32>
    %518 = stablehlo.convert %517 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %519 = stablehlo.sine %516 : tensor<2x1x16x128xf32>
    %520 = stablehlo.convert %519 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %521 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %522 = stablehlo.multiply %488, %521 : tensor<2x4x16x128xf16>
    %523 = stablehlo.slice %488 [0:2, 0:4, 0:16, 0:64] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %524 = stablehlo.slice %488 [0:2, 0:4, 0:16, 64:128] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %525 = stablehlo.negate %524 : tensor<2x4x16x64xf16>
    %526 = stablehlo.concatenate %525, %523, dim = 3 : (tensor<2x4x16x64xf16>, tensor<2x4x16x64xf16>) -> tensor<2x4x16x128xf16>
    %527 = stablehlo.broadcast_in_dim %520, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %528 = stablehlo.multiply %526, %527 : tensor<2x4x16x128xf16>
    %529 = stablehlo.add %522, %528 : tensor<2x4x16x128xf16>
    %530 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %531 = stablehlo.multiply %503, %530 : tensor<2x2x16x128xf16>
    %532 = stablehlo.slice %503 [0:2, 0:2, 0:16, 0:64] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %533 = stablehlo.slice %503 [0:2, 0:2, 0:16, 64:128] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %534 = stablehlo.negate %533 : tensor<2x2x16x64xf16>
    %535 = stablehlo.concatenate %534, %532, dim = 3 : (tensor<2x2x16x64xf16>, tensor<2x2x16x64xf16>) -> tensor<2x2x16x128xf16>
    %536 = stablehlo.broadcast_in_dim %520, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %537 = stablehlo.multiply %535, %536 : tensor<2x2x16x128xf16>
    %538 = stablehlo.add %531, %537 : tensor<2x2x16x128xf16>
    %539 = stablehlo.slice %538 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %540 = stablehlo.slice %538 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %541 = stablehlo.slice %538 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %542 = stablehlo.slice %538 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %543 = stablehlo.concatenate %539, %540, %541, %542, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %544 = stablehlo.slice %473 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %545 = stablehlo.slice %473 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %546 = stablehlo.slice %473 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %547 = stablehlo.slice %473 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %548 = stablehlo.concatenate %544, %545, %546, %547, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %549 = stablehlo.dot_general %529, %543, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x128xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x16xf16>
    %cst_93 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %550 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %551 = stablehlo.multiply %549, %550 : tensor<2x4x16x16xf16>
    %cst_94 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %552 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %553 = stablehlo.divide %551, %552 : tensor<2x4x16x16xf16>
    %554 = stablehlo.tanh %553 : tensor<2x4x16x16xf16>
    %cst_95 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %555 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %556 = stablehlo.multiply %554, %555 : tensor<2x4x16x16xf16>
    %557 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x16xf16>) -> tensor<2x16x1xf16>
    %558 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x16xf16>) -> tensor<2x1x16xf16>
    %559 = stablehlo.broadcast_in_dim %557, dims = [0, 1, 2] : (tensor<2x16x1xf16>) -> tensor<2x16x16xf16>
    %560 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2] : (tensor<2x1x16xf16>) -> tensor<2x16x16xf16>
    %561 = stablehlo.compare GE, %559, %560, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %562 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f16>) -> tensor<2x16x16xf16>
    %563 = stablehlo.compare GT, %arg2, %562, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %564 = stablehlo.and %561, %563 : tensor<2x16x16xi1>
    %565 = stablehlo.broadcast_in_dim %564, dims = [0, 2, 3] : (tensor<2x16x16xi1>) -> tensor<2x1x16x16xi1>
    %cst_97 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %566 = call @_where(%565, %556, %cst_97) : (tensor<2x1x16x16xi1>, tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16x16xf16>
    %cst_98 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %567 = stablehlo.reduce(%566 init: %cst_98) applies stablehlo.maximum across dimensions = [3] : (tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16xf16>
    %cst_99 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %568 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f16>) -> tensor<2x4x16xf16>
    %569 = stablehlo.maximum %568, %567 : tensor<2x4x16xf16>
    %570 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2] : (tensor<2x4x16xf16>) -> tensor<2x4x16x1xf16>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %572 = stablehlo.subtract %566, %571 : tensor<2x4x16x16xf16>
    %573 = stablehlo.exponential %572 : tensor<2x4x16x16xf16>
    %574 = stablehlo.convert %573 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %575 = stablehlo.reduce(%574 init: %cst_100) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %577 = stablehlo.convert %576 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %578 = stablehlo.broadcast_in_dim %577, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %579 = stablehlo.divide %573, %578 : tensor<2x4x16x16xf16>
    %580 = stablehlo.convert %579 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %581 = stablehlo.reduce(%580 init: %cst_101) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %582 = stablehlo.broadcast_in_dim %581, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %583 = stablehlo.convert %582 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %584 = stablehlo.broadcast_in_dim %583, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %585 = stablehlo.divide %579, %584 : tensor<2x4x16x16xf16>
    %586 = stablehlo.dot_general %585, %548, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x16xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf16>
    %587 = stablehlo.transpose %586, dims = [0, 2, 1, 3] : (tensor<2x4x16x128xf16>) -> tensor<2x16x4x128xf16>
    %588 = stablehlo.dot_general %587, %arg27, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x4x128xf16>, tensor<4x128x512xf16>) -> tensor<2x16x512xf16>
    %589 = stablehlo.add %448, %588 : tensor<2x16x512xf16>
    %590 = stablehlo.convert %589 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %591 = stablehlo.multiply %590, %590 : tensor<2x16x512xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %592 = stablehlo.reduce(%591 init: %cst_102) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %593 = stablehlo.broadcast_in_dim %592, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_103 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %594 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %595 = stablehlo.divide %593, %594 : tensor<2x16x1xf32>
    %cst_104 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %596 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %597 = stablehlo.add %595, %596 : tensor<2x16x1xf32>
    %598 = stablehlo.rsqrt %597 : tensor<2x16x1xf32>
    %599 = stablehlo.broadcast_in_dim %598, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %600 = stablehlo.multiply %590, %599 : tensor<2x16x512xf32>
    %601 = stablehlo.convert %600 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %602 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %603 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %604 = stablehlo.multiply %601, %603 : tensor<2x16x512xf16>
    %cst_105 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %605 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %606 = stablehlo.multiply %605, %604 : tensor<2x16x512xf16>
    %607 = stablehlo.add %589, %606 : tensor<2x16x512xf16>
    %608 = stablehlo.convert %607 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %609 = stablehlo.multiply %608, %608 : tensor<2x16x512xf32>
    %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %610 = stablehlo.reduce(%609 init: %cst_106) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %611 = stablehlo.broadcast_in_dim %610, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_107 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %612 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %613 = stablehlo.divide %611, %612 : tensor<2x16x1xf32>
    %cst_108 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %614 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %615 = stablehlo.add %613, %614 : tensor<2x16x1xf32>
    %616 = stablehlo.rsqrt %615 : tensor<2x16x1xf32>
    %617 = stablehlo.broadcast_in_dim %616, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %618 = stablehlo.multiply %608, %617 : tensor<2x16x512xf32>
    %619 = stablehlo.convert %618 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %620 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %621 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %622 = stablehlo.multiply %619, %621 : tensor<2x16x512xf16>
    %623 = stablehlo.dot_general %arg30, %622, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x512x1024xf16>, tensor<2x16x512xf16>) -> tensor<2x1024x2x16xf16>
    %624 = stablehlo.transpose %623, dims = [2, 0, 3, 1] : (tensor<2x1024x2x16xf16>) -> tensor<2x2x16x1024xf16>
    %625 = stablehlo.slice %624 [0:2, 0:1, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %626 = stablehlo.reshape %625 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %627 = stablehlo.slice %624 [0:2, 1:2, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %628 = stablehlo.reshape %627 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %629 = stablehlo.negate %626 : tensor<2x16x1024xf16>
    %630 = stablehlo.exponential %629 : tensor<2x16x1024xf16>
    %cst_109 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %631 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %632 = stablehlo.add %631, %630 : tensor<2x16x1024xf16>
    %cst_110 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %633 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %634 = stablehlo.divide %633, %632 : tensor<2x16x1024xf16>
    %635 = stablehlo.multiply %626, %634 : tensor<2x16x1024xf16>
    %636 = stablehlo.multiply %635, %628 : tensor<2x16x1024xf16>
    %637 = stablehlo.dot_general %636, %arg31, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x1024xf16>, tensor<1024x512xf16>) -> tensor<2x16x512xf16>
    %638 = stablehlo.add %607, %637 : tensor<2x16x512xf16>
    %639 = stablehlo.convert %638 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %640 = stablehlo.multiply %639, %639 : tensor<2x16x512xf32>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %641 = stablehlo.reduce(%640 init: %cst_111) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %642 = stablehlo.broadcast_in_dim %641, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_112 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %643 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %644 = stablehlo.divide %642, %643 : tensor<2x16x1xf32>
    %cst_113 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %645 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %646 = stablehlo.add %644, %645 : tensor<2x16x1xf32>
    %647 = stablehlo.rsqrt %646 : tensor<2x16x1xf32>
    %648 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %649 = stablehlo.multiply %639, %648 : tensor<2x16x512xf32>
    %650 = stablehlo.convert %649 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %651 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %653 = stablehlo.multiply %650, %652 : tensor<2x16x512xf16>
    %cst_114 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %654 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %655 = stablehlo.multiply %654, %653 : tensor<2x16x512xf16>
    %656 = stablehlo.add %638, %655 : tensor<2x16x512xf16>
    %657 = stablehlo.slice %arg2 [0:2, 0:16, 0:1] : (tensor<2x16x16xf16>) -> tensor<2x16x1xf16>
    %658 = stablehlo.reshape %657 : (tensor<2x16x1xf16>) -> tensor<2x16xf16>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %659 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f16>) -> tensor<2x16xf16>
    %660 = stablehlo.compare GT, %658, %659, FLOAT : (tensor<2x16xf16>, tensor<2x16xf16>) -> tensor<2x16xi1>
    %661 = stablehlo.broadcast_in_dim %660, dims = [0, 1] : (tensor<2x16xi1>) -> tensor<2x16x1xi1>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %662 = call @_where_87(%661, %656, %cst_116) : (tensor<2x16x1xi1>, tensor<2x16x512xf16>, tensor<f16>) -> tensor<2x16x512xf16>
    %cst_117 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %663 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %664 = stablehlo.compare GT, %662, %663, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_118 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %665 = call @_where_92(%664, %cst_118, %662) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %cst_119 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %666 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %667 = stablehlo.compare LT, %665, %666, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_120 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %668 = call @_where_92(%667, %cst_120, %665) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %669 = stablehlo.convert %668 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %670 = stablehlo.multiply %669, %669 : tensor<2x16x512xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %671 = stablehlo.reduce(%670 init: %cst_121) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %672 = stablehlo.broadcast_in_dim %671, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_122 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %673 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %674 = stablehlo.divide %672, %673 : tensor<2x16x1xf32>
    %cst_123 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %675 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %676 = stablehlo.add %674, %675 : tensor<2x16x1xf32>
    %677 = stablehlo.rsqrt %676 : tensor<2x16x1xf32>
    %678 = stablehlo.broadcast_in_dim %677, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %679 = stablehlo.multiply %669, %678 : tensor<2x16x512xf32>
    %680 = stablehlo.convert %679 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %681 = stablehlo.broadcast_in_dim %arg43, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %682 = stablehlo.broadcast_in_dim %681, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %683 = stablehlo.multiply %680, %682 : tensor<2x16x512xf16>
    %684 = stablehlo.dot_general %683, %arg40, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16x512xf16>, tensor<4x512x128xf16>) -> tensor<2x16x4x128xf16>
    %685 = stablehlo.transpose %684, dims = [0, 2, 1, 3] : (tensor<2x16x4x128xf16>) -> tensor<2x4x16x128xf16>
    %686 = stablehlo.dot_general %arg39, %683, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x512x128xf16>, tensor<2x16x512xf16>) -> tensor<2x2x128x2x16xf16>
    %687 = stablehlo.transpose %686, dims = [3, 0, 4, 1, 2] : (tensor<2x2x128x2x16xf16>) -> tensor<2x2x16x2x128xf16>
    %688 = stablehlo.slice %687 [0:2, 0:1, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %689 = stablehlo.reshape %688 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %690 = stablehlo.transpose %689, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %691 = stablehlo.slice %687 [0:2, 1:2, 0:16, 0:2, 0:128] : (tensor<2x2x16x2x128xf16>) -> tensor<2x1x16x2x128xf16>
    %692 = stablehlo.reshape %691 : (tensor<2x1x16x2x128xf16>) -> tensor<2x16x2x128xf16>
    %693 = stablehlo.transpose %692, dims = [0, 2, 1, 3] : (tensor<2x16x2x128xf16>) -> tensor<2x2x16x128xf16>
    %694 = stablehlo.convert %685 : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf32>
    %695 = stablehlo.multiply %694, %694 : tensor<2x4x16x128xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %696 = stablehlo.reduce(%695 init: %cst_124) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x128xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %697 = stablehlo.broadcast_in_dim %696, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %cst_125 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %698 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %699 = stablehlo.divide %697, %698 : tensor<2x4x16x1xf32>
    %cst_126 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %700 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<2x4x16x1xf32>
    %701 = stablehlo.add %699, %700 : tensor<2x4x16x1xf32>
    %702 = stablehlo.rsqrt %701 : tensor<2x4x16x1xf32>
    %703 = stablehlo.broadcast_in_dim %702, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x128xf32>
    %704 = stablehlo.multiply %694, %703 : tensor<2x4x16x128xf32>
    %705 = stablehlo.convert %704 : (tensor<2x4x16x128xf32>) -> tensor<2x4x16x128xf16>
    %706 = stablehlo.broadcast_in_dim %arg47, dims = [1, 3] : (tensor<4x128xf16>) -> tensor<1x4x1x128xf16>
    %707 = stablehlo.broadcast_in_dim %706, dims = [0, 1, 2, 3] : (tensor<1x4x1x128xf16>) -> tensor<2x4x16x128xf16>
    %708 = stablehlo.multiply %705, %707 : tensor<2x4x16x128xf16>
    %709 = stablehlo.convert %690 : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x128xf32>
    %710 = stablehlo.multiply %709, %709 : tensor<2x2x16x128xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %711 = stablehlo.reduce(%710 init: %cst_127) applies stablehlo.add across dimensions = [3] : (tensor<2x2x16x128xf32>, tensor<f32>) -> tensor<2x2x16xf32>
    %712 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2] : (tensor<2x2x16xf32>) -> tensor<2x2x16x1xf32>
    %cst_128 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %713 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %714 = stablehlo.divide %712, %713 : tensor<2x2x16x1xf32>
    %cst_129 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %715 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<2x2x16x1xf32>
    %716 = stablehlo.add %714, %715 : tensor<2x2x16x1xf32>
    %717 = stablehlo.rsqrt %716 : tensor<2x2x16x1xf32>
    %718 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2, 3] : (tensor<2x2x16x1xf32>) -> tensor<2x2x16x128xf32>
    %719 = stablehlo.multiply %709, %718 : tensor<2x2x16x128xf32>
    %720 = stablehlo.convert %719 : (tensor<2x2x16x128xf32>) -> tensor<2x2x16x128xf16>
    %721 = stablehlo.broadcast_in_dim %arg48, dims = [1, 3] : (tensor<2x128xf16>) -> tensor<1x2x1x128xf16>
    %722 = stablehlo.broadcast_in_dim %721, dims = [0, 1, 2, 3] : (tensor<1x2x1x128xf16>) -> tensor<2x2x16x128xf16>
    %723 = stablehlo.multiply %720, %722 : tensor<2x2x16x128xf16>
    %724 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_130 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %725 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %726 = stablehlo.divide %724, %725 : tensor<128xf32>
    %cst_131 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %727 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %728 = stablehlo.power %727, %726 : tensor<128xf32>
    %cst_132 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %729 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %730 = stablehlo.divide %729, %728 : tensor<128xf32>
    %731 = stablehlo.convert %arg1 : (tensor<2x16xf16>) -> tensor<2x16xf32>
    %732 = stablehlo.broadcast_in_dim %731, dims = [0, 2] : (tensor<2x16xf32>) -> tensor<2x1x16x1xf32>
    %733 = stablehlo.broadcast_in_dim %730, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %734 = stablehlo.broadcast_in_dim %732, dims = [0, 1, 2, 3] : (tensor<2x1x16x1xf32>) -> tensor<2x1x16x128xf32>
    %735 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x16x128xf32>
    %736 = stablehlo.multiply %734, %735 : tensor<2x1x16x128xf32>
    %737 = stablehlo.cosine %736 : tensor<2x1x16x128xf32>
    %738 = stablehlo.convert %737 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %739 = stablehlo.sine %736 : tensor<2x1x16x128xf32>
    %740 = stablehlo.convert %739 : (tensor<2x1x16x128xf32>) -> tensor<2x1x16x128xf16>
    %741 = stablehlo.broadcast_in_dim %738, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %742 = stablehlo.multiply %708, %741 : tensor<2x4x16x128xf16>
    %743 = stablehlo.slice %708 [0:2, 0:4, 0:16, 0:64] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %744 = stablehlo.slice %708 [0:2, 0:4, 0:16, 64:128] : (tensor<2x4x16x128xf16>) -> tensor<2x4x16x64xf16>
    %745 = stablehlo.negate %744 : tensor<2x4x16x64xf16>
    %746 = stablehlo.concatenate %745, %743, dim = 3 : (tensor<2x4x16x64xf16>, tensor<2x4x16x64xf16>) -> tensor<2x4x16x128xf16>
    %747 = stablehlo.broadcast_in_dim %740, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %748 = stablehlo.multiply %746, %747 : tensor<2x4x16x128xf16>
    %749 = stablehlo.add %742, %748 : tensor<2x4x16x128xf16>
    %750 = stablehlo.broadcast_in_dim %738, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %751 = stablehlo.multiply %723, %750 : tensor<2x2x16x128xf16>
    %752 = stablehlo.slice %723 [0:2, 0:2, 0:16, 0:64] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %753 = stablehlo.slice %723 [0:2, 0:2, 0:16, 64:128] : (tensor<2x2x16x128xf16>) -> tensor<2x2x16x64xf16>
    %754 = stablehlo.negate %753 : tensor<2x2x16x64xf16>
    %755 = stablehlo.concatenate %754, %752, dim = 3 : (tensor<2x2x16x64xf16>, tensor<2x2x16x64xf16>) -> tensor<2x2x16x128xf16>
    %756 = stablehlo.broadcast_in_dim %740, dims = [0, 1, 2, 3] : (tensor<2x1x16x128xf16>) -> tensor<2x2x16x128xf16>
    %757 = stablehlo.multiply %755, %756 : tensor<2x2x16x128xf16>
    %758 = stablehlo.add %751, %757 : tensor<2x2x16x128xf16>
    %759 = stablehlo.slice %758 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %760 = stablehlo.slice %758 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %761 = stablehlo.slice %758 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %762 = stablehlo.slice %758 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %763 = stablehlo.concatenate %759, %760, %761, %762, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %764 = stablehlo.slice %693 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %765 = stablehlo.slice %693 [0:2, 0:1, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %766 = stablehlo.slice %693 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %767 = stablehlo.slice %693 [0:2, 1:2, 0:16, 0:128] : (tensor<2x2x16x128xf16>) -> tensor<2x1x16x128xf16>
    %768 = stablehlo.concatenate %764, %765, %766, %767, dim = 1 : (tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>, tensor<2x1x16x128xf16>) -> tensor<2x4x16x128xf16>
    %769 = stablehlo.dot_general %749, %763, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x128xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x16xf16>
    %cst_133 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %770 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %771 = stablehlo.multiply %769, %770 : tensor<2x4x16x16xf16>
    %cst_134 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %772 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %773 = stablehlo.divide %771, %772 : tensor<2x4x16x16xf16>
    %774 = stablehlo.tanh %773 : tensor<2x4x16x16xf16>
    %cst_135 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %775 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %776 = stablehlo.multiply %774, %775 : tensor<2x4x16x16xf16>
    %777 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x16xf16>) -> tensor<2x16x1xf16>
    %778 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x16xf16>) -> tensor<2x1x16xf16>
    %779 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2] : (tensor<2x16x1xf16>) -> tensor<2x16x16xf16>
    %780 = stablehlo.broadcast_in_dim %778, dims = [0, 1, 2] : (tensor<2x1x16xf16>) -> tensor<2x16x16xf16>
    %781 = stablehlo.compare GE, %779, %780, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %782 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f16>) -> tensor<2x16x16xf16>
    %783 = stablehlo.compare GT, %arg2, %782, FLOAT : (tensor<2x16x16xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xi1>
    %784 = stablehlo.and %781, %783 : tensor<2x16x16xi1>
    %785 = stablehlo.broadcast_in_dim %784, dims = [0, 2, 3] : (tensor<2x16x16xi1>) -> tensor<2x1x16x16xi1>
    %cst_137 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %786 = call @_where(%785, %776, %cst_137) : (tensor<2x1x16x16xi1>, tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16x16xf16>
    %cst_138 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %787 = stablehlo.reduce(%786 init: %cst_138) applies stablehlo.maximum across dimensions = [3] : (tensor<2x4x16x16xf16>, tensor<f16>) -> tensor<2x4x16xf16>
    %cst_139 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %788 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f16>) -> tensor<2x4x16xf16>
    %789 = stablehlo.maximum %788, %787 : tensor<2x4x16xf16>
    %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2] : (tensor<2x4x16xf16>) -> tensor<2x4x16x1xf16>
    %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %792 = stablehlo.subtract %786, %791 : tensor<2x4x16x16xf16>
    %793 = stablehlo.exponential %792 : tensor<2x4x16x16xf16>
    %794 = stablehlo.convert %793 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %795 = stablehlo.reduce(%794 init: %cst_140) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %797 = stablehlo.convert %796 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %798 = stablehlo.broadcast_in_dim %797, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %799 = stablehlo.divide %793, %798 : tensor<2x4x16x16xf16>
    %800 = stablehlo.convert %799 : (tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %801 = stablehlo.reduce(%800 init: %cst_141) applies stablehlo.add across dimensions = [3] : (tensor<2x4x16x16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    %802 = stablehlo.broadcast_in_dim %801, dims = [0, 1, 2] : (tensor<2x4x16xf32>) -> tensor<2x4x16x1xf32>
    %803 = stablehlo.convert %802 : (tensor<2x4x16x1xf32>) -> tensor<2x4x16x1xf16>
    %804 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2, 3] : (tensor<2x4x16x1xf16>) -> tensor<2x4x16x16xf16>
    %805 = stablehlo.divide %799, %804 : tensor<2x4x16x16xf16>
    %806 = stablehlo.dot_general %805, %768, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16x16xf16>, tensor<2x4x16x128xf16>) -> tensor<2x4x16x128xf16>
    %807 = stablehlo.transpose %806, dims = [0, 2, 1, 3] : (tensor<2x4x16x128xf16>) -> tensor<2x16x4x128xf16>
    %808 = stablehlo.dot_general %807, %arg38, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x4x128xf16>, tensor<4x128x512xf16>) -> tensor<2x16x512xf16>
    %809 = stablehlo.add %668, %808 : tensor<2x16x512xf16>
    %810 = stablehlo.convert %809 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %811 = stablehlo.multiply %810, %810 : tensor<2x16x512xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %812 = stablehlo.reduce(%811 init: %cst_142) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %813 = stablehlo.broadcast_in_dim %812, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_143 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %814 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %815 = stablehlo.divide %813, %814 : tensor<2x16x1xf32>
    %cst_144 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %816 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %817 = stablehlo.add %815, %816 : tensor<2x16x1xf32>
    %818 = stablehlo.rsqrt %817 : tensor<2x16x1xf32>
    %819 = stablehlo.broadcast_in_dim %818, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %820 = stablehlo.multiply %810, %819 : tensor<2x16x512xf32>
    %821 = stablehlo.convert %820 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %822 = stablehlo.broadcast_in_dim %arg44, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %823 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %824 = stablehlo.multiply %821, %823 : tensor<2x16x512xf16>
    %cst_145 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %825 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %826 = stablehlo.multiply %825, %824 : tensor<2x16x512xf16>
    %827 = stablehlo.add %809, %826 : tensor<2x16x512xf16>
    %828 = stablehlo.convert %827 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %829 = stablehlo.multiply %828, %828 : tensor<2x16x512xf32>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %830 = stablehlo.reduce(%829 init: %cst_146) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %831 = stablehlo.broadcast_in_dim %830, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_147 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %832 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %833 = stablehlo.divide %831, %832 : tensor<2x16x1xf32>
    %cst_148 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %834 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %835 = stablehlo.add %833, %834 : tensor<2x16x1xf32>
    %836 = stablehlo.rsqrt %835 : tensor<2x16x1xf32>
    %837 = stablehlo.broadcast_in_dim %836, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %838 = stablehlo.multiply %828, %837 : tensor<2x16x512xf32>
    %839 = stablehlo.convert %838 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %840 = stablehlo.broadcast_in_dim %arg45, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %841 = stablehlo.broadcast_in_dim %840, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %842 = stablehlo.multiply %839, %841 : tensor<2x16x512xf16>
    %843 = stablehlo.dot_general %arg41, %842, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x512x1024xf16>, tensor<2x16x512xf16>) -> tensor<2x1024x2x16xf16>
    %844 = stablehlo.transpose %843, dims = [2, 0, 3, 1] : (tensor<2x1024x2x16xf16>) -> tensor<2x2x16x1024xf16>
    %845 = stablehlo.slice %844 [0:2, 0:1, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %846 = stablehlo.reshape %845 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %847 = stablehlo.slice %844 [0:2, 1:2, 0:16, 0:1024] : (tensor<2x2x16x1024xf16>) -> tensor<2x1x16x1024xf16>
    %848 = stablehlo.reshape %847 : (tensor<2x1x16x1024xf16>) -> tensor<2x16x1024xf16>
    %849 = stablehlo.negate %846 : tensor<2x16x1024xf16>
    %850 = stablehlo.exponential %849 : tensor<2x16x1024xf16>
    %cst_149 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %851 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %852 = stablehlo.add %851, %850 : tensor<2x16x1024xf16>
    %cst_150 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %853 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %854 = stablehlo.divide %853, %852 : tensor<2x16x1024xf16>
    %855 = stablehlo.multiply %846, %854 : tensor<2x16x1024xf16>
    %856 = stablehlo.multiply %855, %848 : tensor<2x16x1024xf16>
    %857 = stablehlo.dot_general %856, %arg42, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x16x1024xf16>, tensor<1024x512xf16>) -> tensor<2x16x512xf16>
    %858 = stablehlo.add %827, %857 : tensor<2x16x512xf16>
    %859 = stablehlo.convert %858 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %860 = stablehlo.multiply %859, %859 : tensor<2x16x512xf32>
    %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %861 = stablehlo.reduce(%860 init: %cst_151) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %862 = stablehlo.broadcast_in_dim %861, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_152 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %863 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %864 = stablehlo.divide %862, %863 : tensor<2x16x1xf32>
    %cst_153 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %865 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %866 = stablehlo.add %864, %865 : tensor<2x16x1xf32>
    %867 = stablehlo.rsqrt %866 : tensor<2x16x1xf32>
    %868 = stablehlo.broadcast_in_dim %867, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %869 = stablehlo.multiply %859, %868 : tensor<2x16x512xf32>
    %870 = stablehlo.convert %869 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %871 = stablehlo.broadcast_in_dim %arg46, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %872 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %873 = stablehlo.multiply %870, %872 : tensor<2x16x512xf16>
    %cst_154 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %874 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %875 = stablehlo.multiply %874, %873 : tensor<2x16x512xf16>
    %876 = stablehlo.add %858, %875 : tensor<2x16x512xf16>
    %877 = stablehlo.slice %arg2 [0:2, 0:16, 0:1] : (tensor<2x16x16xf16>) -> tensor<2x16x1xf16>
    %878 = stablehlo.reshape %877 : (tensor<2x16x1xf16>) -> tensor<2x16xf16>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %879 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f16>) -> tensor<2x16xf16>
    %880 = stablehlo.compare GT, %878, %879, FLOAT : (tensor<2x16xf16>, tensor<2x16xf16>) -> tensor<2x16xi1>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1] : (tensor<2x16xi1>) -> tensor<2x16x1xi1>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %882 = call @_where_87(%881, %876, %cst_156) : (tensor<2x16x1xi1>, tensor<2x16x512xf16>, tensor<f16>) -> tensor<2x16x512xf16>
    %cst_157 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %883 = stablehlo.broadcast_in_dim %cst_157, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %884 = stablehlo.compare GT, %882, %883, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_158 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %885 = call @_where_92(%884, %cst_158, %882) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %cst_159 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %886 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %887 = stablehlo.compare LT, %885, %886, FLOAT : (tensor<2x16x512xf16>, tensor<2x16x512xf16>) -> tensor<2x16x512xi1>
    %cst_160 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %888 = call @_where_92(%887, %cst_160, %885) : (tensor<2x16x512xi1>, tensor<f16>, tensor<2x16x512xf16>) -> tensor<2x16x512xf16>
    %889 = stablehlo.convert %888 : (tensor<2x16x512xf16>) -> tensor<2x16x512xf32>
    %890 = stablehlo.multiply %889, %889 : tensor<2x16x512xf32>
    %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %891 = stablehlo.reduce(%890 init: %cst_161) applies stablehlo.add across dimensions = [2] : (tensor<2x16x512xf32>, tensor<f32>) -> tensor<2x16xf32>
    %892 = stablehlo.broadcast_in_dim %891, dims = [0, 1] : (tensor<2x16xf32>) -> tensor<2x16x1xf32>
    %cst_162 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %893 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %894 = stablehlo.divide %892, %893 : tensor<2x16x1xf32>
    %cst_163 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %895 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<2x16x1xf32>
    %896 = stablehlo.add %894, %895 : tensor<2x16x1xf32>
    %897 = stablehlo.rsqrt %896 : tensor<2x16x1xf32>
    %898 = stablehlo.broadcast_in_dim %897, dims = [0, 1, 2] : (tensor<2x16x1xf32>) -> tensor<2x16x512xf32>
    %899 = stablehlo.multiply %889, %898 : tensor<2x16x512xf32>
    %900 = stablehlo.convert %899 : (tensor<2x16x512xf32>) -> tensor<2x16x512xf16>
    %901 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<2x16x512xf16>
    %903 = stablehlo.multiply %900, %902 : tensor<2x16x512xf16>
    %904 = stablehlo.dot_general %903, %arg3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16x512xf16>, tensor<1024x512xf16>) -> tensor<2x16x1024xf16>
    %cst_164 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %905 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %906 = stablehlo.divide %904, %905 : tensor<2x16x1024xf16>
    %907 = stablehlo.tanh %906 : tensor<2x16x1024xf16>
    %cst_165 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %908 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f16>) -> tensor<2x16x1024xf16>
    %909 = stablehlo.multiply %907, %908 : tensor<2x16x1024xf16>
    return %909 : tensor<2x16x1024xf16>
  }
  func.func private @_where(%arg0: tensor<2x1x16x16xi1>, %arg1: tensor<2x4x16x16xf16>, %arg2: tensor<f16>) -> tensor<2x4x16x16xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<2x1x16x16xi1>) -> tensor<2x4x16x16xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<2x4x16x16xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x16x16xi1>, tensor<2x4x16x16xf16>
    return %2 : tensor<2x4x16x16xf16>
  }
  func.func private @_where_87(%arg0: tensor<2x16x1xi1>, %arg1: tensor<2x16x512xf16>, %arg2: tensor<f16>) -> tensor<2x16x512xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x16x1xi1>) -> tensor<2x16x512xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x16x512xi1>, tensor<2x16x512xf16>
    return %2 : tensor<2x16x512xf16>
  }
  func.func private @_where_92(%arg0: tensor<2x16x512xi1>, %arg1: tensor<f16>, %arg2: tensor<2x16x512xf16>) -> tensor<2x16x512xf16> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f16>) -> tensor<2x16x512xf16>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<2x16x512xi1>, tensor<2x16x512xf16>
    return %1 : tensor<2x16x512xf16>
  }
}
