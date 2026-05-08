module @jit_real_shape_mini_gemma attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x8xi32>, %arg1: tensor<1x8xf16>, %arg2: tensor<1x8x8xf16>, %arg3: tensor<4096x2304xf16>, %arg4: tensor<2304xf16>, %arg5: tensor<8x256x2304xf16>, %arg6: tensor<2x4x2304x256xf16>, %arg7: tensor<8x2304x256xf16>, %arg8: tensor<2x2304x9216xf16>, %arg9: tensor<9216x2304xf16>, %arg10: tensor<2304xf16>, %arg11: tensor<2304xf16>, %arg12: tensor<2304xf16>, %arg13: tensor<2304xf16>, %arg14: tensor<8x256xf16>, %arg15: tensor<4x256xf16>) -> (tensor<1x8x4096xf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %c_0 = stablehlo.constant dense<4096> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x8xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<1x8xi1>, tensor<1x8xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %6 = "stablehlo.gather"(%arg3, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2304>}> : (tensor<4096x2304xf16>, tensor<1x8x1xi32>) -> tensor<1x8x2304xf16>
    %cst = stablehlo.constant dense<4.800000e+01> : tensor<f16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<1x8x2304xf16>
    %8 = stablehlo.multiply %6, %7 : tensor<1x8x2304xf16>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %10 = stablehlo.compare NE, %arg0, %9, SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x8xi1>) -> tensor<1x8x1xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %12 = call @_where(%11, %8, %cst_2) : (tensor<1x8x1xi1>, tensor<1x8x2304xf16>, tensor<f16>) -> tensor<1x8x2304xf16>
    %13 = stablehlo.convert %12 : (tensor<1x8x2304xf16>) -> tensor<1x8x2304xf32>
    %14 = chlo.square %13 : tensor<1x8x2304xf32> -> tensor<1x8x2304xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.reduce(%14 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_4 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %18 = stablehlo.divide %16, %17 : tensor<1x8x1xf32>
    %cst_5 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x8x1xf32>
    %21 = stablehlo.rsqrt %20 : tensor<1x8x1xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x2304xf32>
    %23 = stablehlo.multiply %13, %22 : tensor<1x8x2304xf32>
    %24 = stablehlo.convert %23 : (tensor<1x8x2304xf32>) -> tensor<1x8x2304xf16>
    %25 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<2304xf16>) -> tensor<1x1x2304xf16>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x1x2304xf16>) -> tensor<1x8x2304xf16>
    %27 = stablehlo.multiply %24, %26 : tensor<1x8x2304xf16>
    %28 = stablehlo.dot_general %27, %arg7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xf16>, tensor<8x2304x256xf16>) -> tensor<1x8x8x256xf16>
    %29 = stablehlo.transpose %28, dims = [0, 2, 1, 3] : (tensor<1x8x8x256xf16>) -> tensor<1x8x8x256xf16>
    %30 = stablehlo.dot_general %arg6, %27, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xf16>, tensor<1x8x2304xf16>) -> tensor<2x4x256x1x8xf16>
    %31 = stablehlo.transpose %30, dims = [3, 0, 4, 1, 2] : (tensor<2x4x256x1x8xf16>) -> tensor<1x2x8x4x256xf16>
    %32 = stablehlo.slice %31 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<1x2x8x4x256xf16>) -> tensor<1x1x8x4x256xf16>
    %33 = stablehlo.reshape %32 : (tensor<1x1x8x4x256xf16>) -> tensor<1x8x4x256xf16>
    %34 = stablehlo.transpose %33, dims = [0, 2, 1, 3] : (tensor<1x8x4x256xf16>) -> tensor<1x4x8x256xf16>
    %35 = stablehlo.slice %31 [0:1, 1:2, 0:8, 0:4, 0:256] : (tensor<1x2x8x4x256xf16>) -> tensor<1x1x8x4x256xf16>
    %36 = stablehlo.reshape %35 : (tensor<1x1x8x4x256xf16>) -> tensor<1x8x4x256xf16>
    %37 = stablehlo.transpose %36, dims = [0, 2, 1, 3] : (tensor<1x8x4x256xf16>) -> tensor<1x4x8x256xf16>
    %38 = stablehlo.convert %29 : (tensor<1x8x8x256xf16>) -> tensor<1x8x8x256xf32>
    %39 = chlo.square %38 : tensor<1x8x8x256xf32> -> tensor<1x8x8x256xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %40 = stablehlo.reduce(%39 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x256xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %cst_7 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %43 = stablehlo.divide %41, %42 : tensor<1x8x8x1xf32>
    %cst_8 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %45 = stablehlo.add %43, %44 : tensor<1x8x8x1xf32>
    %46 = stablehlo.rsqrt %45 : tensor<1x8x8x1xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x256xf32>
    %48 = stablehlo.multiply %38, %47 : tensor<1x8x8x256xf32>
    %49 = stablehlo.convert %48 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xf16>
    %50 = stablehlo.broadcast_in_dim %arg14, dims = [1, 3] : (tensor<8x256xf16>) -> tensor<1x8x1x256xf16>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x8x1x256xf16>) -> tensor<1x8x8x256xf16>
    %52 = stablehlo.multiply %49, %51 : tensor<1x8x8x256xf16>
    %53 = stablehlo.convert %34 : (tensor<1x4x8x256xf16>) -> tensor<1x4x8x256xf32>
    %54 = chlo.square %53 : tensor<1x4x8x256xf32> -> tensor<1x4x8x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %55 = stablehlo.reduce(%54 init: %cst_9) applies stablehlo.add across dimensions = [3] : (tensor<1x4x8x256xf32>, tensor<f32>) -> tensor<1x4x8xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2] : (tensor<1x4x8xf32>) -> tensor<1x4x8x1xf32>
    %cst_10 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x4x8x1xf32>
    %58 = stablehlo.divide %56, %57 : tensor<1x4x8x1xf32>
    %cst_11 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4x8x1xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x4x8x1xf32>
    %61 = stablehlo.rsqrt %60 : tensor<1x4x8x1xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x4x8x1xf32>) -> tensor<1x4x8x256xf32>
    %63 = stablehlo.multiply %53, %62 : tensor<1x4x8x256xf32>
    %64 = stablehlo.convert %63 : (tensor<1x4x8x256xf32>) -> tensor<1x4x8x256xf16>
    %65 = stablehlo.broadcast_in_dim %arg15, dims = [1, 3] : (tensor<4x256xf16>) -> tensor<1x4x1x256xf16>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<1x4x1x256xf16>) -> tensor<1x4x8x256xf16>
    %67 = stablehlo.multiply %64, %66 : tensor<1x4x8x256xf16>
    %68 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_12 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %70 = stablehlo.divide %68, %69 : tensor<128xf32>
    %cst_13 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %71 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %72 = stablehlo.power %71, %70 : tensor<128xf32>
    %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %74 = stablehlo.divide %73, %72 : tensor<128xf32>
    %75 = stablehlo.convert %arg1 : (tensor<1x8xf16>) -> tensor<1x8xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 2] : (tensor<1x8xf32>) -> tensor<1x1x8x1xf32>
    %77 = stablehlo.broadcast_in_dim %74, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %78 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x1x8x1xf32>) -> tensor<1x1x8x128xf32>
    %79 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x1x8x128xf32>
    %80 = stablehlo.multiply %78, %79 : tensor<1x1x8x128xf32>
    %81 = stablehlo.cosine %80 : tensor<1x1x8x128xf32>
    %82 = stablehlo.convert %81 : (tensor<1x1x8x128xf32>) -> tensor<1x1x8x128xf16>
    %83 = stablehlo.sine %80 : tensor<1x1x8x128xf32>
    %84 = stablehlo.convert %83 : (tensor<1x1x8x128xf32>) -> tensor<1x1x8x128xf16>
    %85 = stablehlo.slice %52 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xf16>) -> tensor<1x8x8x128xf16>
    %86 = stablehlo.slice %52 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xf16>) -> tensor<1x8x8x128xf16>
    %87 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x8x8x128xf16>
    %88 = stablehlo.multiply %85, %87 : tensor<1x8x8x128xf16>
    %89 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x8x8x128xf16>
    %90 = stablehlo.multiply %86, %89 : tensor<1x8x8x128xf16>
    %91 = stablehlo.subtract %88, %90 : tensor<1x8x8x128xf16>
    %92 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x8x8x128xf16>
    %93 = stablehlo.multiply %86, %92 : tensor<1x8x8x128xf16>
    %94 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x8x8x128xf16>
    %95 = stablehlo.multiply %85, %94 : tensor<1x8x8x128xf16>
    %96 = stablehlo.add %93, %95 : tensor<1x8x8x128xf16>
    %97 = stablehlo.concatenate %91, %96, dim = 3 : (tensor<1x8x8x128xf16>, tensor<1x8x8x128xf16>) -> tensor<1x8x8x256xf16>
    %98 = stablehlo.slice %67 [0:1, 0:4, 0:8, 0:128] : (tensor<1x4x8x256xf16>) -> tensor<1x4x8x128xf16>
    %99 = stablehlo.slice %67 [0:1, 0:4, 0:8, 128:256] : (tensor<1x4x8x256xf16>) -> tensor<1x4x8x128xf16>
    %100 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x4x8x128xf16>
    %101 = stablehlo.multiply %98, %100 : tensor<1x4x8x128xf16>
    %102 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x4x8x128xf16>
    %103 = stablehlo.multiply %99, %102 : tensor<1x4x8x128xf16>
    %104 = stablehlo.subtract %101, %103 : tensor<1x4x8x128xf16>
    %105 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x4x8x128xf16>
    %106 = stablehlo.multiply %99, %105 : tensor<1x4x8x128xf16>
    %107 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x8x128xf16>) -> tensor<1x4x8x128xf16>
    %108 = stablehlo.multiply %98, %107 : tensor<1x4x8x128xf16>
    %109 = stablehlo.add %106, %108 : tensor<1x4x8x128xf16>
    %110 = stablehlo.concatenate %104, %109, dim = 3 : (tensor<1x4x8x128xf16>, tensor<1x4x8x128xf16>) -> tensor<1x4x8x256xf16>
    %111 = stablehlo.slice %110 [0:1, 0:1, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %112 = stablehlo.slice %110 [0:1, 0:1, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %113 = stablehlo.slice %110 [0:1, 1:2, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %114 = stablehlo.slice %110 [0:1, 1:2, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %115 = stablehlo.slice %110 [0:1, 2:3, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %116 = stablehlo.slice %110 [0:1, 2:3, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %117 = stablehlo.slice %110 [0:1, 3:4, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %118 = stablehlo.slice %110 [0:1, 3:4, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %119 = stablehlo.concatenate %111, %112, %113, %114, %115, %116, %117, %118, dim = 1 : (tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>) -> tensor<1x8x8x256xf16>
    %120 = stablehlo.slice %37 [0:1, 0:1, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %121 = stablehlo.slice %37 [0:1, 0:1, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %122 = stablehlo.slice %37 [0:1, 1:2, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %123 = stablehlo.slice %37 [0:1, 1:2, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %124 = stablehlo.slice %37 [0:1, 2:3, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %125 = stablehlo.slice %37 [0:1, 2:3, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %126 = stablehlo.slice %37 [0:1, 3:4, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %127 = stablehlo.slice %37 [0:1, 3:4, 0:8, 0:256] : (tensor<1x4x8x256xf16>) -> tensor<1x1x8x256xf16>
    %128 = stablehlo.concatenate %120, %121, %122, %123, %124, %125, %126, %127, dim = 1 : (tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>, tensor<1x1x8x256xf16>) -> tensor<1x8x8x256xf16>
    %129 = stablehlo.dot_general %97, %119, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xf16>, tensor<1x8x8x256xf16>) -> tensor<1x8x8x8xf16>
    %cst_15 = stablehlo.constant dense<6.250000e-02> : tensor<f16>
    %130 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f16>) -> tensor<1x8x8x8xf16>
    %131 = stablehlo.multiply %129, %130 : tensor<1x8x8x8xf16>
    %cst_16 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %132 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f16>) -> tensor<1x8x8x8xf16>
    %133 = stablehlo.divide %131, %132 : tensor<1x8x8x8xf16>
    %134 = stablehlo.tanh %133 : tensor<1x8x8x8xf16>
    %cst_17 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %135 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f16>) -> tensor<1x8x8x8xf16>
    %136 = stablehlo.multiply %134, %135 : tensor<1x8x8x8xf16>
    %137 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x8xf16>) -> tensor<1x8x1xf16>
    %138 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<1x8xf16>) -> tensor<1x1x8xf16>
    %139 = stablehlo.broadcast_in_dim %137, dims = [0, 1, 2] : (tensor<1x8x1xf16>) -> tensor<1x8x8xf16>
    %140 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2] : (tensor<1x1x8xf16>) -> tensor<1x8x8xf16>
    %141 = stablehlo.compare GE, %139, %140, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %142 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f16>) -> tensor<1x8x8xf16>
    %143 = stablehlo.compare GT, %arg2, %142, FLOAT : (tensor<1x8x8xf16>, tensor<1x8x8xf16>) -> tensor<1x8x8xi1>
    %144 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x8xi1>) -> tensor<1x8x1xi1>
    %145 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x1x8xi1>
    %146 = stablehlo.and %141, %143 : tensor<1x8x8xi1>
    %147 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2] : (tensor<1x8x1xi1>) -> tensor<1x8x8xi1>
    %148 = stablehlo.and %146, %147 : tensor<1x8x8xi1>
    %149 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<1x1x8xi1>) -> tensor<1x8x8xi1>
    %150 = stablehlo.and %148, %149 : tensor<1x8x8xi1>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 2, 3] : (tensor<1x8x8xi1>) -> tensor<1x1x8x8xi1>
    %cst_19 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %152 = call @_where_67(%151, %136, %cst_19) : (tensor<1x1x8x8xi1>, tensor<1x8x8x8xf16>, tensor<f16>) -> tensor<1x8x8x8xf16>
    %cst_20 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %153 = stablehlo.reduce(%152 init: %cst_20) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xf16>, tensor<f16>) -> tensor<1x8x8xf16>
    %cst_21 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %154 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f16>) -> tensor<1x8x8xf16>
    %155 = stablehlo.maximum %154, %153 : tensor<1x8x8xf16>
    %156 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2] : (tensor<1x8x8xf16>) -> tensor<1x8x8x1xf16>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf16>) -> tensor<1x8x8x8xf16>
    %158 = stablehlo.subtract %152, %157 : tensor<1x8x8x8xf16>
    %159 = stablehlo.exponential %158 : tensor<1x8x8x8xf16>
    %160 = stablehlo.convert %159 : (tensor<1x8x8x8xf16>) -> tensor<1x8x8x8xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %161 = stablehlo.reduce(%160 init: %cst_22) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %163 = stablehlo.convert %162 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xf16>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf16>) -> tensor<1x8x8x8xf16>
    %165 = stablehlo.divide %159, %164 : tensor<1x8x8x8xf16>
    %166 = stablehlo.convert %165 : (tensor<1x8x8x8xf16>) -> tensor<1x8x8x8xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %167 = stablehlo.reduce(%166 init: %cst_23) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %169 = stablehlo.convert %168 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xf16>
    %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf16>) -> tensor<1x8x8x8xf16>
    %171 = stablehlo.divide %165, %170 : tensor<1x8x8x8xf16>
    %172 = stablehlo.dot_general %171, %128, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x8xf16>, tensor<1x8x8x256xf16>) -> tensor<1x8x8x256xf16>
    %173 = stablehlo.transpose %172, dims = [0, 2, 1, 3] : (tensor<1x8x8x256xf16>) -> tensor<1x8x8x256xf16>
    %174 = stablehlo.dot_general %173, %arg5, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xf16>, tensor<8x256x2304xf16>) -> tensor<1x8x2304xf16>
    %175 = stablehlo.add %12, %174 : tensor<1x8x2304xf16>
    %176 = stablehlo.convert %175 : (tensor<1x8x2304xf16>) -> tensor<1x8x2304xf32>
    %177 = chlo.square %176 : tensor<1x8x2304xf32> -> tensor<1x8x2304xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %178 = stablehlo.reduce(%177 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_25 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %180 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %181 = stablehlo.divide %179, %180 : tensor<1x8x1xf32>
    %cst_26 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %182 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %183 = stablehlo.add %181, %182 : tensor<1x8x1xf32>
    %184 = stablehlo.rsqrt %183 : tensor<1x8x1xf32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x2304xf32>
    %186 = stablehlo.multiply %176, %185 : tensor<1x8x2304xf32>
    %187 = stablehlo.convert %186 : (tensor<1x8x2304xf32>) -> tensor<1x8x2304xf16>
    %188 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<2304xf16>) -> tensor<1x1x2304xf16>
    %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2] : (tensor<1x1x2304xf16>) -> tensor<1x8x2304xf16>
    %190 = stablehlo.multiply %187, %189 : tensor<1x8x2304xf16>
    %cst_27 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %191 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f16>) -> tensor<1x8x2304xf16>
    %192 = stablehlo.multiply %191, %190 : tensor<1x8x2304xf16>
    %193 = stablehlo.add %175, %192 : tensor<1x8x2304xf16>
    %194 = stablehlo.convert %193 : (tensor<1x8x2304xf16>) -> tensor<1x8x2304xf32>
    %195 = chlo.square %194 : tensor<1x8x2304xf32> -> tensor<1x8x2304xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %196 = stablehlo.reduce(%195 init: %cst_28) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %197 = stablehlo.broadcast_in_dim %196, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_29 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %199 = stablehlo.divide %197, %198 : tensor<1x8x1xf32>
    %cst_30 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %200 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %201 = stablehlo.add %199, %200 : tensor<1x8x1xf32>
    %202 = stablehlo.rsqrt %201 : tensor<1x8x1xf32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x2304xf32>
    %204 = stablehlo.multiply %194, %203 : tensor<1x8x2304xf32>
    %205 = stablehlo.convert %204 : (tensor<1x8x2304xf32>) -> tensor<1x8x2304xf16>
    %206 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<2304xf16>) -> tensor<1x1x2304xf16>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2] : (tensor<1x1x2304xf16>) -> tensor<1x8x2304xf16>
    %208 = stablehlo.multiply %205, %207 : tensor<1x8x2304xf16>
    %209 = stablehlo.dot_general %arg8, %208, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2304x9216xf16>, tensor<1x8x2304xf16>) -> tensor<2x9216x1x8xf16>
    %210 = stablehlo.transpose %209, dims = [2, 0, 3, 1] : (tensor<2x9216x1x8xf16>) -> tensor<1x2x8x9216xf16>
    %211 = stablehlo.slice %210 [0:1, 0:1, 0:8, 0:9216] : (tensor<1x2x8x9216xf16>) -> tensor<1x1x8x9216xf16>
    %212 = stablehlo.reshape %211 : (tensor<1x1x8x9216xf16>) -> tensor<1x8x9216xf16>
    %213 = stablehlo.slice %210 [0:1, 1:2, 0:8, 0:9216] : (tensor<1x2x8x9216xf16>) -> tensor<1x1x8x9216xf16>
    %214 = stablehlo.reshape %213 : (tensor<1x1x8x9216xf16>) -> tensor<1x8x9216xf16>
    %215 = stablehlo.negate %212 : tensor<1x8x9216xf16>
    %216 = stablehlo.exponential %215 : tensor<1x8x9216xf16>
    %cst_31 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %217 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f16>) -> tensor<1x8x9216xf16>
    %218 = stablehlo.add %217, %216 : tensor<1x8x9216xf16>
    %cst_32 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %219 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f16>) -> tensor<1x8x9216xf16>
    %220 = stablehlo.divide %219, %218 : tensor<1x8x9216xf16>
    %221 = stablehlo.multiply %212, %220 : tensor<1x8x9216xf16>
    %222 = stablehlo.multiply %221, %214 : tensor<1x8x9216xf16>
    %cst_33 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %223 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f16>) -> tensor<1x8x9216xf16>
    %224 = stablehlo.divide %222, %223 : tensor<1x8x9216xf16>
    %225 = stablehlo.tanh %224 : tensor<1x8x9216xf16>
    %cst_34 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %226 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f16>) -> tensor<1x8x9216xf16>
    %227 = stablehlo.multiply %225, %226 : tensor<1x8x9216xf16>
    %228 = stablehlo.dot_general %227, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xf16>, tensor<9216x2304xf16>) -> tensor<1x8x2304xf16>
    %229 = stablehlo.add %193, %228 : tensor<1x8x2304xf16>
    %230 = stablehlo.convert %229 : (tensor<1x8x2304xf16>) -> tensor<1x8x2304xf32>
    %231 = chlo.square %230 : tensor<1x8x2304xf32> -> tensor<1x8x2304xf32>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %232 = stablehlo.reduce(%231 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %233 = stablehlo.broadcast_in_dim %232, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_36 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %234 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %235 = stablehlo.divide %233, %234 : tensor<1x8x1xf32>
    %cst_37 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %236 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %237 = stablehlo.add %235, %236 : tensor<1x8x1xf32>
    %238 = stablehlo.rsqrt %237 : tensor<1x8x1xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x2304xf32>
    %240 = stablehlo.multiply %230, %239 : tensor<1x8x2304xf32>
    %241 = stablehlo.convert %240 : (tensor<1x8x2304xf32>) -> tensor<1x8x2304xf16>
    %242 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<2304xf16>) -> tensor<1x1x2304xf16>
    %243 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<1x1x2304xf16>) -> tensor<1x8x2304xf16>
    %244 = stablehlo.multiply %241, %243 : tensor<1x8x2304xf16>
    %cst_38 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %245 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f16>) -> tensor<1x8x2304xf16>
    %246 = stablehlo.multiply %245, %244 : tensor<1x8x2304xf16>
    %247 = stablehlo.add %229, %246 : tensor<1x8x2304xf16>
    %248 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x8xi1>) -> tensor<1x8x1xi1>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %249 = call @_where(%248, %247, %cst_39) : (tensor<1x8x1xi1>, tensor<1x8x2304xf16>, tensor<f16>) -> tensor<1x8x2304xf16>
    %cst_40 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %250 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f16>) -> tensor<1x8x2304xf16>
    %251 = stablehlo.compare GT, %249, %250, FLOAT : (tensor<1x8x2304xf16>, tensor<1x8x2304xf16>) -> tensor<1x8x2304xi1>
    %cst_41 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %252 = call @_where_95(%251, %cst_41, %249) : (tensor<1x8x2304xi1>, tensor<f16>, tensor<1x8x2304xf16>) -> tensor<1x8x2304xf16>
    %cst_42 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %253 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f16>) -> tensor<1x8x2304xf16>
    %254 = stablehlo.compare LT, %252, %253, FLOAT : (tensor<1x8x2304xf16>, tensor<1x8x2304xf16>) -> tensor<1x8x2304xi1>
    %cst_43 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %255 = call @_where_95(%254, %cst_43, %252) : (tensor<1x8x2304xi1>, tensor<f16>, tensor<1x8x2304xf16>) -> tensor<1x8x2304xf16>
    %256 = stablehlo.convert %255 : (tensor<1x8x2304xf16>) -> tensor<1x8x2304xf32>
    %257 = chlo.square %256 : tensor<1x8x2304xf32> -> tensor<1x8x2304xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %258 = stablehlo.reduce(%257 init: %cst_44) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_45 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %260 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %261 = stablehlo.divide %259, %260 : tensor<1x8x1xf32>
    %cst_46 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %262 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %263 = stablehlo.add %261, %262 : tensor<1x8x1xf32>
    %264 = stablehlo.rsqrt %263 : tensor<1x8x1xf32>
    %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x2304xf32>
    %266 = stablehlo.multiply %256, %265 : tensor<1x8x2304xf32>
    %267 = stablehlo.convert %266 : (tensor<1x8x2304xf32>) -> tensor<1x8x2304xf16>
    %268 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<2304xf16>) -> tensor<1x1x2304xf16>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2] : (tensor<1x1x2304xf16>) -> tensor<1x8x2304xf16>
    %270 = stablehlo.multiply %267, %269 : tensor<1x8x2304xf16>
    %271 = stablehlo.dot_general %270, %arg3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xf16>, tensor<4096x2304xf16>) -> tensor<1x8x4096xf16>
    %cst_47 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %272 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f16>) -> tensor<1x8x4096xf16>
    %273 = stablehlo.divide %271, %272 : tensor<1x8x4096xf16>
    %274 = stablehlo.tanh %273 : tensor<1x8x4096xf16>
    %cst_48 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %275 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f16>) -> tensor<1x8x4096xf16>
    %276 = stablehlo.multiply %274, %275 : tensor<1x8x4096xf16>
    return %276 : tensor<1x8x4096xf16>
  }
  func.func private @_where(%arg0: tensor<1x8x1xi1>, %arg1: tensor<1x8x2304xf16>, %arg2: tensor<f16>) -> tensor<1x8x2304xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x8x1xi1>) -> tensor<1x8x2304xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<1x8x2304xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<1x8x2304xi1>, tensor<1x8x2304xf16>
    return %2 : tensor<1x8x2304xf16>
  }
  func.func private @_where_67(%arg0: tensor<1x1x8x8xi1>, %arg1: tensor<1x8x8x8xf16>, %arg2: tensor<f16>) -> tensor<1x8x8x8xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<1x8x8x8xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<1x8x8x8xi1>, tensor<1x8x8x8xf16>
    return %2 : tensor<1x8x8x8xf16>
  }
  func.func private @_where_95(%arg0: tensor<1x8x2304xi1>, %arg1: tensor<f16>, %arg2: tensor<1x8x2304xf16>) -> tensor<1x8x2304xf16> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f16>) -> tensor<1x8x2304xf16>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<1x8x2304xi1>, tensor<1x8x2304xf16>
    return %1 : tensor<1x8x2304xf16>
  }
}
