module @jit_tiny_gpt2_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x5xi32>, %arg1: tensor<1x5xi32>, %arg2: tensor<50257x2xf16>, %arg3: tensor<1024x2xf16>, %arg4: tensor<2xf16>, %arg5: tensor<2xf16>, %arg6: tensor<2xf16>, %arg7: tensor<2xf16>, %arg8: tensor<2x6xf16>, %arg9: tensor<6xf16>, %arg10: tensor<2x2xf16>, %arg11: tensor<2xf16>, %arg12: tensor<2xf16>, %arg13: tensor<2xf16>, %arg14: tensor<2x8xf16>, %arg15: tensor<8xf16>, %arg16: tensor<8x2xf16>, %arg17: tensor<2xf16>, %arg18: tensor<2xf16>, %arg19: tensor<2xf16>, %arg20: tensor<2x6xf16>, %arg21: tensor<6xf16>, %arg22: tensor<2x2xf16>, %arg23: tensor<2xf16>, %arg24: tensor<2xf16>, %arg25: tensor<2xf16>, %arg26: tensor<2x8xf16>, %arg27: tensor<8xf16>, %arg28: tensor<8x2xf16>, %arg29: tensor<2xf16>) -> (tensor<1x5x50257xf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x5xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<1x5xi32>, tensor<1x5xi32>) -> tensor<1x5xi1>
    %c_0 = stablehlo.constant dense<50257> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x5xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x5xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<1x5xi1>, tensor<1x5xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x5xi32>) -> tensor<1x5x1xi32>
    %6 = "stablehlo.gather"(%arg2, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2>}> : (tensor<50257x2xf16>, tensor<1x5x1xi32>) -> tensor<1x5x2xf16>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x5xi32>
    %8 = stablehlo.compare LT, %arg1, %7, SIGNED : (tensor<1x5xi32>, tensor<1x5xi32>) -> tensor<1x5xi1>
    %c_2 = stablehlo.constant dense<1024> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x5xi32>
    %10 = stablehlo.add %arg1, %9 : tensor<1x5xi32>
    %11 = stablehlo.select %8, %10, %arg1 : tensor<1x5xi1>, tensor<1x5xi32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x5xi32>) -> tensor<1x5x1xi32>
    %13 = "stablehlo.gather"(%arg3, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2>}> : (tensor<1024x2xf16>, tensor<1x5x1xi32>) -> tensor<1x5x2xf16>
    %14 = stablehlo.add %6, %13 : tensor<1x5x2xf16>
    %15 = stablehlo.convert %14 : (tensor<1x5x2xf16>) -> tensor<1x5x2xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %19 = stablehlo.divide %17, %18 : tensor<1x5x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %21 = stablehlo.subtract %15, %20 : tensor<1x5x2xf32>
    %22 = stablehlo.multiply %21, %21 : tensor<1x5x2xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %23 = stablehlo.reduce(%22 init: %cst_4) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %26 = stablehlo.divide %24, %25 : tensor<1x5x1xf32>
    %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x5x1xf32>
    %29 = stablehlo.rsqrt %28 : tensor<1x5x1xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %31 = stablehlo.multiply %21, %30 : tensor<1x5x2xf32>
    %32 = stablehlo.convert %31 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf16>
    %33 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %35 = stablehlo.multiply %32, %34 : tensor<1x5x2xf16>
    %36 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %38 = stablehlo.add %35, %37 : tensor<1x5x2xf16>
    %39 = stablehlo.dot_general %38, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x2xf16>, tensor<2x6xf16>) -> tensor<1x5x6xf16>
    %40 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<6xf16>) -> tensor<1x1x6xf16>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<1x1x6xf16>) -> tensor<1x5x6xf16>
    %42 = stablehlo.add %39, %41 : tensor<1x5x6xf16>
    %43 = stablehlo.slice %42 [0:1, 0:5, 0:2] : (tensor<1x5x6xf16>) -> tensor<1x5x2xf16>
    %44 = stablehlo.slice %42 [0:1, 0:5, 2:4] : (tensor<1x5x6xf16>) -> tensor<1x5x2xf16>
    %45 = stablehlo.slice %42 [0:1, 0:5, 4:6] : (tensor<1x5x6xf16>) -> tensor<1x5x2xf16>
    %46 = stablehlo.reshape %43 : (tensor<1x5x2xf16>) -> tensor<1x5x2x1xf16>
    %47 = stablehlo.transpose %46, dims = [0, 2, 1, 3] : (tensor<1x5x2x1xf16>) -> tensor<1x2x5x1xf16>
    %48 = stablehlo.reshape %44 : (tensor<1x5x2xf16>) -> tensor<1x5x2x1xf16>
    %49 = stablehlo.transpose %48, dims = [0, 2, 1, 3] : (tensor<1x5x2x1xf16>) -> tensor<1x2x5x1xf16>
    %50 = stablehlo.reshape %45 : (tensor<1x5x2xf16>) -> tensor<1x5x2x1xf16>
    %51 = stablehlo.transpose %50, dims = [0, 2, 1, 3] : (tensor<1x5x2x1xf16>) -> tensor<1x2x5x1xf16>
    %52 = stablehlo.dot_general %47, %49, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x2x5x1xf16>, tensor<1x2x5x1xf16>) -> tensor<1x2x5x5xf16>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %53 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f16>) -> tensor<1x2x5x5xf16>
    %54 = stablehlo.multiply %52, %53 : tensor<1x2x5x5xf16>
    %c_8 = stablehlo.constant dense<true> : tensor<i1>
    %55 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i1>) -> tensor<5x5xi1>
    %56 = call @tril(%55) : (tensor<5x5xi1>) -> tensor<5x5xi1>
    %57 = stablehlo.broadcast_in_dim %56, dims = [2, 3] : (tensor<5x5xi1>) -> tensor<1x1x5x5xi1>
    %cst_9 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %58 = call @_where(%57, %54, %cst_9) : (tensor<1x1x5x5xi1>, tensor<1x2x5x5xf16>, tensor<f16>) -> tensor<1x2x5x5xf16>
    %cst_10 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %59 = stablehlo.reduce(%58 init: %cst_10) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x5x5xf16>, tensor<f16>) -> tensor<1x2x5xf16>
    %cst_11 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %60 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f16>) -> tensor<1x2x5xf16>
    %61 = stablehlo.maximum %60, %59 : tensor<1x2x5xf16>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2] : (tensor<1x2x5xf16>) -> tensor<1x2x5x1xf16>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<1x2x5x1xf16>) -> tensor<1x2x5x5xf16>
    %64 = stablehlo.subtract %58, %63 : tensor<1x2x5x5xf16>
    %65 = stablehlo.exponential %64 : tensor<1x2x5x5xf16>
    %66 = stablehlo.convert %65 : (tensor<1x2x5x5xf16>) -> tensor<1x2x5x5xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %67 = stablehlo.reduce(%66 init: %cst_12) applies stablehlo.add across dimensions = [3] : (tensor<1x2x5x5xf32>, tensor<f32>) -> tensor<1x2x5xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2] : (tensor<1x2x5xf32>) -> tensor<1x2x5x1xf32>
    %69 = stablehlo.convert %68 : (tensor<1x2x5x1xf32>) -> tensor<1x2x5x1xf16>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x2x5x1xf16>) -> tensor<1x2x5x5xf16>
    %71 = stablehlo.divide %65, %70 : tensor<1x2x5x5xf16>
    %72 = stablehlo.dot_general %71, %51, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x2x5x5xf16>, tensor<1x2x5x1xf16>) -> tensor<1x2x5x1xf16>
    %73 = stablehlo.transpose %72, dims = [0, 2, 1, 3] : (tensor<1x2x5x1xf16>) -> tensor<1x5x2x1xf16>
    %74 = stablehlo.reshape %73 : (tensor<1x5x2x1xf16>) -> tensor<1x5x2xf16>
    %75 = stablehlo.dot_general %74, %arg10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x2xf16>, tensor<2x2xf16>) -> tensor<1x5x2xf16>
    %76 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %78 = stablehlo.add %75, %77 : tensor<1x5x2xf16>
    %79 = stablehlo.add %14, %78 : tensor<1x5x2xf16>
    %80 = stablehlo.convert %79 : (tensor<1x5x2xf16>) -> tensor<1x5x2xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %81 = stablehlo.reduce(%80 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_14 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %84 = stablehlo.divide %82, %83 : tensor<1x5x1xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %86 = stablehlo.subtract %80, %85 : tensor<1x5x2xf32>
    %87 = stablehlo.multiply %86, %86 : tensor<1x5x2xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %88 = stablehlo.reduce(%87 init: %cst_15) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_16 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %91 = stablehlo.divide %89, %90 : tensor<1x5x1xf32>
    %cst_17 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %93 = stablehlo.add %91, %92 : tensor<1x5x1xf32>
    %94 = stablehlo.rsqrt %93 : tensor<1x5x1xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %96 = stablehlo.multiply %86, %95 : tensor<1x5x2xf32>
    %97 = stablehlo.convert %96 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf16>
    %98 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %100 = stablehlo.multiply %97, %99 : tensor<1x5x2xf16>
    %101 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %103 = stablehlo.add %100, %102 : tensor<1x5x2xf16>
    %104 = stablehlo.dot_general %103, %arg14, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x2xf16>, tensor<2x8xf16>) -> tensor<1x5x8xf16>
    %105 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<8xf16>) -> tensor<1x1x8xf16>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2] : (tensor<1x1x8xf16>) -> tensor<1x5x8xf16>
    %107 = stablehlo.add %104, %106 : tensor<1x5x8xf16>
    %cst_18 = stablehlo.constant dense<5.000000e-01> : tensor<f16>
    %108 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %109 = stablehlo.multiply %108, %107 : tensor<1x5x8xf16>
    %cst_19 = stablehlo.constant dense<4.470830e-02> : tensor<f16>
    %110 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %111 = stablehlo.multiply %110, %107 : tensor<1x5x8xf16>
    %112 = stablehlo.multiply %111, %107 : tensor<1x5x8xf16>
    %113 = stablehlo.multiply %112, %107 : tensor<1x5x8xf16>
    %114 = stablehlo.add %107, %113 : tensor<1x5x8xf16>
    %cst_20 = stablehlo.constant dense<7.978520e-01> : tensor<f16>
    %115 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %116 = stablehlo.multiply %115, %114 : tensor<1x5x8xf16>
    %117 = stablehlo.tanh %116 : tensor<1x5x8xf16>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %118 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %119 = stablehlo.add %118, %117 : tensor<1x5x8xf16>
    %120 = stablehlo.multiply %109, %119 : tensor<1x5x8xf16>
    %121 = stablehlo.dot_general %120, %arg16, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x8xf16>, tensor<8x2xf16>) -> tensor<1x5x2xf16>
    %122 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %124 = stablehlo.add %121, %123 : tensor<1x5x2xf16>
    %125 = stablehlo.add %79, %124 : tensor<1x5x2xf16>
    %126 = stablehlo.convert %125 : (tensor<1x5x2xf16>) -> tensor<1x5x2xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %127 = stablehlo.reduce(%126 init: %cst_22) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_23 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %129 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %130 = stablehlo.divide %128, %129 : tensor<1x5x1xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %132 = stablehlo.subtract %126, %131 : tensor<1x5x2xf32>
    %133 = stablehlo.multiply %132, %132 : tensor<1x5x2xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %134 = stablehlo.reduce(%133 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %135 = stablehlo.broadcast_in_dim %134, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_25 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %136 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %137 = stablehlo.divide %135, %136 : tensor<1x5x1xf32>
    %cst_26 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %138 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %139 = stablehlo.add %137, %138 : tensor<1x5x1xf32>
    %140 = stablehlo.rsqrt %139 : tensor<1x5x1xf32>
    %141 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %142 = stablehlo.multiply %132, %141 : tensor<1x5x2xf32>
    %143 = stablehlo.convert %142 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf16>
    %144 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %146 = stablehlo.multiply %143, %145 : tensor<1x5x2xf16>
    %147 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %149 = stablehlo.add %146, %148 : tensor<1x5x2xf16>
    %150 = stablehlo.dot_general %149, %arg20, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x2xf16>, tensor<2x6xf16>) -> tensor<1x5x6xf16>
    %151 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<6xf16>) -> tensor<1x1x6xf16>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2] : (tensor<1x1x6xf16>) -> tensor<1x5x6xf16>
    %153 = stablehlo.add %150, %152 : tensor<1x5x6xf16>
    %154 = stablehlo.slice %153 [0:1, 0:5, 0:2] : (tensor<1x5x6xf16>) -> tensor<1x5x2xf16>
    %155 = stablehlo.slice %153 [0:1, 0:5, 2:4] : (tensor<1x5x6xf16>) -> tensor<1x5x2xf16>
    %156 = stablehlo.slice %153 [0:1, 0:5, 4:6] : (tensor<1x5x6xf16>) -> tensor<1x5x2xf16>
    %157 = stablehlo.reshape %154 : (tensor<1x5x2xf16>) -> tensor<1x5x2x1xf16>
    %158 = stablehlo.transpose %157, dims = [0, 2, 1, 3] : (tensor<1x5x2x1xf16>) -> tensor<1x2x5x1xf16>
    %159 = stablehlo.reshape %155 : (tensor<1x5x2xf16>) -> tensor<1x5x2x1xf16>
    %160 = stablehlo.transpose %159, dims = [0, 2, 1, 3] : (tensor<1x5x2x1xf16>) -> tensor<1x2x5x1xf16>
    %161 = stablehlo.reshape %156 : (tensor<1x5x2xf16>) -> tensor<1x5x2x1xf16>
    %162 = stablehlo.transpose %161, dims = [0, 2, 1, 3] : (tensor<1x5x2x1xf16>) -> tensor<1x2x5x1xf16>
    %163 = stablehlo.dot_general %158, %160, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x2x5x1xf16>, tensor<1x2x5x1xf16>) -> tensor<1x2x5x5xf16>
    %cst_27 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %164 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f16>) -> tensor<1x2x5x5xf16>
    %165 = stablehlo.multiply %163, %164 : tensor<1x2x5x5xf16>
    %c_28 = stablehlo.constant dense<true> : tensor<i1>
    %166 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i1>) -> tensor<5x5xi1>
    %167 = call @tril(%166) : (tensor<5x5xi1>) -> tensor<5x5xi1>
    %168 = stablehlo.broadcast_in_dim %167, dims = [2, 3] : (tensor<5x5xi1>) -> tensor<1x1x5x5xi1>
    %cst_29 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %169 = call @_where(%168, %165, %cst_29) : (tensor<1x1x5x5xi1>, tensor<1x2x5x5xf16>, tensor<f16>) -> tensor<1x2x5x5xf16>
    %cst_30 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %170 = stablehlo.reduce(%169 init: %cst_30) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x5x5xf16>, tensor<f16>) -> tensor<1x2x5xf16>
    %cst_31 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %171 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f16>) -> tensor<1x2x5xf16>
    %172 = stablehlo.maximum %171, %170 : tensor<1x2x5xf16>
    %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2] : (tensor<1x2x5xf16>) -> tensor<1x2x5x1xf16>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<1x2x5x1xf16>) -> tensor<1x2x5x5xf16>
    %175 = stablehlo.subtract %169, %174 : tensor<1x2x5x5xf16>
    %176 = stablehlo.exponential %175 : tensor<1x2x5x5xf16>
    %177 = stablehlo.convert %176 : (tensor<1x2x5x5xf16>) -> tensor<1x2x5x5xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %178 = stablehlo.reduce(%177 init: %cst_32) applies stablehlo.add across dimensions = [3] : (tensor<1x2x5x5xf32>, tensor<f32>) -> tensor<1x2x5xf32>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2] : (tensor<1x2x5xf32>) -> tensor<1x2x5x1xf32>
    %180 = stablehlo.convert %179 : (tensor<1x2x5x1xf32>) -> tensor<1x2x5x1xf16>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2, 3] : (tensor<1x2x5x1xf16>) -> tensor<1x2x5x5xf16>
    %182 = stablehlo.divide %176, %181 : tensor<1x2x5x5xf16>
    %183 = stablehlo.dot_general %182, %162, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x2x5x5xf16>, tensor<1x2x5x1xf16>) -> tensor<1x2x5x1xf16>
    %184 = stablehlo.transpose %183, dims = [0, 2, 1, 3] : (tensor<1x2x5x1xf16>) -> tensor<1x5x2x1xf16>
    %185 = stablehlo.reshape %184 : (tensor<1x5x2x1xf16>) -> tensor<1x5x2xf16>
    %186 = stablehlo.dot_general %185, %arg22, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x2xf16>, tensor<2x2xf16>) -> tensor<1x5x2xf16>
    %187 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %189 = stablehlo.add %186, %188 : tensor<1x5x2xf16>
    %190 = stablehlo.add %125, %189 : tensor<1x5x2xf16>
    %191 = stablehlo.convert %190 : (tensor<1x5x2xf16>) -> tensor<1x5x2xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %192 = stablehlo.reduce(%191 init: %cst_33) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_34 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %194 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %195 = stablehlo.divide %193, %194 : tensor<1x5x1xf32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %197 = stablehlo.subtract %191, %196 : tensor<1x5x2xf32>
    %198 = stablehlo.multiply %197, %197 : tensor<1x5x2xf32>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %199 = stablehlo.reduce(%198 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_36 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %201 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %202 = stablehlo.divide %200, %201 : tensor<1x5x1xf32>
    %cst_37 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %203 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %204 = stablehlo.add %202, %203 : tensor<1x5x1xf32>
    %205 = stablehlo.rsqrt %204 : tensor<1x5x1xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %207 = stablehlo.multiply %197, %206 : tensor<1x5x2xf32>
    %208 = stablehlo.convert %207 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf16>
    %209 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %211 = stablehlo.multiply %208, %210 : tensor<1x5x2xf16>
    %212 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %214 = stablehlo.add %211, %213 : tensor<1x5x2xf16>
    %215 = stablehlo.dot_general %214, %arg26, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x2xf16>, tensor<2x8xf16>) -> tensor<1x5x8xf16>
    %216 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<8xf16>) -> tensor<1x1x8xf16>
    %217 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2] : (tensor<1x1x8xf16>) -> tensor<1x5x8xf16>
    %218 = stablehlo.add %215, %217 : tensor<1x5x8xf16>
    %cst_38 = stablehlo.constant dense<5.000000e-01> : tensor<f16>
    %219 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %220 = stablehlo.multiply %219, %218 : tensor<1x5x8xf16>
    %cst_39 = stablehlo.constant dense<4.470830e-02> : tensor<f16>
    %221 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %222 = stablehlo.multiply %221, %218 : tensor<1x5x8xf16>
    %223 = stablehlo.multiply %222, %218 : tensor<1x5x8xf16>
    %224 = stablehlo.multiply %223, %218 : tensor<1x5x8xf16>
    %225 = stablehlo.add %218, %224 : tensor<1x5x8xf16>
    %cst_40 = stablehlo.constant dense<7.978520e-01> : tensor<f16>
    %226 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %227 = stablehlo.multiply %226, %225 : tensor<1x5x8xf16>
    %228 = stablehlo.tanh %227 : tensor<1x5x8xf16>
    %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %229 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f16>) -> tensor<1x5x8xf16>
    %230 = stablehlo.add %229, %228 : tensor<1x5x8xf16>
    %231 = stablehlo.multiply %220, %230 : tensor<1x5x8xf16>
    %232 = stablehlo.dot_general %231, %arg28, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x5x8xf16>, tensor<8x2xf16>) -> tensor<1x5x2xf16>
    %233 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %234 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %235 = stablehlo.add %232, %234 : tensor<1x5x2xf16>
    %236 = stablehlo.add %190, %235 : tensor<1x5x2xf16>
    %237 = stablehlo.convert %236 : (tensor<1x5x2xf16>) -> tensor<1x5x2xf32>
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %238 = stablehlo.reduce(%237 init: %cst_42) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_43 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %240 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %241 = stablehlo.divide %239, %240 : tensor<1x5x1xf32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %243 = stablehlo.subtract %237, %242 : tensor<1x5x2xf32>
    %244 = stablehlo.multiply %243, %243 : tensor<1x5x2xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %245 = stablehlo.reduce(%244 init: %cst_44) applies stablehlo.add across dimensions = [2] : (tensor<1x5x2xf32>, tensor<f32>) -> tensor<1x5xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<1x5x1xf32>
    %cst_45 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %247 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %248 = stablehlo.divide %246, %247 : tensor<1x5x1xf32>
    %cst_46 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %249 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<1x5x1xf32>
    %250 = stablehlo.add %248, %249 : tensor<1x5x1xf32>
    %251 = stablehlo.rsqrt %250 : tensor<1x5x1xf32>
    %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1, 2] : (tensor<1x5x1xf32>) -> tensor<1x5x2xf32>
    %253 = stablehlo.multiply %243, %252 : tensor<1x5x2xf32>
    %254 = stablehlo.convert %253 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf16>
    %255 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %257 = stablehlo.multiply %254, %256 : tensor<1x5x2xf16>
    %258 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<2xf16>) -> tensor<1x1x2xf16>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 2] : (tensor<1x1x2xf16>) -> tensor<1x5x2xf16>
    %260 = stablehlo.add %257, %259 : tensor<1x5x2xf16>
    %261 = stablehlo.dot_general %260, %arg2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x5x2xf16>, tensor<50257x2xf16>) -> tensor<1x5x50257xf16>
    return %261 : tensor<1x5x50257xf16>
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
  func.func private @_where(%arg0: tensor<1x1x5x5xi1>, %arg1: tensor<1x2x5x5xf16>, %arg2: tensor<f16>) -> tensor<1x2x5x5xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x5x5xi1>) -> tensor<1x2x5x5xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<1x2x5x5xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<1x2x5x5xi1>, tensor<1x2x5x5xf16>
    return %2 : tensor<1x2x5x5xf16>
  }
}
