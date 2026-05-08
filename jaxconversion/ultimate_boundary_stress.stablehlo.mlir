module @jit_ultimate_prefill attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x24xi32>, %arg1: tensor<2x24xf16>, %arg2: tensor<2x24x24xf16>, %arg3: tensor<1536x768xf16>, %arg4: tensor<768xf16>, %arg5: tensor<768xf16>, %arg6: tensor<6x128x768xf16>, %arg7: tensor<2x3x768x128xf16>, %arg8: tensor<6x768x128xf16>, %arg9: tensor<2x768x1536xf16>, %arg10: tensor<1536x768xf16>, %arg11: tensor<768xf16>, %arg12: tensor<768xf16>, %arg13: tensor<768xf16>, %arg14: tensor<768xf16>, %arg15: tensor<6x128xf16>, %arg16: tensor<3x128xf16>, %arg17: tensor<768xf16>, %arg18: tensor<768x768xf16>, %arg19: tensor<768xf16>, %arg20: tensor<6x128x768xf16>, %arg21: tensor<2x3x768x128xf16>, %arg22: tensor<6x768x128xf16>, %arg23: tensor<2x768x1536xf16>, %arg24: tensor<1536x768xf16>, %arg25: tensor<768xf16>, %arg26: tensor<768xf16>, %arg27: tensor<768xf16>, %arg28: tensor<768xf16>, %arg29: tensor<6x128xf16>, %arg30: tensor<3x128xf16>, %arg31: tensor<768xf16>, %arg32: tensor<768x768xf16>, %arg33: tensor<768xf16>, %arg34: tensor<6x128x768xf16>, %arg35: tensor<2x3x768x128xf16>, %arg36: tensor<6x768x128xf16>, %arg37: tensor<2x768x1536xf16>, %arg38: tensor<1536x768xf16>, %arg39: tensor<768xf16>, %arg40: tensor<768xf16>, %arg41: tensor<768xf16>, %arg42: tensor<768xf16>, %arg43: tensor<6x128xf16>, %arg44: tensor<3x128xf16>, %arg45: tensor<768xf16>, %arg46: tensor<768x768xf16>, %arg47: tensor<768xf16>, %arg48: tensor<6x128x768xf16>, %arg49: tensor<2x3x768x128xf16>, %arg50: tensor<6x768x128xf16>, %arg51: tensor<2x768x1536xf16>, %arg52: tensor<1536x768xf16>, %arg53: tensor<768xf16>, %arg54: tensor<768xf16>, %arg55: tensor<768xf16>, %arg56: tensor<768xf16>, %arg57: tensor<6x128xf16>, %arg58: tensor<3x128xf16>, %arg59: tensor<768xf16>, %arg60: tensor<768x768xf16>, %arg61: tensor<768xf16>, %arg62: tensor<6x128x768xf16>, %arg63: tensor<2x3x768x128xf16>, %arg64: tensor<6x768x128xf16>, %arg65: tensor<2x768x1536xf16>, %arg66: tensor<1536x768xf16>, %arg67: tensor<768xf16>, %arg68: tensor<768xf16>, %arg69: tensor<768xf16>, %arg70: tensor<768xf16>, %arg71: tensor<6x128xf16>, %arg72: tensor<3x128xf16>, %arg73: tensor<768xf16>, %arg74: tensor<768x768xf16>, %arg75: tensor<768xf16>, %arg76: tensor<6x128x768xf16>, %arg77: tensor<2x3x768x128xf16>, %arg78: tensor<6x768x128xf16>, %arg79: tensor<2x768x1536xf16>, %arg80: tensor<1536x768xf16>, %arg81: tensor<768xf16>, %arg82: tensor<768xf16>, %arg83: tensor<768xf16>, %arg84: tensor<768xf16>, %arg85: tensor<6x128xf16>, %arg86: tensor<3x128xf16>, %arg87: tensor<768xf16>, %arg88: tensor<768x768xf16>, %arg89: tensor<768xf16>) -> (tensor<2x24x1536xf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %c_0 = stablehlo.constant dense<1536> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<2x24xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<2x24xi1>, tensor<2x24xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x24xi32>) -> tensor<2x24x1xi32>
    %6 = "stablehlo.gather"(%arg3, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<1536x768xf16>, tensor<2x24x1xi32>) -> tensor<2x24x768xf16>
    %cst = stablehlo.constant dense<2.771880e+01> : tensor<f16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %8 = stablehlo.multiply %6, %7 : tensor<2x24x768xf16>
    %9 = stablehlo.convert %arg1 : (tensor<2x24xf16>) -> tensor<2x24xf32>
    %cst_1 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x24xf32>
    %11 = stablehlo.divide %9, %10 : tensor<2x24xf32>
    %12 = stablehlo.convert %11 : (tensor<2x24xf32>) -> tensor<2x24xf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x24xf16>) -> tensor<2x24x1xf16>
    %cst_2 = stablehlo.constant dense<1.000210e-02> : tensor<f16>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f16>) -> tensor<2x24x1xf16>
    %15 = stablehlo.multiply %13, %14 : tensor<2x24x1xf16>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x768xf16>
    %17 = stablehlo.add %8, %16 : tensor<2x24x768xf16>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %19 = stablehlo.compare NE, %arg0, %18, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %21 = call @_where(%20, %17, %cst_4) : (tensor<2x24x1xi1>, tensor<2x24x768xf16>, tensor<f16>) -> tensor<2x24x768xf16>
    %22 = stablehlo.convert %21 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %23 = chlo.square %22 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_6 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %27 = stablehlo.divide %25, %26 : tensor<2x24x1xf32>
    %cst_7 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %29 = stablehlo.add %27, %28 : tensor<2x24x1xf32>
    %30 = stablehlo.rsqrt %29 : tensor<2x24x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %32 = stablehlo.multiply %22, %31 : tensor<2x24x768xf32>
    %33 = stablehlo.convert %32 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %34 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %36 = stablehlo.multiply %33, %35 : tensor<2x24x768xf16>
    %37 = stablehlo.dot_general %36, %arg18, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<768x768xf16>) -> tensor<2x24x768xf16>
    %cst_8 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %39 = stablehlo.divide %37, %38 : tensor<2x24x768xf16>
    %40 = stablehlo.tanh %39 : tensor<2x24x768xf16>
    %cst_9 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %41 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %42 = stablehlo.multiply %40, %41 : tensor<2x24x768xf16>
    %cst_10 = stablehlo.constant dense<1.562500e-02> : tensor<f16>
    %43 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %44 = stablehlo.multiply %42, %43 : tensor<2x24x768xf16>
    %45 = stablehlo.add %36, %44 : tensor<2x24x768xf16>
    %46 = stablehlo.dot_general %45, %arg8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<6x768x128xf16>) -> tensor<2x24x6x128xf16>
    %47 = stablehlo.transpose %46, dims = [0, 2, 1, 3] : (tensor<2x24x6x128xf16>) -> tensor<2x6x24x128xf16>
    %48 = stablehlo.dot_general %arg7, %45, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x768x128xf16>, tensor<2x24x768xf16>) -> tensor<2x3x128x2x24xf16>
    %49 = stablehlo.transpose %48, dims = [3, 0, 4, 1, 2] : (tensor<2x3x128x2x24xf16>) -> tensor<2x2x24x3x128xf16>
    %50 = stablehlo.slice %49 [0:2, 0:1, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %51 = stablehlo.reshape %50 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %52 = stablehlo.transpose %51, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %53 = stablehlo.slice %49 [0:2, 1:2, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %54 = stablehlo.reshape %53 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %55 = stablehlo.transpose %54, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %56 = stablehlo.convert %47 : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf32>
    %57 = chlo.square %56 : tensor<2x6x24x128xf32> -> tensor<2x6x24x128xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %58 = stablehlo.reduce(%57 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x128xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %cst_12 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %61 = stablehlo.divide %59, %60 : tensor<2x6x24x1xf32>
    %cst_13 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %63 = stablehlo.add %61, %62 : tensor<2x6x24x1xf32>
    %64 = stablehlo.rsqrt %63 : tensor<2x6x24x1xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x128xf32>
    %66 = stablehlo.multiply %56, %65 : tensor<2x6x24x128xf32>
    %67 = stablehlo.convert %66 : (tensor<2x6x24x128xf32>) -> tensor<2x6x24x128xf16>
    %68 = stablehlo.broadcast_in_dim %arg15, dims = [1, 3] : (tensor<6x128xf16>) -> tensor<1x6x1x128xf16>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2, 3] : (tensor<1x6x1x128xf16>) -> tensor<2x6x24x128xf16>
    %70 = stablehlo.multiply %67, %69 : tensor<2x6x24x128xf16>
    %71 = stablehlo.convert %52 : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x128xf32>
    %72 = chlo.square %71 : tensor<2x3x24x128xf32> -> tensor<2x3x24x128xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %73 = stablehlo.reduce(%72 init: %cst_14) applies stablehlo.add across dimensions = [3] : (tensor<2x3x24x128xf32>, tensor<f32>) -> tensor<2x3x24xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<2x3x24xf32>) -> tensor<2x3x24x1xf32>
    %cst_15 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %76 = stablehlo.divide %74, %75 : tensor<2x3x24x1xf32>
    %cst_16 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %77 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %78 = stablehlo.add %76, %77 : tensor<2x3x24x1xf32>
    %79 = stablehlo.rsqrt %78 : tensor<2x3x24x1xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<2x3x24x1xf32>) -> tensor<2x3x24x128xf32>
    %81 = stablehlo.multiply %71, %80 : tensor<2x3x24x128xf32>
    %82 = stablehlo.convert %81 : (tensor<2x3x24x128xf32>) -> tensor<2x3x24x128xf16>
    %83 = stablehlo.broadcast_in_dim %arg16, dims = [1, 3] : (tensor<3x128xf16>) -> tensor<1x3x1x128xf16>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x3x1x128xf16>) -> tensor<2x3x24x128xf16>
    %85 = stablehlo.multiply %82, %84 : tensor<2x3x24x128xf16>
    %86 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_17 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %88 = stablehlo.divide %86, %87 : tensor<128xf32>
    %cst_18 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %90 = stablehlo.power %89, %88 : tensor<128xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %91 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %92 = stablehlo.divide %91, %90 : tensor<128xf32>
    %93 = stablehlo.convert %arg1 : (tensor<2x24xf16>) -> tensor<2x24xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [0, 2] : (tensor<2x24xf32>) -> tensor<2x1x24x1xf32>
    %95 = stablehlo.broadcast_in_dim %92, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %96 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<2x1x24x1xf32>) -> tensor<2x1x24x128xf32>
    %97 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x24x128xf32>
    %98 = stablehlo.multiply %96, %97 : tensor<2x1x24x128xf32>
    %99 = stablehlo.cosine %98 : tensor<2x1x24x128xf32>
    %100 = stablehlo.convert %99 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %101 = stablehlo.sine %98 : tensor<2x1x24x128xf32>
    %102 = stablehlo.convert %101 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %103 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %104 = stablehlo.multiply %70, %103 : tensor<2x6x24x128xf16>
    %105 = stablehlo.slice %70 [0:2, 0:6, 0:24, 0:64] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %106 = stablehlo.slice %70 [0:2, 0:6, 0:24, 64:128] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %107 = stablehlo.negate %106 : tensor<2x6x24x64xf16>
    %108 = stablehlo.concatenate %107, %105, dim = 3 : (tensor<2x6x24x64xf16>, tensor<2x6x24x64xf16>) -> tensor<2x6x24x128xf16>
    %109 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %110 = stablehlo.multiply %108, %109 : tensor<2x6x24x128xf16>
    %111 = stablehlo.add %104, %110 : tensor<2x6x24x128xf16>
    %112 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %113 = stablehlo.multiply %85, %112 : tensor<2x3x24x128xf16>
    %114 = stablehlo.slice %85 [0:2, 0:3, 0:24, 0:64] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %115 = stablehlo.slice %85 [0:2, 0:3, 0:24, 64:128] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %116 = stablehlo.negate %115 : tensor<2x3x24x64xf16>
    %117 = stablehlo.concatenate %116, %114, dim = 3 : (tensor<2x3x24x64xf16>, tensor<2x3x24x64xf16>) -> tensor<2x3x24x128xf16>
    %118 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %119 = stablehlo.multiply %117, %118 : tensor<2x3x24x128xf16>
    %120 = stablehlo.add %113, %119 : tensor<2x3x24x128xf16>
    %121 = stablehlo.slice %120 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %122 = stablehlo.slice %120 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %123 = stablehlo.slice %120 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %124 = stablehlo.slice %120 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %125 = stablehlo.slice %120 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %126 = stablehlo.slice %120 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %127 = stablehlo.concatenate %121, %122, %123, %124, %125, %126, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %128 = stablehlo.slice %55 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %129 = stablehlo.slice %55 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %130 = stablehlo.slice %55 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %131 = stablehlo.slice %55 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %132 = stablehlo.slice %55 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %133 = stablehlo.slice %55 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %134 = stablehlo.concatenate %128, %129, %130, %131, %132, %133, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %135 = stablehlo.dot_general %111, %127, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x128xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x24xf16>
    %cst_20 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %136 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %137 = stablehlo.multiply %135, %136 : tensor<2x6x24x24xf16>
    %cst_21 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %138 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %139 = stablehlo.divide %137, %138 : tensor<2x6x24x24xf16>
    %140 = stablehlo.tanh %139 : tensor<2x6x24x24xf16>
    %cst_22 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %141 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %142 = stablehlo.multiply %140, %141 : tensor<2x6x24x24xf16>
    %c_23 = stablehlo.constant dense<0> : tensor<i32>
    %143 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %144 = stablehlo.compare NE, %arg0, %143, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %c_24 = stablehlo.constant dense<2> : tensor<i32>
    %145 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %146 = stablehlo.compare EQ, %arg0, %145, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %147 = stablehlo.not %146 : tensor<2x24xi1>
    %148 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x24xf16>) -> tensor<2x24x1xf16>
    %149 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x24xf16>) -> tensor<2x1x24xf16>
    %150 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %151 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %152 = stablehlo.compare GE, %150, %151, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %153 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %154 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %155 = stablehlo.compare GT, %153, %154, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_25 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %156 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f16>) -> tensor<2x24x1xf16>
    %157 = stablehlo.add %148, %156 : tensor<2x24x1xf16>
    %158 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %159 = stablehlo.broadcast_in_dim %157, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %160 = stablehlo.compare LT, %158, %159, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %161 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %162 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %163 = stablehlo.subtract %161, %162 : tensor<2x24x24xf16>
    %cst_26 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %164 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %165 = stablehlo.compare LE, %163, %164, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %166 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %167 = stablehlo.compare GT, %arg2, %166, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %168 = stablehlo.broadcast_in_dim %144, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %169 = stablehlo.broadcast_in_dim %144, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %170 = stablehlo.and %167, %152 : tensor<2x24x24xi1>
    %171 = stablehlo.and %170, %160 : tensor<2x24x24xi1>
    %172 = stablehlo.or %155, %152 : tensor<2x24x24xi1>
    %173 = stablehlo.and %171, %172 : tensor<2x24x24xi1>
    %174 = stablehlo.and %173, %165 : tensor<2x24x24xi1>
    %175 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %176 = stablehlo.and %174, %175 : tensor<2x24x24xi1>
    %177 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %178 = stablehlo.and %176, %177 : tensor<2x24x24xi1>
    %179 = stablehlo.broadcast_in_dim %146, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %180 = stablehlo.broadcast_in_dim %146, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %181 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %182 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %183 = stablehlo.and %181, %182 : tensor<2x24x24xi1>
    %184 = stablehlo.or %178, %183 : tensor<2x24x24xi1>
    %185 = stablehlo.broadcast_in_dim %147, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %186 = call @_where_77(%185, %184, %183) : (tensor<2x24x1xi1>, tensor<2x24x24xi1>, tensor<2x24x24xi1>) -> tensor<2x24x24xi1>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 2, 3] : (tensor<2x24x24xi1>) -> tensor<2x1x24x24xi1>
    %cst_28 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %188 = call @_where_81(%187, %142, %cst_28) : (tensor<2x1x24x24xi1>, tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24x24xf16>
    %cst_29 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %189 = stablehlo.reduce(%188 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %cst_30 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %190 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f16>) -> tensor<2x6x24xf16>
    %191 = stablehlo.maximum %190, %189 : tensor<2x6x24xf16>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %194 = stablehlo.subtract %188, %193 : tensor<2x6x24x24xf16>
    %195 = stablehlo.exponential %194 : tensor<2x6x24x24xf16>
    %196 = stablehlo.convert %195 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %197 = stablehlo.reduce(%196 init: %cst_31) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %199 = stablehlo.convert %198 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %200 = stablehlo.broadcast_in_dim %199, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %201 = stablehlo.divide %195, %200 : tensor<2x6x24x24xf16>
    %202 = stablehlo.convert %201 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %203 = stablehlo.reduce(%202 init: %cst_32) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %205 = stablehlo.convert %204 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %207 = stablehlo.divide %201, %206 : tensor<2x6x24x24xf16>
    %cst_33 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %208 = stablehlo.reduce(%188 init: %cst_33) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %211 = stablehlo.subtract %188, %210 : tensor<2x6x24x24xf16>
    %212 = stablehlo.exponential %211 : tensor<2x6x24x24xf16>
    %213 = stablehlo.convert %212 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %214 = stablehlo.reduce(%213 init: %cst_34) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %216 = stablehlo.convert %215 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %217 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %218 = stablehlo.divide %212, %217 : tensor<2x6x24x24xf16>
    %cst_35 = stablehlo.constant dense<7.500000e-01> : tensor<f16>
    %219 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %220 = stablehlo.multiply %207, %219 : tensor<2x6x24x24xf16>
    %cst_36 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %221 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %222 = stablehlo.multiply %218, %221 : tensor<2x6x24x24xf16>
    %223 = stablehlo.add %220, %222 : tensor<2x6x24x24xf16>
    %224 = stablehlo.dot_general %223, %134, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x24xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf16>
    %225 = stablehlo.transpose %224, dims = [0, 2, 1, 3] : (tensor<2x6x24x128xf16>) -> tensor<2x24x6x128xf16>
    %226 = stablehlo.dot_general %225, %arg6, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x6x128xf16>, tensor<6x128x768xf16>) -> tensor<2x24x768xf16>
    %cst_37 = stablehlo.constant dense<1.250000e-01> : tensor<f16>
    %227 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f16>) -> tensor<768xf16>
    %228 = stablehlo.maximum %arg17, %227 : tensor<768xf16>
    %229 = stablehlo.add %21, %226 : tensor<2x24x768xf16>
    %230 = stablehlo.convert %229 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %231 = chlo.square %230 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %232 = stablehlo.reduce(%231 init: %cst_38) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %233 = stablehlo.broadcast_in_dim %232, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_39 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %234 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %235 = stablehlo.divide %233, %234 : tensor<2x24x1xf32>
    %cst_40 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %236 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %237 = stablehlo.add %235, %236 : tensor<2x24x1xf32>
    %238 = stablehlo.rsqrt %237 : tensor<2x24x1xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %240 = stablehlo.multiply %230, %239 : tensor<2x24x768xf32>
    %241 = stablehlo.convert %240 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %242 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %243 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %244 = stablehlo.multiply %241, %243 : tensor<2x24x768xf16>
    %cst_41 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %245 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %246 = stablehlo.multiply %245, %244 : tensor<2x24x768xf16>
    %247 = stablehlo.add %229, %246 : tensor<2x24x768xf16>
    %248 = stablehlo.broadcast_in_dim %228, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %cst_42 = stablehlo.constant dense<1.000400e-03> : tensor<f16>
    %249 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f16>) -> tensor<1x1x768xf16>
    %250 = stablehlo.multiply %248, %249 : tensor<1x1x768xf16>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %252 = stablehlo.add %247, %251 : tensor<2x24x768xf16>
    %253 = stablehlo.convert %252 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %254 = chlo.square %253 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %255 = stablehlo.reduce(%254 init: %cst_43) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_44 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %257 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %258 = stablehlo.divide %256, %257 : tensor<2x24x1xf32>
    %cst_45 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %259 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %260 = stablehlo.add %258, %259 : tensor<2x24x1xf32>
    %261 = stablehlo.rsqrt %260 : tensor<2x24x1xf32>
    %262 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %263 = stablehlo.multiply %253, %262 : tensor<2x24x768xf32>
    %264 = stablehlo.convert %263 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %265 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %267 = stablehlo.multiply %264, %266 : tensor<2x24x768xf16>
    %268 = stablehlo.dot_general %arg9, %267, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x768x1536xf16>, tensor<2x24x768xf16>) -> tensor<2x1536x2x24xf16>
    %269 = stablehlo.transpose %268, dims = [2, 0, 3, 1] : (tensor<2x1536x2x24xf16>) -> tensor<2x2x24x1536xf16>
    %270 = stablehlo.slice %269 [0:2, 0:1, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %271 = stablehlo.reshape %270 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %272 = stablehlo.slice %269 [0:2, 1:2, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %273 = stablehlo.reshape %272 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %274 = stablehlo.negate %271 : tensor<2x24x1536xf16>
    %275 = stablehlo.exponential %274 : tensor<2x24x1536xf16>
    %cst_46 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %276 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %277 = stablehlo.add %276, %275 : tensor<2x24x1536xf16>
    %cst_47 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %278 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %279 = stablehlo.divide %278, %277 : tensor<2x24x1536xf16>
    %280 = stablehlo.multiply %271, %279 : tensor<2x24x1536xf16>
    %281 = stablehlo.multiply %280, %273 : tensor<2x24x1536xf16>
    %cst_48 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %282 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %283 = stablehlo.divide %281, %282 : tensor<2x24x1536xf16>
    %284 = stablehlo.tanh %283 : tensor<2x24x1536xf16>
    %cst_49 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %285 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %286 = stablehlo.multiply %284, %285 : tensor<2x24x1536xf16>
    %287 = stablehlo.dot_general %286, %arg10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x1536xf16>, tensor<1536x768xf16>) -> tensor<2x24x768xf16>
    %288 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %290 = stablehlo.multiply %287, %289 : tensor<2x24x768xf16>
    %291 = stablehlo.add %252, %290 : tensor<2x24x768xf16>
    %292 = stablehlo.convert %291 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %293 = chlo.square %292 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_50 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %294 = stablehlo.reduce(%293 init: %cst_50) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_51 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %296 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %297 = stablehlo.divide %295, %296 : tensor<2x24x1xf32>
    %cst_52 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %298 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %299 = stablehlo.add %297, %298 : tensor<2x24x1xf32>
    %300 = stablehlo.rsqrt %299 : tensor<2x24x1xf32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %302 = stablehlo.multiply %292, %301 : tensor<2x24x768xf32>
    %303 = stablehlo.convert %302 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %304 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %305 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %306 = stablehlo.multiply %303, %305 : tensor<2x24x768xf16>
    %cst_53 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %307 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %308 = stablehlo.multiply %307, %306 : tensor<2x24x768xf16>
    %309 = stablehlo.add %291, %308 : tensor<2x24x768xf16>
    %c_54 = stablehlo.constant dense<0> : tensor<i32>
    %310 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %311 = stablehlo.compare NE, %arg0, %310, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %313 = call @_where(%312, %309, %cst_55) : (tensor<2x24x1xi1>, tensor<2x24x768xf16>, tensor<f16>) -> tensor<2x24x768xf16>
    %cst_56 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %314 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %315 = stablehlo.compare GT, %313, %314, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_57 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %316 = call @_where_114(%315, %cst_57, %313) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %cst_58 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %317 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %318 = stablehlo.compare LT, %316, %317, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_59 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %319 = call @_where_114(%318, %cst_59, %316) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %320 = stablehlo.convert %319 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %321 = chlo.square %320 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %322 = stablehlo.reduce(%321 init: %cst_60) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_61 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %324 = stablehlo.broadcast_in_dim %cst_61, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %325 = stablehlo.divide %323, %324 : tensor<2x24x1xf32>
    %cst_62 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %326 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %327 = stablehlo.maximum %325, %326 : tensor<2x24x1xf32>
    %328 = stablehlo.sqrt %327 : tensor<2x24x1xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %330 = stablehlo.divide %320, %329 : tensor<2x24x768xf32>
    %331 = stablehlo.convert %330 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %332 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %334 = stablehlo.multiply %331, %333 : tensor<2x24x768xf16>
    %335 = stablehlo.dot_general %334, %arg32, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<768x768xf16>) -> tensor<2x24x768xf16>
    %cst_63 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %336 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %337 = stablehlo.divide %335, %336 : tensor<2x24x768xf16>
    %338 = stablehlo.tanh %337 : tensor<2x24x768xf16>
    %cst_64 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %339 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %340 = stablehlo.multiply %338, %339 : tensor<2x24x768xf16>
    %cst_65 = stablehlo.constant dense<1.562500e-02> : tensor<f16>
    %341 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %342 = stablehlo.multiply %340, %341 : tensor<2x24x768xf16>
    %343 = stablehlo.add %334, %342 : tensor<2x24x768xf16>
    %344 = stablehlo.dot_general %343, %arg22, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<6x768x128xf16>) -> tensor<2x24x6x128xf16>
    %345 = stablehlo.transpose %344, dims = [0, 2, 1, 3] : (tensor<2x24x6x128xf16>) -> tensor<2x6x24x128xf16>
    %346 = stablehlo.dot_general %arg21, %343, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x768x128xf16>, tensor<2x24x768xf16>) -> tensor<2x3x128x2x24xf16>
    %347 = stablehlo.transpose %346, dims = [3, 0, 4, 1, 2] : (tensor<2x3x128x2x24xf16>) -> tensor<2x2x24x3x128xf16>
    %348 = stablehlo.slice %347 [0:2, 0:1, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %349 = stablehlo.reshape %348 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %350 = stablehlo.transpose %349, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %351 = stablehlo.slice %347 [0:2, 1:2, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %352 = stablehlo.reshape %351 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %353 = stablehlo.transpose %352, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %354 = stablehlo.convert %345 : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf32>
    %355 = chlo.square %354 : tensor<2x6x24x128xf32> -> tensor<2x6x24x128xf32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %356 = stablehlo.reduce(%355 init: %cst_66) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x128xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %cst_67 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %358 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %359 = stablehlo.divide %357, %358 : tensor<2x6x24x1xf32>
    %cst_68 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %360 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %361 = stablehlo.add %359, %360 : tensor<2x6x24x1xf32>
    %362 = stablehlo.rsqrt %361 : tensor<2x6x24x1xf32>
    %363 = stablehlo.broadcast_in_dim %362, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x128xf32>
    %364 = stablehlo.multiply %354, %363 : tensor<2x6x24x128xf32>
    %365 = stablehlo.convert %364 : (tensor<2x6x24x128xf32>) -> tensor<2x6x24x128xf16>
    %366 = stablehlo.broadcast_in_dim %arg29, dims = [1, 3] : (tensor<6x128xf16>) -> tensor<1x6x1x128xf16>
    %367 = stablehlo.broadcast_in_dim %366, dims = [0, 1, 2, 3] : (tensor<1x6x1x128xf16>) -> tensor<2x6x24x128xf16>
    %368 = stablehlo.multiply %365, %367 : tensor<2x6x24x128xf16>
    %369 = stablehlo.convert %350 : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x128xf32>
    %370 = chlo.square %369 : tensor<2x3x24x128xf32> -> tensor<2x3x24x128xf32>
    %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %371 = stablehlo.reduce(%370 init: %cst_69) applies stablehlo.add across dimensions = [3] : (tensor<2x3x24x128xf32>, tensor<f32>) -> tensor<2x3x24xf32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2] : (tensor<2x3x24xf32>) -> tensor<2x3x24x1xf32>
    %cst_70 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %373 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %374 = stablehlo.divide %372, %373 : tensor<2x3x24x1xf32>
    %cst_71 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %375 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %376 = stablehlo.add %374, %375 : tensor<2x3x24x1xf32>
    %377 = stablehlo.rsqrt %376 : tensor<2x3x24x1xf32>
    %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1, 2, 3] : (tensor<2x3x24x1xf32>) -> tensor<2x3x24x128xf32>
    %379 = stablehlo.multiply %369, %378 : tensor<2x3x24x128xf32>
    %380 = stablehlo.convert %379 : (tensor<2x3x24x128xf32>) -> tensor<2x3x24x128xf16>
    %381 = stablehlo.broadcast_in_dim %arg30, dims = [1, 3] : (tensor<3x128xf16>) -> tensor<1x3x1x128xf16>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1, 2, 3] : (tensor<1x3x1x128xf16>) -> tensor<2x3x24x128xf16>
    %383 = stablehlo.multiply %380, %382 : tensor<2x3x24x128xf16>
    %384 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_72 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %385 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %386 = stablehlo.divide %384, %385 : tensor<128xf32>
    %cst_73 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %387 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %388 = stablehlo.power %387, %386 : tensor<128xf32>
    %cst_74 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %389 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %390 = stablehlo.divide %389, %388 : tensor<128xf32>
    %391 = stablehlo.convert %arg1 : (tensor<2x24xf16>) -> tensor<2x24xf32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [0, 2] : (tensor<2x24xf32>) -> tensor<2x1x24x1xf32>
    %393 = stablehlo.broadcast_in_dim %390, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %394 = stablehlo.broadcast_in_dim %392, dims = [0, 1, 2, 3] : (tensor<2x1x24x1xf32>) -> tensor<2x1x24x128xf32>
    %395 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x24x128xf32>
    %396 = stablehlo.multiply %394, %395 : tensor<2x1x24x128xf32>
    %397 = stablehlo.cosine %396 : tensor<2x1x24x128xf32>
    %398 = stablehlo.convert %397 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %399 = stablehlo.sine %396 : tensor<2x1x24x128xf32>
    %400 = stablehlo.convert %399 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %401 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %402 = stablehlo.multiply %368, %401 : tensor<2x6x24x128xf16>
    %403 = stablehlo.slice %368 [0:2, 0:6, 0:24, 0:64] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %404 = stablehlo.slice %368 [0:2, 0:6, 0:24, 64:128] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %405 = stablehlo.negate %404 : tensor<2x6x24x64xf16>
    %406 = stablehlo.concatenate %405, %403, dim = 3 : (tensor<2x6x24x64xf16>, tensor<2x6x24x64xf16>) -> tensor<2x6x24x128xf16>
    %407 = stablehlo.broadcast_in_dim %400, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %408 = stablehlo.multiply %406, %407 : tensor<2x6x24x128xf16>
    %409 = stablehlo.add %402, %408 : tensor<2x6x24x128xf16>
    %410 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %411 = stablehlo.multiply %383, %410 : tensor<2x3x24x128xf16>
    %412 = stablehlo.slice %383 [0:2, 0:3, 0:24, 0:64] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %413 = stablehlo.slice %383 [0:2, 0:3, 0:24, 64:128] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %414 = stablehlo.negate %413 : tensor<2x3x24x64xf16>
    %415 = stablehlo.concatenate %414, %412, dim = 3 : (tensor<2x3x24x64xf16>, tensor<2x3x24x64xf16>) -> tensor<2x3x24x128xf16>
    %416 = stablehlo.broadcast_in_dim %400, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %417 = stablehlo.multiply %415, %416 : tensor<2x3x24x128xf16>
    %418 = stablehlo.add %411, %417 : tensor<2x3x24x128xf16>
    %419 = stablehlo.slice %418 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %420 = stablehlo.slice %418 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %421 = stablehlo.slice %418 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %422 = stablehlo.slice %418 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %423 = stablehlo.slice %418 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %424 = stablehlo.slice %418 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %425 = stablehlo.concatenate %419, %420, %421, %422, %423, %424, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %426 = stablehlo.slice %353 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %427 = stablehlo.slice %353 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %428 = stablehlo.slice %353 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %429 = stablehlo.slice %353 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %430 = stablehlo.slice %353 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %431 = stablehlo.slice %353 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %432 = stablehlo.concatenate %426, %427, %428, %429, %430, %431, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %433 = stablehlo.dot_general %409, %425, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x128xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x24xf16>
    %cst_75 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %434 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %435 = stablehlo.multiply %433, %434 : tensor<2x6x24x24xf16>
    %cst_76 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %436 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %437 = stablehlo.divide %435, %436 : tensor<2x6x24x24xf16>
    %438 = stablehlo.tanh %437 : tensor<2x6x24x24xf16>
    %cst_77 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %439 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %440 = stablehlo.multiply %438, %439 : tensor<2x6x24x24xf16>
    %c_78 = stablehlo.constant dense<0> : tensor<i32>
    %441 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %442 = stablehlo.compare NE, %arg0, %441, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %c_79 = stablehlo.constant dense<2> : tensor<i32>
    %443 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %444 = stablehlo.compare EQ, %arg0, %443, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %445 = stablehlo.not %444 : tensor<2x24xi1>
    %446 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x24xf16>) -> tensor<2x24x1xf16>
    %447 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x24xf16>) -> tensor<2x1x24xf16>
    %448 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %449 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %450 = stablehlo.compare GE, %448, %449, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %451 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %452 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %453 = stablehlo.compare GT, %451, %452, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_80 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %454 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f16>) -> tensor<2x24x1xf16>
    %455 = stablehlo.add %446, %454 : tensor<2x24x1xf16>
    %456 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %457 = stablehlo.broadcast_in_dim %455, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %458 = stablehlo.compare LT, %456, %457, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %459 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %460 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %461 = stablehlo.subtract %459, %460 : tensor<2x24x24xf16>
    %cst_81 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %462 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %463 = stablehlo.compare LE, %461, %462, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %464 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %465 = stablehlo.compare GT, %arg2, %464, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %466 = stablehlo.broadcast_in_dim %442, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %467 = stablehlo.broadcast_in_dim %442, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %468 = stablehlo.and %465, %450 : tensor<2x24x24xi1>
    %469 = stablehlo.and %468, %458 : tensor<2x24x24xi1>
    %470 = stablehlo.or %453, %450 : tensor<2x24x24xi1>
    %471 = stablehlo.and %469, %470 : tensor<2x24x24xi1>
    %472 = stablehlo.and %471, %463 : tensor<2x24x24xi1>
    %473 = stablehlo.broadcast_in_dim %466, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %474 = stablehlo.and %472, %473 : tensor<2x24x24xi1>
    %475 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %476 = stablehlo.and %474, %475 : tensor<2x24x24xi1>
    %477 = stablehlo.broadcast_in_dim %444, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %478 = stablehlo.broadcast_in_dim %444, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %479 = stablehlo.broadcast_in_dim %477, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %480 = stablehlo.broadcast_in_dim %478, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %481 = stablehlo.and %479, %480 : tensor<2x24x24xi1>
    %482 = stablehlo.or %476, %481 : tensor<2x24x24xi1>
    %483 = stablehlo.broadcast_in_dim %445, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %484 = call @_where_77(%483, %482, %481) : (tensor<2x24x1xi1>, tensor<2x24x24xi1>, tensor<2x24x24xi1>) -> tensor<2x24x24xi1>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 2, 3] : (tensor<2x24x24xi1>) -> tensor<2x1x24x24xi1>
    %cst_83 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %486 = call @_where_81(%485, %440, %cst_83) : (tensor<2x1x24x24xi1>, tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24x24xf16>
    %cst_84 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %487 = stablehlo.reduce(%486 init: %cst_84) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %cst_85 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %488 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f16>) -> tensor<2x6x24xf16>
    %489 = stablehlo.maximum %488, %487 : tensor<2x6x24xf16>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %491 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %492 = stablehlo.subtract %486, %491 : tensor<2x6x24x24xf16>
    %493 = stablehlo.exponential %492 : tensor<2x6x24x24xf16>
    %494 = stablehlo.convert %493 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %495 = stablehlo.reduce(%494 init: %cst_86) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %497 = stablehlo.convert %496 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %498 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %499 = stablehlo.divide %493, %498 : tensor<2x6x24x24xf16>
    %500 = stablehlo.convert %499 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %501 = stablehlo.reduce(%500 init: %cst_87) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %502 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %503 = stablehlo.convert %502 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %504 = stablehlo.broadcast_in_dim %503, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %505 = stablehlo.divide %499, %504 : tensor<2x6x24x24xf16>
    %cst_88 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %506 = stablehlo.reduce(%486 init: %cst_88) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %507 = stablehlo.broadcast_in_dim %506, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %508 = stablehlo.broadcast_in_dim %507, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %509 = stablehlo.subtract %486, %508 : tensor<2x6x24x24xf16>
    %510 = stablehlo.exponential %509 : tensor<2x6x24x24xf16>
    %511 = stablehlo.convert %510 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %512 = stablehlo.reduce(%511 init: %cst_89) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %513 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %514 = stablehlo.convert %513 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %516 = stablehlo.divide %510, %515 : tensor<2x6x24x24xf16>
    %cst_90 = stablehlo.constant dense<7.500000e-01> : tensor<f16>
    %517 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %518 = stablehlo.multiply %505, %517 : tensor<2x6x24x24xf16>
    %cst_91 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %519 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %520 = stablehlo.multiply %516, %519 : tensor<2x6x24x24xf16>
    %521 = stablehlo.add %518, %520 : tensor<2x6x24x24xf16>
    %522 = stablehlo.dot_general %521, %432, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x24xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf16>
    %523 = stablehlo.transpose %522, dims = [0, 2, 1, 3] : (tensor<2x6x24x128xf16>) -> tensor<2x24x6x128xf16>
    %524 = stablehlo.dot_general %523, %arg20, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x6x128xf16>, tensor<6x128x768xf16>) -> tensor<2x24x768xf16>
    %cst_92 = stablehlo.constant dense<1.250000e-01> : tensor<f16>
    %525 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f16>) -> tensor<768xf16>
    %526 = stablehlo.maximum %arg31, %525 : tensor<768xf16>
    %527 = stablehlo.add %319, %524 : tensor<2x24x768xf16>
    %528 = stablehlo.convert %527 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %529 = chlo.square %528 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %530 = stablehlo.reduce(%529 init: %cst_93) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %531 = stablehlo.broadcast_in_dim %530, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_94 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %532 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %533 = stablehlo.divide %531, %532 : tensor<2x24x1xf32>
    %cst_95 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %534 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %535 = stablehlo.add %533, %534 : tensor<2x24x1xf32>
    %536 = stablehlo.rsqrt %535 : tensor<2x24x1xf32>
    %537 = stablehlo.broadcast_in_dim %536, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %538 = stablehlo.multiply %528, %537 : tensor<2x24x768xf32>
    %539 = stablehlo.convert %538 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %540 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %542 = stablehlo.multiply %539, %541 : tensor<2x24x768xf16>
    %cst_96 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %543 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %544 = stablehlo.multiply %543, %542 : tensor<2x24x768xf16>
    %545 = stablehlo.add %527, %544 : tensor<2x24x768xf16>
    %546 = stablehlo.broadcast_in_dim %526, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %cst_97 = stablehlo.constant dense<1.000400e-03> : tensor<f16>
    %547 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f16>) -> tensor<1x1x768xf16>
    %548 = stablehlo.multiply %546, %547 : tensor<1x1x768xf16>
    %549 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %550 = stablehlo.add %545, %549 : tensor<2x24x768xf16>
    %551 = stablehlo.convert %550 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %552 = chlo.square %551 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %553 = stablehlo.reduce(%552 init: %cst_98) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_99 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %555 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %556 = stablehlo.divide %554, %555 : tensor<2x24x1xf32>
    %cst_100 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %557 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %558 = stablehlo.add %556, %557 : tensor<2x24x1xf32>
    %559 = stablehlo.rsqrt %558 : tensor<2x24x1xf32>
    %560 = stablehlo.broadcast_in_dim %559, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %561 = stablehlo.multiply %551, %560 : tensor<2x24x768xf32>
    %562 = stablehlo.convert %561 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %563 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %564 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %565 = stablehlo.multiply %562, %564 : tensor<2x24x768xf16>
    %566 = stablehlo.dot_general %arg23, %565, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x768x1536xf16>, tensor<2x24x768xf16>) -> tensor<2x1536x2x24xf16>
    %567 = stablehlo.transpose %566, dims = [2, 0, 3, 1] : (tensor<2x1536x2x24xf16>) -> tensor<2x2x24x1536xf16>
    %568 = stablehlo.slice %567 [0:2, 0:1, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %569 = stablehlo.reshape %568 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %570 = stablehlo.slice %567 [0:2, 1:2, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %571 = stablehlo.reshape %570 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %572 = stablehlo.negate %569 : tensor<2x24x1536xf16>
    %573 = stablehlo.exponential %572 : tensor<2x24x1536xf16>
    %cst_101 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %574 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %575 = stablehlo.add %574, %573 : tensor<2x24x1536xf16>
    %cst_102 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %576 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %577 = stablehlo.divide %576, %575 : tensor<2x24x1536xf16>
    %578 = stablehlo.multiply %569, %577 : tensor<2x24x1536xf16>
    %579 = stablehlo.multiply %578, %571 : tensor<2x24x1536xf16>
    %cst_103 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %580 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %581 = stablehlo.divide %579, %580 : tensor<2x24x1536xf16>
    %582 = stablehlo.tanh %581 : tensor<2x24x1536xf16>
    %cst_104 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %583 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %584 = stablehlo.multiply %582, %583 : tensor<2x24x1536xf16>
    %585 = stablehlo.dot_general %584, %arg24, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x1536xf16>, tensor<1536x768xf16>) -> tensor<2x24x768xf16>
    %586 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %587 = stablehlo.broadcast_in_dim %586, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %588 = stablehlo.multiply %585, %587 : tensor<2x24x768xf16>
    %589 = stablehlo.add %550, %588 : tensor<2x24x768xf16>
    %590 = stablehlo.convert %589 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %591 = chlo.square %590 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %592 = stablehlo.reduce(%591 init: %cst_105) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %593 = stablehlo.broadcast_in_dim %592, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_106 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %594 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %595 = stablehlo.divide %593, %594 : tensor<2x24x1xf32>
    %cst_107 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %596 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %597 = stablehlo.add %595, %596 : tensor<2x24x1xf32>
    %598 = stablehlo.rsqrt %597 : tensor<2x24x1xf32>
    %599 = stablehlo.broadcast_in_dim %598, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %600 = stablehlo.multiply %590, %599 : tensor<2x24x768xf32>
    %601 = stablehlo.convert %600 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %602 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %603 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %604 = stablehlo.multiply %601, %603 : tensor<2x24x768xf16>
    %cst_108 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %605 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %606 = stablehlo.multiply %605, %604 : tensor<2x24x768xf16>
    %607 = stablehlo.add %589, %606 : tensor<2x24x768xf16>
    %c_109 = stablehlo.constant dense<0> : tensor<i32>
    %608 = stablehlo.broadcast_in_dim %c_109, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %609 = stablehlo.compare NE, %arg0, %608, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %610 = stablehlo.broadcast_in_dim %609, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %611 = call @_where(%610, %607, %cst_110) : (tensor<2x24x1xi1>, tensor<2x24x768xf16>, tensor<f16>) -> tensor<2x24x768xf16>
    %cst_111 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %612 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %613 = stablehlo.compare GT, %611, %612, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_112 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %614 = call @_where_114(%613, %cst_112, %611) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %cst_113 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %615 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %616 = stablehlo.compare LT, %614, %615, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_114 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %617 = call @_where_114(%616, %cst_114, %614) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %618 = stablehlo.convert %617 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %619 = chlo.square %618 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %620 = stablehlo.reduce(%619 init: %cst_115) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %621 = stablehlo.broadcast_in_dim %620, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_116 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %622 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %623 = stablehlo.divide %621, %622 : tensor<2x24x1xf32>
    %cst_117 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %624 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %625 = stablehlo.add %623, %624 : tensor<2x24x1xf32>
    %626 = stablehlo.rsqrt %625 : tensor<2x24x1xf32>
    %627 = stablehlo.broadcast_in_dim %626, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %628 = stablehlo.multiply %618, %627 : tensor<2x24x768xf32>
    %629 = stablehlo.convert %628 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %630 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %632 = stablehlo.multiply %629, %631 : tensor<2x24x768xf16>
    %633 = stablehlo.dot_general %632, %arg46, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<768x768xf16>) -> tensor<2x24x768xf16>
    %cst_118 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %634 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %635 = stablehlo.divide %633, %634 : tensor<2x24x768xf16>
    %636 = stablehlo.tanh %635 : tensor<2x24x768xf16>
    %cst_119 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %637 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %638 = stablehlo.multiply %636, %637 : tensor<2x24x768xf16>
    %cst_120 = stablehlo.constant dense<1.562500e-02> : tensor<f16>
    %639 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %640 = stablehlo.multiply %638, %639 : tensor<2x24x768xf16>
    %641 = stablehlo.add %632, %640 : tensor<2x24x768xf16>
    %642 = stablehlo.dot_general %641, %arg36, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<6x768x128xf16>) -> tensor<2x24x6x128xf16>
    %643 = stablehlo.transpose %642, dims = [0, 2, 1, 3] : (tensor<2x24x6x128xf16>) -> tensor<2x6x24x128xf16>
    %644 = stablehlo.dot_general %arg35, %641, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x768x128xf16>, tensor<2x24x768xf16>) -> tensor<2x3x128x2x24xf16>
    %645 = stablehlo.transpose %644, dims = [3, 0, 4, 1, 2] : (tensor<2x3x128x2x24xf16>) -> tensor<2x2x24x3x128xf16>
    %646 = stablehlo.slice %645 [0:2, 0:1, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %647 = stablehlo.reshape %646 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %648 = stablehlo.transpose %647, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %649 = stablehlo.slice %645 [0:2, 1:2, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %650 = stablehlo.reshape %649 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %651 = stablehlo.transpose %650, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %652 = stablehlo.convert %643 : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf32>
    %653 = chlo.square %652 : tensor<2x6x24x128xf32> -> tensor<2x6x24x128xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %654 = stablehlo.reduce(%653 init: %cst_121) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x128xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %655 = stablehlo.broadcast_in_dim %654, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %cst_122 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %656 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %657 = stablehlo.divide %655, %656 : tensor<2x6x24x1xf32>
    %cst_123 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %658 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %659 = stablehlo.add %657, %658 : tensor<2x6x24x1xf32>
    %660 = stablehlo.rsqrt %659 : tensor<2x6x24x1xf32>
    %661 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x128xf32>
    %662 = stablehlo.multiply %652, %661 : tensor<2x6x24x128xf32>
    %663 = stablehlo.convert %662 : (tensor<2x6x24x128xf32>) -> tensor<2x6x24x128xf16>
    %664 = stablehlo.broadcast_in_dim %arg43, dims = [1, 3] : (tensor<6x128xf16>) -> tensor<1x6x1x128xf16>
    %665 = stablehlo.broadcast_in_dim %664, dims = [0, 1, 2, 3] : (tensor<1x6x1x128xf16>) -> tensor<2x6x24x128xf16>
    %666 = stablehlo.multiply %663, %665 : tensor<2x6x24x128xf16>
    %667 = stablehlo.convert %648 : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x128xf32>
    %668 = chlo.square %667 : tensor<2x3x24x128xf32> -> tensor<2x3x24x128xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %669 = stablehlo.reduce(%668 init: %cst_124) applies stablehlo.add across dimensions = [3] : (tensor<2x3x24x128xf32>, tensor<f32>) -> tensor<2x3x24xf32>
    %670 = stablehlo.broadcast_in_dim %669, dims = [0, 1, 2] : (tensor<2x3x24xf32>) -> tensor<2x3x24x1xf32>
    %cst_125 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %671 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %672 = stablehlo.divide %670, %671 : tensor<2x3x24x1xf32>
    %cst_126 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %673 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %674 = stablehlo.add %672, %673 : tensor<2x3x24x1xf32>
    %675 = stablehlo.rsqrt %674 : tensor<2x3x24x1xf32>
    %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1, 2, 3] : (tensor<2x3x24x1xf32>) -> tensor<2x3x24x128xf32>
    %677 = stablehlo.multiply %667, %676 : tensor<2x3x24x128xf32>
    %678 = stablehlo.convert %677 : (tensor<2x3x24x128xf32>) -> tensor<2x3x24x128xf16>
    %679 = stablehlo.broadcast_in_dim %arg44, dims = [1, 3] : (tensor<3x128xf16>) -> tensor<1x3x1x128xf16>
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2, 3] : (tensor<1x3x1x128xf16>) -> tensor<2x3x24x128xf16>
    %681 = stablehlo.multiply %678, %680 : tensor<2x3x24x128xf16>
    %682 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_127 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %683 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %684 = stablehlo.divide %682, %683 : tensor<128xf32>
    %cst_128 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %685 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %686 = stablehlo.power %685, %684 : tensor<128xf32>
    %cst_129 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %687 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %688 = stablehlo.divide %687, %686 : tensor<128xf32>
    %689 = stablehlo.convert %arg1 : (tensor<2x24xf16>) -> tensor<2x24xf32>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 2] : (tensor<2x24xf32>) -> tensor<2x1x24x1xf32>
    %691 = stablehlo.broadcast_in_dim %688, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %692 = stablehlo.broadcast_in_dim %690, dims = [0, 1, 2, 3] : (tensor<2x1x24x1xf32>) -> tensor<2x1x24x128xf32>
    %693 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x24x128xf32>
    %694 = stablehlo.multiply %692, %693 : tensor<2x1x24x128xf32>
    %695 = stablehlo.cosine %694 : tensor<2x1x24x128xf32>
    %696 = stablehlo.convert %695 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %697 = stablehlo.sine %694 : tensor<2x1x24x128xf32>
    %698 = stablehlo.convert %697 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %699 = stablehlo.broadcast_in_dim %696, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %700 = stablehlo.multiply %666, %699 : tensor<2x6x24x128xf16>
    %701 = stablehlo.slice %666 [0:2, 0:6, 0:24, 0:64] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %702 = stablehlo.slice %666 [0:2, 0:6, 0:24, 64:128] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %703 = stablehlo.negate %702 : tensor<2x6x24x64xf16>
    %704 = stablehlo.concatenate %703, %701, dim = 3 : (tensor<2x6x24x64xf16>, tensor<2x6x24x64xf16>) -> tensor<2x6x24x128xf16>
    %705 = stablehlo.broadcast_in_dim %698, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %706 = stablehlo.multiply %704, %705 : tensor<2x6x24x128xf16>
    %707 = stablehlo.add %700, %706 : tensor<2x6x24x128xf16>
    %708 = stablehlo.broadcast_in_dim %696, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %709 = stablehlo.multiply %681, %708 : tensor<2x3x24x128xf16>
    %710 = stablehlo.slice %681 [0:2, 0:3, 0:24, 0:64] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %711 = stablehlo.slice %681 [0:2, 0:3, 0:24, 64:128] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %712 = stablehlo.negate %711 : tensor<2x3x24x64xf16>
    %713 = stablehlo.concatenate %712, %710, dim = 3 : (tensor<2x3x24x64xf16>, tensor<2x3x24x64xf16>) -> tensor<2x3x24x128xf16>
    %714 = stablehlo.broadcast_in_dim %698, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %715 = stablehlo.multiply %713, %714 : tensor<2x3x24x128xf16>
    %716 = stablehlo.add %709, %715 : tensor<2x3x24x128xf16>
    %717 = stablehlo.slice %716 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %718 = stablehlo.slice %716 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %719 = stablehlo.slice %716 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %720 = stablehlo.slice %716 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %721 = stablehlo.slice %716 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %722 = stablehlo.slice %716 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %723 = stablehlo.concatenate %717, %718, %719, %720, %721, %722, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %724 = stablehlo.slice %651 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %725 = stablehlo.slice %651 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %726 = stablehlo.slice %651 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %727 = stablehlo.slice %651 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %728 = stablehlo.slice %651 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %729 = stablehlo.slice %651 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %730 = stablehlo.concatenate %724, %725, %726, %727, %728, %729, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %731 = stablehlo.dot_general %707, %723, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x128xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x24xf16>
    %cst_130 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %732 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %733 = stablehlo.multiply %731, %732 : tensor<2x6x24x24xf16>
    %cst_131 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %734 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %735 = stablehlo.divide %733, %734 : tensor<2x6x24x24xf16>
    %736 = stablehlo.tanh %735 : tensor<2x6x24x24xf16>
    %cst_132 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %737 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %738 = stablehlo.multiply %736, %737 : tensor<2x6x24x24xf16>
    %c_133 = stablehlo.constant dense<0> : tensor<i32>
    %739 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %740 = stablehlo.compare NE, %arg0, %739, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %c_134 = stablehlo.constant dense<2> : tensor<i32>
    %741 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %742 = stablehlo.compare EQ, %arg0, %741, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %743 = stablehlo.not %742 : tensor<2x24xi1>
    %744 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x24xf16>) -> tensor<2x24x1xf16>
    %745 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x24xf16>) -> tensor<2x1x24xf16>
    %746 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %747 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %748 = stablehlo.compare GE, %746, %747, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %749 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %750 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %751 = stablehlo.compare GT, %749, %750, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_135 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %752 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f16>) -> tensor<2x24x1xf16>
    %753 = stablehlo.add %744, %752 : tensor<2x24x1xf16>
    %754 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %755 = stablehlo.broadcast_in_dim %753, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %756 = stablehlo.compare LT, %754, %755, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %757 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %758 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %759 = stablehlo.subtract %757, %758 : tensor<2x24x24xf16>
    %cst_136 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %760 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %761 = stablehlo.compare LE, %759, %760, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %762 = stablehlo.broadcast_in_dim %cst_137, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %763 = stablehlo.compare GT, %arg2, %762, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %764 = stablehlo.broadcast_in_dim %740, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %765 = stablehlo.broadcast_in_dim %740, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %766 = stablehlo.and %763, %748 : tensor<2x24x24xi1>
    %767 = stablehlo.and %766, %756 : tensor<2x24x24xi1>
    %768 = stablehlo.or %751, %748 : tensor<2x24x24xi1>
    %769 = stablehlo.and %767, %768 : tensor<2x24x24xi1>
    %770 = stablehlo.and %769, %761 : tensor<2x24x24xi1>
    %771 = stablehlo.broadcast_in_dim %764, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %772 = stablehlo.and %770, %771 : tensor<2x24x24xi1>
    %773 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %774 = stablehlo.and %772, %773 : tensor<2x24x24xi1>
    %775 = stablehlo.broadcast_in_dim %742, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %776 = stablehlo.broadcast_in_dim %742, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %777 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %778 = stablehlo.broadcast_in_dim %776, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %779 = stablehlo.and %777, %778 : tensor<2x24x24xi1>
    %780 = stablehlo.or %774, %779 : tensor<2x24x24xi1>
    %781 = stablehlo.broadcast_in_dim %743, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %782 = call @_where_77(%781, %780, %779) : (tensor<2x24x1xi1>, tensor<2x24x24xi1>, tensor<2x24x24xi1>) -> tensor<2x24x24xi1>
    %783 = stablehlo.broadcast_in_dim %782, dims = [0, 2, 3] : (tensor<2x24x24xi1>) -> tensor<2x1x24x24xi1>
    %cst_138 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %784 = call @_where_81(%783, %738, %cst_138) : (tensor<2x1x24x24xi1>, tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24x24xf16>
    %cst_139 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %785 = stablehlo.reduce(%784 init: %cst_139) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %cst_140 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %786 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f16>) -> tensor<2x6x24xf16>
    %787 = stablehlo.maximum %786, %785 : tensor<2x6x24xf16>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %789 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %790 = stablehlo.subtract %784, %789 : tensor<2x6x24x24xf16>
    %791 = stablehlo.exponential %790 : tensor<2x6x24x24xf16>
    %792 = stablehlo.convert %791 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %793 = stablehlo.reduce(%792 init: %cst_141) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %794 = stablehlo.broadcast_in_dim %793, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %795 = stablehlo.convert %794 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %797 = stablehlo.divide %791, %796 : tensor<2x6x24x24xf16>
    %798 = stablehlo.convert %797 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %799 = stablehlo.reduce(%798 init: %cst_142) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %800 = stablehlo.broadcast_in_dim %799, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %801 = stablehlo.convert %800 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %802 = stablehlo.broadcast_in_dim %801, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %803 = stablehlo.divide %797, %802 : tensor<2x6x24x24xf16>
    %cst_143 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %804 = stablehlo.reduce(%784 init: %cst_143) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %806 = stablehlo.broadcast_in_dim %805, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %807 = stablehlo.subtract %784, %806 : tensor<2x6x24x24xf16>
    %808 = stablehlo.exponential %807 : tensor<2x6x24x24xf16>
    %809 = stablehlo.convert %808 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %810 = stablehlo.reduce(%809 init: %cst_144) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %811 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %812 = stablehlo.convert %811 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %813 = stablehlo.broadcast_in_dim %812, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %814 = stablehlo.divide %808, %813 : tensor<2x6x24x24xf16>
    %cst_145 = stablehlo.constant dense<7.500000e-01> : tensor<f16>
    %815 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %816 = stablehlo.multiply %803, %815 : tensor<2x6x24x24xf16>
    %cst_146 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %817 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %818 = stablehlo.multiply %814, %817 : tensor<2x6x24x24xf16>
    %819 = stablehlo.add %816, %818 : tensor<2x6x24x24xf16>
    %820 = stablehlo.dot_general %819, %730, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x24xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf16>
    %821 = stablehlo.transpose %820, dims = [0, 2, 1, 3] : (tensor<2x6x24x128xf16>) -> tensor<2x24x6x128xf16>
    %822 = stablehlo.dot_general %821, %arg34, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x6x128xf16>, tensor<6x128x768xf16>) -> tensor<2x24x768xf16>
    %cst_147 = stablehlo.constant dense<1.250000e-01> : tensor<f16>
    %823 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f16>) -> tensor<768xf16>
    %824 = stablehlo.maximum %arg45, %823 : tensor<768xf16>
    %825 = stablehlo.add %617, %822 : tensor<2x24x768xf16>
    %826 = stablehlo.convert %825 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %827 = chlo.square %826 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %828 = stablehlo.reduce(%827 init: %cst_148) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_149 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %830 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %831 = stablehlo.divide %829, %830 : tensor<2x24x1xf32>
    %cst_150 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %832 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %833 = stablehlo.add %831, %832 : tensor<2x24x1xf32>
    %834 = stablehlo.rsqrt %833 : tensor<2x24x1xf32>
    %835 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %836 = stablehlo.multiply %826, %835 : tensor<2x24x768xf32>
    %837 = stablehlo.convert %836 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %838 = stablehlo.broadcast_in_dim %arg40, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %839 = stablehlo.broadcast_in_dim %838, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %840 = stablehlo.multiply %837, %839 : tensor<2x24x768xf16>
    %cst_151 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %841 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %842 = stablehlo.multiply %841, %840 : tensor<2x24x768xf16>
    %843 = stablehlo.add %825, %842 : tensor<2x24x768xf16>
    %844 = stablehlo.broadcast_in_dim %824, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %cst_152 = stablehlo.constant dense<1.000400e-03> : tensor<f16>
    %845 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f16>) -> tensor<1x1x768xf16>
    %846 = stablehlo.multiply %844, %845 : tensor<1x1x768xf16>
    %847 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %848 = stablehlo.add %843, %847 : tensor<2x24x768xf16>
    %849 = stablehlo.convert %848 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %850 = chlo.square %849 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %851 = stablehlo.reduce(%850 init: %cst_153) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %852 = stablehlo.broadcast_in_dim %851, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_154 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %853 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %854 = stablehlo.divide %852, %853 : tensor<2x24x1xf32>
    %cst_155 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %855 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %856 = stablehlo.add %854, %855 : tensor<2x24x1xf32>
    %857 = stablehlo.rsqrt %856 : tensor<2x24x1xf32>
    %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %859 = stablehlo.multiply %849, %858 : tensor<2x24x768xf32>
    %860 = stablehlo.convert %859 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %861 = stablehlo.broadcast_in_dim %arg41, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %862 = stablehlo.broadcast_in_dim %861, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %863 = stablehlo.multiply %860, %862 : tensor<2x24x768xf16>
    %864 = stablehlo.dot_general %arg37, %863, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x768x1536xf16>, tensor<2x24x768xf16>) -> tensor<2x1536x2x24xf16>
    %865 = stablehlo.transpose %864, dims = [2, 0, 3, 1] : (tensor<2x1536x2x24xf16>) -> tensor<2x2x24x1536xf16>
    %866 = stablehlo.slice %865 [0:2, 0:1, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %867 = stablehlo.reshape %866 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %868 = stablehlo.slice %865 [0:2, 1:2, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %869 = stablehlo.reshape %868 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %870 = stablehlo.negate %867 : tensor<2x24x1536xf16>
    %871 = stablehlo.exponential %870 : tensor<2x24x1536xf16>
    %cst_156 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %872 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %873 = stablehlo.add %872, %871 : tensor<2x24x1536xf16>
    %cst_157 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %874 = stablehlo.broadcast_in_dim %cst_157, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %875 = stablehlo.divide %874, %873 : tensor<2x24x1536xf16>
    %876 = stablehlo.multiply %867, %875 : tensor<2x24x1536xf16>
    %877 = stablehlo.multiply %876, %869 : tensor<2x24x1536xf16>
    %cst_158 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %878 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %879 = stablehlo.divide %877, %878 : tensor<2x24x1536xf16>
    %880 = stablehlo.tanh %879 : tensor<2x24x1536xf16>
    %cst_159 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %881 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %882 = stablehlo.multiply %880, %881 : tensor<2x24x1536xf16>
    %883 = stablehlo.dot_general %882, %arg38, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x1536xf16>, tensor<1536x768xf16>) -> tensor<2x24x768xf16>
    %884 = stablehlo.broadcast_in_dim %arg47, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %885 = stablehlo.broadcast_in_dim %884, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %886 = stablehlo.multiply %883, %885 : tensor<2x24x768xf16>
    %887 = stablehlo.add %848, %886 : tensor<2x24x768xf16>
    %888 = stablehlo.convert %887 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %889 = chlo.square %888 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %890 = stablehlo.reduce(%889 init: %cst_160) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %891 = stablehlo.broadcast_in_dim %890, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_161 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %892 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %893 = stablehlo.divide %891, %892 : tensor<2x24x1xf32>
    %cst_162 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %894 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %895 = stablehlo.add %893, %894 : tensor<2x24x1xf32>
    %896 = stablehlo.rsqrt %895 : tensor<2x24x1xf32>
    %897 = stablehlo.broadcast_in_dim %896, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %898 = stablehlo.multiply %888, %897 : tensor<2x24x768xf32>
    %899 = stablehlo.convert %898 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %900 = stablehlo.broadcast_in_dim %arg42, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %901 = stablehlo.broadcast_in_dim %900, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %902 = stablehlo.multiply %899, %901 : tensor<2x24x768xf16>
    %cst_163 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %903 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %904 = stablehlo.multiply %903, %902 : tensor<2x24x768xf16>
    %905 = stablehlo.add %887, %904 : tensor<2x24x768xf16>
    %c_164 = stablehlo.constant dense<0> : tensor<i32>
    %906 = stablehlo.broadcast_in_dim %c_164, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %907 = stablehlo.compare NE, %arg0, %906, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %908 = stablehlo.broadcast_in_dim %907, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %909 = call @_where(%908, %905, %cst_165) : (tensor<2x24x1xi1>, tensor<2x24x768xf16>, tensor<f16>) -> tensor<2x24x768xf16>
    %cst_166 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %910 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %911 = stablehlo.compare GT, %909, %910, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_167 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %912 = call @_where_114(%911, %cst_167, %909) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %cst_168 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %913 = stablehlo.broadcast_in_dim %cst_168, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %914 = stablehlo.compare LT, %912, %913, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_169 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %915 = call @_where_114(%914, %cst_169, %912) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %916 = stablehlo.convert %915 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %917 = chlo.square %916 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %918 = stablehlo.reduce(%917 init: %cst_170) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %919 = stablehlo.broadcast_in_dim %918, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_171 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %920 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %921 = stablehlo.divide %919, %920 : tensor<2x24x1xf32>
    %cst_172 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %922 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %923 = stablehlo.maximum %921, %922 : tensor<2x24x1xf32>
    %924 = stablehlo.sqrt %923 : tensor<2x24x1xf32>
    %925 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %926 = stablehlo.divide %916, %925 : tensor<2x24x768xf32>
    %927 = stablehlo.convert %926 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %928 = stablehlo.broadcast_in_dim %arg53, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %929 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %930 = stablehlo.multiply %927, %929 : tensor<2x24x768xf16>
    %931 = stablehlo.dot_general %930, %arg60, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<768x768xf16>) -> tensor<2x24x768xf16>
    %cst_173 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %932 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %933 = stablehlo.divide %931, %932 : tensor<2x24x768xf16>
    %934 = stablehlo.tanh %933 : tensor<2x24x768xf16>
    %cst_174 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %935 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %936 = stablehlo.multiply %934, %935 : tensor<2x24x768xf16>
    %cst_175 = stablehlo.constant dense<1.562500e-02> : tensor<f16>
    %937 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %938 = stablehlo.multiply %936, %937 : tensor<2x24x768xf16>
    %939 = stablehlo.add %930, %938 : tensor<2x24x768xf16>
    %940 = stablehlo.dot_general %939, %arg50, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<6x768x128xf16>) -> tensor<2x24x6x128xf16>
    %941 = stablehlo.transpose %940, dims = [0, 2, 1, 3] : (tensor<2x24x6x128xf16>) -> tensor<2x6x24x128xf16>
    %942 = stablehlo.dot_general %arg49, %939, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x768x128xf16>, tensor<2x24x768xf16>) -> tensor<2x3x128x2x24xf16>
    %943 = stablehlo.transpose %942, dims = [3, 0, 4, 1, 2] : (tensor<2x3x128x2x24xf16>) -> tensor<2x2x24x3x128xf16>
    %944 = stablehlo.slice %943 [0:2, 0:1, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %945 = stablehlo.reshape %944 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %946 = stablehlo.transpose %945, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %947 = stablehlo.slice %943 [0:2, 1:2, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %948 = stablehlo.reshape %947 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %949 = stablehlo.transpose %948, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %950 = stablehlo.convert %941 : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf32>
    %951 = chlo.square %950 : tensor<2x6x24x128xf32> -> tensor<2x6x24x128xf32>
    %cst_176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %952 = stablehlo.reduce(%951 init: %cst_176) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x128xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %953 = stablehlo.broadcast_in_dim %952, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %cst_177 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %954 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %955 = stablehlo.divide %953, %954 : tensor<2x6x24x1xf32>
    %cst_178 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %956 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %957 = stablehlo.add %955, %956 : tensor<2x6x24x1xf32>
    %958 = stablehlo.rsqrt %957 : tensor<2x6x24x1xf32>
    %959 = stablehlo.broadcast_in_dim %958, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x128xf32>
    %960 = stablehlo.multiply %950, %959 : tensor<2x6x24x128xf32>
    %961 = stablehlo.convert %960 : (tensor<2x6x24x128xf32>) -> tensor<2x6x24x128xf16>
    %962 = stablehlo.broadcast_in_dim %arg57, dims = [1, 3] : (tensor<6x128xf16>) -> tensor<1x6x1x128xf16>
    %963 = stablehlo.broadcast_in_dim %962, dims = [0, 1, 2, 3] : (tensor<1x6x1x128xf16>) -> tensor<2x6x24x128xf16>
    %964 = stablehlo.multiply %961, %963 : tensor<2x6x24x128xf16>
    %965 = stablehlo.convert %946 : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x128xf32>
    %966 = chlo.square %965 : tensor<2x3x24x128xf32> -> tensor<2x3x24x128xf32>
    %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %967 = stablehlo.reduce(%966 init: %cst_179) applies stablehlo.add across dimensions = [3] : (tensor<2x3x24x128xf32>, tensor<f32>) -> tensor<2x3x24xf32>
    %968 = stablehlo.broadcast_in_dim %967, dims = [0, 1, 2] : (tensor<2x3x24xf32>) -> tensor<2x3x24x1xf32>
    %cst_180 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %969 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %970 = stablehlo.divide %968, %969 : tensor<2x3x24x1xf32>
    %cst_181 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %971 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %972 = stablehlo.add %970, %971 : tensor<2x3x24x1xf32>
    %973 = stablehlo.rsqrt %972 : tensor<2x3x24x1xf32>
    %974 = stablehlo.broadcast_in_dim %973, dims = [0, 1, 2, 3] : (tensor<2x3x24x1xf32>) -> tensor<2x3x24x128xf32>
    %975 = stablehlo.multiply %965, %974 : tensor<2x3x24x128xf32>
    %976 = stablehlo.convert %975 : (tensor<2x3x24x128xf32>) -> tensor<2x3x24x128xf16>
    %977 = stablehlo.broadcast_in_dim %arg58, dims = [1, 3] : (tensor<3x128xf16>) -> tensor<1x3x1x128xf16>
    %978 = stablehlo.broadcast_in_dim %977, dims = [0, 1, 2, 3] : (tensor<1x3x1x128xf16>) -> tensor<2x3x24x128xf16>
    %979 = stablehlo.multiply %976, %978 : tensor<2x3x24x128xf16>
    %980 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_182 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %981 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %982 = stablehlo.divide %980, %981 : tensor<128xf32>
    %cst_183 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %983 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %984 = stablehlo.power %983, %982 : tensor<128xf32>
    %cst_184 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %985 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %986 = stablehlo.divide %985, %984 : tensor<128xf32>
    %987 = stablehlo.convert %arg1 : (tensor<2x24xf16>) -> tensor<2x24xf32>
    %988 = stablehlo.broadcast_in_dim %987, dims = [0, 2] : (tensor<2x24xf32>) -> tensor<2x1x24x1xf32>
    %989 = stablehlo.broadcast_in_dim %986, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %990 = stablehlo.broadcast_in_dim %988, dims = [0, 1, 2, 3] : (tensor<2x1x24x1xf32>) -> tensor<2x1x24x128xf32>
    %991 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x24x128xf32>
    %992 = stablehlo.multiply %990, %991 : tensor<2x1x24x128xf32>
    %993 = stablehlo.cosine %992 : tensor<2x1x24x128xf32>
    %994 = stablehlo.convert %993 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %995 = stablehlo.sine %992 : tensor<2x1x24x128xf32>
    %996 = stablehlo.convert %995 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %997 = stablehlo.broadcast_in_dim %994, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %998 = stablehlo.multiply %964, %997 : tensor<2x6x24x128xf16>
    %999 = stablehlo.slice %964 [0:2, 0:6, 0:24, 0:64] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %1000 = stablehlo.slice %964 [0:2, 0:6, 0:24, 64:128] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %1001 = stablehlo.negate %1000 : tensor<2x6x24x64xf16>
    %1002 = stablehlo.concatenate %1001, %999, dim = 3 : (tensor<2x6x24x64xf16>, tensor<2x6x24x64xf16>) -> tensor<2x6x24x128xf16>
    %1003 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1004 = stablehlo.multiply %1002, %1003 : tensor<2x6x24x128xf16>
    %1005 = stablehlo.add %998, %1004 : tensor<2x6x24x128xf16>
    %1006 = stablehlo.broadcast_in_dim %994, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %1007 = stablehlo.multiply %979, %1006 : tensor<2x3x24x128xf16>
    %1008 = stablehlo.slice %979 [0:2, 0:3, 0:24, 0:64] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %1009 = stablehlo.slice %979 [0:2, 0:3, 0:24, 64:128] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %1010 = stablehlo.negate %1009 : tensor<2x3x24x64xf16>
    %1011 = stablehlo.concatenate %1010, %1008, dim = 3 : (tensor<2x3x24x64xf16>, tensor<2x3x24x64xf16>) -> tensor<2x3x24x128xf16>
    %1012 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %1013 = stablehlo.multiply %1011, %1012 : tensor<2x3x24x128xf16>
    %1014 = stablehlo.add %1007, %1013 : tensor<2x3x24x128xf16>
    %1015 = stablehlo.slice %1014 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1016 = stablehlo.slice %1014 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1017 = stablehlo.slice %1014 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1018 = stablehlo.slice %1014 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1019 = stablehlo.slice %1014 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1020 = stablehlo.slice %1014 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1021 = stablehlo.concatenate %1015, %1016, %1017, %1018, %1019, %1020, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1022 = stablehlo.slice %949 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1023 = stablehlo.slice %949 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1024 = stablehlo.slice %949 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1025 = stablehlo.slice %949 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1026 = stablehlo.slice %949 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1027 = stablehlo.slice %949 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1028 = stablehlo.concatenate %1022, %1023, %1024, %1025, %1026, %1027, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1029 = stablehlo.dot_general %1005, %1021, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x128xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x24xf16>
    %cst_185 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %1030 = stablehlo.broadcast_in_dim %cst_185, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1031 = stablehlo.multiply %1029, %1030 : tensor<2x6x24x24xf16>
    %cst_186 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %1032 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1033 = stablehlo.divide %1031, %1032 : tensor<2x6x24x24xf16>
    %1034 = stablehlo.tanh %1033 : tensor<2x6x24x24xf16>
    %cst_187 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %1035 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1036 = stablehlo.multiply %1034, %1035 : tensor<2x6x24x24xf16>
    %c_188 = stablehlo.constant dense<0> : tensor<i32>
    %1037 = stablehlo.broadcast_in_dim %c_188, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1038 = stablehlo.compare NE, %arg0, %1037, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %c_189 = stablehlo.constant dense<2> : tensor<i32>
    %1039 = stablehlo.broadcast_in_dim %c_189, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1040 = stablehlo.compare EQ, %arg0, %1039, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %1041 = stablehlo.not %1040 : tensor<2x24xi1>
    %1042 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x24xf16>) -> tensor<2x24x1xf16>
    %1043 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x24xf16>) -> tensor<2x1x24xf16>
    %1044 = stablehlo.broadcast_in_dim %1042, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1045 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1046 = stablehlo.compare GE, %1044, %1045, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1047 = stablehlo.broadcast_in_dim %1042, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1048 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1049 = stablehlo.compare GT, %1047, %1048, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_190 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1050 = stablehlo.broadcast_in_dim %cst_190, dims = [] : (tensor<f16>) -> tensor<2x24x1xf16>
    %1051 = stablehlo.add %1042, %1050 : tensor<2x24x1xf16>
    %1052 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1053 = stablehlo.broadcast_in_dim %1051, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1054 = stablehlo.compare LT, %1052, %1053, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1055 = stablehlo.broadcast_in_dim %1042, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1056 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1057 = stablehlo.subtract %1055, %1056 : tensor<2x24x24xf16>
    %cst_191 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1058 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %1059 = stablehlo.compare LE, %1057, %1058, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_192 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1060 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %1061 = stablehlo.compare GT, %arg2, %1060, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1062 = stablehlo.broadcast_in_dim %1038, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1063 = stablehlo.broadcast_in_dim %1038, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %1064 = stablehlo.and %1061, %1046 : tensor<2x24x24xi1>
    %1065 = stablehlo.and %1064, %1054 : tensor<2x24x24xi1>
    %1066 = stablehlo.or %1049, %1046 : tensor<2x24x24xi1>
    %1067 = stablehlo.and %1065, %1066 : tensor<2x24x24xi1>
    %1068 = stablehlo.and %1067, %1059 : tensor<2x24x24xi1>
    %1069 = stablehlo.broadcast_in_dim %1062, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %1070 = stablehlo.and %1068, %1069 : tensor<2x24x24xi1>
    %1071 = stablehlo.broadcast_in_dim %1063, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %1072 = stablehlo.and %1070, %1071 : tensor<2x24x24xi1>
    %1073 = stablehlo.broadcast_in_dim %1040, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1074 = stablehlo.broadcast_in_dim %1040, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %1075 = stablehlo.broadcast_in_dim %1073, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %1076 = stablehlo.broadcast_in_dim %1074, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %1077 = stablehlo.and %1075, %1076 : tensor<2x24x24xi1>
    %1078 = stablehlo.or %1072, %1077 : tensor<2x24x24xi1>
    %1079 = stablehlo.broadcast_in_dim %1041, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1080 = call @_where_77(%1079, %1078, %1077) : (tensor<2x24x1xi1>, tensor<2x24x24xi1>, tensor<2x24x24xi1>) -> tensor<2x24x24xi1>
    %1081 = stablehlo.broadcast_in_dim %1080, dims = [0, 2, 3] : (tensor<2x24x24xi1>) -> tensor<2x1x24x24xi1>
    %cst_193 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %1082 = call @_where_81(%1081, %1036, %cst_193) : (tensor<2x1x24x24xi1>, tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24x24xf16>
    %cst_194 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1083 = stablehlo.reduce(%1082 init: %cst_194) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %cst_195 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1084 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f16>) -> tensor<2x6x24xf16>
    %1085 = stablehlo.maximum %1084, %1083 : tensor<2x6x24xf16>
    %1086 = stablehlo.broadcast_in_dim %1085, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %1087 = stablehlo.broadcast_in_dim %1086, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1088 = stablehlo.subtract %1082, %1087 : tensor<2x6x24x24xf16>
    %1089 = stablehlo.exponential %1088 : tensor<2x6x24x24xf16>
    %1090 = stablehlo.convert %1089 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_196 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1091 = stablehlo.reduce(%1090 init: %cst_196) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1092 = stablehlo.broadcast_in_dim %1091, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1093 = stablehlo.convert %1092 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1094 = stablehlo.broadcast_in_dim %1093, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1095 = stablehlo.divide %1089, %1094 : tensor<2x6x24x24xf16>
    %1096 = stablehlo.convert %1095 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1097 = stablehlo.reduce(%1096 init: %cst_197) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1098 = stablehlo.broadcast_in_dim %1097, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1099 = stablehlo.convert %1098 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1100 = stablehlo.broadcast_in_dim %1099, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1101 = stablehlo.divide %1095, %1100 : tensor<2x6x24x24xf16>
    %cst_198 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1102 = stablehlo.reduce(%1082 init: %cst_198) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %1103 = stablehlo.broadcast_in_dim %1102, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %1104 = stablehlo.broadcast_in_dim %1103, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1105 = stablehlo.subtract %1082, %1104 : tensor<2x6x24x24xf16>
    %1106 = stablehlo.exponential %1105 : tensor<2x6x24x24xf16>
    %1107 = stablehlo.convert %1106 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1108 = stablehlo.reduce(%1107 init: %cst_199) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1109 = stablehlo.broadcast_in_dim %1108, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1110 = stablehlo.convert %1109 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1111 = stablehlo.broadcast_in_dim %1110, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1112 = stablehlo.divide %1106, %1111 : tensor<2x6x24x24xf16>
    %cst_200 = stablehlo.constant dense<7.500000e-01> : tensor<f16>
    %1113 = stablehlo.broadcast_in_dim %cst_200, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1114 = stablehlo.multiply %1101, %1113 : tensor<2x6x24x24xf16>
    %cst_201 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %1115 = stablehlo.broadcast_in_dim %cst_201, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1116 = stablehlo.multiply %1112, %1115 : tensor<2x6x24x24xf16>
    %1117 = stablehlo.add %1114, %1116 : tensor<2x6x24x24xf16>
    %1118 = stablehlo.dot_general %1117, %1028, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x24xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1119 = stablehlo.transpose %1118, dims = [0, 2, 1, 3] : (tensor<2x6x24x128xf16>) -> tensor<2x24x6x128xf16>
    %1120 = stablehlo.dot_general %1119, %arg48, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x6x128xf16>, tensor<6x128x768xf16>) -> tensor<2x24x768xf16>
    %cst_202 = stablehlo.constant dense<1.250000e-01> : tensor<f16>
    %1121 = stablehlo.broadcast_in_dim %cst_202, dims = [] : (tensor<f16>) -> tensor<768xf16>
    %1122 = stablehlo.maximum %arg59, %1121 : tensor<768xf16>
    %1123 = stablehlo.add %915, %1120 : tensor<2x24x768xf16>
    %1124 = stablehlo.convert %1123 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1125 = chlo.square %1124 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1126 = stablehlo.reduce(%1125 init: %cst_203) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1127 = stablehlo.broadcast_in_dim %1126, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_204 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1128 = stablehlo.broadcast_in_dim %cst_204, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1129 = stablehlo.divide %1127, %1128 : tensor<2x24x1xf32>
    %cst_205 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1130 = stablehlo.broadcast_in_dim %cst_205, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1131 = stablehlo.add %1129, %1130 : tensor<2x24x1xf32>
    %1132 = stablehlo.rsqrt %1131 : tensor<2x24x1xf32>
    %1133 = stablehlo.broadcast_in_dim %1132, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1134 = stablehlo.multiply %1124, %1133 : tensor<2x24x768xf32>
    %1135 = stablehlo.convert %1134 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1136 = stablehlo.broadcast_in_dim %arg54, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1138 = stablehlo.multiply %1135, %1137 : tensor<2x24x768xf16>
    %cst_206 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %1139 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1140 = stablehlo.multiply %1139, %1138 : tensor<2x24x768xf16>
    %1141 = stablehlo.add %1123, %1140 : tensor<2x24x768xf16>
    %1142 = stablehlo.broadcast_in_dim %1122, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %cst_207 = stablehlo.constant dense<1.000400e-03> : tensor<f16>
    %1143 = stablehlo.broadcast_in_dim %cst_207, dims = [] : (tensor<f16>) -> tensor<1x1x768xf16>
    %1144 = stablehlo.multiply %1142, %1143 : tensor<1x1x768xf16>
    %1145 = stablehlo.broadcast_in_dim %1144, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1146 = stablehlo.add %1141, %1145 : tensor<2x24x768xf16>
    %1147 = stablehlo.convert %1146 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1148 = chlo.square %1147 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_208 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1149 = stablehlo.reduce(%1148 init: %cst_208) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1150 = stablehlo.broadcast_in_dim %1149, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_209 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1151 = stablehlo.broadcast_in_dim %cst_209, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1152 = stablehlo.divide %1150, %1151 : tensor<2x24x1xf32>
    %cst_210 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1153 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1154 = stablehlo.add %1152, %1153 : tensor<2x24x1xf32>
    %1155 = stablehlo.rsqrt %1154 : tensor<2x24x1xf32>
    %1156 = stablehlo.broadcast_in_dim %1155, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1157 = stablehlo.multiply %1147, %1156 : tensor<2x24x768xf32>
    %1158 = stablehlo.convert %1157 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1159 = stablehlo.broadcast_in_dim %arg55, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1160 = stablehlo.broadcast_in_dim %1159, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1161 = stablehlo.multiply %1158, %1160 : tensor<2x24x768xf16>
    %1162 = stablehlo.dot_general %arg51, %1161, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x768x1536xf16>, tensor<2x24x768xf16>) -> tensor<2x1536x2x24xf16>
    %1163 = stablehlo.transpose %1162, dims = [2, 0, 3, 1] : (tensor<2x1536x2x24xf16>) -> tensor<2x2x24x1536xf16>
    %1164 = stablehlo.slice %1163 [0:2, 0:1, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %1165 = stablehlo.reshape %1164 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %1166 = stablehlo.slice %1163 [0:2, 1:2, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %1167 = stablehlo.reshape %1166 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %1168 = stablehlo.negate %1165 : tensor<2x24x1536xf16>
    %1169 = stablehlo.exponential %1168 : tensor<2x24x1536xf16>
    %cst_211 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1170 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1171 = stablehlo.add %1170, %1169 : tensor<2x24x1536xf16>
    %cst_212 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1172 = stablehlo.broadcast_in_dim %cst_212, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1173 = stablehlo.divide %1172, %1171 : tensor<2x24x1536xf16>
    %1174 = stablehlo.multiply %1165, %1173 : tensor<2x24x1536xf16>
    %1175 = stablehlo.multiply %1174, %1167 : tensor<2x24x1536xf16>
    %cst_213 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %1176 = stablehlo.broadcast_in_dim %cst_213, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1177 = stablehlo.divide %1175, %1176 : tensor<2x24x1536xf16>
    %1178 = stablehlo.tanh %1177 : tensor<2x24x1536xf16>
    %cst_214 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %1179 = stablehlo.broadcast_in_dim %cst_214, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1180 = stablehlo.multiply %1178, %1179 : tensor<2x24x1536xf16>
    %1181 = stablehlo.dot_general %1180, %arg52, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x1536xf16>, tensor<1536x768xf16>) -> tensor<2x24x768xf16>
    %1182 = stablehlo.broadcast_in_dim %arg61, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1183 = stablehlo.broadcast_in_dim %1182, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1184 = stablehlo.multiply %1181, %1183 : tensor<2x24x768xf16>
    %1185 = stablehlo.add %1146, %1184 : tensor<2x24x768xf16>
    %1186 = stablehlo.convert %1185 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1187 = chlo.square %1186 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1188 = stablehlo.reduce(%1187 init: %cst_215) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1189 = stablehlo.broadcast_in_dim %1188, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_216 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1190 = stablehlo.broadcast_in_dim %cst_216, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1191 = stablehlo.divide %1189, %1190 : tensor<2x24x1xf32>
    %cst_217 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1192 = stablehlo.broadcast_in_dim %cst_217, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1193 = stablehlo.add %1191, %1192 : tensor<2x24x1xf32>
    %1194 = stablehlo.rsqrt %1193 : tensor<2x24x1xf32>
    %1195 = stablehlo.broadcast_in_dim %1194, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1196 = stablehlo.multiply %1186, %1195 : tensor<2x24x768xf32>
    %1197 = stablehlo.convert %1196 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1198 = stablehlo.broadcast_in_dim %arg56, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1199 = stablehlo.broadcast_in_dim %1198, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1200 = stablehlo.multiply %1197, %1199 : tensor<2x24x768xf16>
    %cst_218 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %1201 = stablehlo.broadcast_in_dim %cst_218, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1202 = stablehlo.multiply %1201, %1200 : tensor<2x24x768xf16>
    %1203 = stablehlo.add %1185, %1202 : tensor<2x24x768xf16>
    %c_219 = stablehlo.constant dense<0> : tensor<i32>
    %1204 = stablehlo.broadcast_in_dim %c_219, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1205 = stablehlo.compare NE, %arg0, %1204, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %1206 = stablehlo.broadcast_in_dim %1205, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %cst_220 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1207 = call @_where(%1206, %1203, %cst_220) : (tensor<2x24x1xi1>, tensor<2x24x768xf16>, tensor<f16>) -> tensor<2x24x768xf16>
    %cst_221 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1208 = stablehlo.broadcast_in_dim %cst_221, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1209 = stablehlo.compare GT, %1207, %1208, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_222 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1210 = call @_where_114(%1209, %cst_222, %1207) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %cst_223 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %1211 = stablehlo.broadcast_in_dim %cst_223, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1212 = stablehlo.compare LT, %1210, %1211, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_224 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %1213 = call @_where_114(%1212, %cst_224, %1210) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %1214 = stablehlo.convert %1213 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1215 = chlo.square %1214 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1216 = stablehlo.reduce(%1215 init: %cst_225) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_226 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1218 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1219 = stablehlo.divide %1217, %1218 : tensor<2x24x1xf32>
    %cst_227 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1220 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1221 = stablehlo.add %1219, %1220 : tensor<2x24x1xf32>
    %1222 = stablehlo.rsqrt %1221 : tensor<2x24x1xf32>
    %1223 = stablehlo.broadcast_in_dim %1222, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1224 = stablehlo.multiply %1214, %1223 : tensor<2x24x768xf32>
    %1225 = stablehlo.convert %1224 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1226 = stablehlo.broadcast_in_dim %arg67, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1227 = stablehlo.broadcast_in_dim %1226, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1228 = stablehlo.multiply %1225, %1227 : tensor<2x24x768xf16>
    %1229 = stablehlo.dot_general %1228, %arg74, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<768x768xf16>) -> tensor<2x24x768xf16>
    %cst_228 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1230 = stablehlo.broadcast_in_dim %cst_228, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1231 = stablehlo.divide %1229, %1230 : tensor<2x24x768xf16>
    %1232 = stablehlo.tanh %1231 : tensor<2x24x768xf16>
    %cst_229 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1233 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1234 = stablehlo.multiply %1232, %1233 : tensor<2x24x768xf16>
    %cst_230 = stablehlo.constant dense<1.562500e-02> : tensor<f16>
    %1235 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1236 = stablehlo.multiply %1234, %1235 : tensor<2x24x768xf16>
    %1237 = stablehlo.add %1228, %1236 : tensor<2x24x768xf16>
    %1238 = stablehlo.dot_general %1237, %arg64, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<6x768x128xf16>) -> tensor<2x24x6x128xf16>
    %1239 = stablehlo.transpose %1238, dims = [0, 2, 1, 3] : (tensor<2x24x6x128xf16>) -> tensor<2x6x24x128xf16>
    %1240 = stablehlo.dot_general %arg63, %1237, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x768x128xf16>, tensor<2x24x768xf16>) -> tensor<2x3x128x2x24xf16>
    %1241 = stablehlo.transpose %1240, dims = [3, 0, 4, 1, 2] : (tensor<2x3x128x2x24xf16>) -> tensor<2x2x24x3x128xf16>
    %1242 = stablehlo.slice %1241 [0:2, 0:1, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %1243 = stablehlo.reshape %1242 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %1244 = stablehlo.transpose %1243, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %1245 = stablehlo.slice %1241 [0:2, 1:2, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %1246 = stablehlo.reshape %1245 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %1247 = stablehlo.transpose %1246, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %1248 = stablehlo.convert %1239 : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf32>
    %1249 = chlo.square %1248 : tensor<2x6x24x128xf32> -> tensor<2x6x24x128xf32>
    %cst_231 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1250 = stablehlo.reduce(%1249 init: %cst_231) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x128xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1251 = stablehlo.broadcast_in_dim %1250, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %cst_232 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1252 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %1253 = stablehlo.divide %1251, %1252 : tensor<2x6x24x1xf32>
    %cst_233 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1254 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %1255 = stablehlo.add %1253, %1254 : tensor<2x6x24x1xf32>
    %1256 = stablehlo.rsqrt %1255 : tensor<2x6x24x1xf32>
    %1257 = stablehlo.broadcast_in_dim %1256, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x128xf32>
    %1258 = stablehlo.multiply %1248, %1257 : tensor<2x6x24x128xf32>
    %1259 = stablehlo.convert %1258 : (tensor<2x6x24x128xf32>) -> tensor<2x6x24x128xf16>
    %1260 = stablehlo.broadcast_in_dim %arg71, dims = [1, 3] : (tensor<6x128xf16>) -> tensor<1x6x1x128xf16>
    %1261 = stablehlo.broadcast_in_dim %1260, dims = [0, 1, 2, 3] : (tensor<1x6x1x128xf16>) -> tensor<2x6x24x128xf16>
    %1262 = stablehlo.multiply %1259, %1261 : tensor<2x6x24x128xf16>
    %1263 = stablehlo.convert %1244 : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x128xf32>
    %1264 = chlo.square %1263 : tensor<2x3x24x128xf32> -> tensor<2x3x24x128xf32>
    %cst_234 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1265 = stablehlo.reduce(%1264 init: %cst_234) applies stablehlo.add across dimensions = [3] : (tensor<2x3x24x128xf32>, tensor<f32>) -> tensor<2x3x24xf32>
    %1266 = stablehlo.broadcast_in_dim %1265, dims = [0, 1, 2] : (tensor<2x3x24xf32>) -> tensor<2x3x24x1xf32>
    %cst_235 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1267 = stablehlo.broadcast_in_dim %cst_235, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %1268 = stablehlo.divide %1266, %1267 : tensor<2x3x24x1xf32>
    %cst_236 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1269 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %1270 = stablehlo.add %1268, %1269 : tensor<2x3x24x1xf32>
    %1271 = stablehlo.rsqrt %1270 : tensor<2x3x24x1xf32>
    %1272 = stablehlo.broadcast_in_dim %1271, dims = [0, 1, 2, 3] : (tensor<2x3x24x1xf32>) -> tensor<2x3x24x128xf32>
    %1273 = stablehlo.multiply %1263, %1272 : tensor<2x3x24x128xf32>
    %1274 = stablehlo.convert %1273 : (tensor<2x3x24x128xf32>) -> tensor<2x3x24x128xf16>
    %1275 = stablehlo.broadcast_in_dim %arg72, dims = [1, 3] : (tensor<3x128xf16>) -> tensor<1x3x1x128xf16>
    %1276 = stablehlo.broadcast_in_dim %1275, dims = [0, 1, 2, 3] : (tensor<1x3x1x128xf16>) -> tensor<2x3x24x128xf16>
    %1277 = stablehlo.multiply %1274, %1276 : tensor<2x3x24x128xf16>
    %1278 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_237 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1279 = stablehlo.broadcast_in_dim %cst_237, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1280 = stablehlo.divide %1278, %1279 : tensor<128xf32>
    %cst_238 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1281 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1282 = stablehlo.power %1281, %1280 : tensor<128xf32>
    %cst_239 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1283 = stablehlo.broadcast_in_dim %cst_239, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1284 = stablehlo.divide %1283, %1282 : tensor<128xf32>
    %1285 = stablehlo.convert %arg1 : (tensor<2x24xf16>) -> tensor<2x24xf32>
    %1286 = stablehlo.broadcast_in_dim %1285, dims = [0, 2] : (tensor<2x24xf32>) -> tensor<2x1x24x1xf32>
    %1287 = stablehlo.broadcast_in_dim %1284, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1288 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<2x1x24x1xf32>) -> tensor<2x1x24x128xf32>
    %1289 = stablehlo.broadcast_in_dim %1287, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x24x128xf32>
    %1290 = stablehlo.multiply %1288, %1289 : tensor<2x1x24x128xf32>
    %1291 = stablehlo.cosine %1290 : tensor<2x1x24x128xf32>
    %1292 = stablehlo.convert %1291 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %1293 = stablehlo.sine %1290 : tensor<2x1x24x128xf32>
    %1294 = stablehlo.convert %1293 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %1295 = stablehlo.broadcast_in_dim %1292, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1296 = stablehlo.multiply %1262, %1295 : tensor<2x6x24x128xf16>
    %1297 = stablehlo.slice %1262 [0:2, 0:6, 0:24, 0:64] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %1298 = stablehlo.slice %1262 [0:2, 0:6, 0:24, 64:128] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %1299 = stablehlo.negate %1298 : tensor<2x6x24x64xf16>
    %1300 = stablehlo.concatenate %1299, %1297, dim = 3 : (tensor<2x6x24x64xf16>, tensor<2x6x24x64xf16>) -> tensor<2x6x24x128xf16>
    %1301 = stablehlo.broadcast_in_dim %1294, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1302 = stablehlo.multiply %1300, %1301 : tensor<2x6x24x128xf16>
    %1303 = stablehlo.add %1296, %1302 : tensor<2x6x24x128xf16>
    %1304 = stablehlo.broadcast_in_dim %1292, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %1305 = stablehlo.multiply %1277, %1304 : tensor<2x3x24x128xf16>
    %1306 = stablehlo.slice %1277 [0:2, 0:3, 0:24, 0:64] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %1307 = stablehlo.slice %1277 [0:2, 0:3, 0:24, 64:128] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %1308 = stablehlo.negate %1307 : tensor<2x3x24x64xf16>
    %1309 = stablehlo.concatenate %1308, %1306, dim = 3 : (tensor<2x3x24x64xf16>, tensor<2x3x24x64xf16>) -> tensor<2x3x24x128xf16>
    %1310 = stablehlo.broadcast_in_dim %1294, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %1311 = stablehlo.multiply %1309, %1310 : tensor<2x3x24x128xf16>
    %1312 = stablehlo.add %1305, %1311 : tensor<2x3x24x128xf16>
    %1313 = stablehlo.slice %1312 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1314 = stablehlo.slice %1312 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1315 = stablehlo.slice %1312 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1316 = stablehlo.slice %1312 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1317 = stablehlo.slice %1312 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1318 = stablehlo.slice %1312 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1319 = stablehlo.concatenate %1313, %1314, %1315, %1316, %1317, %1318, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1320 = stablehlo.slice %1247 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1321 = stablehlo.slice %1247 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1322 = stablehlo.slice %1247 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1323 = stablehlo.slice %1247 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1324 = stablehlo.slice %1247 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1325 = stablehlo.slice %1247 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1326 = stablehlo.concatenate %1320, %1321, %1322, %1323, %1324, %1325, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1327 = stablehlo.dot_general %1303, %1319, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x128xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x24xf16>
    %cst_240 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %1328 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1329 = stablehlo.multiply %1327, %1328 : tensor<2x6x24x24xf16>
    %cst_241 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %1330 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1331 = stablehlo.divide %1329, %1330 : tensor<2x6x24x24xf16>
    %1332 = stablehlo.tanh %1331 : tensor<2x6x24x24xf16>
    %cst_242 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %1333 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1334 = stablehlo.multiply %1332, %1333 : tensor<2x6x24x24xf16>
    %c_243 = stablehlo.constant dense<0> : tensor<i32>
    %1335 = stablehlo.broadcast_in_dim %c_243, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1336 = stablehlo.compare NE, %arg0, %1335, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %c_244 = stablehlo.constant dense<2> : tensor<i32>
    %1337 = stablehlo.broadcast_in_dim %c_244, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1338 = stablehlo.compare EQ, %arg0, %1337, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %1339 = stablehlo.not %1338 : tensor<2x24xi1>
    %1340 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x24xf16>) -> tensor<2x24x1xf16>
    %1341 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x24xf16>) -> tensor<2x1x24xf16>
    %1342 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1343 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1344 = stablehlo.compare GE, %1342, %1343, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1345 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1346 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1347 = stablehlo.compare GT, %1345, %1346, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_245 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1348 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<f16>) -> tensor<2x24x1xf16>
    %1349 = stablehlo.add %1340, %1348 : tensor<2x24x1xf16>
    %1350 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1351 = stablehlo.broadcast_in_dim %1349, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1352 = stablehlo.compare LT, %1350, %1351, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1353 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1354 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1355 = stablehlo.subtract %1353, %1354 : tensor<2x24x24xf16>
    %cst_246 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1356 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %1357 = stablehlo.compare LE, %1355, %1356, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1358 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %1359 = stablehlo.compare GT, %arg2, %1358, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1360 = stablehlo.broadcast_in_dim %1336, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1361 = stablehlo.broadcast_in_dim %1336, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %1362 = stablehlo.and %1359, %1344 : tensor<2x24x24xi1>
    %1363 = stablehlo.and %1362, %1352 : tensor<2x24x24xi1>
    %1364 = stablehlo.or %1347, %1344 : tensor<2x24x24xi1>
    %1365 = stablehlo.and %1363, %1364 : tensor<2x24x24xi1>
    %1366 = stablehlo.and %1365, %1357 : tensor<2x24x24xi1>
    %1367 = stablehlo.broadcast_in_dim %1360, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %1368 = stablehlo.and %1366, %1367 : tensor<2x24x24xi1>
    %1369 = stablehlo.broadcast_in_dim %1361, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %1370 = stablehlo.and %1368, %1369 : tensor<2x24x24xi1>
    %1371 = stablehlo.broadcast_in_dim %1338, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1372 = stablehlo.broadcast_in_dim %1338, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %1373 = stablehlo.broadcast_in_dim %1371, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %1374 = stablehlo.broadcast_in_dim %1372, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %1375 = stablehlo.and %1373, %1374 : tensor<2x24x24xi1>
    %1376 = stablehlo.or %1370, %1375 : tensor<2x24x24xi1>
    %1377 = stablehlo.broadcast_in_dim %1339, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1378 = call @_where_77(%1377, %1376, %1375) : (tensor<2x24x1xi1>, tensor<2x24x24xi1>, tensor<2x24x24xi1>) -> tensor<2x24x24xi1>
    %1379 = stablehlo.broadcast_in_dim %1378, dims = [0, 2, 3] : (tensor<2x24x24xi1>) -> tensor<2x1x24x24xi1>
    %cst_248 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %1380 = call @_where_81(%1379, %1334, %cst_248) : (tensor<2x1x24x24xi1>, tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24x24xf16>
    %cst_249 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1381 = stablehlo.reduce(%1380 init: %cst_249) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %cst_250 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1382 = stablehlo.broadcast_in_dim %cst_250, dims = [] : (tensor<f16>) -> tensor<2x6x24xf16>
    %1383 = stablehlo.maximum %1382, %1381 : tensor<2x6x24xf16>
    %1384 = stablehlo.broadcast_in_dim %1383, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %1385 = stablehlo.broadcast_in_dim %1384, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1386 = stablehlo.subtract %1380, %1385 : tensor<2x6x24x24xf16>
    %1387 = stablehlo.exponential %1386 : tensor<2x6x24x24xf16>
    %1388 = stablehlo.convert %1387 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1389 = stablehlo.reduce(%1388 init: %cst_251) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1390 = stablehlo.broadcast_in_dim %1389, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1391 = stablehlo.convert %1390 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1392 = stablehlo.broadcast_in_dim %1391, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1393 = stablehlo.divide %1387, %1392 : tensor<2x6x24x24xf16>
    %1394 = stablehlo.convert %1393 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1395 = stablehlo.reduce(%1394 init: %cst_252) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1396 = stablehlo.broadcast_in_dim %1395, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1397 = stablehlo.convert %1396 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1398 = stablehlo.broadcast_in_dim %1397, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1399 = stablehlo.divide %1393, %1398 : tensor<2x6x24x24xf16>
    %cst_253 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1400 = stablehlo.reduce(%1380 init: %cst_253) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %1401 = stablehlo.broadcast_in_dim %1400, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %1402 = stablehlo.broadcast_in_dim %1401, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1403 = stablehlo.subtract %1380, %1402 : tensor<2x6x24x24xf16>
    %1404 = stablehlo.exponential %1403 : tensor<2x6x24x24xf16>
    %1405 = stablehlo.convert %1404 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1406 = stablehlo.reduce(%1405 init: %cst_254) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1407 = stablehlo.broadcast_in_dim %1406, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1408 = stablehlo.convert %1407 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1409 = stablehlo.broadcast_in_dim %1408, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1410 = stablehlo.divide %1404, %1409 : tensor<2x6x24x24xf16>
    %cst_255 = stablehlo.constant dense<7.500000e-01> : tensor<f16>
    %1411 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1412 = stablehlo.multiply %1399, %1411 : tensor<2x6x24x24xf16>
    %cst_256 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %1413 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1414 = stablehlo.multiply %1410, %1413 : tensor<2x6x24x24xf16>
    %1415 = stablehlo.add %1412, %1414 : tensor<2x6x24x24xf16>
    %1416 = stablehlo.dot_general %1415, %1326, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x24xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1417 = stablehlo.transpose %1416, dims = [0, 2, 1, 3] : (tensor<2x6x24x128xf16>) -> tensor<2x24x6x128xf16>
    %1418 = stablehlo.dot_general %1417, %arg62, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x6x128xf16>, tensor<6x128x768xf16>) -> tensor<2x24x768xf16>
    %cst_257 = stablehlo.constant dense<1.250000e-01> : tensor<f16>
    %1419 = stablehlo.broadcast_in_dim %cst_257, dims = [] : (tensor<f16>) -> tensor<768xf16>
    %1420 = stablehlo.maximum %arg73, %1419 : tensor<768xf16>
    %1421 = stablehlo.add %1213, %1418 : tensor<2x24x768xf16>
    %1422 = stablehlo.convert %1421 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1423 = chlo.square %1422 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_258 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1424 = stablehlo.reduce(%1423 init: %cst_258) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1425 = stablehlo.broadcast_in_dim %1424, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_259 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1426 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1427 = stablehlo.divide %1425, %1426 : tensor<2x24x1xf32>
    %cst_260 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1428 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1429 = stablehlo.add %1427, %1428 : tensor<2x24x1xf32>
    %1430 = stablehlo.rsqrt %1429 : tensor<2x24x1xf32>
    %1431 = stablehlo.broadcast_in_dim %1430, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1432 = stablehlo.multiply %1422, %1431 : tensor<2x24x768xf32>
    %1433 = stablehlo.convert %1432 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1434 = stablehlo.broadcast_in_dim %arg68, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1435 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1436 = stablehlo.multiply %1433, %1435 : tensor<2x24x768xf16>
    %cst_261 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %1437 = stablehlo.broadcast_in_dim %cst_261, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1438 = stablehlo.multiply %1437, %1436 : tensor<2x24x768xf16>
    %1439 = stablehlo.add %1421, %1438 : tensor<2x24x768xf16>
    %1440 = stablehlo.broadcast_in_dim %1420, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %cst_262 = stablehlo.constant dense<1.000400e-03> : tensor<f16>
    %1441 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<f16>) -> tensor<1x1x768xf16>
    %1442 = stablehlo.multiply %1440, %1441 : tensor<1x1x768xf16>
    %1443 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1444 = stablehlo.add %1439, %1443 : tensor<2x24x768xf16>
    %1445 = stablehlo.convert %1444 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1446 = chlo.square %1445 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1447 = stablehlo.reduce(%1446 init: %cst_263) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1448 = stablehlo.broadcast_in_dim %1447, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_264 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1449 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1450 = stablehlo.divide %1448, %1449 : tensor<2x24x1xf32>
    %cst_265 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1451 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1452 = stablehlo.add %1450, %1451 : tensor<2x24x1xf32>
    %1453 = stablehlo.rsqrt %1452 : tensor<2x24x1xf32>
    %1454 = stablehlo.broadcast_in_dim %1453, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1455 = stablehlo.multiply %1445, %1454 : tensor<2x24x768xf32>
    %1456 = stablehlo.convert %1455 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1457 = stablehlo.broadcast_in_dim %arg69, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1458 = stablehlo.broadcast_in_dim %1457, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1459 = stablehlo.multiply %1456, %1458 : tensor<2x24x768xf16>
    %1460 = stablehlo.dot_general %arg65, %1459, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x768x1536xf16>, tensor<2x24x768xf16>) -> tensor<2x1536x2x24xf16>
    %1461 = stablehlo.transpose %1460, dims = [2, 0, 3, 1] : (tensor<2x1536x2x24xf16>) -> tensor<2x2x24x1536xf16>
    %1462 = stablehlo.slice %1461 [0:2, 0:1, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %1463 = stablehlo.reshape %1462 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %1464 = stablehlo.slice %1461 [0:2, 1:2, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %1465 = stablehlo.reshape %1464 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %1466 = stablehlo.negate %1463 : tensor<2x24x1536xf16>
    %1467 = stablehlo.exponential %1466 : tensor<2x24x1536xf16>
    %cst_266 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1468 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1469 = stablehlo.add %1468, %1467 : tensor<2x24x1536xf16>
    %cst_267 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1470 = stablehlo.broadcast_in_dim %cst_267, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1471 = stablehlo.divide %1470, %1469 : tensor<2x24x1536xf16>
    %1472 = stablehlo.multiply %1463, %1471 : tensor<2x24x1536xf16>
    %1473 = stablehlo.multiply %1472, %1465 : tensor<2x24x1536xf16>
    %cst_268 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %1474 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1475 = stablehlo.divide %1473, %1474 : tensor<2x24x1536xf16>
    %1476 = stablehlo.tanh %1475 : tensor<2x24x1536xf16>
    %cst_269 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %1477 = stablehlo.broadcast_in_dim %cst_269, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1478 = stablehlo.multiply %1476, %1477 : tensor<2x24x1536xf16>
    %1479 = stablehlo.dot_general %1478, %arg66, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x1536xf16>, tensor<1536x768xf16>) -> tensor<2x24x768xf16>
    %1480 = stablehlo.broadcast_in_dim %arg75, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1481 = stablehlo.broadcast_in_dim %1480, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1482 = stablehlo.multiply %1479, %1481 : tensor<2x24x768xf16>
    %1483 = stablehlo.add %1444, %1482 : tensor<2x24x768xf16>
    %1484 = stablehlo.convert %1483 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1485 = chlo.square %1484 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_270 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1486 = stablehlo.reduce(%1485 init: %cst_270) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1487 = stablehlo.broadcast_in_dim %1486, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_271 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1488 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1489 = stablehlo.divide %1487, %1488 : tensor<2x24x1xf32>
    %cst_272 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1490 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1491 = stablehlo.add %1489, %1490 : tensor<2x24x1xf32>
    %1492 = stablehlo.rsqrt %1491 : tensor<2x24x1xf32>
    %1493 = stablehlo.broadcast_in_dim %1492, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1494 = stablehlo.multiply %1484, %1493 : tensor<2x24x768xf32>
    %1495 = stablehlo.convert %1494 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1496 = stablehlo.broadcast_in_dim %arg70, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1497 = stablehlo.broadcast_in_dim %1496, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1498 = stablehlo.multiply %1495, %1497 : tensor<2x24x768xf16>
    %cst_273 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %1499 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1500 = stablehlo.multiply %1499, %1498 : tensor<2x24x768xf16>
    %1501 = stablehlo.add %1483, %1500 : tensor<2x24x768xf16>
    %c_274 = stablehlo.constant dense<0> : tensor<i32>
    %1502 = stablehlo.broadcast_in_dim %c_274, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1503 = stablehlo.compare NE, %arg0, %1502, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %1504 = stablehlo.broadcast_in_dim %1503, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %cst_275 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1505 = call @_where(%1504, %1501, %cst_275) : (tensor<2x24x1xi1>, tensor<2x24x768xf16>, tensor<f16>) -> tensor<2x24x768xf16>
    %cst_276 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1506 = stablehlo.broadcast_in_dim %cst_276, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1507 = stablehlo.compare GT, %1505, %1506, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_277 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1508 = call @_where_114(%1507, %cst_277, %1505) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %cst_278 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %1509 = stablehlo.broadcast_in_dim %cst_278, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1510 = stablehlo.compare LT, %1508, %1509, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_279 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %1511 = call @_where_114(%1510, %cst_279, %1508) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %1512 = stablehlo.convert %1511 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1513 = chlo.square %1512 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1514 = stablehlo.reduce(%1513 init: %cst_280) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1515 = stablehlo.broadcast_in_dim %1514, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_281 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1516 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1517 = stablehlo.divide %1515, %1516 : tensor<2x24x1xf32>
    %cst_282 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1518 = stablehlo.broadcast_in_dim %cst_282, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1519 = stablehlo.maximum %1517, %1518 : tensor<2x24x1xf32>
    %1520 = stablehlo.sqrt %1519 : tensor<2x24x1xf32>
    %1521 = stablehlo.broadcast_in_dim %1520, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1522 = stablehlo.divide %1512, %1521 : tensor<2x24x768xf32>
    %1523 = stablehlo.convert %1522 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1524 = stablehlo.broadcast_in_dim %arg81, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1525 = stablehlo.broadcast_in_dim %1524, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1526 = stablehlo.multiply %1523, %1525 : tensor<2x24x768xf16>
    %1527 = stablehlo.dot_general %1526, %arg88, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<768x768xf16>) -> tensor<2x24x768xf16>
    %cst_283 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1528 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1529 = stablehlo.divide %1527, %1528 : tensor<2x24x768xf16>
    %1530 = stablehlo.tanh %1529 : tensor<2x24x768xf16>
    %cst_284 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1531 = stablehlo.broadcast_in_dim %cst_284, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1532 = stablehlo.multiply %1530, %1531 : tensor<2x24x768xf16>
    %cst_285 = stablehlo.constant dense<1.562500e-02> : tensor<f16>
    %1533 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1534 = stablehlo.multiply %1532, %1533 : tensor<2x24x768xf16>
    %1535 = stablehlo.add %1526, %1534 : tensor<2x24x768xf16>
    %1536 = stablehlo.dot_general %1535, %arg78, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<6x768x128xf16>) -> tensor<2x24x6x128xf16>
    %1537 = stablehlo.transpose %1536, dims = [0, 2, 1, 3] : (tensor<2x24x6x128xf16>) -> tensor<2x6x24x128xf16>
    %1538 = stablehlo.dot_general %arg77, %1535, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x768x128xf16>, tensor<2x24x768xf16>) -> tensor<2x3x128x2x24xf16>
    %1539 = stablehlo.transpose %1538, dims = [3, 0, 4, 1, 2] : (tensor<2x3x128x2x24xf16>) -> tensor<2x2x24x3x128xf16>
    %1540 = stablehlo.slice %1539 [0:2, 0:1, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %1541 = stablehlo.reshape %1540 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %1542 = stablehlo.transpose %1541, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %1543 = stablehlo.slice %1539 [0:2, 1:2, 0:24, 0:3, 0:128] : (tensor<2x2x24x3x128xf16>) -> tensor<2x1x24x3x128xf16>
    %1544 = stablehlo.reshape %1543 : (tensor<2x1x24x3x128xf16>) -> tensor<2x24x3x128xf16>
    %1545 = stablehlo.transpose %1544, dims = [0, 2, 1, 3] : (tensor<2x24x3x128xf16>) -> tensor<2x3x24x128xf16>
    %1546 = stablehlo.convert %1537 : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf32>
    %1547 = chlo.square %1546 : tensor<2x6x24x128xf32> -> tensor<2x6x24x128xf32>
    %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1548 = stablehlo.reduce(%1547 init: %cst_286) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x128xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1549 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %cst_287 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1550 = stablehlo.broadcast_in_dim %cst_287, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %1551 = stablehlo.divide %1549, %1550 : tensor<2x6x24x1xf32>
    %cst_288 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1552 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<2x6x24x1xf32>
    %1553 = stablehlo.add %1551, %1552 : tensor<2x6x24x1xf32>
    %1554 = stablehlo.rsqrt %1553 : tensor<2x6x24x1xf32>
    %1555 = stablehlo.broadcast_in_dim %1554, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x128xf32>
    %1556 = stablehlo.multiply %1546, %1555 : tensor<2x6x24x128xf32>
    %1557 = stablehlo.convert %1556 : (tensor<2x6x24x128xf32>) -> tensor<2x6x24x128xf16>
    %1558 = stablehlo.broadcast_in_dim %arg85, dims = [1, 3] : (tensor<6x128xf16>) -> tensor<1x6x1x128xf16>
    %1559 = stablehlo.broadcast_in_dim %1558, dims = [0, 1, 2, 3] : (tensor<1x6x1x128xf16>) -> tensor<2x6x24x128xf16>
    %1560 = stablehlo.multiply %1557, %1559 : tensor<2x6x24x128xf16>
    %1561 = stablehlo.convert %1542 : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x128xf32>
    %1562 = chlo.square %1561 : tensor<2x3x24x128xf32> -> tensor<2x3x24x128xf32>
    %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1563 = stablehlo.reduce(%1562 init: %cst_289) applies stablehlo.add across dimensions = [3] : (tensor<2x3x24x128xf32>, tensor<f32>) -> tensor<2x3x24xf32>
    %1564 = stablehlo.broadcast_in_dim %1563, dims = [0, 1, 2] : (tensor<2x3x24xf32>) -> tensor<2x3x24x1xf32>
    %cst_290 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1565 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %1566 = stablehlo.divide %1564, %1565 : tensor<2x3x24x1xf32>
    %cst_291 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1567 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f32>) -> tensor<2x3x24x1xf32>
    %1568 = stablehlo.add %1566, %1567 : tensor<2x3x24x1xf32>
    %1569 = stablehlo.rsqrt %1568 : tensor<2x3x24x1xf32>
    %1570 = stablehlo.broadcast_in_dim %1569, dims = [0, 1, 2, 3] : (tensor<2x3x24x1xf32>) -> tensor<2x3x24x128xf32>
    %1571 = stablehlo.multiply %1561, %1570 : tensor<2x3x24x128xf32>
    %1572 = stablehlo.convert %1571 : (tensor<2x3x24x128xf32>) -> tensor<2x3x24x128xf16>
    %1573 = stablehlo.broadcast_in_dim %arg86, dims = [1, 3] : (tensor<3x128xf16>) -> tensor<1x3x1x128xf16>
    %1574 = stablehlo.broadcast_in_dim %1573, dims = [0, 1, 2, 3] : (tensor<1x3x1x128xf16>) -> tensor<2x3x24x128xf16>
    %1575 = stablehlo.multiply %1572, %1574 : tensor<2x3x24x128xf16>
    %1576 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_292 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1577 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1578 = stablehlo.divide %1576, %1577 : tensor<128xf32>
    %cst_293 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1579 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1580 = stablehlo.power %1579, %1578 : tensor<128xf32>
    %cst_294 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1581 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1582 = stablehlo.divide %1581, %1580 : tensor<128xf32>
    %1583 = stablehlo.convert %arg1 : (tensor<2x24xf16>) -> tensor<2x24xf32>
    %1584 = stablehlo.broadcast_in_dim %1583, dims = [0, 2] : (tensor<2x24xf32>) -> tensor<2x1x24x1xf32>
    %1585 = stablehlo.broadcast_in_dim %1582, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1586 = stablehlo.broadcast_in_dim %1584, dims = [0, 1, 2, 3] : (tensor<2x1x24x1xf32>) -> tensor<2x1x24x128xf32>
    %1587 = stablehlo.broadcast_in_dim %1585, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<2x1x24x128xf32>
    %1588 = stablehlo.multiply %1586, %1587 : tensor<2x1x24x128xf32>
    %1589 = stablehlo.cosine %1588 : tensor<2x1x24x128xf32>
    %1590 = stablehlo.convert %1589 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %1591 = stablehlo.sine %1588 : tensor<2x1x24x128xf32>
    %1592 = stablehlo.convert %1591 : (tensor<2x1x24x128xf32>) -> tensor<2x1x24x128xf16>
    %1593 = stablehlo.broadcast_in_dim %1590, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1594 = stablehlo.multiply %1560, %1593 : tensor<2x6x24x128xf16>
    %1595 = stablehlo.slice %1560 [0:2, 0:6, 0:24, 0:64] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %1596 = stablehlo.slice %1560 [0:2, 0:6, 0:24, 64:128] : (tensor<2x6x24x128xf16>) -> tensor<2x6x24x64xf16>
    %1597 = stablehlo.negate %1596 : tensor<2x6x24x64xf16>
    %1598 = stablehlo.concatenate %1597, %1595, dim = 3 : (tensor<2x6x24x64xf16>, tensor<2x6x24x64xf16>) -> tensor<2x6x24x128xf16>
    %1599 = stablehlo.broadcast_in_dim %1592, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1600 = stablehlo.multiply %1598, %1599 : tensor<2x6x24x128xf16>
    %1601 = stablehlo.add %1594, %1600 : tensor<2x6x24x128xf16>
    %1602 = stablehlo.broadcast_in_dim %1590, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %1603 = stablehlo.multiply %1575, %1602 : tensor<2x3x24x128xf16>
    %1604 = stablehlo.slice %1575 [0:2, 0:3, 0:24, 0:64] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %1605 = stablehlo.slice %1575 [0:2, 0:3, 0:24, 64:128] : (tensor<2x3x24x128xf16>) -> tensor<2x3x24x64xf16>
    %1606 = stablehlo.negate %1605 : tensor<2x3x24x64xf16>
    %1607 = stablehlo.concatenate %1606, %1604, dim = 3 : (tensor<2x3x24x64xf16>, tensor<2x3x24x64xf16>) -> tensor<2x3x24x128xf16>
    %1608 = stablehlo.broadcast_in_dim %1592, dims = [0, 1, 2, 3] : (tensor<2x1x24x128xf16>) -> tensor<2x3x24x128xf16>
    %1609 = stablehlo.multiply %1607, %1608 : tensor<2x3x24x128xf16>
    %1610 = stablehlo.add %1603, %1609 : tensor<2x3x24x128xf16>
    %1611 = stablehlo.slice %1610 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1612 = stablehlo.slice %1610 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1613 = stablehlo.slice %1610 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1614 = stablehlo.slice %1610 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1615 = stablehlo.slice %1610 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1616 = stablehlo.slice %1610 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1617 = stablehlo.concatenate %1611, %1612, %1613, %1614, %1615, %1616, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1618 = stablehlo.slice %1545 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1619 = stablehlo.slice %1545 [0:2, 0:1, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1620 = stablehlo.slice %1545 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1621 = stablehlo.slice %1545 [0:2, 1:2, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1622 = stablehlo.slice %1545 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1623 = stablehlo.slice %1545 [0:2, 2:3, 0:24, 0:128] : (tensor<2x3x24x128xf16>) -> tensor<2x1x24x128xf16>
    %1624 = stablehlo.concatenate %1618, %1619, %1620, %1621, %1622, %1623, dim = 1 : (tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>, tensor<2x1x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1625 = stablehlo.dot_general %1601, %1617, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x128xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x24xf16>
    %cst_295 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %1626 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1627 = stablehlo.multiply %1625, %1626 : tensor<2x6x24x24xf16>
    %cst_296 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %1628 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1629 = stablehlo.divide %1627, %1628 : tensor<2x6x24x24xf16>
    %1630 = stablehlo.tanh %1629 : tensor<2x6x24x24xf16>
    %cst_297 = stablehlo.constant dense<5.000000e+01> : tensor<f16>
    %1631 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1632 = stablehlo.multiply %1630, %1631 : tensor<2x6x24x24xf16>
    %c_298 = stablehlo.constant dense<0> : tensor<i32>
    %1633 = stablehlo.broadcast_in_dim %c_298, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1634 = stablehlo.compare NE, %arg0, %1633, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %c_299 = stablehlo.constant dense<2> : tensor<i32>
    %1635 = stablehlo.broadcast_in_dim %c_299, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1636 = stablehlo.compare EQ, %arg0, %1635, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %1637 = stablehlo.not %1636 : tensor<2x24xi1>
    %1638 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x24xf16>) -> tensor<2x24x1xf16>
    %1639 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<2x24xf16>) -> tensor<2x1x24xf16>
    %1640 = stablehlo.broadcast_in_dim %1638, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1641 = stablehlo.broadcast_in_dim %1639, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1642 = stablehlo.compare GE, %1640, %1641, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1643 = stablehlo.broadcast_in_dim %1638, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1644 = stablehlo.broadcast_in_dim %1639, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1645 = stablehlo.compare GT, %1643, %1644, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_300 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1646 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f16>) -> tensor<2x24x1xf16>
    %1647 = stablehlo.add %1638, %1646 : tensor<2x24x1xf16>
    %1648 = stablehlo.broadcast_in_dim %1639, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1649 = stablehlo.broadcast_in_dim %1647, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1650 = stablehlo.compare LT, %1648, %1649, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1651 = stablehlo.broadcast_in_dim %1638, dims = [0, 1, 2] : (tensor<2x24x1xf16>) -> tensor<2x24x24xf16>
    %1652 = stablehlo.broadcast_in_dim %1639, dims = [0, 1, 2] : (tensor<2x1x24xf16>) -> tensor<2x24x24xf16>
    %1653 = stablehlo.subtract %1651, %1652 : tensor<2x24x24xf16>
    %cst_301 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1654 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %1655 = stablehlo.compare LE, %1653, %1654, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1656 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f16>) -> tensor<2x24x24xf16>
    %1657 = stablehlo.compare GT, %arg2, %1656, FLOAT : (tensor<2x24x24xf16>, tensor<2x24x24xf16>) -> tensor<2x24x24xi1>
    %1658 = stablehlo.broadcast_in_dim %1634, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1659 = stablehlo.broadcast_in_dim %1634, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %1660 = stablehlo.and %1657, %1642 : tensor<2x24x24xi1>
    %1661 = stablehlo.and %1660, %1650 : tensor<2x24x24xi1>
    %1662 = stablehlo.or %1645, %1642 : tensor<2x24x24xi1>
    %1663 = stablehlo.and %1661, %1662 : tensor<2x24x24xi1>
    %1664 = stablehlo.and %1663, %1655 : tensor<2x24x24xi1>
    %1665 = stablehlo.broadcast_in_dim %1658, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %1666 = stablehlo.and %1664, %1665 : tensor<2x24x24xi1>
    %1667 = stablehlo.broadcast_in_dim %1659, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %1668 = stablehlo.and %1666, %1667 : tensor<2x24x24xi1>
    %1669 = stablehlo.broadcast_in_dim %1636, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1670 = stablehlo.broadcast_in_dim %1636, dims = [0, 2] : (tensor<2x24xi1>) -> tensor<2x1x24xi1>
    %1671 = stablehlo.broadcast_in_dim %1669, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %1672 = stablehlo.broadcast_in_dim %1670, dims = [0, 1, 2] : (tensor<2x1x24xi1>) -> tensor<2x24x24xi1>
    %1673 = stablehlo.and %1671, %1672 : tensor<2x24x24xi1>
    %1674 = stablehlo.or %1668, %1673 : tensor<2x24x24xi1>
    %1675 = stablehlo.broadcast_in_dim %1637, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %1676 = call @_where_77(%1675, %1674, %1673) : (tensor<2x24x1xi1>, tensor<2x24x24xi1>, tensor<2x24x24xi1>) -> tensor<2x24x24xi1>
    %1677 = stablehlo.broadcast_in_dim %1676, dims = [0, 2, 3] : (tensor<2x24x24xi1>) -> tensor<2x1x24x24xi1>
    %cst_303 = stablehlo.constant dense<-1.000000e+04> : tensor<f16>
    %1678 = call @_where_81(%1677, %1632, %cst_303) : (tensor<2x1x24x24xi1>, tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24x24xf16>
    %cst_304 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1679 = stablehlo.reduce(%1678 init: %cst_304) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %cst_305 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1680 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f16>) -> tensor<2x6x24xf16>
    %1681 = stablehlo.maximum %1680, %1679 : tensor<2x6x24xf16>
    %1682 = stablehlo.broadcast_in_dim %1681, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %1683 = stablehlo.broadcast_in_dim %1682, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1684 = stablehlo.subtract %1678, %1683 : tensor<2x6x24x24xf16>
    %1685 = stablehlo.exponential %1684 : tensor<2x6x24x24xf16>
    %1686 = stablehlo.convert %1685 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_306 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1687 = stablehlo.reduce(%1686 init: %cst_306) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1689 = stablehlo.convert %1688 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1690 = stablehlo.broadcast_in_dim %1689, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1691 = stablehlo.divide %1685, %1690 : tensor<2x6x24x24xf16>
    %1692 = stablehlo.convert %1691 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_307 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1693 = stablehlo.reduce(%1692 init: %cst_307) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1694 = stablehlo.broadcast_in_dim %1693, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1695 = stablehlo.convert %1694 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1697 = stablehlo.divide %1691, %1696 : tensor<2x6x24x24xf16>
    %cst_308 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1698 = stablehlo.reduce(%1678 init: %cst_308) applies stablehlo.maximum across dimensions = [3] : (tensor<2x6x24x24xf16>, tensor<f16>) -> tensor<2x6x24xf16>
    %1699 = stablehlo.broadcast_in_dim %1698, dims = [0, 1, 2] : (tensor<2x6x24xf16>) -> tensor<2x6x24x1xf16>
    %1700 = stablehlo.broadcast_in_dim %1699, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1701 = stablehlo.subtract %1678, %1700 : tensor<2x6x24x24xf16>
    %1702 = stablehlo.exponential %1701 : tensor<2x6x24x24xf16>
    %1703 = stablehlo.convert %1702 : (tensor<2x6x24x24xf16>) -> tensor<2x6x24x24xf32>
    %cst_309 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1704 = stablehlo.reduce(%1703 init: %cst_309) applies stablehlo.add across dimensions = [3] : (tensor<2x6x24x24xf32>, tensor<f32>) -> tensor<2x6x24xf32>
    %1705 = stablehlo.broadcast_in_dim %1704, dims = [0, 1, 2] : (tensor<2x6x24xf32>) -> tensor<2x6x24x1xf32>
    %1706 = stablehlo.convert %1705 : (tensor<2x6x24x1xf32>) -> tensor<2x6x24x1xf16>
    %1707 = stablehlo.broadcast_in_dim %1706, dims = [0, 1, 2, 3] : (tensor<2x6x24x1xf16>) -> tensor<2x6x24x24xf16>
    %1708 = stablehlo.divide %1702, %1707 : tensor<2x6x24x24xf16>
    %cst_310 = stablehlo.constant dense<7.500000e-01> : tensor<f16>
    %1709 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1710 = stablehlo.multiply %1697, %1709 : tensor<2x6x24x24xf16>
    %cst_311 = stablehlo.constant dense<2.500000e-01> : tensor<f16>
    %1711 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %1712 = stablehlo.multiply %1708, %1711 : tensor<2x6x24x24xf16>
    %1713 = stablehlo.add %1710, %1712 : tensor<2x6x24x24xf16>
    %1714 = stablehlo.dot_general %1713, %1624, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x6x24x24xf16>, tensor<2x6x24x128xf16>) -> tensor<2x6x24x128xf16>
    %1715 = stablehlo.transpose %1714, dims = [0, 2, 1, 3] : (tensor<2x6x24x128xf16>) -> tensor<2x24x6x128xf16>
    %1716 = stablehlo.dot_general %1715, %arg76, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x6x128xf16>, tensor<6x128x768xf16>) -> tensor<2x24x768xf16>
    %cst_312 = stablehlo.constant dense<1.250000e-01> : tensor<f16>
    %1717 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f16>) -> tensor<768xf16>
    %1718 = stablehlo.maximum %arg87, %1717 : tensor<768xf16>
    %1719 = stablehlo.add %1511, %1716 : tensor<2x24x768xf16>
    %1720 = stablehlo.convert %1719 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1721 = chlo.square %1720 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_313 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1722 = stablehlo.reduce(%1721 init: %cst_313) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1723 = stablehlo.broadcast_in_dim %1722, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_314 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1724 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1725 = stablehlo.divide %1723, %1724 : tensor<2x24x1xf32>
    %cst_315 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1726 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1727 = stablehlo.add %1725, %1726 : tensor<2x24x1xf32>
    %1728 = stablehlo.rsqrt %1727 : tensor<2x24x1xf32>
    %1729 = stablehlo.broadcast_in_dim %1728, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1730 = stablehlo.multiply %1720, %1729 : tensor<2x24x768xf32>
    %1731 = stablehlo.convert %1730 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1732 = stablehlo.broadcast_in_dim %arg82, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1733 = stablehlo.broadcast_in_dim %1732, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1734 = stablehlo.multiply %1731, %1733 : tensor<2x24x768xf16>
    %cst_316 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %1735 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1736 = stablehlo.multiply %1735, %1734 : tensor<2x24x768xf16>
    %1737 = stablehlo.add %1719, %1736 : tensor<2x24x768xf16>
    %1738 = stablehlo.broadcast_in_dim %1718, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %cst_317 = stablehlo.constant dense<1.000400e-03> : tensor<f16>
    %1739 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f16>) -> tensor<1x1x768xf16>
    %1740 = stablehlo.multiply %1738, %1739 : tensor<1x1x768xf16>
    %1741 = stablehlo.broadcast_in_dim %1740, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1742 = stablehlo.add %1737, %1741 : tensor<2x24x768xf16>
    %1743 = stablehlo.convert %1742 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1744 = chlo.square %1743 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1745 = stablehlo.reduce(%1744 init: %cst_318) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1746 = stablehlo.broadcast_in_dim %1745, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_319 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1747 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1748 = stablehlo.divide %1746, %1747 : tensor<2x24x1xf32>
    %cst_320 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1749 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1750 = stablehlo.add %1748, %1749 : tensor<2x24x1xf32>
    %1751 = stablehlo.rsqrt %1750 : tensor<2x24x1xf32>
    %1752 = stablehlo.broadcast_in_dim %1751, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1753 = stablehlo.multiply %1743, %1752 : tensor<2x24x768xf32>
    %1754 = stablehlo.convert %1753 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1755 = stablehlo.broadcast_in_dim %arg83, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1756 = stablehlo.broadcast_in_dim %1755, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1757 = stablehlo.multiply %1754, %1756 : tensor<2x24x768xf16>
    %1758 = stablehlo.dot_general %arg79, %1757, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x768x1536xf16>, tensor<2x24x768xf16>) -> tensor<2x1536x2x24xf16>
    %1759 = stablehlo.transpose %1758, dims = [2, 0, 3, 1] : (tensor<2x1536x2x24xf16>) -> tensor<2x2x24x1536xf16>
    %1760 = stablehlo.slice %1759 [0:2, 0:1, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %1761 = stablehlo.reshape %1760 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %1762 = stablehlo.slice %1759 [0:2, 1:2, 0:24, 0:1536] : (tensor<2x2x24x1536xf16>) -> tensor<2x1x24x1536xf16>
    %1763 = stablehlo.reshape %1762 : (tensor<2x1x24x1536xf16>) -> tensor<2x24x1536xf16>
    %1764 = stablehlo.negate %1761 : tensor<2x24x1536xf16>
    %1765 = stablehlo.exponential %1764 : tensor<2x24x1536xf16>
    %cst_321 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1766 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1767 = stablehlo.add %1766, %1765 : tensor<2x24x1536xf16>
    %cst_322 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1768 = stablehlo.broadcast_in_dim %cst_322, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1769 = stablehlo.divide %1768, %1767 : tensor<2x24x1536xf16>
    %1770 = stablehlo.multiply %1761, %1769 : tensor<2x24x1536xf16>
    %1771 = stablehlo.multiply %1770, %1763 : tensor<2x24x1536xf16>
    %cst_323 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %1772 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1773 = stablehlo.divide %1771, %1772 : tensor<2x24x1536xf16>
    %1774 = stablehlo.tanh %1773 : tensor<2x24x1536xf16>
    %cst_324 = stablehlo.constant dense<1.200000e+01> : tensor<f16>
    %1775 = stablehlo.broadcast_in_dim %cst_324, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1776 = stablehlo.multiply %1774, %1775 : tensor<2x24x1536xf16>
    %1777 = stablehlo.dot_general %1776, %arg80, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x24x1536xf16>, tensor<1536x768xf16>) -> tensor<2x24x768xf16>
    %1778 = stablehlo.broadcast_in_dim %arg89, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1779 = stablehlo.broadcast_in_dim %1778, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1780 = stablehlo.multiply %1777, %1779 : tensor<2x24x768xf16>
    %1781 = stablehlo.add %1742, %1780 : tensor<2x24x768xf16>
    %1782 = stablehlo.convert %1781 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1783 = chlo.square %1782 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1784 = stablehlo.reduce(%1783 init: %cst_325) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1785 = stablehlo.broadcast_in_dim %1784, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_326 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1786 = stablehlo.broadcast_in_dim %cst_326, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1787 = stablehlo.divide %1785, %1786 : tensor<2x24x1xf32>
    %cst_327 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1788 = stablehlo.broadcast_in_dim %cst_327, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1789 = stablehlo.add %1787, %1788 : tensor<2x24x1xf32>
    %1790 = stablehlo.rsqrt %1789 : tensor<2x24x1xf32>
    %1791 = stablehlo.broadcast_in_dim %1790, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1792 = stablehlo.multiply %1782, %1791 : tensor<2x24x768xf32>
    %1793 = stablehlo.convert %1792 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1794 = stablehlo.broadcast_in_dim %arg84, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1795 = stablehlo.broadcast_in_dim %1794, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1796 = stablehlo.multiply %1793, %1795 : tensor<2x24x768xf16>
    %cst_328 = stablehlo.constant dense<3.125000e-02> : tensor<f16>
    %1797 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1798 = stablehlo.multiply %1797, %1796 : tensor<2x24x768xf16>
    %1799 = stablehlo.add %1781, %1798 : tensor<2x24x768xf16>
    %c_329 = stablehlo.constant dense<0> : tensor<i32>
    %1800 = stablehlo.broadcast_in_dim %c_329, dims = [] : (tensor<i32>) -> tensor<2x24xi32>
    %1801 = stablehlo.compare NE, %arg0, %1800, SIGNED : (tensor<2x24xi32>, tensor<2x24xi32>) -> tensor<2x24xi1>
    %1802 = stablehlo.broadcast_in_dim %1801, dims = [0, 1] : (tensor<2x24xi1>) -> tensor<2x24x1xi1>
    %cst_330 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1803 = call @_where(%1802, %1799, %cst_330) : (tensor<2x24x1xi1>, tensor<2x24x768xf16>, tensor<f16>) -> tensor<2x24x768xf16>
    %cst_331 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1804 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1805 = stablehlo.compare GT, %1803, %1804, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_332 = stablehlo.constant dense<8.000000e+00> : tensor<f16>
    %1806 = call @_where_114(%1805, %cst_332, %1803) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %cst_333 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %1807 = stablehlo.broadcast_in_dim %cst_333, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1808 = stablehlo.compare LT, %1806, %1807, FLOAT : (tensor<2x24x768xf16>, tensor<2x24x768xf16>) -> tensor<2x24x768xi1>
    %cst_334 = stablehlo.constant dense<-8.000000e+00> : tensor<f16>
    %1809 = call @_where_114(%1808, %cst_334, %1806) : (tensor<2x24x768xi1>, tensor<f16>, tensor<2x24x768xf16>) -> tensor<2x24x768xf16>
    %1810 = stablehlo.convert %1809 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1811 = chlo.square %1810 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_335 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1812 = stablehlo.reduce(%1811 init: %cst_335) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1813 = stablehlo.broadcast_in_dim %1812, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_336 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1814 = stablehlo.broadcast_in_dim %cst_336, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1815 = stablehlo.divide %1813, %1814 : tensor<2x24x1xf32>
    %cst_337 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1816 = stablehlo.broadcast_in_dim %cst_337, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1817 = stablehlo.add %1815, %1816 : tensor<2x24x1xf32>
    %1818 = stablehlo.rsqrt %1817 : tensor<2x24x1xf32>
    %1819 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1820 = stablehlo.multiply %1810, %1819 : tensor<2x24x768xf32>
    %1821 = stablehlo.convert %1820 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1822 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1823 = stablehlo.broadcast_in_dim %1822, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1824 = stablehlo.multiply %1821, %1823 : tensor<2x24x768xf16>
    %1825 = stablehlo.convert %1809 : (tensor<2x24x768xf16>) -> tensor<2x24x768xf32>
    %1826 = chlo.square %1825 : tensor<2x24x768xf32> -> tensor<2x24x768xf32>
    %cst_338 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1827 = stablehlo.reduce(%1826 init: %cst_338) applies stablehlo.add across dimensions = [2] : (tensor<2x24x768xf32>, tensor<f32>) -> tensor<2x24xf32>
    %1828 = stablehlo.broadcast_in_dim %1827, dims = [0, 1] : (tensor<2x24xf32>) -> tensor<2x24x1xf32>
    %cst_339 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1829 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1830 = stablehlo.divide %1828, %1829 : tensor<2x24x1xf32>
    %cst_340 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1831 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<f32>) -> tensor<2x24x1xf32>
    %1832 = stablehlo.maximum %1830, %1831 : tensor<2x24x1xf32>
    %1833 = stablehlo.sqrt %1832 : tensor<2x24x1xf32>
    %1834 = stablehlo.broadcast_in_dim %1833, dims = [0, 1, 2] : (tensor<2x24x1xf32>) -> tensor<2x24x768xf32>
    %1835 = stablehlo.divide %1825, %1834 : tensor<2x24x768xf32>
    %1836 = stablehlo.convert %1835 : (tensor<2x24x768xf32>) -> tensor<2x24x768xf16>
    %1837 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1838 = stablehlo.broadcast_in_dim %1837, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<2x24x768xf16>
    %1839 = stablehlo.multiply %1836, %1838 : tensor<2x24x768xf16>
    %cst_341 = stablehlo.constant dense<1.562500e-02> : tensor<f16>
    %1840 = stablehlo.broadcast_in_dim %cst_341, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1841 = stablehlo.multiply %1839, %1840 : tensor<2x24x768xf16>
    %1842 = stablehlo.add %1824, %1841 : tensor<2x24x768xf16>
    %1843 = stablehlo.dot_general %1842, %arg3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24x768xf16>, tensor<1536x768xf16>) -> tensor<2x24x1536xf16>
    %cst_342 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %1844 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1845 = stablehlo.divide %1843, %1844 : tensor<2x24x1536xf16>
    %1846 = stablehlo.tanh %1845 : tensor<2x24x1536xf16>
    %cst_343 = stablehlo.constant dense<3.000000e+01> : tensor<f16>
    %1847 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<f16>) -> tensor<2x24x1536xf16>
    %1848 = stablehlo.multiply %1846, %1847 : tensor<2x24x1536xf16>
    return %1848 : tensor<2x24x1536xf16>
  }
  func.func private @_where(%arg0: tensor<2x24x1xi1>, %arg1: tensor<2x24x768xf16>, %arg2: tensor<f16>) -> tensor<2x24x768xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x768xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x24x768xi1>, tensor<2x24x768xf16>
    return %2 : tensor<2x24x768xf16>
  }
  func.func private @_where_77(%arg0: tensor<2x24x1xi1>, %arg1: tensor<2x24x24xi1>, %arg2: tensor<2x24x24xi1>) -> tensor<2x24x24xi1> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x24x1xi1>) -> tensor<2x24x24xi1>
    %1 = stablehlo.select %0, %arg1, %arg2 : tensor<2x24x24xi1>, tensor<2x24x24xi1>
    return %1 : tensor<2x24x24xi1>
  }
  func.func private @_where_81(%arg0: tensor<2x1x24x24xi1>, %arg1: tensor<2x6x24x24xf16>, %arg2: tensor<f16>) -> tensor<2x6x24x24xf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<2x1x24x24xi1>) -> tensor<2x6x24x24xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<2x6x24x24xf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x6x24x24xi1>, tensor<2x6x24x24xf16>
    return %2 : tensor<2x6x24x24xf16>
  }
  func.func private @_where_114(%arg0: tensor<2x24x768xi1>, %arg1: tensor<f16>, %arg2: tensor<2x24x768xf16>) -> tensor<2x24x768xf16> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f16>) -> tensor<2x24x768xf16>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<2x24x768xi1>, tensor<2x24x768xf16>
    return %1 : tensor<2x24x768xf16>
  }
}
