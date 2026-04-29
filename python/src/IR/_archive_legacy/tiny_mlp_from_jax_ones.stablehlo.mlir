module @jit_tiny_mlp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x16xf16> {mhlo.layout_mode = "default"}, %arg1: tensor<16x32xf16> {mhlo.layout_mode = "default"}, %arg2: tensor<32xf16> {mhlo.layout_mode = "default"}, %arg3: tensor<32x16xf16> {mhlo.layout_mode = "default"}, %arg4: tensor<16xf16> {mhlo.layout_mode = "default"}) -> (tensor<1x16xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16xf16>, tensor<16x32xf16>) -> tensor<1x32xf16>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf16>) -> tensor<1x32xf16>
    %2 = stablehlo.add %0, %1 : tensor<1x32xf16>
    %3 = call @relu(%2) : (tensor<1x32xf16>) -> tensor<1x32xf16>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x32xf16>, tensor<32x16xf16>) -> tensor<1x16xf16>
    %5 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<16xf16>) -> tensor<1x16xf16>
    %6 = stablehlo.add %4, %5 : tensor<1x16xf16>
    %7 = call @relu_0(%6) : (tensor<1x16xf16>) -> tensor<1x16xf16>
    return %7 : tensor<1x16xf16>
  }
  func.func private @relu(%arg0: tensor<1x32xf16> {mhlo.layout_mode = "default"}) -> (tensor<1x32xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<1x32xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x32xf16>
    return %1 : tensor<1x32xf16>
  }
  func.func private @relu_0(%arg0: tensor<1x16xf16> {mhlo.layout_mode = "default"}) -> (tensor<1x16xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<1x16xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x16xf16>
    return %1 : tensor<1x16xf16>
  }
}
