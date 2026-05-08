module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256128x2304xbf16>, %arg1: tensor<2304xbf16>, %arg2: tensor<8x256x2304xbf16>, %arg3: tensor<2x4x2304x256xbf16>, %arg4: tensor<8x2304x256xbf16>, %arg5: tensor<2x2304x9216xbf16>, %arg6: tensor<9216x2304xbf16>, %arg7: tensor<2304xbf16>, %arg8: tensor<2304xbf16>, %arg9: tensor<2304xbf16>, %arg10: tensor<2304xbf16>, %arg11: tensor<8x256x2304xbf16>, %arg12: tensor<2x4x2304x256xbf16>, %arg13: tensor<8x2304x256xbf16>, %arg14: tensor<2x2304x9216xbf16>, %arg15: tensor<9216x2304xbf16>, %arg16: tensor<2304xbf16>, %arg17: tensor<2304xbf16>, %arg18: tensor<2304xbf16>, %arg19: tensor<2304xbf16>, %arg20: tensor<8x256x2304xbf16>, %arg21: tensor<2x4x2304x256xbf16>, %arg22: tensor<8x2304x256xbf16>, %arg23: tensor<2x2304x9216xbf16>, %arg24: tensor<9216x2304xbf16>, %arg25: tensor<2304xbf16>, %arg26: tensor<2304xbf16>, %arg27: tensor<2304xbf16>, %arg28: tensor<2304xbf16>, %arg29: tensor<8x256x2304xbf16>, %arg30: tensor<2x4x2304x256xbf16>, %arg31: tensor<8x2304x256xbf16>, %arg32: tensor<2x2304x9216xbf16>, %arg33: tensor<9216x2304xbf16>, %arg34: tensor<2304xbf16>, %arg35: tensor<2304xbf16>, %arg36: tensor<2304xbf16>, %arg37: tensor<2304xbf16>, %arg38: tensor<8x256x2304xbf16>, %arg39: tensor<2x4x2304x256xbf16>, %arg40: tensor<8x2304x256xbf16>, %arg41: tensor<2x2304x9216xbf16>, %arg42: tensor<9216x2304xbf16>, %arg43: tensor<2304xbf16>, %arg44: tensor<2304xbf16>, %arg45: tensor<2304xbf16>, %arg46: tensor<2304xbf16>, %arg47: tensor<8x256x2304xbf16>, %arg48: tensor<2x4x2304x256xbf16>, %arg49: tensor<8x2304x256xbf16>, %arg50: tensor<2x2304x9216xbf16>, %arg51: tensor<9216x2304xbf16>, %arg52: tensor<2304xbf16>, %arg53: tensor<2304xbf16>, %arg54: tensor<2304xbf16>, %arg55: tensor<2304xbf16>, %arg56: tensor<8x256x2304xbf16>, %arg57: tensor<2x4x2304x256xbf16>, %arg58: tensor<8x2304x256xbf16>, %arg59: tensor<2x2304x9216xbf16>, %arg60: tensor<9216x2304xbf16>, %arg61: tensor<2304xbf16>, %arg62: tensor<2304xbf16>, %arg63: tensor<2304xbf16>, %arg64: tensor<2304xbf16>, %arg65: tensor<8x256x2304xbf16>, %arg66: tensor<2x4x2304x256xbf16>, %arg67: tensor<8x2304x256xbf16>, %arg68: tensor<2x2304x9216xbf16>, %arg69: tensor<9216x2304xbf16>, %arg70: tensor<2304xbf16>, %arg71: tensor<2304xbf16>, %arg72: tensor<2304xbf16>, %arg73: tensor<2304xbf16>, %arg74: tensor<8x256x2304xbf16>, %arg75: tensor<2x4x2304x256xbf16>, %arg76: tensor<8x2304x256xbf16>, %arg77: tensor<2x2304x9216xbf16>, %arg78: tensor<9216x2304xbf16>, %arg79: tensor<2304xbf16>, %arg80: tensor<2304xbf16>, %arg81: tensor<2304xbf16>, %arg82: tensor<2304xbf16>, %arg83: tensor<8x256x2304xbf16>, %arg84: tensor<2x4x2304x256xbf16>, %arg85: tensor<8x2304x256xbf16>, %arg86: tensor<2x2304x9216xbf16>, %arg87: tensor<9216x2304xbf16>, %arg88: tensor<2304xbf16>, %arg89: tensor<2304xbf16>, %arg90: tensor<2304xbf16>, %arg91: tensor<2304xbf16>, %arg92: tensor<8x256x2304xbf16>, %arg93: tensor<2x4x2304x256xbf16>, %arg94: tensor<8x2304x256xbf16>, %arg95: tensor<2x2304x9216xbf16>, %arg96: tensor<9216x2304xbf16>, %arg97: tensor<2304xbf16>, %arg98: tensor<2304xbf16>, %arg99: tensor<2304xbf16>, %arg100: tensor<2304xbf16>, %arg101: tensor<8x256x2304xbf16>, %arg102: tensor<2x4x2304x256xbf16>, %arg103: tensor<8x2304x256xbf16>, %arg104: tensor<2x2304x9216xbf16>, %arg105: tensor<9216x2304xbf16>, %arg106: tensor<2304xbf16>, %arg107: tensor<2304xbf16>, %arg108: tensor<2304xbf16>, %arg109: tensor<2304xbf16>, %arg110: tensor<8x256x2304xbf16>, %arg111: tensor<2x4x2304x256xbf16>, %arg112: tensor<8x2304x256xbf16>, %arg113: tensor<2x2304x9216xbf16>, %arg114: tensor<9216x2304xbf16>, %arg115: tensor<2304xbf16>, %arg116: tensor<2304xbf16>, %arg117: tensor<2304xbf16>, %arg118: tensor<2304xbf16>, %arg119: tensor<8x256x2304xbf16>, %arg120: tensor<2x4x2304x256xbf16>, %arg121: tensor<8x2304x256xbf16>, %arg122: tensor<2x2304x9216xbf16>, %arg123: tensor<9216x2304xbf16>, %arg124: tensor<2304xbf16>, %arg125: tensor<2304xbf16>, %arg126: tensor<2304xbf16>, %arg127: tensor<2304xbf16>, %arg128: tensor<8x256x2304xbf16>, %arg129: tensor<2x4x2304x256xbf16>, %arg130: tensor<8x2304x256xbf16>, %arg131: tensor<2x2304x9216xbf16>, %arg132: tensor<9216x2304xbf16>, %arg133: tensor<2304xbf16>, %arg134: tensor<2304xbf16>, %arg135: tensor<2304xbf16>, %arg136: tensor<2304xbf16>, %arg137: tensor<8x256x2304xbf16>, %arg138: tensor<2x4x2304x256xbf16>, %arg139: tensor<8x2304x256xbf16>, %arg140: tensor<2x2304x9216xbf16>, %arg141: tensor<9216x2304xbf16>, %arg142: tensor<2304xbf16>, %arg143: tensor<2304xbf16>, %arg144: tensor<2304xbf16>, %arg145: tensor<2304xbf16>, %arg146: tensor<8x256x2304xbf16>, %arg147: tensor<2x4x2304x256xbf16>, %arg148: tensor<8x2304x256xbf16>, %arg149: tensor<2x2304x9216xbf16>, %arg150: tensor<9216x2304xbf16>, %arg151: tensor<2304xbf16>, %arg152: tensor<2304xbf16>, %arg153: tensor<2304xbf16>, %arg154: tensor<2304xbf16>, %arg155: tensor<8x256x2304xbf16>, %arg156: tensor<2x4x2304x256xbf16>, %arg157: tensor<8x2304x256xbf16>, %arg158: tensor<2x2304x9216xbf16>, %arg159: tensor<9216x2304xbf16>, %arg160: tensor<2304xbf16>, %arg161: tensor<2304xbf16>, %arg162: tensor<2304xbf16>, %arg163: tensor<2304xbf16>, %arg164: tensor<8x256x2304xbf16>, %arg165: tensor<2x4x2304x256xbf16>, %arg166: tensor<8x2304x256xbf16>, %arg167: tensor<2x2304x9216xbf16>, %arg168: tensor<9216x2304xbf16>, %arg169: tensor<2304xbf16>, %arg170: tensor<2304xbf16>, %arg171: tensor<2304xbf16>, %arg172: tensor<2304xbf16>, %arg173: tensor<8x256x2304xbf16>, %arg174: tensor<2x4x2304x256xbf16>, %arg175: tensor<8x2304x256xbf16>, %arg176: tensor<2x2304x9216xbf16>, %arg177: tensor<9216x2304xbf16>, %arg178: tensor<2304xbf16>, %arg179: tensor<2304xbf16>, %arg180: tensor<2304xbf16>, %arg181: tensor<2304xbf16>, %arg182: tensor<8x256x2304xbf16>, %arg183: tensor<2x4x2304x256xbf16>, %arg184: tensor<8x2304x256xbf16>, %arg185: tensor<2x2304x9216xbf16>, %arg186: tensor<9216x2304xbf16>, %arg187: tensor<2304xbf16>, %arg188: tensor<2304xbf16>, %arg189: tensor<2304xbf16>, %arg190: tensor<2304xbf16>, %arg191: tensor<8x256x2304xbf16>, %arg192: tensor<2x4x2304x256xbf16>, %arg193: tensor<8x2304x256xbf16>, %arg194: tensor<2x2304x9216xbf16>, %arg195: tensor<9216x2304xbf16>, %arg196: tensor<2304xbf16>, %arg197: tensor<2304xbf16>, %arg198: tensor<2304xbf16>, %arg199: tensor<2304xbf16>, %arg200: tensor<8x256x2304xbf16>, %arg201: tensor<2x4x2304x256xbf16>, %arg202: tensor<8x2304x256xbf16>, %arg203: tensor<2x2304x9216xbf16>, %arg204: tensor<9216x2304xbf16>, %arg205: tensor<2304xbf16>, %arg206: tensor<2304xbf16>, %arg207: tensor<2304xbf16>, %arg208: tensor<2304xbf16>, %arg209: tensor<8x256x2304xbf16>, %arg210: tensor<2x4x2304x256xbf16>, %arg211: tensor<8x2304x256xbf16>, %arg212: tensor<2x2304x9216xbf16>, %arg213: tensor<9216x2304xbf16>, %arg214: tensor<2304xbf16>, %arg215: tensor<2304xbf16>, %arg216: tensor<2304xbf16>, %arg217: tensor<2304xbf16>, %arg218: tensor<8x256x2304xbf16>, %arg219: tensor<2x4x2304x256xbf16>, %arg220: tensor<8x2304x256xbf16>, %arg221: tensor<2x2304x9216xbf16>, %arg222: tensor<9216x2304xbf16>, %arg223: tensor<2304xbf16>, %arg224: tensor<2304xbf16>, %arg225: tensor<2304xbf16>, %arg226: tensor<2304xbf16>, %arg227: tensor<8x256x2304xbf16>, %arg228: tensor<2x4x2304x256xbf16>, %arg229: tensor<8x2304x256xbf16>, %arg230: tensor<2x2304x9216xbf16>, %arg231: tensor<9216x2304xbf16>, %arg232: tensor<2304xbf16>, %arg233: tensor<2304xbf16>, %arg234: tensor<2304xbf16>, %arg235: tensor<2304xbf16>, %arg236: tensor<1x8xi32>, %arg237: tensor<1x8xi32>, %arg238: tensor<1x8x8xi1>) -> (tensor<1x8x256128xbf16> {jax.result_info = "result"}) {
    %0 = call @__call__(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg204, %arg205, %arg206, %arg207, %arg208, %arg209, %arg210, %arg211, %arg212, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %arg231, %arg232, %arg233, %arg234, %arg235, %arg236, %arg238, %arg237) : (tensor<256128x2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<8x256x2304xbf16>, tensor<2x4x2304x256xbf16>, tensor<8x2304x256xbf16>, tensor<2x2304x9216xbf16>, tensor<9216x2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<2304xbf16>, tensor<1x8xi32>, tensor<1x8x8xi1>, tensor<1x8xi32>) -> tensor<1x8x256128xbf16>
    return %0 : tensor<1x8x256128xbf16>
  }
  func.func private @__call__(%arg0: tensor<256128x2304xbf16>, %arg1: tensor<2304xbf16>, %arg2: tensor<8x256x2304xbf16>, %arg3: tensor<2x4x2304x256xbf16>, %arg4: tensor<8x2304x256xbf16>, %arg5: tensor<2x2304x9216xbf16>, %arg6: tensor<9216x2304xbf16>, %arg7: tensor<2304xbf16>, %arg8: tensor<2304xbf16>, %arg9: tensor<2304xbf16>, %arg10: tensor<2304xbf16>, %arg11: tensor<8x256x2304xbf16>, %arg12: tensor<2x4x2304x256xbf16>, %arg13: tensor<8x2304x256xbf16>, %arg14: tensor<2x2304x9216xbf16>, %arg15: tensor<9216x2304xbf16>, %arg16: tensor<2304xbf16>, %arg17: tensor<2304xbf16>, %arg18: tensor<2304xbf16>, %arg19: tensor<2304xbf16>, %arg20: tensor<8x256x2304xbf16>, %arg21: tensor<2x4x2304x256xbf16>, %arg22: tensor<8x2304x256xbf16>, %arg23: tensor<2x2304x9216xbf16>, %arg24: tensor<9216x2304xbf16>, %arg25: tensor<2304xbf16>, %arg26: tensor<2304xbf16>, %arg27: tensor<2304xbf16>, %arg28: tensor<2304xbf16>, %arg29: tensor<8x256x2304xbf16>, %arg30: tensor<2x4x2304x256xbf16>, %arg31: tensor<8x2304x256xbf16>, %arg32: tensor<2x2304x9216xbf16>, %arg33: tensor<9216x2304xbf16>, %arg34: tensor<2304xbf16>, %arg35: tensor<2304xbf16>, %arg36: tensor<2304xbf16>, %arg37: tensor<2304xbf16>, %arg38: tensor<8x256x2304xbf16>, %arg39: tensor<2x4x2304x256xbf16>, %arg40: tensor<8x2304x256xbf16>, %arg41: tensor<2x2304x9216xbf16>, %arg42: tensor<9216x2304xbf16>, %arg43: tensor<2304xbf16>, %arg44: tensor<2304xbf16>, %arg45: tensor<2304xbf16>, %arg46: tensor<2304xbf16>, %arg47: tensor<8x256x2304xbf16>, %arg48: tensor<2x4x2304x256xbf16>, %arg49: tensor<8x2304x256xbf16>, %arg50: tensor<2x2304x9216xbf16>, %arg51: tensor<9216x2304xbf16>, %arg52: tensor<2304xbf16>, %arg53: tensor<2304xbf16>, %arg54: tensor<2304xbf16>, %arg55: tensor<2304xbf16>, %arg56: tensor<8x256x2304xbf16>, %arg57: tensor<2x4x2304x256xbf16>, %arg58: tensor<8x2304x256xbf16>, %arg59: tensor<2x2304x9216xbf16>, %arg60: tensor<9216x2304xbf16>, %arg61: tensor<2304xbf16>, %arg62: tensor<2304xbf16>, %arg63: tensor<2304xbf16>, %arg64: tensor<2304xbf16>, %arg65: tensor<8x256x2304xbf16>, %arg66: tensor<2x4x2304x256xbf16>, %arg67: tensor<8x2304x256xbf16>, %arg68: tensor<2x2304x9216xbf16>, %arg69: tensor<9216x2304xbf16>, %arg70: tensor<2304xbf16>, %arg71: tensor<2304xbf16>, %arg72: tensor<2304xbf16>, %arg73: tensor<2304xbf16>, %arg74: tensor<8x256x2304xbf16>, %arg75: tensor<2x4x2304x256xbf16>, %arg76: tensor<8x2304x256xbf16>, %arg77: tensor<2x2304x9216xbf16>, %arg78: tensor<9216x2304xbf16>, %arg79: tensor<2304xbf16>, %arg80: tensor<2304xbf16>, %arg81: tensor<2304xbf16>, %arg82: tensor<2304xbf16>, %arg83: tensor<8x256x2304xbf16>, %arg84: tensor<2x4x2304x256xbf16>, %arg85: tensor<8x2304x256xbf16>, %arg86: tensor<2x2304x9216xbf16>, %arg87: tensor<9216x2304xbf16>, %arg88: tensor<2304xbf16>, %arg89: tensor<2304xbf16>, %arg90: tensor<2304xbf16>, %arg91: tensor<2304xbf16>, %arg92: tensor<8x256x2304xbf16>, %arg93: tensor<2x4x2304x256xbf16>, %arg94: tensor<8x2304x256xbf16>, %arg95: tensor<2x2304x9216xbf16>, %arg96: tensor<9216x2304xbf16>, %arg97: tensor<2304xbf16>, %arg98: tensor<2304xbf16>, %arg99: tensor<2304xbf16>, %arg100: tensor<2304xbf16>, %arg101: tensor<8x256x2304xbf16>, %arg102: tensor<2x4x2304x256xbf16>, %arg103: tensor<8x2304x256xbf16>, %arg104: tensor<2x2304x9216xbf16>, %arg105: tensor<9216x2304xbf16>, %arg106: tensor<2304xbf16>, %arg107: tensor<2304xbf16>, %arg108: tensor<2304xbf16>, %arg109: tensor<2304xbf16>, %arg110: tensor<8x256x2304xbf16>, %arg111: tensor<2x4x2304x256xbf16>, %arg112: tensor<8x2304x256xbf16>, %arg113: tensor<2x2304x9216xbf16>, %arg114: tensor<9216x2304xbf16>, %arg115: tensor<2304xbf16>, %arg116: tensor<2304xbf16>, %arg117: tensor<2304xbf16>, %arg118: tensor<2304xbf16>, %arg119: tensor<8x256x2304xbf16>, %arg120: tensor<2x4x2304x256xbf16>, %arg121: tensor<8x2304x256xbf16>, %arg122: tensor<2x2304x9216xbf16>, %arg123: tensor<9216x2304xbf16>, %arg124: tensor<2304xbf16>, %arg125: tensor<2304xbf16>, %arg126: tensor<2304xbf16>, %arg127: tensor<2304xbf16>, %arg128: tensor<8x256x2304xbf16>, %arg129: tensor<2x4x2304x256xbf16>, %arg130: tensor<8x2304x256xbf16>, %arg131: tensor<2x2304x9216xbf16>, %arg132: tensor<9216x2304xbf16>, %arg133: tensor<2304xbf16>, %arg134: tensor<2304xbf16>, %arg135: tensor<2304xbf16>, %arg136: tensor<2304xbf16>, %arg137: tensor<8x256x2304xbf16>, %arg138: tensor<2x4x2304x256xbf16>, %arg139: tensor<8x2304x256xbf16>, %arg140: tensor<2x2304x9216xbf16>, %arg141: tensor<9216x2304xbf16>, %arg142: tensor<2304xbf16>, %arg143: tensor<2304xbf16>, %arg144: tensor<2304xbf16>, %arg145: tensor<2304xbf16>, %arg146: tensor<8x256x2304xbf16>, %arg147: tensor<2x4x2304x256xbf16>, %arg148: tensor<8x2304x256xbf16>, %arg149: tensor<2x2304x9216xbf16>, %arg150: tensor<9216x2304xbf16>, %arg151: tensor<2304xbf16>, %arg152: tensor<2304xbf16>, %arg153: tensor<2304xbf16>, %arg154: tensor<2304xbf16>, %arg155: tensor<8x256x2304xbf16>, %arg156: tensor<2x4x2304x256xbf16>, %arg157: tensor<8x2304x256xbf16>, %arg158: tensor<2x2304x9216xbf16>, %arg159: tensor<9216x2304xbf16>, %arg160: tensor<2304xbf16>, %arg161: tensor<2304xbf16>, %arg162: tensor<2304xbf16>, %arg163: tensor<2304xbf16>, %arg164: tensor<8x256x2304xbf16>, %arg165: tensor<2x4x2304x256xbf16>, %arg166: tensor<8x2304x256xbf16>, %arg167: tensor<2x2304x9216xbf16>, %arg168: tensor<9216x2304xbf16>, %arg169: tensor<2304xbf16>, %arg170: tensor<2304xbf16>, %arg171: tensor<2304xbf16>, %arg172: tensor<2304xbf16>, %arg173: tensor<8x256x2304xbf16>, %arg174: tensor<2x4x2304x256xbf16>, %arg175: tensor<8x2304x256xbf16>, %arg176: tensor<2x2304x9216xbf16>, %arg177: tensor<9216x2304xbf16>, %arg178: tensor<2304xbf16>, %arg179: tensor<2304xbf16>, %arg180: tensor<2304xbf16>, %arg181: tensor<2304xbf16>, %arg182: tensor<8x256x2304xbf16>, %arg183: tensor<2x4x2304x256xbf16>, %arg184: tensor<8x2304x256xbf16>, %arg185: tensor<2x2304x9216xbf16>, %arg186: tensor<9216x2304xbf16>, %arg187: tensor<2304xbf16>, %arg188: tensor<2304xbf16>, %arg189: tensor<2304xbf16>, %arg190: tensor<2304xbf16>, %arg191: tensor<8x256x2304xbf16>, %arg192: tensor<2x4x2304x256xbf16>, %arg193: tensor<8x2304x256xbf16>, %arg194: tensor<2x2304x9216xbf16>, %arg195: tensor<9216x2304xbf16>, %arg196: tensor<2304xbf16>, %arg197: tensor<2304xbf16>, %arg198: tensor<2304xbf16>, %arg199: tensor<2304xbf16>, %arg200: tensor<8x256x2304xbf16>, %arg201: tensor<2x4x2304x256xbf16>, %arg202: tensor<8x2304x256xbf16>, %arg203: tensor<2x2304x9216xbf16>, %arg204: tensor<9216x2304xbf16>, %arg205: tensor<2304xbf16>, %arg206: tensor<2304xbf16>, %arg207: tensor<2304xbf16>, %arg208: tensor<2304xbf16>, %arg209: tensor<8x256x2304xbf16>, %arg210: tensor<2x4x2304x256xbf16>, %arg211: tensor<8x2304x256xbf16>, %arg212: tensor<2x2304x9216xbf16>, %arg213: tensor<9216x2304xbf16>, %arg214: tensor<2304xbf16>, %arg215: tensor<2304xbf16>, %arg216: tensor<2304xbf16>, %arg217: tensor<2304xbf16>, %arg218: tensor<8x256x2304xbf16>, %arg219: tensor<2x4x2304x256xbf16>, %arg220: tensor<8x2304x256xbf16>, %arg221: tensor<2x2304x9216xbf16>, %arg222: tensor<9216x2304xbf16>, %arg223: tensor<2304xbf16>, %arg224: tensor<2304xbf16>, %arg225: tensor<2304xbf16>, %arg226: tensor<2304xbf16>, %arg227: tensor<8x256x2304xbf16>, %arg228: tensor<2x4x2304x256xbf16>, %arg229: tensor<8x2304x256xbf16>, %arg230: tensor<2x2304x9216xbf16>, %arg231: tensor<9216x2304xbf16>, %arg232: tensor<2304xbf16>, %arg233: tensor<2304xbf16>, %arg234: tensor<2304xbf16>, %arg235: tensor<2304xbf16>, %arg236: tensor<1x8xi32>, %arg237: tensor<1x8x8xi1>, %arg238: tensor<1x8xi32>) -> tensor<1x8x256128xbf16> {
    %0 = call @tokens_with_mm(%arg236) : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2 = stablehlo.compare LT, %0, %1, SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %c_0 = stablehlo.constant dense<256128> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4 = stablehlo.add %0, %3 : tensor<1x8xi32>
    %5 = stablehlo.select %2, %4, %0 : tensor<1x8xi1>, tensor<1x8xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2304>}> : (tensor<256128x2304xbf16>, tensor<1x8x1xi32>) -> tensor<1x8x2304xbf16>
    %cst = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %8 = stablehlo.sqrt %cst : tensor<f32>
    %9 = stablehlo.convert %8 : (tensor<f32>) -> tensor<bf16>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<bf16>) -> tensor<1x8x2304xbf16>
    %11 = stablehlo.multiply %7, %10 : tensor<1x8x2304xbf16>
    %12 = chlo.square %11 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %13 = stablehlo.convert %12 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_2 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %17 = stablehlo.divide %15, %16 : tensor<1x8x1xf32>
    %18 = stablehlo.convert %17 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_3 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %20 = stablehlo.add %18, %19 : tensor<1x8x1xbf16>
    %21 = stablehlo.rsqrt %20 : tensor<1x8x1xbf16>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %23 = stablehlo.multiply %11, %22 : tensor<1x8x2304xbf16>
    %24 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %25 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %26 = stablehlo.add %25, %24 : tensor<1x1x2304xbf16>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %28 = stablehlo.multiply %23, %27 : tensor<1x8x2304xbf16>
    %29 = stablehlo.dot_general %28, %arg4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %30 = stablehlo.dot_general %arg3, %28, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %31 = stablehlo.transpose %30, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %32 = stablehlo.slice %31 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %33 = stablehlo.reshape %32 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %34 = stablehlo.slice %31 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %35 = stablehlo.reshape %34 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %36 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_5 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %38 = stablehlo.multiply %37, %36 : tensor<128xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %40 = stablehlo.power %39, %38 : tensor<128xf32>
    %41 = call @_pad(%40) : (tensor<128xf32>) -> tensor<128xf32>
    %42 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %43 = stablehlo.broadcast_in_dim %41, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %44 = stablehlo.convert %42 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %46 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %47 = stablehlo.divide %45, %46 : tensor<1x8x128xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %50 = stablehlo.divide %48, %49 : tensor<1x8x1x128xf32>
    %51 = stablehlo.sine %50 : tensor<1x8x1x128xf32>
    %52 = stablehlo.cosine %50 : tensor<1x8x1x128xf32>
    %53 = stablehlo.slice %29 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %54 = stablehlo.slice %29 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %55 = stablehlo.convert %53 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %56 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %57 = stablehlo.multiply %55, %56 : tensor<1x8x8x128xf32>
    %58 = stablehlo.convert %54 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %59 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %60 = stablehlo.multiply %58, %59 : tensor<1x8x8x128xf32>
    %61 = stablehlo.subtract %57, %60 : tensor<1x8x8x128xf32>
    %62 = stablehlo.convert %54 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %63 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %64 = stablehlo.multiply %62, %63 : tensor<1x8x8x128xf32>
    %65 = stablehlo.convert %53 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %66 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %67 = stablehlo.multiply %65, %66 : tensor<1x8x8x128xf32>
    %68 = stablehlo.add %64, %67 : tensor<1x8x8x128xf32>
    %69 = stablehlo.concatenate %61, %68, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %70 = stablehlo.convert %69 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_8 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %71 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %72 = stablehlo.multiply %70, %71 : tensor<1x8x8x256xbf16>
    %73 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_9 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %74 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %75 = stablehlo.multiply %74, %73 : tensor<128xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %77 = stablehlo.power %76, %75 : tensor<128xf32>
    %78 = call @_pad(%77) : (tensor<128xf32>) -> tensor<128xf32>
    %79 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %80 = stablehlo.broadcast_in_dim %78, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %81 = stablehlo.convert %79 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %83 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %84 = stablehlo.divide %82, %83 : tensor<1x8x128xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %87 = stablehlo.divide %85, %86 : tensor<1x8x1x128xf32>
    %88 = stablehlo.sine %87 : tensor<1x8x1x128xf32>
    %89 = stablehlo.cosine %87 : tensor<1x8x1x128xf32>
    %90 = stablehlo.slice %33 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %91 = stablehlo.slice %33 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %92 = stablehlo.convert %90 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %93 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %94 = stablehlo.multiply %92, %93 : tensor<1x8x4x128xf32>
    %95 = stablehlo.convert %91 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %96 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %97 = stablehlo.multiply %95, %96 : tensor<1x8x4x128xf32>
    %98 = stablehlo.subtract %94, %97 : tensor<1x8x4x128xf32>
    %99 = stablehlo.convert %91 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %100 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %101 = stablehlo.multiply %99, %100 : tensor<1x8x4x128xf32>
    %102 = stablehlo.convert %90 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %103 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %104 = stablehlo.multiply %102, %103 : tensor<1x8x4x128xf32>
    %105 = stablehlo.add %101, %104 : tensor<1x8x4x128xf32>
    %106 = stablehlo.concatenate %98, %105, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %107 = stablehlo.convert %106 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %108 = stablehlo.reshape %72 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %109 = stablehlo.dot_general %107, %108, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %110 = stablehlo.transpose %109, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %111 = stablehlo.reshape %110 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_12 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %112 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %113 = stablehlo.divide %111, %112 : tensor<1x8x8x8xbf16>
    %114 = stablehlo.tanh %113 : tensor<1x8x8x8xbf16>
    %cst_13 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %115 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %116 = stablehlo.multiply %114, %115 : tensor<1x8x8x8xbf16>
    %117 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %118 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_14 = stablehlo.constant dense<4096> : tensor<i32>
    %119 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %120 = stablehlo.subtract %118, %119 : tensor<1x8x1xi32>
    %121 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %122 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %123 = stablehlo.compare GT, %121, %122, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_15 = stablehlo.constant dense<4096> : tensor<i32>
    %124 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %125 = stablehlo.add %118, %124 : tensor<1x8x1xi32>
    %126 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %127 = stablehlo.broadcast_in_dim %125, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %128 = stablehlo.compare LT, %126, %127, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %129 = stablehlo.and %123, %128 : tensor<1x8x8xi1>
    %130 = stablehlo.and %arg237, %129 : tensor<1x8x8xi1>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_16 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %132 = call @_where(%131, %116, %cst_16) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_17 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %133 = stablehlo.reduce(%132 init: %cst_17) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_18 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %134 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %135 = stablehlo.maximum %134, %133 : tensor<1x8x8xbf16>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %137 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %138 = stablehlo.subtract %132, %137 : tensor<1x8x8x8xbf16>
    %139 = stablehlo.exponential %138 : tensor<1x8x8x8xbf16>
    %140 = stablehlo.convert %139 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %141 = stablehlo.reduce(%140 init: %cst_19) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %143 = stablehlo.convert %142 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %145 = stablehlo.divide %139, %144 : tensor<1x8x8x8xbf16>
    %146 = stablehlo.reshape %145 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %147 = stablehlo.dot_general %35, %146, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %148 = stablehlo.transpose %147, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %149 = stablehlo.reshape %148 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %150 = stablehlo.dot_general %149, %arg2, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %151 = chlo.square %150 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %152 = stablehlo.convert %151 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %153 = stablehlo.reduce(%152 init: %cst_20) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_21 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %155 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %156 = stablehlo.divide %154, %155 : tensor<1x8x1xf32>
    %157 = stablehlo.convert %156 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_22 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %158 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %159 = stablehlo.add %157, %158 : tensor<1x8x1xbf16>
    %160 = stablehlo.rsqrt %159 : tensor<1x8x1xbf16>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %162 = stablehlo.multiply %150, %161 : tensor<1x8x2304xbf16>
    %163 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %164 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %165 = stablehlo.add %164, %163 : tensor<1x1x2304xbf16>
    %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %167 = stablehlo.multiply %162, %166 : tensor<1x8x2304xbf16>
    %168 = stablehlo.add %167, %11 : tensor<1x8x2304xbf16>
    %169 = chlo.square %168 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %170 = stablehlo.convert %169 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %171 = stablehlo.reduce(%170 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_25 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %173 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %174 = stablehlo.divide %172, %173 : tensor<1x8x1xf32>
    %175 = stablehlo.convert %174 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_26 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %176 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %177 = stablehlo.add %175, %176 : tensor<1x8x1xbf16>
    %178 = stablehlo.rsqrt %177 : tensor<1x8x1xbf16>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %180 = stablehlo.multiply %168, %179 : tensor<1x8x2304xbf16>
    %181 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_27 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %182 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %183 = stablehlo.add %182, %181 : tensor<1x1x2304xbf16>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %185 = stablehlo.multiply %180, %184 : tensor<1x8x2304xbf16>
    %186 = stablehlo.dot_general %185, %arg5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %187 = stablehlo.slice %186 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %188 = stablehlo.reshape %187 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %189 = stablehlo.multiply %188, %188 : tensor<1x8x9216xbf16>
    %190 = stablehlo.multiply %189, %188 : tensor<1x8x9216xbf16>
    %cst_28 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %191 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %192 = stablehlo.multiply %191, %190 : tensor<1x8x9216xbf16>
    %193 = stablehlo.add %188, %192 : tensor<1x8x9216xbf16>
    %cst_29 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %194 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %195 = stablehlo.multiply %194, %193 : tensor<1x8x9216xbf16>
    %196 = stablehlo.tanh %195 : tensor<1x8x9216xbf16>
    %cst_30 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %197 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %198 = stablehlo.add %197, %196 : tensor<1x8x9216xbf16>
    %cst_31 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %199 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %200 = stablehlo.multiply %199, %198 : tensor<1x8x9216xbf16>
    %201 = stablehlo.multiply %188, %200 : tensor<1x8x9216xbf16>
    %202 = stablehlo.slice %186 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %203 = stablehlo.reshape %202 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %204 = stablehlo.multiply %201, %203 : tensor<1x8x9216xbf16>
    %205 = stablehlo.dot_general %204, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %206 = chlo.square %205 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %207 = stablehlo.convert %206 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %208 = stablehlo.reduce(%207 init: %cst_32) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_33 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %210 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %211 = stablehlo.divide %209, %210 : tensor<1x8x1xf32>
    %212 = stablehlo.convert %211 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_34 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %213 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %214 = stablehlo.add %212, %213 : tensor<1x8x1xbf16>
    %215 = stablehlo.rsqrt %214 : tensor<1x8x1xbf16>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %217 = stablehlo.multiply %205, %216 : tensor<1x8x2304xbf16>
    %218 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_35 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %219 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %220 = stablehlo.add %219, %218 : tensor<1x1x2304xbf16>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %222 = stablehlo.multiply %217, %221 : tensor<1x8x2304xbf16>
    %223 = stablehlo.add %222, %168 : tensor<1x8x2304xbf16>
    %224 = chlo.square %223 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %225 = stablehlo.convert %224 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %226 = stablehlo.reduce(%225 init: %cst_36) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_37 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %228 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %229 = stablehlo.divide %227, %228 : tensor<1x8x1xf32>
    %230 = stablehlo.convert %229 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_38 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %231 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %232 = stablehlo.add %230, %231 : tensor<1x8x1xbf16>
    %233 = stablehlo.rsqrt %232 : tensor<1x8x1xbf16>
    %234 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %235 = stablehlo.multiply %223, %234 : tensor<1x8x2304xbf16>
    %236 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_39 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %237 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %238 = stablehlo.add %237, %236 : tensor<1x1x2304xbf16>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %240 = stablehlo.multiply %235, %239 : tensor<1x8x2304xbf16>
    %241 = stablehlo.dot_general %240, %arg13, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %242 = stablehlo.dot_general %arg12, %240, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %243 = stablehlo.transpose %242, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %244 = stablehlo.slice %243 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %245 = stablehlo.reshape %244 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %246 = stablehlo.slice %243 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %247 = stablehlo.reshape %246 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %248 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_40 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %249 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %250 = stablehlo.multiply %249, %248 : tensor<128xf32>
    %cst_41 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %251 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %252 = stablehlo.power %251, %250 : tensor<128xf32>
    %253 = call @_pad(%252) : (tensor<128xf32>) -> tensor<128xf32>
    %254 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %255 = stablehlo.broadcast_in_dim %253, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %256 = stablehlo.convert %254 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %258 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %259 = stablehlo.divide %257, %258 : tensor<1x8x128xf32>
    %260 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_42 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %261 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %262 = stablehlo.divide %260, %261 : tensor<1x8x1x128xf32>
    %263 = stablehlo.sine %262 : tensor<1x8x1x128xf32>
    %264 = stablehlo.cosine %262 : tensor<1x8x1x128xf32>
    %265 = stablehlo.slice %241 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %266 = stablehlo.slice %241 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %267 = stablehlo.convert %265 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %268 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %269 = stablehlo.multiply %267, %268 : tensor<1x8x8x128xf32>
    %270 = stablehlo.convert %266 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %271 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %272 = stablehlo.multiply %270, %271 : tensor<1x8x8x128xf32>
    %273 = stablehlo.subtract %269, %272 : tensor<1x8x8x128xf32>
    %274 = stablehlo.convert %266 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %275 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %276 = stablehlo.multiply %274, %275 : tensor<1x8x8x128xf32>
    %277 = stablehlo.convert %265 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %278 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %279 = stablehlo.multiply %277, %278 : tensor<1x8x8x128xf32>
    %280 = stablehlo.add %276, %279 : tensor<1x8x8x128xf32>
    %281 = stablehlo.concatenate %273, %280, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %282 = stablehlo.convert %281 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_43 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %283 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %284 = stablehlo.multiply %282, %283 : tensor<1x8x8x256xbf16>
    %285 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_44 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %286 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %287 = stablehlo.multiply %286, %285 : tensor<128xf32>
    %cst_45 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %288 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %289 = stablehlo.power %288, %287 : tensor<128xf32>
    %290 = call @_pad(%289) : (tensor<128xf32>) -> tensor<128xf32>
    %291 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %292 = stablehlo.broadcast_in_dim %290, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %293 = stablehlo.convert %291 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %295 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %296 = stablehlo.divide %294, %295 : tensor<1x8x128xf32>
    %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_46 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %298 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %299 = stablehlo.divide %297, %298 : tensor<1x8x1x128xf32>
    %300 = stablehlo.sine %299 : tensor<1x8x1x128xf32>
    %301 = stablehlo.cosine %299 : tensor<1x8x1x128xf32>
    %302 = stablehlo.slice %245 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %303 = stablehlo.slice %245 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %304 = stablehlo.convert %302 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %305 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %306 = stablehlo.multiply %304, %305 : tensor<1x8x4x128xf32>
    %307 = stablehlo.convert %303 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %308 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %309 = stablehlo.multiply %307, %308 : tensor<1x8x4x128xf32>
    %310 = stablehlo.subtract %306, %309 : tensor<1x8x4x128xf32>
    %311 = stablehlo.convert %303 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %312 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %313 = stablehlo.multiply %311, %312 : tensor<1x8x4x128xf32>
    %314 = stablehlo.convert %302 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %315 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %316 = stablehlo.multiply %314, %315 : tensor<1x8x4x128xf32>
    %317 = stablehlo.add %313, %316 : tensor<1x8x4x128xf32>
    %318 = stablehlo.concatenate %310, %317, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %319 = stablehlo.convert %318 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %320 = stablehlo.reshape %284 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %321 = stablehlo.dot_general %319, %320, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %322 = stablehlo.transpose %321, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %323 = stablehlo.reshape %322 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_47 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %324 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %325 = stablehlo.divide %323, %324 : tensor<1x8x8x8xbf16>
    %326 = stablehlo.tanh %325 : tensor<1x8x8x8xbf16>
    %cst_48 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %327 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %328 = stablehlo.multiply %326, %327 : tensor<1x8x8x8xbf16>
    %329 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_49 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %330 = call @_where(%329, %328, %cst_49) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_50 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %331 = stablehlo.reduce(%330 init: %cst_50) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_51 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %332 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %333 = stablehlo.maximum %332, %331 : tensor<1x8x8xbf16>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %336 = stablehlo.subtract %330, %335 : tensor<1x8x8x8xbf16>
    %337 = stablehlo.exponential %336 : tensor<1x8x8x8xbf16>
    %338 = stablehlo.convert %337 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %339 = stablehlo.reduce(%338 init: %cst_52) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %340 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %341 = stablehlo.convert %340 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %343 = stablehlo.divide %337, %342 : tensor<1x8x8x8xbf16>
    %344 = stablehlo.reshape %343 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %345 = stablehlo.dot_general %247, %344, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %346 = stablehlo.transpose %345, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %347 = stablehlo.reshape %346 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %348 = stablehlo.dot_general %347, %arg11, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %349 = chlo.square %348 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %350 = stablehlo.convert %349 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %351 = stablehlo.reduce(%350 init: %cst_53) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %352 = stablehlo.broadcast_in_dim %351, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_54 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %353 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %354 = stablehlo.divide %352, %353 : tensor<1x8x1xf32>
    %355 = stablehlo.convert %354 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_55 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %356 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %357 = stablehlo.add %355, %356 : tensor<1x8x1xbf16>
    %358 = stablehlo.rsqrt %357 : tensor<1x8x1xbf16>
    %359 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %360 = stablehlo.multiply %348, %359 : tensor<1x8x2304xbf16>
    %361 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_56 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %362 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %363 = stablehlo.add %362, %361 : tensor<1x1x2304xbf16>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %365 = stablehlo.multiply %360, %364 : tensor<1x8x2304xbf16>
    %366 = stablehlo.add %365, %223 : tensor<1x8x2304xbf16>
    %367 = chlo.square %366 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %368 = stablehlo.convert %367 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %369 = stablehlo.reduce(%368 init: %cst_57) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %370 = stablehlo.broadcast_in_dim %369, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_58 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %371 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %372 = stablehlo.divide %370, %371 : tensor<1x8x1xf32>
    %373 = stablehlo.convert %372 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_59 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %374 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %375 = stablehlo.add %373, %374 : tensor<1x8x1xbf16>
    %376 = stablehlo.rsqrt %375 : tensor<1x8x1xbf16>
    %377 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %378 = stablehlo.multiply %366, %377 : tensor<1x8x2304xbf16>
    %379 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_60 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %380 = stablehlo.broadcast_in_dim %cst_60, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %381 = stablehlo.add %380, %379 : tensor<1x1x2304xbf16>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %383 = stablehlo.multiply %378, %382 : tensor<1x8x2304xbf16>
    %384 = stablehlo.dot_general %383, %arg14, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %385 = stablehlo.slice %384 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %386 = stablehlo.reshape %385 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %387 = stablehlo.multiply %386, %386 : tensor<1x8x9216xbf16>
    %388 = stablehlo.multiply %387, %386 : tensor<1x8x9216xbf16>
    %cst_61 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %389 = stablehlo.broadcast_in_dim %cst_61, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %390 = stablehlo.multiply %389, %388 : tensor<1x8x9216xbf16>
    %391 = stablehlo.add %386, %390 : tensor<1x8x9216xbf16>
    %cst_62 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %392 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %393 = stablehlo.multiply %392, %391 : tensor<1x8x9216xbf16>
    %394 = stablehlo.tanh %393 : tensor<1x8x9216xbf16>
    %cst_63 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %395 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %396 = stablehlo.add %395, %394 : tensor<1x8x9216xbf16>
    %cst_64 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %397 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %398 = stablehlo.multiply %397, %396 : tensor<1x8x9216xbf16>
    %399 = stablehlo.multiply %386, %398 : tensor<1x8x9216xbf16>
    %400 = stablehlo.slice %384 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %401 = stablehlo.reshape %400 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %402 = stablehlo.multiply %399, %401 : tensor<1x8x9216xbf16>
    %403 = stablehlo.dot_general %402, %arg15, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %404 = chlo.square %403 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %405 = stablehlo.convert %404 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %406 = stablehlo.reduce(%405 init: %cst_65) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %407 = stablehlo.broadcast_in_dim %406, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_66 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %408 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %409 = stablehlo.divide %407, %408 : tensor<1x8x1xf32>
    %410 = stablehlo.convert %409 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_67 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %411 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %412 = stablehlo.add %410, %411 : tensor<1x8x1xbf16>
    %413 = stablehlo.rsqrt %412 : tensor<1x8x1xbf16>
    %414 = stablehlo.broadcast_in_dim %413, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %415 = stablehlo.multiply %403, %414 : tensor<1x8x2304xbf16>
    %416 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_68 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %417 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %418 = stablehlo.add %417, %416 : tensor<1x1x2304xbf16>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %420 = stablehlo.multiply %415, %419 : tensor<1x8x2304xbf16>
    %421 = stablehlo.add %420, %366 : tensor<1x8x2304xbf16>
    %422 = chlo.square %421 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %423 = stablehlo.convert %422 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %424 = stablehlo.reduce(%423 init: %cst_69) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_70 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %426 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %427 = stablehlo.divide %425, %426 : tensor<1x8x1xf32>
    %428 = stablehlo.convert %427 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_71 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %429 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %430 = stablehlo.add %428, %429 : tensor<1x8x1xbf16>
    %431 = stablehlo.rsqrt %430 : tensor<1x8x1xbf16>
    %432 = stablehlo.broadcast_in_dim %431, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %433 = stablehlo.multiply %421, %432 : tensor<1x8x2304xbf16>
    %434 = stablehlo.broadcast_in_dim %arg117, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_72 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %435 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %436 = stablehlo.add %435, %434 : tensor<1x1x2304xbf16>
    %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %438 = stablehlo.multiply %433, %437 : tensor<1x8x2304xbf16>
    %439 = stablehlo.dot_general %438, %arg112, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %440 = stablehlo.dot_general %arg111, %438, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %441 = stablehlo.transpose %440, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %442 = stablehlo.slice %441 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %443 = stablehlo.reshape %442 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %444 = stablehlo.slice %441 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %445 = stablehlo.reshape %444 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %446 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_73 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %447 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %448 = stablehlo.multiply %447, %446 : tensor<128xf32>
    %cst_74 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %449 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %450 = stablehlo.power %449, %448 : tensor<128xf32>
    %451 = call @_pad(%450) : (tensor<128xf32>) -> tensor<128xf32>
    %452 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %453 = stablehlo.broadcast_in_dim %451, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %454 = stablehlo.convert %452 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %455 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %456 = stablehlo.broadcast_in_dim %453, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %457 = stablehlo.divide %455, %456 : tensor<1x8x128xf32>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_75 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %459 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %460 = stablehlo.divide %458, %459 : tensor<1x8x1x128xf32>
    %461 = stablehlo.sine %460 : tensor<1x8x1x128xf32>
    %462 = stablehlo.cosine %460 : tensor<1x8x1x128xf32>
    %463 = stablehlo.slice %439 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %464 = stablehlo.slice %439 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %465 = stablehlo.convert %463 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %466 = stablehlo.broadcast_in_dim %462, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %467 = stablehlo.multiply %465, %466 : tensor<1x8x8x128xf32>
    %468 = stablehlo.convert %464 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %469 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %470 = stablehlo.multiply %468, %469 : tensor<1x8x8x128xf32>
    %471 = stablehlo.subtract %467, %470 : tensor<1x8x8x128xf32>
    %472 = stablehlo.convert %464 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %473 = stablehlo.broadcast_in_dim %462, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %474 = stablehlo.multiply %472, %473 : tensor<1x8x8x128xf32>
    %475 = stablehlo.convert %463 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %476 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %477 = stablehlo.multiply %475, %476 : tensor<1x8x8x128xf32>
    %478 = stablehlo.add %474, %477 : tensor<1x8x8x128xf32>
    %479 = stablehlo.concatenate %471, %478, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %480 = stablehlo.convert %479 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_76 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %481 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %482 = stablehlo.multiply %480, %481 : tensor<1x8x8x256xbf16>
    %483 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_77 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %484 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %485 = stablehlo.multiply %484, %483 : tensor<128xf32>
    %cst_78 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %486 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %487 = stablehlo.power %486, %485 : tensor<128xf32>
    %488 = call @_pad(%487) : (tensor<128xf32>) -> tensor<128xf32>
    %489 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %490 = stablehlo.broadcast_in_dim %488, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %491 = stablehlo.convert %489 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %492 = stablehlo.broadcast_in_dim %491, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %493 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %494 = stablehlo.divide %492, %493 : tensor<1x8x128xf32>
    %495 = stablehlo.broadcast_in_dim %494, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_79 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %496 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %497 = stablehlo.divide %495, %496 : tensor<1x8x1x128xf32>
    %498 = stablehlo.sine %497 : tensor<1x8x1x128xf32>
    %499 = stablehlo.cosine %497 : tensor<1x8x1x128xf32>
    %500 = stablehlo.slice %443 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %501 = stablehlo.slice %443 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %502 = stablehlo.convert %500 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %503 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %504 = stablehlo.multiply %502, %503 : tensor<1x8x4x128xf32>
    %505 = stablehlo.convert %501 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %506 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %507 = stablehlo.multiply %505, %506 : tensor<1x8x4x128xf32>
    %508 = stablehlo.subtract %504, %507 : tensor<1x8x4x128xf32>
    %509 = stablehlo.convert %501 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %510 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %511 = stablehlo.multiply %509, %510 : tensor<1x8x4x128xf32>
    %512 = stablehlo.convert %500 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %513 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %514 = stablehlo.multiply %512, %513 : tensor<1x8x4x128xf32>
    %515 = stablehlo.add %511, %514 : tensor<1x8x4x128xf32>
    %516 = stablehlo.concatenate %508, %515, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %517 = stablehlo.convert %516 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %518 = stablehlo.reshape %482 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %519 = stablehlo.dot_general %517, %518, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %520 = stablehlo.transpose %519, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %521 = stablehlo.reshape %520 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_80 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %522 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %523 = stablehlo.divide %521, %522 : tensor<1x8x8x8xbf16>
    %524 = stablehlo.tanh %523 : tensor<1x8x8x8xbf16>
    %cst_81 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %525 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %526 = stablehlo.multiply %524, %525 : tensor<1x8x8x8xbf16>
    %527 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %528 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_82 = stablehlo.constant dense<4096> : tensor<i32>
    %529 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %530 = stablehlo.subtract %528, %529 : tensor<1x8x1xi32>
    %531 = stablehlo.broadcast_in_dim %527, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %532 = stablehlo.broadcast_in_dim %530, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %533 = stablehlo.compare GT, %531, %532, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_83 = stablehlo.constant dense<4096> : tensor<i32>
    %534 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %535 = stablehlo.add %528, %534 : tensor<1x8x1xi32>
    %536 = stablehlo.broadcast_in_dim %527, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %537 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %538 = stablehlo.compare LT, %536, %537, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %539 = stablehlo.and %533, %538 : tensor<1x8x8xi1>
    %540 = stablehlo.and %arg237, %539 : tensor<1x8x8xi1>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_84 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %542 = call @_where(%541, %526, %cst_84) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_85 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %543 = stablehlo.reduce(%542 init: %cst_85) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_86 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %544 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %545 = stablehlo.maximum %544, %543 : tensor<1x8x8xbf16>
    %546 = stablehlo.broadcast_in_dim %545, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %547 = stablehlo.broadcast_in_dim %546, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %548 = stablehlo.subtract %542, %547 : tensor<1x8x8x8xbf16>
    %549 = stablehlo.exponential %548 : tensor<1x8x8x8xbf16>
    %550 = stablehlo.convert %549 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %551 = stablehlo.reduce(%550 init: %cst_87) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %552 = stablehlo.broadcast_in_dim %551, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %553 = stablehlo.convert %552 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %555 = stablehlo.divide %549, %554 : tensor<1x8x8x8xbf16>
    %556 = stablehlo.reshape %555 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %557 = stablehlo.dot_general %445, %556, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %558 = stablehlo.transpose %557, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %559 = stablehlo.reshape %558 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %560 = stablehlo.dot_general %559, %arg110, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %561 = chlo.square %560 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %562 = stablehlo.convert %561 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %563 = stablehlo.reduce(%562 init: %cst_88) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %564 = stablehlo.broadcast_in_dim %563, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_89 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %565 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %566 = stablehlo.divide %564, %565 : tensor<1x8x1xf32>
    %567 = stablehlo.convert %566 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_90 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %568 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %569 = stablehlo.add %567, %568 : tensor<1x8x1xbf16>
    %570 = stablehlo.rsqrt %569 : tensor<1x8x1xbf16>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %572 = stablehlo.multiply %560, %571 : tensor<1x8x2304xbf16>
    %573 = stablehlo.broadcast_in_dim %arg115, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_91 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %574 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %575 = stablehlo.add %574, %573 : tensor<1x1x2304xbf16>
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %577 = stablehlo.multiply %572, %576 : tensor<1x8x2304xbf16>
    %578 = stablehlo.add %577, %421 : tensor<1x8x2304xbf16>
    %579 = chlo.square %578 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %580 = stablehlo.convert %579 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %581 = stablehlo.reduce(%580 init: %cst_92) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %582 = stablehlo.broadcast_in_dim %581, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_93 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %583 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %584 = stablehlo.divide %582, %583 : tensor<1x8x1xf32>
    %585 = stablehlo.convert %584 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_94 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %586 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %587 = stablehlo.add %585, %586 : tensor<1x8x1xbf16>
    %588 = stablehlo.rsqrt %587 : tensor<1x8x1xbf16>
    %589 = stablehlo.broadcast_in_dim %588, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %590 = stablehlo.multiply %578, %589 : tensor<1x8x2304xbf16>
    %591 = stablehlo.broadcast_in_dim %arg118, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_95 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %592 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %593 = stablehlo.add %592, %591 : tensor<1x1x2304xbf16>
    %594 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %595 = stablehlo.multiply %590, %594 : tensor<1x8x2304xbf16>
    %596 = stablehlo.dot_general %595, %arg113, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %597 = stablehlo.slice %596 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %598 = stablehlo.reshape %597 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %599 = stablehlo.multiply %598, %598 : tensor<1x8x9216xbf16>
    %600 = stablehlo.multiply %599, %598 : tensor<1x8x9216xbf16>
    %cst_96 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %601 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %602 = stablehlo.multiply %601, %600 : tensor<1x8x9216xbf16>
    %603 = stablehlo.add %598, %602 : tensor<1x8x9216xbf16>
    %cst_97 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %604 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %605 = stablehlo.multiply %604, %603 : tensor<1x8x9216xbf16>
    %606 = stablehlo.tanh %605 : tensor<1x8x9216xbf16>
    %cst_98 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %607 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %608 = stablehlo.add %607, %606 : tensor<1x8x9216xbf16>
    %cst_99 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %609 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %610 = stablehlo.multiply %609, %608 : tensor<1x8x9216xbf16>
    %611 = stablehlo.multiply %598, %610 : tensor<1x8x9216xbf16>
    %612 = stablehlo.slice %596 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %613 = stablehlo.reshape %612 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %614 = stablehlo.multiply %611, %613 : tensor<1x8x9216xbf16>
    %615 = stablehlo.dot_general %614, %arg114, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %616 = chlo.square %615 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %617 = stablehlo.convert %616 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %618 = stablehlo.reduce(%617 init: %cst_100) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_101 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %620 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %621 = stablehlo.divide %619, %620 : tensor<1x8x1xf32>
    %622 = stablehlo.convert %621 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_102 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %623 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %624 = stablehlo.add %622, %623 : tensor<1x8x1xbf16>
    %625 = stablehlo.rsqrt %624 : tensor<1x8x1xbf16>
    %626 = stablehlo.broadcast_in_dim %625, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %627 = stablehlo.multiply %615, %626 : tensor<1x8x2304xbf16>
    %628 = stablehlo.broadcast_in_dim %arg116, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_103 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %629 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %630 = stablehlo.add %629, %628 : tensor<1x1x2304xbf16>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %632 = stablehlo.multiply %627, %631 : tensor<1x8x2304xbf16>
    %633 = stablehlo.add %632, %578 : tensor<1x8x2304xbf16>
    %634 = chlo.square %633 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %635 = stablehlo.convert %634 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %636 = stablehlo.reduce(%635 init: %cst_104) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %637 = stablehlo.broadcast_in_dim %636, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_105 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %638 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %639 = stablehlo.divide %637, %638 : tensor<1x8x1xf32>
    %640 = stablehlo.convert %639 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_106 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %641 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %642 = stablehlo.add %640, %641 : tensor<1x8x1xbf16>
    %643 = stablehlo.rsqrt %642 : tensor<1x8x1xbf16>
    %644 = stablehlo.broadcast_in_dim %643, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %645 = stablehlo.multiply %633, %644 : tensor<1x8x2304xbf16>
    %646 = stablehlo.broadcast_in_dim %arg180, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_107 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %647 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %648 = stablehlo.add %647, %646 : tensor<1x1x2304xbf16>
    %649 = stablehlo.broadcast_in_dim %648, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %650 = stablehlo.multiply %645, %649 : tensor<1x8x2304xbf16>
    %651 = stablehlo.dot_general %650, %arg175, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %652 = stablehlo.dot_general %arg174, %650, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %653 = stablehlo.transpose %652, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %654 = stablehlo.slice %653 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %655 = stablehlo.reshape %654 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %656 = stablehlo.slice %653 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %657 = stablehlo.reshape %656 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %658 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_108 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %659 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %660 = stablehlo.multiply %659, %658 : tensor<128xf32>
    %cst_109 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %661 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %662 = stablehlo.power %661, %660 : tensor<128xf32>
    %663 = call @_pad(%662) : (tensor<128xf32>) -> tensor<128xf32>
    %664 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %665 = stablehlo.broadcast_in_dim %663, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %666 = stablehlo.convert %664 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %667 = stablehlo.broadcast_in_dim %666, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %668 = stablehlo.broadcast_in_dim %665, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %669 = stablehlo.divide %667, %668 : tensor<1x8x128xf32>
    %670 = stablehlo.broadcast_in_dim %669, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_110 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %671 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %672 = stablehlo.divide %670, %671 : tensor<1x8x1x128xf32>
    %673 = stablehlo.sine %672 : tensor<1x8x1x128xf32>
    %674 = stablehlo.cosine %672 : tensor<1x8x1x128xf32>
    %675 = stablehlo.slice %651 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %676 = stablehlo.slice %651 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %677 = stablehlo.convert %675 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %678 = stablehlo.broadcast_in_dim %674, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %679 = stablehlo.multiply %677, %678 : tensor<1x8x8x128xf32>
    %680 = stablehlo.convert %676 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %681 = stablehlo.broadcast_in_dim %673, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %682 = stablehlo.multiply %680, %681 : tensor<1x8x8x128xf32>
    %683 = stablehlo.subtract %679, %682 : tensor<1x8x8x128xf32>
    %684 = stablehlo.convert %676 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %685 = stablehlo.broadcast_in_dim %674, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %686 = stablehlo.multiply %684, %685 : tensor<1x8x8x128xf32>
    %687 = stablehlo.convert %675 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %688 = stablehlo.broadcast_in_dim %673, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %689 = stablehlo.multiply %687, %688 : tensor<1x8x8x128xf32>
    %690 = stablehlo.add %686, %689 : tensor<1x8x8x128xf32>
    %691 = stablehlo.concatenate %683, %690, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %692 = stablehlo.convert %691 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_111 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %693 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %694 = stablehlo.multiply %692, %693 : tensor<1x8x8x256xbf16>
    %695 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_112 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %696 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %697 = stablehlo.multiply %696, %695 : tensor<128xf32>
    %cst_113 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %698 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %699 = stablehlo.power %698, %697 : tensor<128xf32>
    %700 = call @_pad(%699) : (tensor<128xf32>) -> tensor<128xf32>
    %701 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %702 = stablehlo.broadcast_in_dim %700, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %703 = stablehlo.convert %701 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %704 = stablehlo.broadcast_in_dim %703, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %705 = stablehlo.broadcast_in_dim %702, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %706 = stablehlo.divide %704, %705 : tensor<1x8x128xf32>
    %707 = stablehlo.broadcast_in_dim %706, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_114 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %708 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %709 = stablehlo.divide %707, %708 : tensor<1x8x1x128xf32>
    %710 = stablehlo.sine %709 : tensor<1x8x1x128xf32>
    %711 = stablehlo.cosine %709 : tensor<1x8x1x128xf32>
    %712 = stablehlo.slice %655 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %713 = stablehlo.slice %655 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %714 = stablehlo.convert %712 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %715 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %716 = stablehlo.multiply %714, %715 : tensor<1x8x4x128xf32>
    %717 = stablehlo.convert %713 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %718 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %719 = stablehlo.multiply %717, %718 : tensor<1x8x4x128xf32>
    %720 = stablehlo.subtract %716, %719 : tensor<1x8x4x128xf32>
    %721 = stablehlo.convert %713 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %722 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %723 = stablehlo.multiply %721, %722 : tensor<1x8x4x128xf32>
    %724 = stablehlo.convert %712 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %725 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %726 = stablehlo.multiply %724, %725 : tensor<1x8x4x128xf32>
    %727 = stablehlo.add %723, %726 : tensor<1x8x4x128xf32>
    %728 = stablehlo.concatenate %720, %727, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %729 = stablehlo.convert %728 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %730 = stablehlo.reshape %694 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %731 = stablehlo.dot_general %729, %730, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %732 = stablehlo.transpose %731, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %733 = stablehlo.reshape %732 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_115 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %734 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %735 = stablehlo.divide %733, %734 : tensor<1x8x8x8xbf16>
    %736 = stablehlo.tanh %735 : tensor<1x8x8x8xbf16>
    %cst_116 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %737 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %738 = stablehlo.multiply %736, %737 : tensor<1x8x8x8xbf16>
    %739 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_117 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %740 = call @_where(%739, %738, %cst_117) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_118 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %741 = stablehlo.reduce(%740 init: %cst_118) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_119 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %742 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %743 = stablehlo.maximum %742, %741 : tensor<1x8x8xbf16>
    %744 = stablehlo.broadcast_in_dim %743, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %745 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %746 = stablehlo.subtract %740, %745 : tensor<1x8x8x8xbf16>
    %747 = stablehlo.exponential %746 : tensor<1x8x8x8xbf16>
    %748 = stablehlo.convert %747 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %749 = stablehlo.reduce(%748 init: %cst_120) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %751 = stablehlo.convert %750 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %752 = stablehlo.broadcast_in_dim %751, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %753 = stablehlo.divide %747, %752 : tensor<1x8x8x8xbf16>
    %754 = stablehlo.reshape %753 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %755 = stablehlo.dot_general %657, %754, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %756 = stablehlo.transpose %755, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %757 = stablehlo.reshape %756 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %758 = stablehlo.dot_general %757, %arg173, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %759 = chlo.square %758 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %760 = stablehlo.convert %759 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %761 = stablehlo.reduce(%760 init: %cst_121) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %762 = stablehlo.broadcast_in_dim %761, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_122 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %763 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %764 = stablehlo.divide %762, %763 : tensor<1x8x1xf32>
    %765 = stablehlo.convert %764 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_123 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %766 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %767 = stablehlo.add %765, %766 : tensor<1x8x1xbf16>
    %768 = stablehlo.rsqrt %767 : tensor<1x8x1xbf16>
    %769 = stablehlo.broadcast_in_dim %768, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %770 = stablehlo.multiply %758, %769 : tensor<1x8x2304xbf16>
    %771 = stablehlo.broadcast_in_dim %arg178, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_124 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %772 = stablehlo.broadcast_in_dim %cst_124, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %773 = stablehlo.add %772, %771 : tensor<1x1x2304xbf16>
    %774 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %775 = stablehlo.multiply %770, %774 : tensor<1x8x2304xbf16>
    %776 = stablehlo.add %775, %633 : tensor<1x8x2304xbf16>
    %777 = chlo.square %776 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %778 = stablehlo.convert %777 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %779 = stablehlo.reduce(%778 init: %cst_125) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %780 = stablehlo.broadcast_in_dim %779, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_126 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %781 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %782 = stablehlo.divide %780, %781 : tensor<1x8x1xf32>
    %783 = stablehlo.convert %782 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_127 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %784 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %785 = stablehlo.add %783, %784 : tensor<1x8x1xbf16>
    %786 = stablehlo.rsqrt %785 : tensor<1x8x1xbf16>
    %787 = stablehlo.broadcast_in_dim %786, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %788 = stablehlo.multiply %776, %787 : tensor<1x8x2304xbf16>
    %789 = stablehlo.broadcast_in_dim %arg181, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_128 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %790 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %791 = stablehlo.add %790, %789 : tensor<1x1x2304xbf16>
    %792 = stablehlo.broadcast_in_dim %791, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %793 = stablehlo.multiply %788, %792 : tensor<1x8x2304xbf16>
    %794 = stablehlo.dot_general %793, %arg176, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %795 = stablehlo.slice %794 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %796 = stablehlo.reshape %795 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %797 = stablehlo.multiply %796, %796 : tensor<1x8x9216xbf16>
    %798 = stablehlo.multiply %797, %796 : tensor<1x8x9216xbf16>
    %cst_129 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %799 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %800 = stablehlo.multiply %799, %798 : tensor<1x8x9216xbf16>
    %801 = stablehlo.add %796, %800 : tensor<1x8x9216xbf16>
    %cst_130 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %802 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %803 = stablehlo.multiply %802, %801 : tensor<1x8x9216xbf16>
    %804 = stablehlo.tanh %803 : tensor<1x8x9216xbf16>
    %cst_131 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %805 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %806 = stablehlo.add %805, %804 : tensor<1x8x9216xbf16>
    %cst_132 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %807 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %808 = stablehlo.multiply %807, %806 : tensor<1x8x9216xbf16>
    %809 = stablehlo.multiply %796, %808 : tensor<1x8x9216xbf16>
    %810 = stablehlo.slice %794 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %811 = stablehlo.reshape %810 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %812 = stablehlo.multiply %809, %811 : tensor<1x8x9216xbf16>
    %813 = stablehlo.dot_general %812, %arg177, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %814 = chlo.square %813 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %815 = stablehlo.convert %814 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %816 = stablehlo.reduce(%815 init: %cst_133) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %817 = stablehlo.broadcast_in_dim %816, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_134 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %818 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %819 = stablehlo.divide %817, %818 : tensor<1x8x1xf32>
    %820 = stablehlo.convert %819 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_135 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %821 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %822 = stablehlo.add %820, %821 : tensor<1x8x1xbf16>
    %823 = stablehlo.rsqrt %822 : tensor<1x8x1xbf16>
    %824 = stablehlo.broadcast_in_dim %823, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %825 = stablehlo.multiply %813, %824 : tensor<1x8x2304xbf16>
    %826 = stablehlo.broadcast_in_dim %arg179, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_136 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %827 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %828 = stablehlo.add %827, %826 : tensor<1x1x2304xbf16>
    %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %830 = stablehlo.multiply %825, %829 : tensor<1x8x2304xbf16>
    %831 = stablehlo.add %830, %776 : tensor<1x8x2304xbf16>
    %832 = chlo.square %831 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %833 = stablehlo.convert %832 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %834 = stablehlo.reduce(%833 init: %cst_137) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %835 = stablehlo.broadcast_in_dim %834, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_138 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %836 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %837 = stablehlo.divide %835, %836 : tensor<1x8x1xf32>
    %838 = stablehlo.convert %837 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_139 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %839 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %840 = stablehlo.add %838, %839 : tensor<1x8x1xbf16>
    %841 = stablehlo.rsqrt %840 : tensor<1x8x1xbf16>
    %842 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %843 = stablehlo.multiply %831, %842 : tensor<1x8x2304xbf16>
    %844 = stablehlo.broadcast_in_dim %arg189, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_140 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %845 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %846 = stablehlo.add %845, %844 : tensor<1x1x2304xbf16>
    %847 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %848 = stablehlo.multiply %843, %847 : tensor<1x8x2304xbf16>
    %849 = stablehlo.dot_general %848, %arg184, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %850 = stablehlo.dot_general %arg183, %848, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %851 = stablehlo.transpose %850, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %852 = stablehlo.slice %851 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %853 = stablehlo.reshape %852 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %854 = stablehlo.slice %851 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %855 = stablehlo.reshape %854 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %856 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_141 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %857 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %858 = stablehlo.multiply %857, %856 : tensor<128xf32>
    %cst_142 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %859 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %860 = stablehlo.power %859, %858 : tensor<128xf32>
    %861 = call @_pad(%860) : (tensor<128xf32>) -> tensor<128xf32>
    %862 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %863 = stablehlo.broadcast_in_dim %861, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %864 = stablehlo.convert %862 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %865 = stablehlo.broadcast_in_dim %864, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %866 = stablehlo.broadcast_in_dim %863, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %867 = stablehlo.divide %865, %866 : tensor<1x8x128xf32>
    %868 = stablehlo.broadcast_in_dim %867, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_143 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %869 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %870 = stablehlo.divide %868, %869 : tensor<1x8x1x128xf32>
    %871 = stablehlo.sine %870 : tensor<1x8x1x128xf32>
    %872 = stablehlo.cosine %870 : tensor<1x8x1x128xf32>
    %873 = stablehlo.slice %849 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %874 = stablehlo.slice %849 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %875 = stablehlo.convert %873 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %876 = stablehlo.broadcast_in_dim %872, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %877 = stablehlo.multiply %875, %876 : tensor<1x8x8x128xf32>
    %878 = stablehlo.convert %874 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %879 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %880 = stablehlo.multiply %878, %879 : tensor<1x8x8x128xf32>
    %881 = stablehlo.subtract %877, %880 : tensor<1x8x8x128xf32>
    %882 = stablehlo.convert %874 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %883 = stablehlo.broadcast_in_dim %872, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %884 = stablehlo.multiply %882, %883 : tensor<1x8x8x128xf32>
    %885 = stablehlo.convert %873 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %886 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %887 = stablehlo.multiply %885, %886 : tensor<1x8x8x128xf32>
    %888 = stablehlo.add %884, %887 : tensor<1x8x8x128xf32>
    %889 = stablehlo.concatenate %881, %888, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %890 = stablehlo.convert %889 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_144 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %891 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %892 = stablehlo.multiply %890, %891 : tensor<1x8x8x256xbf16>
    %893 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_145 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %894 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %895 = stablehlo.multiply %894, %893 : tensor<128xf32>
    %cst_146 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %896 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %897 = stablehlo.power %896, %895 : tensor<128xf32>
    %898 = call @_pad(%897) : (tensor<128xf32>) -> tensor<128xf32>
    %899 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %900 = stablehlo.broadcast_in_dim %898, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %901 = stablehlo.convert %899 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %903 = stablehlo.broadcast_in_dim %900, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %904 = stablehlo.divide %902, %903 : tensor<1x8x128xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_147 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %906 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %907 = stablehlo.divide %905, %906 : tensor<1x8x1x128xf32>
    %908 = stablehlo.sine %907 : tensor<1x8x1x128xf32>
    %909 = stablehlo.cosine %907 : tensor<1x8x1x128xf32>
    %910 = stablehlo.slice %853 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %911 = stablehlo.slice %853 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %912 = stablehlo.convert %910 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %913 = stablehlo.broadcast_in_dim %909, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %914 = stablehlo.multiply %912, %913 : tensor<1x8x4x128xf32>
    %915 = stablehlo.convert %911 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %916 = stablehlo.broadcast_in_dim %908, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %917 = stablehlo.multiply %915, %916 : tensor<1x8x4x128xf32>
    %918 = stablehlo.subtract %914, %917 : tensor<1x8x4x128xf32>
    %919 = stablehlo.convert %911 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %920 = stablehlo.broadcast_in_dim %909, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %921 = stablehlo.multiply %919, %920 : tensor<1x8x4x128xf32>
    %922 = stablehlo.convert %910 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %923 = stablehlo.broadcast_in_dim %908, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %924 = stablehlo.multiply %922, %923 : tensor<1x8x4x128xf32>
    %925 = stablehlo.add %921, %924 : tensor<1x8x4x128xf32>
    %926 = stablehlo.concatenate %918, %925, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %927 = stablehlo.convert %926 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %928 = stablehlo.reshape %892 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %929 = stablehlo.dot_general %927, %928, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %930 = stablehlo.transpose %929, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %931 = stablehlo.reshape %930 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_148 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %932 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %933 = stablehlo.divide %931, %932 : tensor<1x8x8x8xbf16>
    %934 = stablehlo.tanh %933 : tensor<1x8x8x8xbf16>
    %cst_149 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %935 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %936 = stablehlo.multiply %934, %935 : tensor<1x8x8x8xbf16>
    %937 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %938 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_150 = stablehlo.constant dense<4096> : tensor<i32>
    %939 = stablehlo.broadcast_in_dim %c_150, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %940 = stablehlo.subtract %938, %939 : tensor<1x8x1xi32>
    %941 = stablehlo.broadcast_in_dim %937, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %942 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %943 = stablehlo.compare GT, %941, %942, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_151 = stablehlo.constant dense<4096> : tensor<i32>
    %944 = stablehlo.broadcast_in_dim %c_151, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %945 = stablehlo.add %938, %944 : tensor<1x8x1xi32>
    %946 = stablehlo.broadcast_in_dim %937, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %947 = stablehlo.broadcast_in_dim %945, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %948 = stablehlo.compare LT, %946, %947, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %949 = stablehlo.and %943, %948 : tensor<1x8x8xi1>
    %950 = stablehlo.and %arg237, %949 : tensor<1x8x8xi1>
    %951 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_152 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %952 = call @_where(%951, %936, %cst_152) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_153 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %953 = stablehlo.reduce(%952 init: %cst_153) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_154 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %954 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %955 = stablehlo.maximum %954, %953 : tensor<1x8x8xbf16>
    %956 = stablehlo.broadcast_in_dim %955, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %957 = stablehlo.broadcast_in_dim %956, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %958 = stablehlo.subtract %952, %957 : tensor<1x8x8x8xbf16>
    %959 = stablehlo.exponential %958 : tensor<1x8x8x8xbf16>
    %960 = stablehlo.convert %959 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %961 = stablehlo.reduce(%960 init: %cst_155) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %962 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %963 = stablehlo.convert %962 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %964 = stablehlo.broadcast_in_dim %963, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %965 = stablehlo.divide %959, %964 : tensor<1x8x8x8xbf16>
    %966 = stablehlo.reshape %965 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %967 = stablehlo.dot_general %855, %966, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %968 = stablehlo.transpose %967, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %969 = stablehlo.reshape %968 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %970 = stablehlo.dot_general %969, %arg182, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %971 = chlo.square %970 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %972 = stablehlo.convert %971 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %973 = stablehlo.reduce(%972 init: %cst_156) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %974 = stablehlo.broadcast_in_dim %973, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_157 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %975 = stablehlo.broadcast_in_dim %cst_157, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %976 = stablehlo.divide %974, %975 : tensor<1x8x1xf32>
    %977 = stablehlo.convert %976 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_158 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %978 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %979 = stablehlo.add %977, %978 : tensor<1x8x1xbf16>
    %980 = stablehlo.rsqrt %979 : tensor<1x8x1xbf16>
    %981 = stablehlo.broadcast_in_dim %980, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %982 = stablehlo.multiply %970, %981 : tensor<1x8x2304xbf16>
    %983 = stablehlo.broadcast_in_dim %arg187, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_159 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %984 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %985 = stablehlo.add %984, %983 : tensor<1x1x2304xbf16>
    %986 = stablehlo.broadcast_in_dim %985, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %987 = stablehlo.multiply %982, %986 : tensor<1x8x2304xbf16>
    %988 = stablehlo.add %987, %831 : tensor<1x8x2304xbf16>
    %989 = chlo.square %988 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %990 = stablehlo.convert %989 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %991 = stablehlo.reduce(%990 init: %cst_160) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %992 = stablehlo.broadcast_in_dim %991, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_161 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %993 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %994 = stablehlo.divide %992, %993 : tensor<1x8x1xf32>
    %995 = stablehlo.convert %994 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_162 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %996 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %997 = stablehlo.add %995, %996 : tensor<1x8x1xbf16>
    %998 = stablehlo.rsqrt %997 : tensor<1x8x1xbf16>
    %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1000 = stablehlo.multiply %988, %999 : tensor<1x8x2304xbf16>
    %1001 = stablehlo.broadcast_in_dim %arg190, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_163 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1002 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1003 = stablehlo.add %1002, %1001 : tensor<1x1x2304xbf16>
    %1004 = stablehlo.broadcast_in_dim %1003, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1005 = stablehlo.multiply %1000, %1004 : tensor<1x8x2304xbf16>
    %1006 = stablehlo.dot_general %1005, %arg185, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %1007 = stablehlo.slice %1006 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1008 = stablehlo.reshape %1007 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1009 = stablehlo.multiply %1008, %1008 : tensor<1x8x9216xbf16>
    %1010 = stablehlo.multiply %1009, %1008 : tensor<1x8x9216xbf16>
    %cst_164 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %1011 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1012 = stablehlo.multiply %1011, %1010 : tensor<1x8x9216xbf16>
    %1013 = stablehlo.add %1008, %1012 : tensor<1x8x9216xbf16>
    %cst_165 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %1014 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1015 = stablehlo.multiply %1014, %1013 : tensor<1x8x9216xbf16>
    %1016 = stablehlo.tanh %1015 : tensor<1x8x9216xbf16>
    %cst_166 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1017 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1018 = stablehlo.add %1017, %1016 : tensor<1x8x9216xbf16>
    %cst_167 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %1019 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1020 = stablehlo.multiply %1019, %1018 : tensor<1x8x9216xbf16>
    %1021 = stablehlo.multiply %1008, %1020 : tensor<1x8x9216xbf16>
    %1022 = stablehlo.slice %1006 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1023 = stablehlo.reshape %1022 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1024 = stablehlo.multiply %1021, %1023 : tensor<1x8x9216xbf16>
    %1025 = stablehlo.dot_general %1024, %arg186, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1026 = chlo.square %1025 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1027 = stablehlo.convert %1026 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1028 = stablehlo.reduce(%1027 init: %cst_168) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1029 = stablehlo.broadcast_in_dim %1028, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_169 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1030 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1031 = stablehlo.divide %1029, %1030 : tensor<1x8x1xf32>
    %1032 = stablehlo.convert %1031 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_170 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1033 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1034 = stablehlo.add %1032, %1033 : tensor<1x8x1xbf16>
    %1035 = stablehlo.rsqrt %1034 : tensor<1x8x1xbf16>
    %1036 = stablehlo.broadcast_in_dim %1035, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1037 = stablehlo.multiply %1025, %1036 : tensor<1x8x2304xbf16>
    %1038 = stablehlo.broadcast_in_dim %arg188, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_171 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1039 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1040 = stablehlo.add %1039, %1038 : tensor<1x1x2304xbf16>
    %1041 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1042 = stablehlo.multiply %1037, %1041 : tensor<1x8x2304xbf16>
    %1043 = stablehlo.add %1042, %988 : tensor<1x8x2304xbf16>
    %1044 = chlo.square %1043 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1045 = stablehlo.convert %1044 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1046 = stablehlo.reduce(%1045 init: %cst_172) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1047 = stablehlo.broadcast_in_dim %1046, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_173 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1048 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1049 = stablehlo.divide %1047, %1048 : tensor<1x8x1xf32>
    %1050 = stablehlo.convert %1049 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_174 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1051 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1052 = stablehlo.add %1050, %1051 : tensor<1x8x1xbf16>
    %1053 = stablehlo.rsqrt %1052 : tensor<1x8x1xbf16>
    %1054 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1055 = stablehlo.multiply %1043, %1054 : tensor<1x8x2304xbf16>
    %1056 = stablehlo.broadcast_in_dim %arg198, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_175 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1057 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1058 = stablehlo.add %1057, %1056 : tensor<1x1x2304xbf16>
    %1059 = stablehlo.broadcast_in_dim %1058, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1060 = stablehlo.multiply %1055, %1059 : tensor<1x8x2304xbf16>
    %1061 = stablehlo.dot_general %1060, %arg193, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1062 = stablehlo.dot_general %arg192, %1060, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %1063 = stablehlo.transpose %1062, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %1064 = stablehlo.slice %1063 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1065 = stablehlo.reshape %1064 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1066 = stablehlo.slice %1063 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1067 = stablehlo.reshape %1066 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1068 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_176 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1069 = stablehlo.broadcast_in_dim %cst_176, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1070 = stablehlo.multiply %1069, %1068 : tensor<128xf32>
    %cst_177 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1071 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1072 = stablehlo.power %1071, %1070 : tensor<128xf32>
    %1073 = call @_pad(%1072) : (tensor<128xf32>) -> tensor<128xf32>
    %1074 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1075 = stablehlo.broadcast_in_dim %1073, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1076 = stablehlo.convert %1074 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1077 = stablehlo.broadcast_in_dim %1076, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1078 = stablehlo.broadcast_in_dim %1075, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1079 = stablehlo.divide %1077, %1078 : tensor<1x8x128xf32>
    %1080 = stablehlo.broadcast_in_dim %1079, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_178 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1081 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1082 = stablehlo.divide %1080, %1081 : tensor<1x8x1x128xf32>
    %1083 = stablehlo.sine %1082 : tensor<1x8x1x128xf32>
    %1084 = stablehlo.cosine %1082 : tensor<1x8x1x128xf32>
    %1085 = stablehlo.slice %1061 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1086 = stablehlo.slice %1061 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1087 = stablehlo.convert %1085 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1088 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1089 = stablehlo.multiply %1087, %1088 : tensor<1x8x8x128xf32>
    %1090 = stablehlo.convert %1086 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1091 = stablehlo.broadcast_in_dim %1083, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1092 = stablehlo.multiply %1090, %1091 : tensor<1x8x8x128xf32>
    %1093 = stablehlo.subtract %1089, %1092 : tensor<1x8x8x128xf32>
    %1094 = stablehlo.convert %1086 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1095 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1096 = stablehlo.multiply %1094, %1095 : tensor<1x8x8x128xf32>
    %1097 = stablehlo.convert %1085 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1098 = stablehlo.broadcast_in_dim %1083, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1099 = stablehlo.multiply %1097, %1098 : tensor<1x8x8x128xf32>
    %1100 = stablehlo.add %1096, %1099 : tensor<1x8x8x128xf32>
    %1101 = stablehlo.concatenate %1093, %1100, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %1102 = stablehlo.convert %1101 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_179 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %1103 = stablehlo.broadcast_in_dim %cst_179, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %1104 = stablehlo.multiply %1102, %1103 : tensor<1x8x8x256xbf16>
    %1105 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_180 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1106 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1107 = stablehlo.multiply %1106, %1105 : tensor<128xf32>
    %cst_181 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1108 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1109 = stablehlo.power %1108, %1107 : tensor<128xf32>
    %1110 = call @_pad(%1109) : (tensor<128xf32>) -> tensor<128xf32>
    %1111 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1112 = stablehlo.broadcast_in_dim %1110, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1113 = stablehlo.convert %1111 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1114 = stablehlo.broadcast_in_dim %1113, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1115 = stablehlo.broadcast_in_dim %1112, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1116 = stablehlo.divide %1114, %1115 : tensor<1x8x128xf32>
    %1117 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_182 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1118 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1119 = stablehlo.divide %1117, %1118 : tensor<1x8x1x128xf32>
    %1120 = stablehlo.sine %1119 : tensor<1x8x1x128xf32>
    %1121 = stablehlo.cosine %1119 : tensor<1x8x1x128xf32>
    %1122 = stablehlo.slice %1065 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1123 = stablehlo.slice %1065 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1124 = stablehlo.convert %1122 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1125 = stablehlo.broadcast_in_dim %1121, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1126 = stablehlo.multiply %1124, %1125 : tensor<1x8x4x128xf32>
    %1127 = stablehlo.convert %1123 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1128 = stablehlo.broadcast_in_dim %1120, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1129 = stablehlo.multiply %1127, %1128 : tensor<1x8x4x128xf32>
    %1130 = stablehlo.subtract %1126, %1129 : tensor<1x8x4x128xf32>
    %1131 = stablehlo.convert %1123 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1132 = stablehlo.broadcast_in_dim %1121, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1133 = stablehlo.multiply %1131, %1132 : tensor<1x8x4x128xf32>
    %1134 = stablehlo.convert %1122 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1135 = stablehlo.broadcast_in_dim %1120, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1136 = stablehlo.multiply %1134, %1135 : tensor<1x8x4x128xf32>
    %1137 = stablehlo.add %1133, %1136 : tensor<1x8x4x128xf32>
    %1138 = stablehlo.concatenate %1130, %1137, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %1139 = stablehlo.convert %1138 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %1140 = stablehlo.reshape %1104 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1141 = stablehlo.dot_general %1139, %1140, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %1142 = stablehlo.transpose %1141, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1143 = stablehlo.reshape %1142 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_183 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1144 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1145 = stablehlo.divide %1143, %1144 : tensor<1x8x8x8xbf16>
    %1146 = stablehlo.tanh %1145 : tensor<1x8x8x8xbf16>
    %cst_184 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1147 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1148 = stablehlo.multiply %1146, %1147 : tensor<1x8x8x8xbf16>
    %1149 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_185 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1150 = call @_where(%1149, %1148, %cst_185) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_186 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1151 = stablehlo.reduce(%1150 init: %cst_186) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_187 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1152 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %1153 = stablehlo.maximum %1152, %1151 : tensor<1x8x8xbf16>
    %1154 = stablehlo.broadcast_in_dim %1153, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1156 = stablehlo.subtract %1150, %1155 : tensor<1x8x8x8xbf16>
    %1157 = stablehlo.exponential %1156 : tensor<1x8x8x8xbf16>
    %1158 = stablehlo.convert %1157 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_188 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1159 = stablehlo.reduce(%1158 init: %cst_188) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1160 = stablehlo.broadcast_in_dim %1159, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1161 = stablehlo.convert %1160 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %1162 = stablehlo.broadcast_in_dim %1161, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1163 = stablehlo.divide %1157, %1162 : tensor<1x8x8x8xbf16>
    %1164 = stablehlo.reshape %1163 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1165 = stablehlo.dot_general %1067, %1164, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %1166 = stablehlo.transpose %1165, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1167 = stablehlo.reshape %1166 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1168 = stablehlo.dot_general %1167, %arg191, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1169 = chlo.square %1168 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1170 = stablehlo.convert %1169 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1171 = stablehlo.reduce(%1170 init: %cst_189) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1172 = stablehlo.broadcast_in_dim %1171, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_190 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1173 = stablehlo.broadcast_in_dim %cst_190, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1174 = stablehlo.divide %1172, %1173 : tensor<1x8x1xf32>
    %1175 = stablehlo.convert %1174 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_191 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1176 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1177 = stablehlo.add %1175, %1176 : tensor<1x8x1xbf16>
    %1178 = stablehlo.rsqrt %1177 : tensor<1x8x1xbf16>
    %1179 = stablehlo.broadcast_in_dim %1178, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1180 = stablehlo.multiply %1168, %1179 : tensor<1x8x2304xbf16>
    %1181 = stablehlo.broadcast_in_dim %arg196, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_192 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1182 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1183 = stablehlo.add %1182, %1181 : tensor<1x1x2304xbf16>
    %1184 = stablehlo.broadcast_in_dim %1183, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1185 = stablehlo.multiply %1180, %1184 : tensor<1x8x2304xbf16>
    %1186 = stablehlo.add %1185, %1043 : tensor<1x8x2304xbf16>
    %1187 = chlo.square %1186 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1188 = stablehlo.convert %1187 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1189 = stablehlo.reduce(%1188 init: %cst_193) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1190 = stablehlo.broadcast_in_dim %1189, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_194 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1191 = stablehlo.broadcast_in_dim %cst_194, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1192 = stablehlo.divide %1190, %1191 : tensor<1x8x1xf32>
    %1193 = stablehlo.convert %1192 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_195 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1194 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1195 = stablehlo.add %1193, %1194 : tensor<1x8x1xbf16>
    %1196 = stablehlo.rsqrt %1195 : tensor<1x8x1xbf16>
    %1197 = stablehlo.broadcast_in_dim %1196, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1198 = stablehlo.multiply %1186, %1197 : tensor<1x8x2304xbf16>
    %1199 = stablehlo.broadcast_in_dim %arg199, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_196 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1200 = stablehlo.broadcast_in_dim %cst_196, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1201 = stablehlo.add %1200, %1199 : tensor<1x1x2304xbf16>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1203 = stablehlo.multiply %1198, %1202 : tensor<1x8x2304xbf16>
    %1204 = stablehlo.dot_general %1203, %arg194, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %1205 = stablehlo.slice %1204 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1206 = stablehlo.reshape %1205 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1207 = stablehlo.multiply %1206, %1206 : tensor<1x8x9216xbf16>
    %1208 = stablehlo.multiply %1207, %1206 : tensor<1x8x9216xbf16>
    %cst_197 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %1209 = stablehlo.broadcast_in_dim %cst_197, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1210 = stablehlo.multiply %1209, %1208 : tensor<1x8x9216xbf16>
    %1211 = stablehlo.add %1206, %1210 : tensor<1x8x9216xbf16>
    %cst_198 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %1212 = stablehlo.broadcast_in_dim %cst_198, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1213 = stablehlo.multiply %1212, %1211 : tensor<1x8x9216xbf16>
    %1214 = stablehlo.tanh %1213 : tensor<1x8x9216xbf16>
    %cst_199 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1215 = stablehlo.broadcast_in_dim %cst_199, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1216 = stablehlo.add %1215, %1214 : tensor<1x8x9216xbf16>
    %cst_200 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %1217 = stablehlo.broadcast_in_dim %cst_200, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1218 = stablehlo.multiply %1217, %1216 : tensor<1x8x9216xbf16>
    %1219 = stablehlo.multiply %1206, %1218 : tensor<1x8x9216xbf16>
    %1220 = stablehlo.slice %1204 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1221 = stablehlo.reshape %1220 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1222 = stablehlo.multiply %1219, %1221 : tensor<1x8x9216xbf16>
    %1223 = stablehlo.dot_general %1222, %arg195, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1224 = chlo.square %1223 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1225 = stablehlo.convert %1224 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1226 = stablehlo.reduce(%1225 init: %cst_201) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1227 = stablehlo.broadcast_in_dim %1226, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_202 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1228 = stablehlo.broadcast_in_dim %cst_202, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1229 = stablehlo.divide %1227, %1228 : tensor<1x8x1xf32>
    %1230 = stablehlo.convert %1229 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_203 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1231 = stablehlo.broadcast_in_dim %cst_203, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1232 = stablehlo.add %1230, %1231 : tensor<1x8x1xbf16>
    %1233 = stablehlo.rsqrt %1232 : tensor<1x8x1xbf16>
    %1234 = stablehlo.broadcast_in_dim %1233, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1235 = stablehlo.multiply %1223, %1234 : tensor<1x8x2304xbf16>
    %1236 = stablehlo.broadcast_in_dim %arg197, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_204 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1237 = stablehlo.broadcast_in_dim %cst_204, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1238 = stablehlo.add %1237, %1236 : tensor<1x1x2304xbf16>
    %1239 = stablehlo.broadcast_in_dim %1238, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1240 = stablehlo.multiply %1235, %1239 : tensor<1x8x2304xbf16>
    %1241 = stablehlo.add %1240, %1186 : tensor<1x8x2304xbf16>
    %1242 = chlo.square %1241 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1243 = stablehlo.convert %1242 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1244 = stablehlo.reduce(%1243 init: %cst_205) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1245 = stablehlo.broadcast_in_dim %1244, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_206 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1246 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1247 = stablehlo.divide %1245, %1246 : tensor<1x8x1xf32>
    %1248 = stablehlo.convert %1247 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_207 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1249 = stablehlo.broadcast_in_dim %cst_207, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1250 = stablehlo.add %1248, %1249 : tensor<1x8x1xbf16>
    %1251 = stablehlo.rsqrt %1250 : tensor<1x8x1xbf16>
    %1252 = stablehlo.broadcast_in_dim %1251, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1253 = stablehlo.multiply %1241, %1252 : tensor<1x8x2304xbf16>
    %1254 = stablehlo.broadcast_in_dim %arg207, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_208 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1255 = stablehlo.broadcast_in_dim %cst_208, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1256 = stablehlo.add %1255, %1254 : tensor<1x1x2304xbf16>
    %1257 = stablehlo.broadcast_in_dim %1256, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1258 = stablehlo.multiply %1253, %1257 : tensor<1x8x2304xbf16>
    %1259 = stablehlo.dot_general %1258, %arg202, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1260 = stablehlo.dot_general %arg201, %1258, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %1261 = stablehlo.transpose %1260, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %1262 = stablehlo.slice %1261 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1263 = stablehlo.reshape %1262 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1264 = stablehlo.slice %1261 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1265 = stablehlo.reshape %1264 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1266 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_209 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1267 = stablehlo.broadcast_in_dim %cst_209, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1268 = stablehlo.multiply %1267, %1266 : tensor<128xf32>
    %cst_210 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1269 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1270 = stablehlo.power %1269, %1268 : tensor<128xf32>
    %1271 = call @_pad(%1270) : (tensor<128xf32>) -> tensor<128xf32>
    %1272 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1273 = stablehlo.broadcast_in_dim %1271, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1274 = stablehlo.convert %1272 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1275 = stablehlo.broadcast_in_dim %1274, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1276 = stablehlo.broadcast_in_dim %1273, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1277 = stablehlo.divide %1275, %1276 : tensor<1x8x128xf32>
    %1278 = stablehlo.broadcast_in_dim %1277, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_211 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1279 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1280 = stablehlo.divide %1278, %1279 : tensor<1x8x1x128xf32>
    %1281 = stablehlo.sine %1280 : tensor<1x8x1x128xf32>
    %1282 = stablehlo.cosine %1280 : tensor<1x8x1x128xf32>
    %1283 = stablehlo.slice %1259 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1284 = stablehlo.slice %1259 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1285 = stablehlo.convert %1283 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1286 = stablehlo.broadcast_in_dim %1282, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1287 = stablehlo.multiply %1285, %1286 : tensor<1x8x8x128xf32>
    %1288 = stablehlo.convert %1284 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1289 = stablehlo.broadcast_in_dim %1281, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1290 = stablehlo.multiply %1288, %1289 : tensor<1x8x8x128xf32>
    %1291 = stablehlo.subtract %1287, %1290 : tensor<1x8x8x128xf32>
    %1292 = stablehlo.convert %1284 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1293 = stablehlo.broadcast_in_dim %1282, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1294 = stablehlo.multiply %1292, %1293 : tensor<1x8x8x128xf32>
    %1295 = stablehlo.convert %1283 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1296 = stablehlo.broadcast_in_dim %1281, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1297 = stablehlo.multiply %1295, %1296 : tensor<1x8x8x128xf32>
    %1298 = stablehlo.add %1294, %1297 : tensor<1x8x8x128xf32>
    %1299 = stablehlo.concatenate %1291, %1298, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %1300 = stablehlo.convert %1299 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_212 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %1301 = stablehlo.broadcast_in_dim %cst_212, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %1302 = stablehlo.multiply %1300, %1301 : tensor<1x8x8x256xbf16>
    %1303 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_213 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1304 = stablehlo.broadcast_in_dim %cst_213, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1305 = stablehlo.multiply %1304, %1303 : tensor<128xf32>
    %cst_214 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1306 = stablehlo.broadcast_in_dim %cst_214, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1307 = stablehlo.power %1306, %1305 : tensor<128xf32>
    %1308 = call @_pad(%1307) : (tensor<128xf32>) -> tensor<128xf32>
    %1309 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1310 = stablehlo.broadcast_in_dim %1308, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1311 = stablehlo.convert %1309 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1312 = stablehlo.broadcast_in_dim %1311, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1313 = stablehlo.broadcast_in_dim %1310, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1314 = stablehlo.divide %1312, %1313 : tensor<1x8x128xf32>
    %1315 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_215 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1316 = stablehlo.broadcast_in_dim %cst_215, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1317 = stablehlo.divide %1315, %1316 : tensor<1x8x1x128xf32>
    %1318 = stablehlo.sine %1317 : tensor<1x8x1x128xf32>
    %1319 = stablehlo.cosine %1317 : tensor<1x8x1x128xf32>
    %1320 = stablehlo.slice %1263 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1321 = stablehlo.slice %1263 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1322 = stablehlo.convert %1320 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1323 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1324 = stablehlo.multiply %1322, %1323 : tensor<1x8x4x128xf32>
    %1325 = stablehlo.convert %1321 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1326 = stablehlo.broadcast_in_dim %1318, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1327 = stablehlo.multiply %1325, %1326 : tensor<1x8x4x128xf32>
    %1328 = stablehlo.subtract %1324, %1327 : tensor<1x8x4x128xf32>
    %1329 = stablehlo.convert %1321 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1330 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1331 = stablehlo.multiply %1329, %1330 : tensor<1x8x4x128xf32>
    %1332 = stablehlo.convert %1320 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1333 = stablehlo.broadcast_in_dim %1318, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1334 = stablehlo.multiply %1332, %1333 : tensor<1x8x4x128xf32>
    %1335 = stablehlo.add %1331, %1334 : tensor<1x8x4x128xf32>
    %1336 = stablehlo.concatenate %1328, %1335, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %1337 = stablehlo.convert %1336 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %1338 = stablehlo.reshape %1302 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1339 = stablehlo.dot_general %1337, %1338, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %1340 = stablehlo.transpose %1339, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1341 = stablehlo.reshape %1340 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_216 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1342 = stablehlo.broadcast_in_dim %cst_216, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1343 = stablehlo.divide %1341, %1342 : tensor<1x8x8x8xbf16>
    %1344 = stablehlo.tanh %1343 : tensor<1x8x8x8xbf16>
    %cst_217 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1345 = stablehlo.broadcast_in_dim %cst_217, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1346 = stablehlo.multiply %1344, %1345 : tensor<1x8x8x8xbf16>
    %1347 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %1348 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_218 = stablehlo.constant dense<4096> : tensor<i32>
    %1349 = stablehlo.broadcast_in_dim %c_218, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %1350 = stablehlo.subtract %1348, %1349 : tensor<1x8x1xi32>
    %1351 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %1352 = stablehlo.broadcast_in_dim %1350, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %1353 = stablehlo.compare GT, %1351, %1352, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_219 = stablehlo.constant dense<4096> : tensor<i32>
    %1354 = stablehlo.broadcast_in_dim %c_219, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %1355 = stablehlo.add %1348, %1354 : tensor<1x8x1xi32>
    %1356 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %1357 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %1358 = stablehlo.compare LT, %1356, %1357, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %1359 = stablehlo.and %1353, %1358 : tensor<1x8x8xi1>
    %1360 = stablehlo.and %arg237, %1359 : tensor<1x8x8xi1>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_220 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1362 = call @_where(%1361, %1346, %cst_220) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_221 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1363 = stablehlo.reduce(%1362 init: %cst_221) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_222 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1364 = stablehlo.broadcast_in_dim %cst_222, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %1365 = stablehlo.maximum %1364, %1363 : tensor<1x8x8xbf16>
    %1366 = stablehlo.broadcast_in_dim %1365, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %1367 = stablehlo.broadcast_in_dim %1366, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1368 = stablehlo.subtract %1362, %1367 : tensor<1x8x8x8xbf16>
    %1369 = stablehlo.exponential %1368 : tensor<1x8x8x8xbf16>
    %1370 = stablehlo.convert %1369 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1371 = stablehlo.reduce(%1370 init: %cst_223) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1372 = stablehlo.broadcast_in_dim %1371, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1373 = stablehlo.convert %1372 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %1374 = stablehlo.broadcast_in_dim %1373, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1375 = stablehlo.divide %1369, %1374 : tensor<1x8x8x8xbf16>
    %1376 = stablehlo.reshape %1375 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1377 = stablehlo.dot_general %1265, %1376, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %1378 = stablehlo.transpose %1377, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1379 = stablehlo.reshape %1378 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1380 = stablehlo.dot_general %1379, %arg200, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1381 = chlo.square %1380 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1382 = stablehlo.convert %1381 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_224 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1383 = stablehlo.reduce(%1382 init: %cst_224) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1384 = stablehlo.broadcast_in_dim %1383, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_225 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1385 = stablehlo.broadcast_in_dim %cst_225, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1386 = stablehlo.divide %1384, %1385 : tensor<1x8x1xf32>
    %1387 = stablehlo.convert %1386 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_226 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1388 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1389 = stablehlo.add %1387, %1388 : tensor<1x8x1xbf16>
    %1390 = stablehlo.rsqrt %1389 : tensor<1x8x1xbf16>
    %1391 = stablehlo.broadcast_in_dim %1390, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1392 = stablehlo.multiply %1380, %1391 : tensor<1x8x2304xbf16>
    %1393 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_227 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1394 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1395 = stablehlo.add %1394, %1393 : tensor<1x1x2304xbf16>
    %1396 = stablehlo.broadcast_in_dim %1395, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1397 = stablehlo.multiply %1392, %1396 : tensor<1x8x2304xbf16>
    %1398 = stablehlo.add %1397, %1241 : tensor<1x8x2304xbf16>
    %1399 = chlo.square %1398 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1400 = stablehlo.convert %1399 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1401 = stablehlo.reduce(%1400 init: %cst_228) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1402 = stablehlo.broadcast_in_dim %1401, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_229 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1403 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1404 = stablehlo.divide %1402, %1403 : tensor<1x8x1xf32>
    %1405 = stablehlo.convert %1404 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_230 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1406 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1407 = stablehlo.add %1405, %1406 : tensor<1x8x1xbf16>
    %1408 = stablehlo.rsqrt %1407 : tensor<1x8x1xbf16>
    %1409 = stablehlo.broadcast_in_dim %1408, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1410 = stablehlo.multiply %1398, %1409 : tensor<1x8x2304xbf16>
    %1411 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_231 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1412 = stablehlo.broadcast_in_dim %cst_231, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1413 = stablehlo.add %1412, %1411 : tensor<1x1x2304xbf16>
    %1414 = stablehlo.broadcast_in_dim %1413, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1415 = stablehlo.multiply %1410, %1414 : tensor<1x8x2304xbf16>
    %1416 = stablehlo.dot_general %1415, %arg203, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %1417 = stablehlo.slice %1416 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1418 = stablehlo.reshape %1417 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1419 = stablehlo.multiply %1418, %1418 : tensor<1x8x9216xbf16>
    %1420 = stablehlo.multiply %1419, %1418 : tensor<1x8x9216xbf16>
    %cst_232 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %1421 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1422 = stablehlo.multiply %1421, %1420 : tensor<1x8x9216xbf16>
    %1423 = stablehlo.add %1418, %1422 : tensor<1x8x9216xbf16>
    %cst_233 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %1424 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1425 = stablehlo.multiply %1424, %1423 : tensor<1x8x9216xbf16>
    %1426 = stablehlo.tanh %1425 : tensor<1x8x9216xbf16>
    %cst_234 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1427 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1428 = stablehlo.add %1427, %1426 : tensor<1x8x9216xbf16>
    %cst_235 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %1429 = stablehlo.broadcast_in_dim %cst_235, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1430 = stablehlo.multiply %1429, %1428 : tensor<1x8x9216xbf16>
    %1431 = stablehlo.multiply %1418, %1430 : tensor<1x8x9216xbf16>
    %1432 = stablehlo.slice %1416 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1433 = stablehlo.reshape %1432 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1434 = stablehlo.multiply %1431, %1433 : tensor<1x8x9216xbf16>
    %1435 = stablehlo.dot_general %1434, %arg204, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1436 = chlo.square %1435 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1437 = stablehlo.convert %1436 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_236 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1438 = stablehlo.reduce(%1437 init: %cst_236) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1439 = stablehlo.broadcast_in_dim %1438, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_237 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1440 = stablehlo.broadcast_in_dim %cst_237, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1441 = stablehlo.divide %1439, %1440 : tensor<1x8x1xf32>
    %1442 = stablehlo.convert %1441 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_238 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1443 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1444 = stablehlo.add %1442, %1443 : tensor<1x8x1xbf16>
    %1445 = stablehlo.rsqrt %1444 : tensor<1x8x1xbf16>
    %1446 = stablehlo.broadcast_in_dim %1445, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1447 = stablehlo.multiply %1435, %1446 : tensor<1x8x2304xbf16>
    %1448 = stablehlo.broadcast_in_dim %arg206, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_239 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1449 = stablehlo.broadcast_in_dim %cst_239, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1450 = stablehlo.add %1449, %1448 : tensor<1x1x2304xbf16>
    %1451 = stablehlo.broadcast_in_dim %1450, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1452 = stablehlo.multiply %1447, %1451 : tensor<1x8x2304xbf16>
    %1453 = stablehlo.add %1452, %1398 : tensor<1x8x2304xbf16>
    %1454 = chlo.square %1453 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1455 = stablehlo.convert %1454 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1456 = stablehlo.reduce(%1455 init: %cst_240) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1457 = stablehlo.broadcast_in_dim %1456, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_241 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1458 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1459 = stablehlo.divide %1457, %1458 : tensor<1x8x1xf32>
    %1460 = stablehlo.convert %1459 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_242 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1461 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1462 = stablehlo.add %1460, %1461 : tensor<1x8x1xbf16>
    %1463 = stablehlo.rsqrt %1462 : tensor<1x8x1xbf16>
    %1464 = stablehlo.broadcast_in_dim %1463, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1465 = stablehlo.multiply %1453, %1464 : tensor<1x8x2304xbf16>
    %1466 = stablehlo.broadcast_in_dim %arg216, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_243 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1467 = stablehlo.broadcast_in_dim %cst_243, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1468 = stablehlo.add %1467, %1466 : tensor<1x1x2304xbf16>
    %1469 = stablehlo.broadcast_in_dim %1468, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1470 = stablehlo.multiply %1465, %1469 : tensor<1x8x2304xbf16>
    %1471 = stablehlo.dot_general %1470, %arg211, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1472 = stablehlo.dot_general %arg210, %1470, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %1473 = stablehlo.transpose %1472, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %1474 = stablehlo.slice %1473 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1475 = stablehlo.reshape %1474 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1476 = stablehlo.slice %1473 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1477 = stablehlo.reshape %1476 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1478 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_244 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1479 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1480 = stablehlo.multiply %1479, %1478 : tensor<128xf32>
    %cst_245 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1481 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1482 = stablehlo.power %1481, %1480 : tensor<128xf32>
    %1483 = call @_pad(%1482) : (tensor<128xf32>) -> tensor<128xf32>
    %1484 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1485 = stablehlo.broadcast_in_dim %1483, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1486 = stablehlo.convert %1484 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1487 = stablehlo.broadcast_in_dim %1486, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1488 = stablehlo.broadcast_in_dim %1485, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1489 = stablehlo.divide %1487, %1488 : tensor<1x8x128xf32>
    %1490 = stablehlo.broadcast_in_dim %1489, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_246 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1491 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1492 = stablehlo.divide %1490, %1491 : tensor<1x8x1x128xf32>
    %1493 = stablehlo.sine %1492 : tensor<1x8x1x128xf32>
    %1494 = stablehlo.cosine %1492 : tensor<1x8x1x128xf32>
    %1495 = stablehlo.slice %1471 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1496 = stablehlo.slice %1471 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1497 = stablehlo.convert %1495 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1498 = stablehlo.broadcast_in_dim %1494, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1499 = stablehlo.multiply %1497, %1498 : tensor<1x8x8x128xf32>
    %1500 = stablehlo.convert %1496 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1501 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1502 = stablehlo.multiply %1500, %1501 : tensor<1x8x8x128xf32>
    %1503 = stablehlo.subtract %1499, %1502 : tensor<1x8x8x128xf32>
    %1504 = stablehlo.convert %1496 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1505 = stablehlo.broadcast_in_dim %1494, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1506 = stablehlo.multiply %1504, %1505 : tensor<1x8x8x128xf32>
    %1507 = stablehlo.convert %1495 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1508 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1509 = stablehlo.multiply %1507, %1508 : tensor<1x8x8x128xf32>
    %1510 = stablehlo.add %1506, %1509 : tensor<1x8x8x128xf32>
    %1511 = stablehlo.concatenate %1503, %1510, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %1512 = stablehlo.convert %1511 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_247 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %1513 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %1514 = stablehlo.multiply %1512, %1513 : tensor<1x8x8x256xbf16>
    %1515 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_248 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1516 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1517 = stablehlo.multiply %1516, %1515 : tensor<128xf32>
    %cst_249 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1518 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1519 = stablehlo.power %1518, %1517 : tensor<128xf32>
    %1520 = call @_pad(%1519) : (tensor<128xf32>) -> tensor<128xf32>
    %1521 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1522 = stablehlo.broadcast_in_dim %1520, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1523 = stablehlo.convert %1521 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1524 = stablehlo.broadcast_in_dim %1523, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1525 = stablehlo.broadcast_in_dim %1522, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1526 = stablehlo.divide %1524, %1525 : tensor<1x8x128xf32>
    %1527 = stablehlo.broadcast_in_dim %1526, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_250 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1528 = stablehlo.broadcast_in_dim %cst_250, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1529 = stablehlo.divide %1527, %1528 : tensor<1x8x1x128xf32>
    %1530 = stablehlo.sine %1529 : tensor<1x8x1x128xf32>
    %1531 = stablehlo.cosine %1529 : tensor<1x8x1x128xf32>
    %1532 = stablehlo.slice %1475 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1533 = stablehlo.slice %1475 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1534 = stablehlo.convert %1532 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1535 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1536 = stablehlo.multiply %1534, %1535 : tensor<1x8x4x128xf32>
    %1537 = stablehlo.convert %1533 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1538 = stablehlo.broadcast_in_dim %1530, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1539 = stablehlo.multiply %1537, %1538 : tensor<1x8x4x128xf32>
    %1540 = stablehlo.subtract %1536, %1539 : tensor<1x8x4x128xf32>
    %1541 = stablehlo.convert %1533 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1542 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1543 = stablehlo.multiply %1541, %1542 : tensor<1x8x4x128xf32>
    %1544 = stablehlo.convert %1532 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1545 = stablehlo.broadcast_in_dim %1530, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1546 = stablehlo.multiply %1544, %1545 : tensor<1x8x4x128xf32>
    %1547 = stablehlo.add %1543, %1546 : tensor<1x8x4x128xf32>
    %1548 = stablehlo.concatenate %1540, %1547, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %1549 = stablehlo.convert %1548 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %1550 = stablehlo.reshape %1514 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1551 = stablehlo.dot_general %1549, %1550, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %1552 = stablehlo.transpose %1551, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1553 = stablehlo.reshape %1552 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_251 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1554 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1555 = stablehlo.divide %1553, %1554 : tensor<1x8x8x8xbf16>
    %1556 = stablehlo.tanh %1555 : tensor<1x8x8x8xbf16>
    %cst_252 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1557 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1558 = stablehlo.multiply %1556, %1557 : tensor<1x8x8x8xbf16>
    %1559 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_253 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1560 = call @_where(%1559, %1558, %cst_253) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_254 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1561 = stablehlo.reduce(%1560 init: %cst_254) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_255 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1562 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %1563 = stablehlo.maximum %1562, %1561 : tensor<1x8x8xbf16>
    %1564 = stablehlo.broadcast_in_dim %1563, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %1565 = stablehlo.broadcast_in_dim %1564, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1566 = stablehlo.subtract %1560, %1565 : tensor<1x8x8x8xbf16>
    %1567 = stablehlo.exponential %1566 : tensor<1x8x8x8xbf16>
    %1568 = stablehlo.convert %1567 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_256 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1569 = stablehlo.reduce(%1568 init: %cst_256) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1570 = stablehlo.broadcast_in_dim %1569, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1571 = stablehlo.convert %1570 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %1572 = stablehlo.broadcast_in_dim %1571, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1573 = stablehlo.divide %1567, %1572 : tensor<1x8x8x8xbf16>
    %1574 = stablehlo.reshape %1573 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1575 = stablehlo.dot_general %1477, %1574, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %1576 = stablehlo.transpose %1575, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1577 = stablehlo.reshape %1576 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1578 = stablehlo.dot_general %1577, %arg209, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1579 = chlo.square %1578 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1580 = stablehlo.convert %1579 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_257 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1581 = stablehlo.reduce(%1580 init: %cst_257) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1582 = stablehlo.broadcast_in_dim %1581, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_258 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1583 = stablehlo.broadcast_in_dim %cst_258, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1584 = stablehlo.divide %1582, %1583 : tensor<1x8x1xf32>
    %1585 = stablehlo.convert %1584 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_259 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1586 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1587 = stablehlo.add %1585, %1586 : tensor<1x8x1xbf16>
    %1588 = stablehlo.rsqrt %1587 : tensor<1x8x1xbf16>
    %1589 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1590 = stablehlo.multiply %1578, %1589 : tensor<1x8x2304xbf16>
    %1591 = stablehlo.broadcast_in_dim %arg214, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_260 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1592 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1593 = stablehlo.add %1592, %1591 : tensor<1x1x2304xbf16>
    %1594 = stablehlo.broadcast_in_dim %1593, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1595 = stablehlo.multiply %1590, %1594 : tensor<1x8x2304xbf16>
    %1596 = stablehlo.add %1595, %1453 : tensor<1x8x2304xbf16>
    %1597 = chlo.square %1596 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1598 = stablehlo.convert %1597 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_261 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1599 = stablehlo.reduce(%1598 init: %cst_261) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1600 = stablehlo.broadcast_in_dim %1599, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_262 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1601 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1602 = stablehlo.divide %1600, %1601 : tensor<1x8x1xf32>
    %1603 = stablehlo.convert %1602 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_263 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1604 = stablehlo.broadcast_in_dim %cst_263, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1605 = stablehlo.add %1603, %1604 : tensor<1x8x1xbf16>
    %1606 = stablehlo.rsqrt %1605 : tensor<1x8x1xbf16>
    %1607 = stablehlo.broadcast_in_dim %1606, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1608 = stablehlo.multiply %1596, %1607 : tensor<1x8x2304xbf16>
    %1609 = stablehlo.broadcast_in_dim %arg217, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_264 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1610 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1611 = stablehlo.add %1610, %1609 : tensor<1x1x2304xbf16>
    %1612 = stablehlo.broadcast_in_dim %1611, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1613 = stablehlo.multiply %1608, %1612 : tensor<1x8x2304xbf16>
    %1614 = stablehlo.dot_general %1613, %arg212, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %1615 = stablehlo.slice %1614 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1616 = stablehlo.reshape %1615 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1617 = stablehlo.multiply %1616, %1616 : tensor<1x8x9216xbf16>
    %1618 = stablehlo.multiply %1617, %1616 : tensor<1x8x9216xbf16>
    %cst_265 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %1619 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1620 = stablehlo.multiply %1619, %1618 : tensor<1x8x9216xbf16>
    %1621 = stablehlo.add %1616, %1620 : tensor<1x8x9216xbf16>
    %cst_266 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %1622 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1623 = stablehlo.multiply %1622, %1621 : tensor<1x8x9216xbf16>
    %1624 = stablehlo.tanh %1623 : tensor<1x8x9216xbf16>
    %cst_267 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1625 = stablehlo.broadcast_in_dim %cst_267, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1626 = stablehlo.add %1625, %1624 : tensor<1x8x9216xbf16>
    %cst_268 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %1627 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1628 = stablehlo.multiply %1627, %1626 : tensor<1x8x9216xbf16>
    %1629 = stablehlo.multiply %1616, %1628 : tensor<1x8x9216xbf16>
    %1630 = stablehlo.slice %1614 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1631 = stablehlo.reshape %1630 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1632 = stablehlo.multiply %1629, %1631 : tensor<1x8x9216xbf16>
    %1633 = stablehlo.dot_general %1632, %arg213, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1634 = chlo.square %1633 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1635 = stablehlo.convert %1634 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1636 = stablehlo.reduce(%1635 init: %cst_269) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1637 = stablehlo.broadcast_in_dim %1636, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_270 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1638 = stablehlo.broadcast_in_dim %cst_270, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1639 = stablehlo.divide %1637, %1638 : tensor<1x8x1xf32>
    %1640 = stablehlo.convert %1639 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_271 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1641 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1642 = stablehlo.add %1640, %1641 : tensor<1x8x1xbf16>
    %1643 = stablehlo.rsqrt %1642 : tensor<1x8x1xbf16>
    %1644 = stablehlo.broadcast_in_dim %1643, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1645 = stablehlo.multiply %1633, %1644 : tensor<1x8x2304xbf16>
    %1646 = stablehlo.broadcast_in_dim %arg215, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_272 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1647 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1648 = stablehlo.add %1647, %1646 : tensor<1x1x2304xbf16>
    %1649 = stablehlo.broadcast_in_dim %1648, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1650 = stablehlo.multiply %1645, %1649 : tensor<1x8x2304xbf16>
    %1651 = stablehlo.add %1650, %1596 : tensor<1x8x2304xbf16>
    %1652 = chlo.square %1651 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1653 = stablehlo.convert %1652 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_273 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1654 = stablehlo.reduce(%1653 init: %cst_273) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1655 = stablehlo.broadcast_in_dim %1654, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_274 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1656 = stablehlo.broadcast_in_dim %cst_274, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1657 = stablehlo.divide %1655, %1656 : tensor<1x8x1xf32>
    %1658 = stablehlo.convert %1657 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_275 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1659 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1660 = stablehlo.add %1658, %1659 : tensor<1x8x1xbf16>
    %1661 = stablehlo.rsqrt %1660 : tensor<1x8x1xbf16>
    %1662 = stablehlo.broadcast_in_dim %1661, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1663 = stablehlo.multiply %1651, %1662 : tensor<1x8x2304xbf16>
    %1664 = stablehlo.broadcast_in_dim %arg225, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_276 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1665 = stablehlo.broadcast_in_dim %cst_276, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1666 = stablehlo.add %1665, %1664 : tensor<1x1x2304xbf16>
    %1667 = stablehlo.broadcast_in_dim %1666, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1668 = stablehlo.multiply %1663, %1667 : tensor<1x8x2304xbf16>
    %1669 = stablehlo.dot_general %1668, %arg220, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1670 = stablehlo.dot_general %arg219, %1668, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %1671 = stablehlo.transpose %1670, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %1672 = stablehlo.slice %1671 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1673 = stablehlo.reshape %1672 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1674 = stablehlo.slice %1671 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1675 = stablehlo.reshape %1674 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1676 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_277 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1677 = stablehlo.broadcast_in_dim %cst_277, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1678 = stablehlo.multiply %1677, %1676 : tensor<128xf32>
    %cst_278 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1679 = stablehlo.broadcast_in_dim %cst_278, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1680 = stablehlo.power %1679, %1678 : tensor<128xf32>
    %1681 = call @_pad(%1680) : (tensor<128xf32>) -> tensor<128xf32>
    %1682 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1683 = stablehlo.broadcast_in_dim %1681, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1684 = stablehlo.convert %1682 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1685 = stablehlo.broadcast_in_dim %1684, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1686 = stablehlo.broadcast_in_dim %1683, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1687 = stablehlo.divide %1685, %1686 : tensor<1x8x128xf32>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_279 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1689 = stablehlo.broadcast_in_dim %cst_279, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1690 = stablehlo.divide %1688, %1689 : tensor<1x8x1x128xf32>
    %1691 = stablehlo.sine %1690 : tensor<1x8x1x128xf32>
    %1692 = stablehlo.cosine %1690 : tensor<1x8x1x128xf32>
    %1693 = stablehlo.slice %1669 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1694 = stablehlo.slice %1669 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1695 = stablehlo.convert %1693 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1696 = stablehlo.broadcast_in_dim %1692, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1697 = stablehlo.multiply %1695, %1696 : tensor<1x8x8x128xf32>
    %1698 = stablehlo.convert %1694 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1699 = stablehlo.broadcast_in_dim %1691, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1700 = stablehlo.multiply %1698, %1699 : tensor<1x8x8x128xf32>
    %1701 = stablehlo.subtract %1697, %1700 : tensor<1x8x8x128xf32>
    %1702 = stablehlo.convert %1694 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1703 = stablehlo.broadcast_in_dim %1692, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1704 = stablehlo.multiply %1702, %1703 : tensor<1x8x8x128xf32>
    %1705 = stablehlo.convert %1693 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1706 = stablehlo.broadcast_in_dim %1691, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1707 = stablehlo.multiply %1705, %1706 : tensor<1x8x8x128xf32>
    %1708 = stablehlo.add %1704, %1707 : tensor<1x8x8x128xf32>
    %1709 = stablehlo.concatenate %1701, %1708, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %1710 = stablehlo.convert %1709 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_280 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %1711 = stablehlo.broadcast_in_dim %cst_280, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %1712 = stablehlo.multiply %1710, %1711 : tensor<1x8x8x256xbf16>
    %1713 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_281 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1714 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1715 = stablehlo.multiply %1714, %1713 : tensor<128xf32>
    %cst_282 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1716 = stablehlo.broadcast_in_dim %cst_282, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1717 = stablehlo.power %1716, %1715 : tensor<128xf32>
    %1718 = call @_pad(%1717) : (tensor<128xf32>) -> tensor<128xf32>
    %1719 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1720 = stablehlo.broadcast_in_dim %1718, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1721 = stablehlo.convert %1719 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1722 = stablehlo.broadcast_in_dim %1721, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1723 = stablehlo.broadcast_in_dim %1720, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1724 = stablehlo.divide %1722, %1723 : tensor<1x8x128xf32>
    %1725 = stablehlo.broadcast_in_dim %1724, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_283 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1726 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1727 = stablehlo.divide %1725, %1726 : tensor<1x8x1x128xf32>
    %1728 = stablehlo.sine %1727 : tensor<1x8x1x128xf32>
    %1729 = stablehlo.cosine %1727 : tensor<1x8x1x128xf32>
    %1730 = stablehlo.slice %1673 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1731 = stablehlo.slice %1673 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1732 = stablehlo.convert %1730 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1733 = stablehlo.broadcast_in_dim %1729, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1734 = stablehlo.multiply %1732, %1733 : tensor<1x8x4x128xf32>
    %1735 = stablehlo.convert %1731 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1736 = stablehlo.broadcast_in_dim %1728, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1737 = stablehlo.multiply %1735, %1736 : tensor<1x8x4x128xf32>
    %1738 = stablehlo.subtract %1734, %1737 : tensor<1x8x4x128xf32>
    %1739 = stablehlo.convert %1731 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1740 = stablehlo.broadcast_in_dim %1729, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1741 = stablehlo.multiply %1739, %1740 : tensor<1x8x4x128xf32>
    %1742 = stablehlo.convert %1730 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1743 = stablehlo.broadcast_in_dim %1728, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1744 = stablehlo.multiply %1742, %1743 : tensor<1x8x4x128xf32>
    %1745 = stablehlo.add %1741, %1744 : tensor<1x8x4x128xf32>
    %1746 = stablehlo.concatenate %1738, %1745, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %1747 = stablehlo.convert %1746 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %1748 = stablehlo.reshape %1712 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1749 = stablehlo.dot_general %1747, %1748, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %1750 = stablehlo.transpose %1749, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1751 = stablehlo.reshape %1750 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_284 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1752 = stablehlo.broadcast_in_dim %cst_284, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1753 = stablehlo.divide %1751, %1752 : tensor<1x8x8x8xbf16>
    %1754 = stablehlo.tanh %1753 : tensor<1x8x8x8xbf16>
    %cst_285 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1755 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1756 = stablehlo.multiply %1754, %1755 : tensor<1x8x8x8xbf16>
    %1757 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %1758 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_286 = stablehlo.constant dense<4096> : tensor<i32>
    %1759 = stablehlo.broadcast_in_dim %c_286, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %1760 = stablehlo.subtract %1758, %1759 : tensor<1x8x1xi32>
    %1761 = stablehlo.broadcast_in_dim %1757, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %1762 = stablehlo.broadcast_in_dim %1760, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %1763 = stablehlo.compare GT, %1761, %1762, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_287 = stablehlo.constant dense<4096> : tensor<i32>
    %1764 = stablehlo.broadcast_in_dim %c_287, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %1765 = stablehlo.add %1758, %1764 : tensor<1x8x1xi32>
    %1766 = stablehlo.broadcast_in_dim %1757, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %1767 = stablehlo.broadcast_in_dim %1765, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %1768 = stablehlo.compare LT, %1766, %1767, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %1769 = stablehlo.and %1763, %1768 : tensor<1x8x8xi1>
    %1770 = stablehlo.and %arg237, %1769 : tensor<1x8x8xi1>
    %1771 = stablehlo.broadcast_in_dim %1770, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_288 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1772 = call @_where(%1771, %1756, %cst_288) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_289 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1773 = stablehlo.reduce(%1772 init: %cst_289) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_290 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1774 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %1775 = stablehlo.maximum %1774, %1773 : tensor<1x8x8xbf16>
    %1776 = stablehlo.broadcast_in_dim %1775, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %1777 = stablehlo.broadcast_in_dim %1776, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1778 = stablehlo.subtract %1772, %1777 : tensor<1x8x8x8xbf16>
    %1779 = stablehlo.exponential %1778 : tensor<1x8x8x8xbf16>
    %1780 = stablehlo.convert %1779 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_291 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1781 = stablehlo.reduce(%1780 init: %cst_291) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1782 = stablehlo.broadcast_in_dim %1781, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1783 = stablehlo.convert %1782 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %1784 = stablehlo.broadcast_in_dim %1783, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1785 = stablehlo.divide %1779, %1784 : tensor<1x8x8x8xbf16>
    %1786 = stablehlo.reshape %1785 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1787 = stablehlo.dot_general %1675, %1786, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %1788 = stablehlo.transpose %1787, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1789 = stablehlo.reshape %1788 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1790 = stablehlo.dot_general %1789, %arg218, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1791 = chlo.square %1790 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1792 = stablehlo.convert %1791 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_292 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1793 = stablehlo.reduce(%1792 init: %cst_292) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1794 = stablehlo.broadcast_in_dim %1793, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_293 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1795 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1796 = stablehlo.divide %1794, %1795 : tensor<1x8x1xf32>
    %1797 = stablehlo.convert %1796 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_294 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1798 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1799 = stablehlo.add %1797, %1798 : tensor<1x8x1xbf16>
    %1800 = stablehlo.rsqrt %1799 : tensor<1x8x1xbf16>
    %1801 = stablehlo.broadcast_in_dim %1800, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1802 = stablehlo.multiply %1790, %1801 : tensor<1x8x2304xbf16>
    %1803 = stablehlo.broadcast_in_dim %arg223, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_295 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1804 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1805 = stablehlo.add %1804, %1803 : tensor<1x1x2304xbf16>
    %1806 = stablehlo.broadcast_in_dim %1805, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1807 = stablehlo.multiply %1802, %1806 : tensor<1x8x2304xbf16>
    %1808 = stablehlo.add %1807, %1651 : tensor<1x8x2304xbf16>
    %1809 = chlo.square %1808 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1810 = stablehlo.convert %1809 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_296 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1811 = stablehlo.reduce(%1810 init: %cst_296) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1812 = stablehlo.broadcast_in_dim %1811, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_297 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1813 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1814 = stablehlo.divide %1812, %1813 : tensor<1x8x1xf32>
    %1815 = stablehlo.convert %1814 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_298 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1816 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1817 = stablehlo.add %1815, %1816 : tensor<1x8x1xbf16>
    %1818 = stablehlo.rsqrt %1817 : tensor<1x8x1xbf16>
    %1819 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1820 = stablehlo.multiply %1808, %1819 : tensor<1x8x2304xbf16>
    %1821 = stablehlo.broadcast_in_dim %arg226, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_299 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1822 = stablehlo.broadcast_in_dim %cst_299, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1823 = stablehlo.add %1822, %1821 : tensor<1x1x2304xbf16>
    %1824 = stablehlo.broadcast_in_dim %1823, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1825 = stablehlo.multiply %1820, %1824 : tensor<1x8x2304xbf16>
    %1826 = stablehlo.dot_general %1825, %arg221, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %1827 = stablehlo.slice %1826 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1828 = stablehlo.reshape %1827 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1829 = stablehlo.multiply %1828, %1828 : tensor<1x8x9216xbf16>
    %1830 = stablehlo.multiply %1829, %1828 : tensor<1x8x9216xbf16>
    %cst_300 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %1831 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1832 = stablehlo.multiply %1831, %1830 : tensor<1x8x9216xbf16>
    %1833 = stablehlo.add %1828, %1832 : tensor<1x8x9216xbf16>
    %cst_301 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %1834 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1835 = stablehlo.multiply %1834, %1833 : tensor<1x8x9216xbf16>
    %1836 = stablehlo.tanh %1835 : tensor<1x8x9216xbf16>
    %cst_302 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1837 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1838 = stablehlo.add %1837, %1836 : tensor<1x8x9216xbf16>
    %cst_303 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %1839 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %1840 = stablehlo.multiply %1839, %1838 : tensor<1x8x9216xbf16>
    %1841 = stablehlo.multiply %1828, %1840 : tensor<1x8x9216xbf16>
    %1842 = stablehlo.slice %1826 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %1843 = stablehlo.reshape %1842 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %1844 = stablehlo.multiply %1841, %1843 : tensor<1x8x9216xbf16>
    %1845 = stablehlo.dot_general %1844, %arg222, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1846 = chlo.square %1845 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1847 = stablehlo.convert %1846 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1848 = stablehlo.reduce(%1847 init: %cst_304) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1849 = stablehlo.broadcast_in_dim %1848, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_305 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1850 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1851 = stablehlo.divide %1849, %1850 : tensor<1x8x1xf32>
    %1852 = stablehlo.convert %1851 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_306 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1853 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1854 = stablehlo.add %1852, %1853 : tensor<1x8x1xbf16>
    %1855 = stablehlo.rsqrt %1854 : tensor<1x8x1xbf16>
    %1856 = stablehlo.broadcast_in_dim %1855, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1857 = stablehlo.multiply %1845, %1856 : tensor<1x8x2304xbf16>
    %1858 = stablehlo.broadcast_in_dim %arg224, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_307 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1859 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1860 = stablehlo.add %1859, %1858 : tensor<1x1x2304xbf16>
    %1861 = stablehlo.broadcast_in_dim %1860, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1862 = stablehlo.multiply %1857, %1861 : tensor<1x8x2304xbf16>
    %1863 = stablehlo.add %1862, %1808 : tensor<1x8x2304xbf16>
    %1864 = chlo.square %1863 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1865 = stablehlo.convert %1864 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_308 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1866 = stablehlo.reduce(%1865 init: %cst_308) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1867 = stablehlo.broadcast_in_dim %1866, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_309 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1868 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1869 = stablehlo.divide %1867, %1868 : tensor<1x8x1xf32>
    %1870 = stablehlo.convert %1869 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_310 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1871 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1872 = stablehlo.add %1870, %1871 : tensor<1x8x1xbf16>
    %1873 = stablehlo.rsqrt %1872 : tensor<1x8x1xbf16>
    %1874 = stablehlo.broadcast_in_dim %1873, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %1875 = stablehlo.multiply %1863, %1874 : tensor<1x8x2304xbf16>
    %1876 = stablehlo.broadcast_in_dim %arg234, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_311 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1877 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %1878 = stablehlo.add %1877, %1876 : tensor<1x1x2304xbf16>
    %1879 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1880 = stablehlo.multiply %1875, %1879 : tensor<1x8x2304xbf16>
    %1881 = stablehlo.dot_general %1880, %arg229, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1882 = stablehlo.dot_general %arg228, %1880, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %1883 = stablehlo.transpose %1882, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %1884 = stablehlo.slice %1883 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1885 = stablehlo.reshape %1884 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1886 = stablehlo.slice %1883 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %1887 = stablehlo.reshape %1886 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %1888 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_312 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1889 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1890 = stablehlo.multiply %1889, %1888 : tensor<128xf32>
    %cst_313 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1891 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1892 = stablehlo.power %1891, %1890 : tensor<128xf32>
    %1893 = call @_pad(%1892) : (tensor<128xf32>) -> tensor<128xf32>
    %1894 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1895 = stablehlo.broadcast_in_dim %1893, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1896 = stablehlo.convert %1894 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1897 = stablehlo.broadcast_in_dim %1896, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1898 = stablehlo.broadcast_in_dim %1895, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1899 = stablehlo.divide %1897, %1898 : tensor<1x8x128xf32>
    %1900 = stablehlo.broadcast_in_dim %1899, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_314 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1901 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1902 = stablehlo.divide %1900, %1901 : tensor<1x8x1x128xf32>
    %1903 = stablehlo.sine %1902 : tensor<1x8x1x128xf32>
    %1904 = stablehlo.cosine %1902 : tensor<1x8x1x128xf32>
    %1905 = stablehlo.slice %1881 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1906 = stablehlo.slice %1881 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %1907 = stablehlo.convert %1905 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1908 = stablehlo.broadcast_in_dim %1904, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1909 = stablehlo.multiply %1907, %1908 : tensor<1x8x8x128xf32>
    %1910 = stablehlo.convert %1906 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1911 = stablehlo.broadcast_in_dim %1903, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1912 = stablehlo.multiply %1910, %1911 : tensor<1x8x8x128xf32>
    %1913 = stablehlo.subtract %1909, %1912 : tensor<1x8x8x128xf32>
    %1914 = stablehlo.convert %1906 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1915 = stablehlo.broadcast_in_dim %1904, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1916 = stablehlo.multiply %1914, %1915 : tensor<1x8x8x128xf32>
    %1917 = stablehlo.convert %1905 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1918 = stablehlo.broadcast_in_dim %1903, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %1919 = stablehlo.multiply %1917, %1918 : tensor<1x8x8x128xf32>
    %1920 = stablehlo.add %1916, %1919 : tensor<1x8x8x128xf32>
    %1921 = stablehlo.concatenate %1913, %1920, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %1922 = stablehlo.convert %1921 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_315 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %1923 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %1924 = stablehlo.multiply %1922, %1923 : tensor<1x8x8x256xbf16>
    %1925 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_316 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %1926 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1927 = stablehlo.multiply %1926, %1925 : tensor<128xf32>
    %cst_317 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %1928 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1929 = stablehlo.power %1928, %1927 : tensor<128xf32>
    %1930 = call @_pad(%1929) : (tensor<128xf32>) -> tensor<128xf32>
    %1931 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1932 = stablehlo.broadcast_in_dim %1930, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %1933 = stablehlo.convert %1931 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %1934 = stablehlo.broadcast_in_dim %1933, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %1935 = stablehlo.broadcast_in_dim %1932, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %1936 = stablehlo.divide %1934, %1935 : tensor<1x8x128xf32>
    %1937 = stablehlo.broadcast_in_dim %1936, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_318 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1938 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %1939 = stablehlo.divide %1937, %1938 : tensor<1x8x1x128xf32>
    %1940 = stablehlo.sine %1939 : tensor<1x8x1x128xf32>
    %1941 = stablehlo.cosine %1939 : tensor<1x8x1x128xf32>
    %1942 = stablehlo.slice %1885 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1943 = stablehlo.slice %1885 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %1944 = stablehlo.convert %1942 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1945 = stablehlo.broadcast_in_dim %1941, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1946 = stablehlo.multiply %1944, %1945 : tensor<1x8x4x128xf32>
    %1947 = stablehlo.convert %1943 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1948 = stablehlo.broadcast_in_dim %1940, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1949 = stablehlo.multiply %1947, %1948 : tensor<1x8x4x128xf32>
    %1950 = stablehlo.subtract %1946, %1949 : tensor<1x8x4x128xf32>
    %1951 = stablehlo.convert %1943 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1952 = stablehlo.broadcast_in_dim %1941, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1953 = stablehlo.multiply %1951, %1952 : tensor<1x8x4x128xf32>
    %1954 = stablehlo.convert %1942 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %1955 = stablehlo.broadcast_in_dim %1940, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %1956 = stablehlo.multiply %1954, %1955 : tensor<1x8x4x128xf32>
    %1957 = stablehlo.add %1953, %1956 : tensor<1x8x4x128xf32>
    %1958 = stablehlo.concatenate %1950, %1957, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %1959 = stablehlo.convert %1958 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %1960 = stablehlo.reshape %1924 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1961 = stablehlo.dot_general %1959, %1960, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %1962 = stablehlo.transpose %1961, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1963 = stablehlo.reshape %1962 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_319 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1964 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1965 = stablehlo.divide %1963, %1964 : tensor<1x8x8x8xbf16>
    %1966 = stablehlo.tanh %1965 : tensor<1x8x8x8xbf16>
    %cst_320 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %1967 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %1968 = stablehlo.multiply %1966, %1967 : tensor<1x8x8x8xbf16>
    %1969 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_321 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1970 = call @_where(%1969, %1968, %cst_321) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_322 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1971 = stablehlo.reduce(%1970 init: %cst_322) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_323 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1972 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %1973 = stablehlo.maximum %1972, %1971 : tensor<1x8x8xbf16>
    %1974 = stablehlo.broadcast_in_dim %1973, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %1975 = stablehlo.broadcast_in_dim %1974, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1976 = stablehlo.subtract %1970, %1975 : tensor<1x8x8x8xbf16>
    %1977 = stablehlo.exponential %1976 : tensor<1x8x8x8xbf16>
    %1978 = stablehlo.convert %1977 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_324 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1979 = stablehlo.reduce(%1978 init: %cst_324) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1980 = stablehlo.broadcast_in_dim %1979, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1981 = stablehlo.convert %1980 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %1982 = stablehlo.broadcast_in_dim %1981, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %1983 = stablehlo.divide %1977, %1982 : tensor<1x8x8x8xbf16>
    %1984 = stablehlo.reshape %1983 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %1985 = stablehlo.dot_general %1887, %1984, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %1986 = stablehlo.transpose %1985, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %1987 = stablehlo.reshape %1986 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %1988 = stablehlo.dot_general %1987, %arg227, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %1989 = chlo.square %1988 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %1990 = stablehlo.convert %1989 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1991 = stablehlo.reduce(%1990 init: %cst_325) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1992 = stablehlo.broadcast_in_dim %1991, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_326 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %1993 = stablehlo.broadcast_in_dim %cst_326, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1994 = stablehlo.divide %1992, %1993 : tensor<1x8x1xf32>
    %1995 = stablehlo.convert %1994 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_327 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1996 = stablehlo.broadcast_in_dim %cst_327, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %1997 = stablehlo.add %1995, %1996 : tensor<1x8x1xbf16>
    %1998 = stablehlo.rsqrt %1997 : tensor<1x8x1xbf16>
    %1999 = stablehlo.broadcast_in_dim %1998, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2000 = stablehlo.multiply %1988, %1999 : tensor<1x8x2304xbf16>
    %2001 = stablehlo.broadcast_in_dim %arg232, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_328 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2002 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2003 = stablehlo.add %2002, %2001 : tensor<1x1x2304xbf16>
    %2004 = stablehlo.broadcast_in_dim %2003, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2005 = stablehlo.multiply %2000, %2004 : tensor<1x8x2304xbf16>
    %2006 = stablehlo.add %2005, %1863 : tensor<1x8x2304xbf16>
    %2007 = chlo.square %2006 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2008 = stablehlo.convert %2007 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_329 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2009 = stablehlo.reduce(%2008 init: %cst_329) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2010 = stablehlo.broadcast_in_dim %2009, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_330 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2011 = stablehlo.broadcast_in_dim %cst_330, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2012 = stablehlo.divide %2010, %2011 : tensor<1x8x1xf32>
    %2013 = stablehlo.convert %2012 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_331 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2014 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2015 = stablehlo.add %2013, %2014 : tensor<1x8x1xbf16>
    %2016 = stablehlo.rsqrt %2015 : tensor<1x8x1xbf16>
    %2017 = stablehlo.broadcast_in_dim %2016, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2018 = stablehlo.multiply %2006, %2017 : tensor<1x8x2304xbf16>
    %2019 = stablehlo.broadcast_in_dim %arg235, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_332 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2020 = stablehlo.broadcast_in_dim %cst_332, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2021 = stablehlo.add %2020, %2019 : tensor<1x1x2304xbf16>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2023 = stablehlo.multiply %2018, %2022 : tensor<1x8x2304xbf16>
    %2024 = stablehlo.dot_general %2023, %arg230, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %2025 = stablehlo.slice %2024 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2026 = stablehlo.reshape %2025 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2027 = stablehlo.multiply %2026, %2026 : tensor<1x8x9216xbf16>
    %2028 = stablehlo.multiply %2027, %2026 : tensor<1x8x9216xbf16>
    %cst_333 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %2029 = stablehlo.broadcast_in_dim %cst_333, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2030 = stablehlo.multiply %2029, %2028 : tensor<1x8x9216xbf16>
    %2031 = stablehlo.add %2026, %2030 : tensor<1x8x9216xbf16>
    %cst_334 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %2032 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2033 = stablehlo.multiply %2032, %2031 : tensor<1x8x9216xbf16>
    %2034 = stablehlo.tanh %2033 : tensor<1x8x9216xbf16>
    %cst_335 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2035 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2036 = stablehlo.add %2035, %2034 : tensor<1x8x9216xbf16>
    %cst_336 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %2037 = stablehlo.broadcast_in_dim %cst_336, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2038 = stablehlo.multiply %2037, %2036 : tensor<1x8x9216xbf16>
    %2039 = stablehlo.multiply %2026, %2038 : tensor<1x8x9216xbf16>
    %2040 = stablehlo.slice %2024 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2041 = stablehlo.reshape %2040 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2042 = stablehlo.multiply %2039, %2041 : tensor<1x8x9216xbf16>
    %2043 = stablehlo.dot_general %2042, %arg231, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2044 = chlo.square %2043 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2045 = stablehlo.convert %2044 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_337 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2046 = stablehlo.reduce(%2045 init: %cst_337) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2047 = stablehlo.broadcast_in_dim %2046, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_338 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2048 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2049 = stablehlo.divide %2047, %2048 : tensor<1x8x1xf32>
    %2050 = stablehlo.convert %2049 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_339 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2051 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2052 = stablehlo.add %2050, %2051 : tensor<1x8x1xbf16>
    %2053 = stablehlo.rsqrt %2052 : tensor<1x8x1xbf16>
    %2054 = stablehlo.broadcast_in_dim %2053, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2055 = stablehlo.multiply %2043, %2054 : tensor<1x8x2304xbf16>
    %2056 = stablehlo.broadcast_in_dim %arg233, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_340 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2057 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2058 = stablehlo.add %2057, %2056 : tensor<1x1x2304xbf16>
    %2059 = stablehlo.broadcast_in_dim %2058, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2060 = stablehlo.multiply %2055, %2059 : tensor<1x8x2304xbf16>
    %2061 = stablehlo.add %2060, %2006 : tensor<1x8x2304xbf16>
    %2062 = chlo.square %2061 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2063 = stablehlo.convert %2062 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_341 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2064 = stablehlo.reduce(%2063 init: %cst_341) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2065 = stablehlo.broadcast_in_dim %2064, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_342 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2066 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2067 = stablehlo.divide %2065, %2066 : tensor<1x8x1xf32>
    %2068 = stablehlo.convert %2067 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_343 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2069 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2070 = stablehlo.add %2068, %2069 : tensor<1x8x1xbf16>
    %2071 = stablehlo.rsqrt %2070 : tensor<1x8x1xbf16>
    %2072 = stablehlo.broadcast_in_dim %2071, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2073 = stablehlo.multiply %2061, %2072 : tensor<1x8x2304xbf16>
    %2074 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_344 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2075 = stablehlo.broadcast_in_dim %cst_344, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2076 = stablehlo.add %2075, %2074 : tensor<1x1x2304xbf16>
    %2077 = stablehlo.broadcast_in_dim %2076, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2078 = stablehlo.multiply %2073, %2077 : tensor<1x8x2304xbf16>
    %2079 = stablehlo.dot_general %2078, %arg22, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2080 = stablehlo.dot_general %arg21, %2078, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %2081 = stablehlo.transpose %2080, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %2082 = stablehlo.slice %2081 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2083 = stablehlo.reshape %2082 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2084 = stablehlo.slice %2081 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2085 = stablehlo.reshape %2084 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2086 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_345 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2087 = stablehlo.broadcast_in_dim %cst_345, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2088 = stablehlo.multiply %2087, %2086 : tensor<128xf32>
    %cst_346 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2089 = stablehlo.broadcast_in_dim %cst_346, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2090 = stablehlo.power %2089, %2088 : tensor<128xf32>
    %2091 = call @_pad(%2090) : (tensor<128xf32>) -> tensor<128xf32>
    %2092 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2093 = stablehlo.broadcast_in_dim %2091, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2094 = stablehlo.convert %2092 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2095 = stablehlo.broadcast_in_dim %2094, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2096 = stablehlo.broadcast_in_dim %2093, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2097 = stablehlo.divide %2095, %2096 : tensor<1x8x128xf32>
    %2098 = stablehlo.broadcast_in_dim %2097, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_347 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2099 = stablehlo.broadcast_in_dim %cst_347, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2100 = stablehlo.divide %2098, %2099 : tensor<1x8x1x128xf32>
    %2101 = stablehlo.sine %2100 : tensor<1x8x1x128xf32>
    %2102 = stablehlo.cosine %2100 : tensor<1x8x1x128xf32>
    %2103 = stablehlo.slice %2079 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2104 = stablehlo.slice %2079 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2105 = stablehlo.convert %2103 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2106 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2107 = stablehlo.multiply %2105, %2106 : tensor<1x8x8x128xf32>
    %2108 = stablehlo.convert %2104 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2109 = stablehlo.broadcast_in_dim %2101, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2110 = stablehlo.multiply %2108, %2109 : tensor<1x8x8x128xf32>
    %2111 = stablehlo.subtract %2107, %2110 : tensor<1x8x8x128xf32>
    %2112 = stablehlo.convert %2104 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2113 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2114 = stablehlo.multiply %2112, %2113 : tensor<1x8x8x128xf32>
    %2115 = stablehlo.convert %2103 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2116 = stablehlo.broadcast_in_dim %2101, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2117 = stablehlo.multiply %2115, %2116 : tensor<1x8x8x128xf32>
    %2118 = stablehlo.add %2114, %2117 : tensor<1x8x8x128xf32>
    %2119 = stablehlo.concatenate %2111, %2118, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %2120 = stablehlo.convert %2119 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_348 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %2121 = stablehlo.broadcast_in_dim %cst_348, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %2122 = stablehlo.multiply %2120, %2121 : tensor<1x8x8x256xbf16>
    %2123 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_349 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2124 = stablehlo.broadcast_in_dim %cst_349, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2125 = stablehlo.multiply %2124, %2123 : tensor<128xf32>
    %cst_350 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2126 = stablehlo.broadcast_in_dim %cst_350, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2127 = stablehlo.power %2126, %2125 : tensor<128xf32>
    %2128 = call @_pad(%2127) : (tensor<128xf32>) -> tensor<128xf32>
    %2129 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2130 = stablehlo.broadcast_in_dim %2128, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2131 = stablehlo.convert %2129 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2132 = stablehlo.broadcast_in_dim %2131, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2133 = stablehlo.broadcast_in_dim %2130, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2134 = stablehlo.divide %2132, %2133 : tensor<1x8x128xf32>
    %2135 = stablehlo.broadcast_in_dim %2134, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_351 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2136 = stablehlo.broadcast_in_dim %cst_351, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2137 = stablehlo.divide %2135, %2136 : tensor<1x8x1x128xf32>
    %2138 = stablehlo.sine %2137 : tensor<1x8x1x128xf32>
    %2139 = stablehlo.cosine %2137 : tensor<1x8x1x128xf32>
    %2140 = stablehlo.slice %2083 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2141 = stablehlo.slice %2083 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2142 = stablehlo.convert %2140 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2143 = stablehlo.broadcast_in_dim %2139, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2144 = stablehlo.multiply %2142, %2143 : tensor<1x8x4x128xf32>
    %2145 = stablehlo.convert %2141 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2146 = stablehlo.broadcast_in_dim %2138, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2147 = stablehlo.multiply %2145, %2146 : tensor<1x8x4x128xf32>
    %2148 = stablehlo.subtract %2144, %2147 : tensor<1x8x4x128xf32>
    %2149 = stablehlo.convert %2141 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2150 = stablehlo.broadcast_in_dim %2139, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2151 = stablehlo.multiply %2149, %2150 : tensor<1x8x4x128xf32>
    %2152 = stablehlo.convert %2140 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2153 = stablehlo.broadcast_in_dim %2138, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2154 = stablehlo.multiply %2152, %2153 : tensor<1x8x4x128xf32>
    %2155 = stablehlo.add %2151, %2154 : tensor<1x8x4x128xf32>
    %2156 = stablehlo.concatenate %2148, %2155, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %2157 = stablehlo.convert %2156 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %2158 = stablehlo.reshape %2122 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2159 = stablehlo.dot_general %2157, %2158, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %2160 = stablehlo.transpose %2159, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2161 = stablehlo.reshape %2160 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_352 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2162 = stablehlo.broadcast_in_dim %cst_352, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2163 = stablehlo.divide %2161, %2162 : tensor<1x8x8x8xbf16>
    %2164 = stablehlo.tanh %2163 : tensor<1x8x8x8xbf16>
    %cst_353 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2165 = stablehlo.broadcast_in_dim %cst_353, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2166 = stablehlo.multiply %2164, %2165 : tensor<1x8x8x8xbf16>
    %2167 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %2168 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_354 = stablehlo.constant dense<4096> : tensor<i32>
    %2169 = stablehlo.broadcast_in_dim %c_354, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %2170 = stablehlo.subtract %2168, %2169 : tensor<1x8x1xi32>
    %2171 = stablehlo.broadcast_in_dim %2167, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %2172 = stablehlo.broadcast_in_dim %2170, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %2173 = stablehlo.compare GT, %2171, %2172, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_355 = stablehlo.constant dense<4096> : tensor<i32>
    %2174 = stablehlo.broadcast_in_dim %c_355, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %2175 = stablehlo.add %2168, %2174 : tensor<1x8x1xi32>
    %2176 = stablehlo.broadcast_in_dim %2167, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %2177 = stablehlo.broadcast_in_dim %2175, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %2178 = stablehlo.compare LT, %2176, %2177, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %2179 = stablehlo.and %2173, %2178 : tensor<1x8x8xi1>
    %2180 = stablehlo.and %arg237, %2179 : tensor<1x8x8xi1>
    %2181 = stablehlo.broadcast_in_dim %2180, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_356 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2182 = call @_where(%2181, %2166, %cst_356) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_357 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2183 = stablehlo.reduce(%2182 init: %cst_357) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_358 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2184 = stablehlo.broadcast_in_dim %cst_358, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %2185 = stablehlo.maximum %2184, %2183 : tensor<1x8x8xbf16>
    %2186 = stablehlo.broadcast_in_dim %2185, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %2187 = stablehlo.broadcast_in_dim %2186, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2188 = stablehlo.subtract %2182, %2187 : tensor<1x8x8x8xbf16>
    %2189 = stablehlo.exponential %2188 : tensor<1x8x8x8xbf16>
    %2190 = stablehlo.convert %2189 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_359 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2191 = stablehlo.reduce(%2190 init: %cst_359) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2193 = stablehlo.convert %2192 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %2194 = stablehlo.broadcast_in_dim %2193, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2195 = stablehlo.divide %2189, %2194 : tensor<1x8x8x8xbf16>
    %2196 = stablehlo.reshape %2195 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2197 = stablehlo.dot_general %2085, %2196, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %2198 = stablehlo.transpose %2197, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2199 = stablehlo.reshape %2198 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2200 = stablehlo.dot_general %2199, %arg20, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2201 = chlo.square %2200 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2202 = stablehlo.convert %2201 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_360 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2203 = stablehlo.reduce(%2202 init: %cst_360) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2204 = stablehlo.broadcast_in_dim %2203, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_361 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2205 = stablehlo.broadcast_in_dim %cst_361, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2206 = stablehlo.divide %2204, %2205 : tensor<1x8x1xf32>
    %2207 = stablehlo.convert %2206 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_362 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2208 = stablehlo.broadcast_in_dim %cst_362, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2209 = stablehlo.add %2207, %2208 : tensor<1x8x1xbf16>
    %2210 = stablehlo.rsqrt %2209 : tensor<1x8x1xbf16>
    %2211 = stablehlo.broadcast_in_dim %2210, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2212 = stablehlo.multiply %2200, %2211 : tensor<1x8x2304xbf16>
    %2213 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_363 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2214 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2215 = stablehlo.add %2214, %2213 : tensor<1x1x2304xbf16>
    %2216 = stablehlo.broadcast_in_dim %2215, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2217 = stablehlo.multiply %2212, %2216 : tensor<1x8x2304xbf16>
    %2218 = stablehlo.add %2217, %2061 : tensor<1x8x2304xbf16>
    %2219 = chlo.square %2218 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2220 = stablehlo.convert %2219 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_364 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2221 = stablehlo.reduce(%2220 init: %cst_364) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2222 = stablehlo.broadcast_in_dim %2221, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_365 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2223 = stablehlo.broadcast_in_dim %cst_365, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2224 = stablehlo.divide %2222, %2223 : tensor<1x8x1xf32>
    %2225 = stablehlo.convert %2224 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_366 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2226 = stablehlo.broadcast_in_dim %cst_366, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2227 = stablehlo.add %2225, %2226 : tensor<1x8x1xbf16>
    %2228 = stablehlo.rsqrt %2227 : tensor<1x8x1xbf16>
    %2229 = stablehlo.broadcast_in_dim %2228, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2230 = stablehlo.multiply %2218, %2229 : tensor<1x8x2304xbf16>
    %2231 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_367 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2232 = stablehlo.broadcast_in_dim %cst_367, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2233 = stablehlo.add %2232, %2231 : tensor<1x1x2304xbf16>
    %2234 = stablehlo.broadcast_in_dim %2233, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2235 = stablehlo.multiply %2230, %2234 : tensor<1x8x2304xbf16>
    %2236 = stablehlo.dot_general %2235, %arg23, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %2237 = stablehlo.slice %2236 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2238 = stablehlo.reshape %2237 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2239 = stablehlo.multiply %2238, %2238 : tensor<1x8x9216xbf16>
    %2240 = stablehlo.multiply %2239, %2238 : tensor<1x8x9216xbf16>
    %cst_368 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %2241 = stablehlo.broadcast_in_dim %cst_368, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2242 = stablehlo.multiply %2241, %2240 : tensor<1x8x9216xbf16>
    %2243 = stablehlo.add %2238, %2242 : tensor<1x8x9216xbf16>
    %cst_369 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %2244 = stablehlo.broadcast_in_dim %cst_369, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2245 = stablehlo.multiply %2244, %2243 : tensor<1x8x9216xbf16>
    %2246 = stablehlo.tanh %2245 : tensor<1x8x9216xbf16>
    %cst_370 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2247 = stablehlo.broadcast_in_dim %cst_370, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2248 = stablehlo.add %2247, %2246 : tensor<1x8x9216xbf16>
    %cst_371 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %2249 = stablehlo.broadcast_in_dim %cst_371, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2250 = stablehlo.multiply %2249, %2248 : tensor<1x8x9216xbf16>
    %2251 = stablehlo.multiply %2238, %2250 : tensor<1x8x9216xbf16>
    %2252 = stablehlo.slice %2236 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2253 = stablehlo.reshape %2252 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2254 = stablehlo.multiply %2251, %2253 : tensor<1x8x9216xbf16>
    %2255 = stablehlo.dot_general %2254, %arg24, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2256 = chlo.square %2255 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2257 = stablehlo.convert %2256 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_372 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2258 = stablehlo.reduce(%2257 init: %cst_372) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2259 = stablehlo.broadcast_in_dim %2258, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_373 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2260 = stablehlo.broadcast_in_dim %cst_373, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2261 = stablehlo.divide %2259, %2260 : tensor<1x8x1xf32>
    %2262 = stablehlo.convert %2261 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_374 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2263 = stablehlo.broadcast_in_dim %cst_374, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2264 = stablehlo.add %2262, %2263 : tensor<1x8x1xbf16>
    %2265 = stablehlo.rsqrt %2264 : tensor<1x8x1xbf16>
    %2266 = stablehlo.broadcast_in_dim %2265, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2267 = stablehlo.multiply %2255, %2266 : tensor<1x8x2304xbf16>
    %2268 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_375 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2269 = stablehlo.broadcast_in_dim %cst_375, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2270 = stablehlo.add %2269, %2268 : tensor<1x1x2304xbf16>
    %2271 = stablehlo.broadcast_in_dim %2270, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2272 = stablehlo.multiply %2267, %2271 : tensor<1x8x2304xbf16>
    %2273 = stablehlo.add %2272, %2218 : tensor<1x8x2304xbf16>
    %2274 = chlo.square %2273 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2275 = stablehlo.convert %2274 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_376 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2276 = stablehlo.reduce(%2275 init: %cst_376) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2277 = stablehlo.broadcast_in_dim %2276, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_377 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2278 = stablehlo.broadcast_in_dim %cst_377, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2279 = stablehlo.divide %2277, %2278 : tensor<1x8x1xf32>
    %2280 = stablehlo.convert %2279 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_378 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2281 = stablehlo.broadcast_in_dim %cst_378, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2282 = stablehlo.add %2280, %2281 : tensor<1x8x1xbf16>
    %2283 = stablehlo.rsqrt %2282 : tensor<1x8x1xbf16>
    %2284 = stablehlo.broadcast_in_dim %2283, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2285 = stablehlo.multiply %2273, %2284 : tensor<1x8x2304xbf16>
    %2286 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_379 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2287 = stablehlo.broadcast_in_dim %cst_379, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2288 = stablehlo.add %2287, %2286 : tensor<1x1x2304xbf16>
    %2289 = stablehlo.broadcast_in_dim %2288, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2290 = stablehlo.multiply %2285, %2289 : tensor<1x8x2304xbf16>
    %2291 = stablehlo.dot_general %2290, %arg31, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2292 = stablehlo.dot_general %arg30, %2290, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %2293 = stablehlo.transpose %2292, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %2294 = stablehlo.slice %2293 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2295 = stablehlo.reshape %2294 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2296 = stablehlo.slice %2293 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2297 = stablehlo.reshape %2296 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2298 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_380 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2299 = stablehlo.broadcast_in_dim %cst_380, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2300 = stablehlo.multiply %2299, %2298 : tensor<128xf32>
    %cst_381 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2301 = stablehlo.broadcast_in_dim %cst_381, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2302 = stablehlo.power %2301, %2300 : tensor<128xf32>
    %2303 = call @_pad(%2302) : (tensor<128xf32>) -> tensor<128xf32>
    %2304 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2305 = stablehlo.broadcast_in_dim %2303, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2306 = stablehlo.convert %2304 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2307 = stablehlo.broadcast_in_dim %2306, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2308 = stablehlo.broadcast_in_dim %2305, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2309 = stablehlo.divide %2307, %2308 : tensor<1x8x128xf32>
    %2310 = stablehlo.broadcast_in_dim %2309, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_382 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2311 = stablehlo.broadcast_in_dim %cst_382, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2312 = stablehlo.divide %2310, %2311 : tensor<1x8x1x128xf32>
    %2313 = stablehlo.sine %2312 : tensor<1x8x1x128xf32>
    %2314 = stablehlo.cosine %2312 : tensor<1x8x1x128xf32>
    %2315 = stablehlo.slice %2291 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2316 = stablehlo.slice %2291 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2317 = stablehlo.convert %2315 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2318 = stablehlo.broadcast_in_dim %2314, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2319 = stablehlo.multiply %2317, %2318 : tensor<1x8x8x128xf32>
    %2320 = stablehlo.convert %2316 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2321 = stablehlo.broadcast_in_dim %2313, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2322 = stablehlo.multiply %2320, %2321 : tensor<1x8x8x128xf32>
    %2323 = stablehlo.subtract %2319, %2322 : tensor<1x8x8x128xf32>
    %2324 = stablehlo.convert %2316 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2325 = stablehlo.broadcast_in_dim %2314, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2326 = stablehlo.multiply %2324, %2325 : tensor<1x8x8x128xf32>
    %2327 = stablehlo.convert %2315 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2328 = stablehlo.broadcast_in_dim %2313, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2329 = stablehlo.multiply %2327, %2328 : tensor<1x8x8x128xf32>
    %2330 = stablehlo.add %2326, %2329 : tensor<1x8x8x128xf32>
    %2331 = stablehlo.concatenate %2323, %2330, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %2332 = stablehlo.convert %2331 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_383 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %2333 = stablehlo.broadcast_in_dim %cst_383, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %2334 = stablehlo.multiply %2332, %2333 : tensor<1x8x8x256xbf16>
    %2335 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_384 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2336 = stablehlo.broadcast_in_dim %cst_384, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2337 = stablehlo.multiply %2336, %2335 : tensor<128xf32>
    %cst_385 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2338 = stablehlo.broadcast_in_dim %cst_385, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2339 = stablehlo.power %2338, %2337 : tensor<128xf32>
    %2340 = call @_pad(%2339) : (tensor<128xf32>) -> tensor<128xf32>
    %2341 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2342 = stablehlo.broadcast_in_dim %2340, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2343 = stablehlo.convert %2341 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2344 = stablehlo.broadcast_in_dim %2343, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2345 = stablehlo.broadcast_in_dim %2342, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2346 = stablehlo.divide %2344, %2345 : tensor<1x8x128xf32>
    %2347 = stablehlo.broadcast_in_dim %2346, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_386 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2348 = stablehlo.broadcast_in_dim %cst_386, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2349 = stablehlo.divide %2347, %2348 : tensor<1x8x1x128xf32>
    %2350 = stablehlo.sine %2349 : tensor<1x8x1x128xf32>
    %2351 = stablehlo.cosine %2349 : tensor<1x8x1x128xf32>
    %2352 = stablehlo.slice %2295 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2353 = stablehlo.slice %2295 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2354 = stablehlo.convert %2352 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2355 = stablehlo.broadcast_in_dim %2351, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2356 = stablehlo.multiply %2354, %2355 : tensor<1x8x4x128xf32>
    %2357 = stablehlo.convert %2353 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2358 = stablehlo.broadcast_in_dim %2350, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2359 = stablehlo.multiply %2357, %2358 : tensor<1x8x4x128xf32>
    %2360 = stablehlo.subtract %2356, %2359 : tensor<1x8x4x128xf32>
    %2361 = stablehlo.convert %2353 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2362 = stablehlo.broadcast_in_dim %2351, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2363 = stablehlo.multiply %2361, %2362 : tensor<1x8x4x128xf32>
    %2364 = stablehlo.convert %2352 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2365 = stablehlo.broadcast_in_dim %2350, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2366 = stablehlo.multiply %2364, %2365 : tensor<1x8x4x128xf32>
    %2367 = stablehlo.add %2363, %2366 : tensor<1x8x4x128xf32>
    %2368 = stablehlo.concatenate %2360, %2367, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %2369 = stablehlo.convert %2368 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %2370 = stablehlo.reshape %2334 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2371 = stablehlo.dot_general %2369, %2370, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %2372 = stablehlo.transpose %2371, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2373 = stablehlo.reshape %2372 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_387 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2374 = stablehlo.broadcast_in_dim %cst_387, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2375 = stablehlo.divide %2373, %2374 : tensor<1x8x8x8xbf16>
    %2376 = stablehlo.tanh %2375 : tensor<1x8x8x8xbf16>
    %cst_388 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2377 = stablehlo.broadcast_in_dim %cst_388, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2378 = stablehlo.multiply %2376, %2377 : tensor<1x8x8x8xbf16>
    %2379 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_389 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2380 = call @_where(%2379, %2378, %cst_389) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_390 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2381 = stablehlo.reduce(%2380 init: %cst_390) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_391 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2382 = stablehlo.broadcast_in_dim %cst_391, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %2383 = stablehlo.maximum %2382, %2381 : tensor<1x8x8xbf16>
    %2384 = stablehlo.broadcast_in_dim %2383, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %2385 = stablehlo.broadcast_in_dim %2384, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2386 = stablehlo.subtract %2380, %2385 : tensor<1x8x8x8xbf16>
    %2387 = stablehlo.exponential %2386 : tensor<1x8x8x8xbf16>
    %2388 = stablehlo.convert %2387 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_392 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2389 = stablehlo.reduce(%2388 init: %cst_392) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2390 = stablehlo.broadcast_in_dim %2389, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2391 = stablehlo.convert %2390 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %2392 = stablehlo.broadcast_in_dim %2391, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2393 = stablehlo.divide %2387, %2392 : tensor<1x8x8x8xbf16>
    %2394 = stablehlo.reshape %2393 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2395 = stablehlo.dot_general %2297, %2394, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %2396 = stablehlo.transpose %2395, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2397 = stablehlo.reshape %2396 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2398 = stablehlo.dot_general %2397, %arg29, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2399 = chlo.square %2398 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2400 = stablehlo.convert %2399 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_393 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2401 = stablehlo.reduce(%2400 init: %cst_393) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2402 = stablehlo.broadcast_in_dim %2401, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_394 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2403 = stablehlo.broadcast_in_dim %cst_394, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2404 = stablehlo.divide %2402, %2403 : tensor<1x8x1xf32>
    %2405 = stablehlo.convert %2404 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_395 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2406 = stablehlo.broadcast_in_dim %cst_395, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2407 = stablehlo.add %2405, %2406 : tensor<1x8x1xbf16>
    %2408 = stablehlo.rsqrt %2407 : tensor<1x8x1xbf16>
    %2409 = stablehlo.broadcast_in_dim %2408, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2410 = stablehlo.multiply %2398, %2409 : tensor<1x8x2304xbf16>
    %2411 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_396 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2412 = stablehlo.broadcast_in_dim %cst_396, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2413 = stablehlo.add %2412, %2411 : tensor<1x1x2304xbf16>
    %2414 = stablehlo.broadcast_in_dim %2413, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2415 = stablehlo.multiply %2410, %2414 : tensor<1x8x2304xbf16>
    %2416 = stablehlo.add %2415, %2273 : tensor<1x8x2304xbf16>
    %2417 = chlo.square %2416 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2418 = stablehlo.convert %2417 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_397 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2419 = stablehlo.reduce(%2418 init: %cst_397) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2420 = stablehlo.broadcast_in_dim %2419, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_398 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2421 = stablehlo.broadcast_in_dim %cst_398, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2422 = stablehlo.divide %2420, %2421 : tensor<1x8x1xf32>
    %2423 = stablehlo.convert %2422 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_399 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2424 = stablehlo.broadcast_in_dim %cst_399, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2425 = stablehlo.add %2423, %2424 : tensor<1x8x1xbf16>
    %2426 = stablehlo.rsqrt %2425 : tensor<1x8x1xbf16>
    %2427 = stablehlo.broadcast_in_dim %2426, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2428 = stablehlo.multiply %2416, %2427 : tensor<1x8x2304xbf16>
    %2429 = stablehlo.broadcast_in_dim %arg37, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_400 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2430 = stablehlo.broadcast_in_dim %cst_400, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2431 = stablehlo.add %2430, %2429 : tensor<1x1x2304xbf16>
    %2432 = stablehlo.broadcast_in_dim %2431, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2433 = stablehlo.multiply %2428, %2432 : tensor<1x8x2304xbf16>
    %2434 = stablehlo.dot_general %2433, %arg32, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %2435 = stablehlo.slice %2434 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2436 = stablehlo.reshape %2435 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2437 = stablehlo.multiply %2436, %2436 : tensor<1x8x9216xbf16>
    %2438 = stablehlo.multiply %2437, %2436 : tensor<1x8x9216xbf16>
    %cst_401 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %2439 = stablehlo.broadcast_in_dim %cst_401, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2440 = stablehlo.multiply %2439, %2438 : tensor<1x8x9216xbf16>
    %2441 = stablehlo.add %2436, %2440 : tensor<1x8x9216xbf16>
    %cst_402 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %2442 = stablehlo.broadcast_in_dim %cst_402, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2443 = stablehlo.multiply %2442, %2441 : tensor<1x8x9216xbf16>
    %2444 = stablehlo.tanh %2443 : tensor<1x8x9216xbf16>
    %cst_403 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2445 = stablehlo.broadcast_in_dim %cst_403, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2446 = stablehlo.add %2445, %2444 : tensor<1x8x9216xbf16>
    %cst_404 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %2447 = stablehlo.broadcast_in_dim %cst_404, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2448 = stablehlo.multiply %2447, %2446 : tensor<1x8x9216xbf16>
    %2449 = stablehlo.multiply %2436, %2448 : tensor<1x8x9216xbf16>
    %2450 = stablehlo.slice %2434 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2451 = stablehlo.reshape %2450 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2452 = stablehlo.multiply %2449, %2451 : tensor<1x8x9216xbf16>
    %2453 = stablehlo.dot_general %2452, %arg33, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2454 = chlo.square %2453 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2455 = stablehlo.convert %2454 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_405 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2456 = stablehlo.reduce(%2455 init: %cst_405) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2457 = stablehlo.broadcast_in_dim %2456, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_406 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2458 = stablehlo.broadcast_in_dim %cst_406, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2459 = stablehlo.divide %2457, %2458 : tensor<1x8x1xf32>
    %2460 = stablehlo.convert %2459 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_407 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2461 = stablehlo.broadcast_in_dim %cst_407, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2462 = stablehlo.add %2460, %2461 : tensor<1x8x1xbf16>
    %2463 = stablehlo.rsqrt %2462 : tensor<1x8x1xbf16>
    %2464 = stablehlo.broadcast_in_dim %2463, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2465 = stablehlo.multiply %2453, %2464 : tensor<1x8x2304xbf16>
    %2466 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_408 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2467 = stablehlo.broadcast_in_dim %cst_408, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2468 = stablehlo.add %2467, %2466 : tensor<1x1x2304xbf16>
    %2469 = stablehlo.broadcast_in_dim %2468, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2470 = stablehlo.multiply %2465, %2469 : tensor<1x8x2304xbf16>
    %2471 = stablehlo.add %2470, %2416 : tensor<1x8x2304xbf16>
    %2472 = chlo.square %2471 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2473 = stablehlo.convert %2472 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_409 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2474 = stablehlo.reduce(%2473 init: %cst_409) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2475 = stablehlo.broadcast_in_dim %2474, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_410 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2476 = stablehlo.broadcast_in_dim %cst_410, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2477 = stablehlo.divide %2475, %2476 : tensor<1x8x1xf32>
    %2478 = stablehlo.convert %2477 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_411 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2479 = stablehlo.broadcast_in_dim %cst_411, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2480 = stablehlo.add %2478, %2479 : tensor<1x8x1xbf16>
    %2481 = stablehlo.rsqrt %2480 : tensor<1x8x1xbf16>
    %2482 = stablehlo.broadcast_in_dim %2481, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2483 = stablehlo.multiply %2471, %2482 : tensor<1x8x2304xbf16>
    %2484 = stablehlo.broadcast_in_dim %arg45, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_412 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2485 = stablehlo.broadcast_in_dim %cst_412, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2486 = stablehlo.add %2485, %2484 : tensor<1x1x2304xbf16>
    %2487 = stablehlo.broadcast_in_dim %2486, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2488 = stablehlo.multiply %2483, %2487 : tensor<1x8x2304xbf16>
    %2489 = stablehlo.dot_general %2488, %arg40, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2490 = stablehlo.dot_general %arg39, %2488, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %2491 = stablehlo.transpose %2490, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %2492 = stablehlo.slice %2491 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2493 = stablehlo.reshape %2492 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2494 = stablehlo.slice %2491 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2495 = stablehlo.reshape %2494 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2496 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_413 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2497 = stablehlo.broadcast_in_dim %cst_413, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2498 = stablehlo.multiply %2497, %2496 : tensor<128xf32>
    %cst_414 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2499 = stablehlo.broadcast_in_dim %cst_414, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2500 = stablehlo.power %2499, %2498 : tensor<128xf32>
    %2501 = call @_pad(%2500) : (tensor<128xf32>) -> tensor<128xf32>
    %2502 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2503 = stablehlo.broadcast_in_dim %2501, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2504 = stablehlo.convert %2502 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2505 = stablehlo.broadcast_in_dim %2504, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2506 = stablehlo.broadcast_in_dim %2503, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2507 = stablehlo.divide %2505, %2506 : tensor<1x8x128xf32>
    %2508 = stablehlo.broadcast_in_dim %2507, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_415 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2509 = stablehlo.broadcast_in_dim %cst_415, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2510 = stablehlo.divide %2508, %2509 : tensor<1x8x1x128xf32>
    %2511 = stablehlo.sine %2510 : tensor<1x8x1x128xf32>
    %2512 = stablehlo.cosine %2510 : tensor<1x8x1x128xf32>
    %2513 = stablehlo.slice %2489 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2514 = stablehlo.slice %2489 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2515 = stablehlo.convert %2513 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2516 = stablehlo.broadcast_in_dim %2512, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2517 = stablehlo.multiply %2515, %2516 : tensor<1x8x8x128xf32>
    %2518 = stablehlo.convert %2514 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2519 = stablehlo.broadcast_in_dim %2511, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2520 = stablehlo.multiply %2518, %2519 : tensor<1x8x8x128xf32>
    %2521 = stablehlo.subtract %2517, %2520 : tensor<1x8x8x128xf32>
    %2522 = stablehlo.convert %2514 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2523 = stablehlo.broadcast_in_dim %2512, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2524 = stablehlo.multiply %2522, %2523 : tensor<1x8x8x128xf32>
    %2525 = stablehlo.convert %2513 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2526 = stablehlo.broadcast_in_dim %2511, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2527 = stablehlo.multiply %2525, %2526 : tensor<1x8x8x128xf32>
    %2528 = stablehlo.add %2524, %2527 : tensor<1x8x8x128xf32>
    %2529 = stablehlo.concatenate %2521, %2528, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %2530 = stablehlo.convert %2529 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_416 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %2531 = stablehlo.broadcast_in_dim %cst_416, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %2532 = stablehlo.multiply %2530, %2531 : tensor<1x8x8x256xbf16>
    %2533 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_417 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2534 = stablehlo.broadcast_in_dim %cst_417, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2535 = stablehlo.multiply %2534, %2533 : tensor<128xf32>
    %cst_418 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2536 = stablehlo.broadcast_in_dim %cst_418, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2537 = stablehlo.power %2536, %2535 : tensor<128xf32>
    %2538 = call @_pad(%2537) : (tensor<128xf32>) -> tensor<128xf32>
    %2539 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2540 = stablehlo.broadcast_in_dim %2538, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2541 = stablehlo.convert %2539 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2542 = stablehlo.broadcast_in_dim %2541, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2543 = stablehlo.broadcast_in_dim %2540, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2544 = stablehlo.divide %2542, %2543 : tensor<1x8x128xf32>
    %2545 = stablehlo.broadcast_in_dim %2544, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_419 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2546 = stablehlo.broadcast_in_dim %cst_419, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2547 = stablehlo.divide %2545, %2546 : tensor<1x8x1x128xf32>
    %2548 = stablehlo.sine %2547 : tensor<1x8x1x128xf32>
    %2549 = stablehlo.cosine %2547 : tensor<1x8x1x128xf32>
    %2550 = stablehlo.slice %2493 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2551 = stablehlo.slice %2493 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2552 = stablehlo.convert %2550 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2553 = stablehlo.broadcast_in_dim %2549, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2554 = stablehlo.multiply %2552, %2553 : tensor<1x8x4x128xf32>
    %2555 = stablehlo.convert %2551 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2556 = stablehlo.broadcast_in_dim %2548, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2557 = stablehlo.multiply %2555, %2556 : tensor<1x8x4x128xf32>
    %2558 = stablehlo.subtract %2554, %2557 : tensor<1x8x4x128xf32>
    %2559 = stablehlo.convert %2551 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2560 = stablehlo.broadcast_in_dim %2549, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2561 = stablehlo.multiply %2559, %2560 : tensor<1x8x4x128xf32>
    %2562 = stablehlo.convert %2550 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2563 = stablehlo.broadcast_in_dim %2548, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2564 = stablehlo.multiply %2562, %2563 : tensor<1x8x4x128xf32>
    %2565 = stablehlo.add %2561, %2564 : tensor<1x8x4x128xf32>
    %2566 = stablehlo.concatenate %2558, %2565, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %2567 = stablehlo.convert %2566 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %2568 = stablehlo.reshape %2532 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2569 = stablehlo.dot_general %2567, %2568, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %2570 = stablehlo.transpose %2569, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2571 = stablehlo.reshape %2570 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_420 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2572 = stablehlo.broadcast_in_dim %cst_420, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2573 = stablehlo.divide %2571, %2572 : tensor<1x8x8x8xbf16>
    %2574 = stablehlo.tanh %2573 : tensor<1x8x8x8xbf16>
    %cst_421 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2575 = stablehlo.broadcast_in_dim %cst_421, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2576 = stablehlo.multiply %2574, %2575 : tensor<1x8x8x8xbf16>
    %2577 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %2578 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_422 = stablehlo.constant dense<4096> : tensor<i32>
    %2579 = stablehlo.broadcast_in_dim %c_422, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %2580 = stablehlo.subtract %2578, %2579 : tensor<1x8x1xi32>
    %2581 = stablehlo.broadcast_in_dim %2577, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %2582 = stablehlo.broadcast_in_dim %2580, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %2583 = stablehlo.compare GT, %2581, %2582, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_423 = stablehlo.constant dense<4096> : tensor<i32>
    %2584 = stablehlo.broadcast_in_dim %c_423, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %2585 = stablehlo.add %2578, %2584 : tensor<1x8x1xi32>
    %2586 = stablehlo.broadcast_in_dim %2577, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %2587 = stablehlo.broadcast_in_dim %2585, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %2588 = stablehlo.compare LT, %2586, %2587, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %2589 = stablehlo.and %2583, %2588 : tensor<1x8x8xi1>
    %2590 = stablehlo.and %arg237, %2589 : tensor<1x8x8xi1>
    %2591 = stablehlo.broadcast_in_dim %2590, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_424 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2592 = call @_where(%2591, %2576, %cst_424) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_425 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2593 = stablehlo.reduce(%2592 init: %cst_425) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_426 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2594 = stablehlo.broadcast_in_dim %cst_426, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %2595 = stablehlo.maximum %2594, %2593 : tensor<1x8x8xbf16>
    %2596 = stablehlo.broadcast_in_dim %2595, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %2597 = stablehlo.broadcast_in_dim %2596, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2598 = stablehlo.subtract %2592, %2597 : tensor<1x8x8x8xbf16>
    %2599 = stablehlo.exponential %2598 : tensor<1x8x8x8xbf16>
    %2600 = stablehlo.convert %2599 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_427 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2601 = stablehlo.reduce(%2600 init: %cst_427) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2602 = stablehlo.broadcast_in_dim %2601, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2603 = stablehlo.convert %2602 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %2604 = stablehlo.broadcast_in_dim %2603, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2605 = stablehlo.divide %2599, %2604 : tensor<1x8x8x8xbf16>
    %2606 = stablehlo.reshape %2605 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2607 = stablehlo.dot_general %2495, %2606, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %2608 = stablehlo.transpose %2607, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2609 = stablehlo.reshape %2608 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2610 = stablehlo.dot_general %2609, %arg38, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2611 = chlo.square %2610 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2612 = stablehlo.convert %2611 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_428 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2613 = stablehlo.reduce(%2612 init: %cst_428) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2614 = stablehlo.broadcast_in_dim %2613, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_429 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2615 = stablehlo.broadcast_in_dim %cst_429, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2616 = stablehlo.divide %2614, %2615 : tensor<1x8x1xf32>
    %2617 = stablehlo.convert %2616 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_430 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2618 = stablehlo.broadcast_in_dim %cst_430, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2619 = stablehlo.add %2617, %2618 : tensor<1x8x1xbf16>
    %2620 = stablehlo.rsqrt %2619 : tensor<1x8x1xbf16>
    %2621 = stablehlo.broadcast_in_dim %2620, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2622 = stablehlo.multiply %2610, %2621 : tensor<1x8x2304xbf16>
    %2623 = stablehlo.broadcast_in_dim %arg43, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_431 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2624 = stablehlo.broadcast_in_dim %cst_431, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2625 = stablehlo.add %2624, %2623 : tensor<1x1x2304xbf16>
    %2626 = stablehlo.broadcast_in_dim %2625, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2627 = stablehlo.multiply %2622, %2626 : tensor<1x8x2304xbf16>
    %2628 = stablehlo.add %2627, %2471 : tensor<1x8x2304xbf16>
    %2629 = chlo.square %2628 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2630 = stablehlo.convert %2629 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_432 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2631 = stablehlo.reduce(%2630 init: %cst_432) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2632 = stablehlo.broadcast_in_dim %2631, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_433 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2633 = stablehlo.broadcast_in_dim %cst_433, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2634 = stablehlo.divide %2632, %2633 : tensor<1x8x1xf32>
    %2635 = stablehlo.convert %2634 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_434 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2636 = stablehlo.broadcast_in_dim %cst_434, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2637 = stablehlo.add %2635, %2636 : tensor<1x8x1xbf16>
    %2638 = stablehlo.rsqrt %2637 : tensor<1x8x1xbf16>
    %2639 = stablehlo.broadcast_in_dim %2638, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2640 = stablehlo.multiply %2628, %2639 : tensor<1x8x2304xbf16>
    %2641 = stablehlo.broadcast_in_dim %arg46, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_435 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2642 = stablehlo.broadcast_in_dim %cst_435, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2643 = stablehlo.add %2642, %2641 : tensor<1x1x2304xbf16>
    %2644 = stablehlo.broadcast_in_dim %2643, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2645 = stablehlo.multiply %2640, %2644 : tensor<1x8x2304xbf16>
    %2646 = stablehlo.dot_general %2645, %arg41, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %2647 = stablehlo.slice %2646 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2648 = stablehlo.reshape %2647 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2649 = stablehlo.multiply %2648, %2648 : tensor<1x8x9216xbf16>
    %2650 = stablehlo.multiply %2649, %2648 : tensor<1x8x9216xbf16>
    %cst_436 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %2651 = stablehlo.broadcast_in_dim %cst_436, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2652 = stablehlo.multiply %2651, %2650 : tensor<1x8x9216xbf16>
    %2653 = stablehlo.add %2648, %2652 : tensor<1x8x9216xbf16>
    %cst_437 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %2654 = stablehlo.broadcast_in_dim %cst_437, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2655 = stablehlo.multiply %2654, %2653 : tensor<1x8x9216xbf16>
    %2656 = stablehlo.tanh %2655 : tensor<1x8x9216xbf16>
    %cst_438 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2657 = stablehlo.broadcast_in_dim %cst_438, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2658 = stablehlo.add %2657, %2656 : tensor<1x8x9216xbf16>
    %cst_439 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %2659 = stablehlo.broadcast_in_dim %cst_439, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2660 = stablehlo.multiply %2659, %2658 : tensor<1x8x9216xbf16>
    %2661 = stablehlo.multiply %2648, %2660 : tensor<1x8x9216xbf16>
    %2662 = stablehlo.slice %2646 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2663 = stablehlo.reshape %2662 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2664 = stablehlo.multiply %2661, %2663 : tensor<1x8x9216xbf16>
    %2665 = stablehlo.dot_general %2664, %arg42, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2666 = chlo.square %2665 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2667 = stablehlo.convert %2666 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_440 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2668 = stablehlo.reduce(%2667 init: %cst_440) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2669 = stablehlo.broadcast_in_dim %2668, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_441 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2670 = stablehlo.broadcast_in_dim %cst_441, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2671 = stablehlo.divide %2669, %2670 : tensor<1x8x1xf32>
    %2672 = stablehlo.convert %2671 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_442 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2673 = stablehlo.broadcast_in_dim %cst_442, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2674 = stablehlo.add %2672, %2673 : tensor<1x8x1xbf16>
    %2675 = stablehlo.rsqrt %2674 : tensor<1x8x1xbf16>
    %2676 = stablehlo.broadcast_in_dim %2675, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2677 = stablehlo.multiply %2665, %2676 : tensor<1x8x2304xbf16>
    %2678 = stablehlo.broadcast_in_dim %arg44, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_443 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2679 = stablehlo.broadcast_in_dim %cst_443, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2680 = stablehlo.add %2679, %2678 : tensor<1x1x2304xbf16>
    %2681 = stablehlo.broadcast_in_dim %2680, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2682 = stablehlo.multiply %2677, %2681 : tensor<1x8x2304xbf16>
    %2683 = stablehlo.add %2682, %2628 : tensor<1x8x2304xbf16>
    %2684 = chlo.square %2683 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2685 = stablehlo.convert %2684 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_444 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2686 = stablehlo.reduce(%2685 init: %cst_444) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2687 = stablehlo.broadcast_in_dim %2686, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_445 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2688 = stablehlo.broadcast_in_dim %cst_445, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2689 = stablehlo.divide %2687, %2688 : tensor<1x8x1xf32>
    %2690 = stablehlo.convert %2689 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_446 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2691 = stablehlo.broadcast_in_dim %cst_446, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2692 = stablehlo.add %2690, %2691 : tensor<1x8x1xbf16>
    %2693 = stablehlo.rsqrt %2692 : tensor<1x8x1xbf16>
    %2694 = stablehlo.broadcast_in_dim %2693, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2695 = stablehlo.multiply %2683, %2694 : tensor<1x8x2304xbf16>
    %2696 = stablehlo.broadcast_in_dim %arg54, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_447 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2697 = stablehlo.broadcast_in_dim %cst_447, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2698 = stablehlo.add %2697, %2696 : tensor<1x1x2304xbf16>
    %2699 = stablehlo.broadcast_in_dim %2698, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2700 = stablehlo.multiply %2695, %2699 : tensor<1x8x2304xbf16>
    %2701 = stablehlo.dot_general %2700, %arg49, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2702 = stablehlo.dot_general %arg48, %2700, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %2703 = stablehlo.transpose %2702, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %2704 = stablehlo.slice %2703 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2705 = stablehlo.reshape %2704 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2706 = stablehlo.slice %2703 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2707 = stablehlo.reshape %2706 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2708 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_448 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2709 = stablehlo.broadcast_in_dim %cst_448, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2710 = stablehlo.multiply %2709, %2708 : tensor<128xf32>
    %cst_449 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2711 = stablehlo.broadcast_in_dim %cst_449, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2712 = stablehlo.power %2711, %2710 : tensor<128xf32>
    %2713 = call @_pad(%2712) : (tensor<128xf32>) -> tensor<128xf32>
    %2714 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2715 = stablehlo.broadcast_in_dim %2713, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2716 = stablehlo.convert %2714 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2717 = stablehlo.broadcast_in_dim %2716, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2718 = stablehlo.broadcast_in_dim %2715, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2719 = stablehlo.divide %2717, %2718 : tensor<1x8x128xf32>
    %2720 = stablehlo.broadcast_in_dim %2719, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_450 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2721 = stablehlo.broadcast_in_dim %cst_450, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2722 = stablehlo.divide %2720, %2721 : tensor<1x8x1x128xf32>
    %2723 = stablehlo.sine %2722 : tensor<1x8x1x128xf32>
    %2724 = stablehlo.cosine %2722 : tensor<1x8x1x128xf32>
    %2725 = stablehlo.slice %2701 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2726 = stablehlo.slice %2701 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2727 = stablehlo.convert %2725 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2728 = stablehlo.broadcast_in_dim %2724, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2729 = stablehlo.multiply %2727, %2728 : tensor<1x8x8x128xf32>
    %2730 = stablehlo.convert %2726 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2731 = stablehlo.broadcast_in_dim %2723, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2732 = stablehlo.multiply %2730, %2731 : tensor<1x8x8x128xf32>
    %2733 = stablehlo.subtract %2729, %2732 : tensor<1x8x8x128xf32>
    %2734 = stablehlo.convert %2726 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2735 = stablehlo.broadcast_in_dim %2724, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2736 = stablehlo.multiply %2734, %2735 : tensor<1x8x8x128xf32>
    %2737 = stablehlo.convert %2725 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2738 = stablehlo.broadcast_in_dim %2723, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2739 = stablehlo.multiply %2737, %2738 : tensor<1x8x8x128xf32>
    %2740 = stablehlo.add %2736, %2739 : tensor<1x8x8x128xf32>
    %2741 = stablehlo.concatenate %2733, %2740, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %2742 = stablehlo.convert %2741 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_451 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %2743 = stablehlo.broadcast_in_dim %cst_451, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %2744 = stablehlo.multiply %2742, %2743 : tensor<1x8x8x256xbf16>
    %2745 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_452 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2746 = stablehlo.broadcast_in_dim %cst_452, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2747 = stablehlo.multiply %2746, %2745 : tensor<128xf32>
    %cst_453 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2748 = stablehlo.broadcast_in_dim %cst_453, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2749 = stablehlo.power %2748, %2747 : tensor<128xf32>
    %2750 = call @_pad(%2749) : (tensor<128xf32>) -> tensor<128xf32>
    %2751 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2752 = stablehlo.broadcast_in_dim %2750, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2753 = stablehlo.convert %2751 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2754 = stablehlo.broadcast_in_dim %2753, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2755 = stablehlo.broadcast_in_dim %2752, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2756 = stablehlo.divide %2754, %2755 : tensor<1x8x128xf32>
    %2757 = stablehlo.broadcast_in_dim %2756, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_454 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2758 = stablehlo.broadcast_in_dim %cst_454, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2759 = stablehlo.divide %2757, %2758 : tensor<1x8x1x128xf32>
    %2760 = stablehlo.sine %2759 : tensor<1x8x1x128xf32>
    %2761 = stablehlo.cosine %2759 : tensor<1x8x1x128xf32>
    %2762 = stablehlo.slice %2705 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2763 = stablehlo.slice %2705 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2764 = stablehlo.convert %2762 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2765 = stablehlo.broadcast_in_dim %2761, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2766 = stablehlo.multiply %2764, %2765 : tensor<1x8x4x128xf32>
    %2767 = stablehlo.convert %2763 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2768 = stablehlo.broadcast_in_dim %2760, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2769 = stablehlo.multiply %2767, %2768 : tensor<1x8x4x128xf32>
    %2770 = stablehlo.subtract %2766, %2769 : tensor<1x8x4x128xf32>
    %2771 = stablehlo.convert %2763 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2772 = stablehlo.broadcast_in_dim %2761, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2773 = stablehlo.multiply %2771, %2772 : tensor<1x8x4x128xf32>
    %2774 = stablehlo.convert %2762 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2775 = stablehlo.broadcast_in_dim %2760, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2776 = stablehlo.multiply %2774, %2775 : tensor<1x8x4x128xf32>
    %2777 = stablehlo.add %2773, %2776 : tensor<1x8x4x128xf32>
    %2778 = stablehlo.concatenate %2770, %2777, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %2779 = stablehlo.convert %2778 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %2780 = stablehlo.reshape %2744 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2781 = stablehlo.dot_general %2779, %2780, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %2782 = stablehlo.transpose %2781, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2783 = stablehlo.reshape %2782 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_455 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2784 = stablehlo.broadcast_in_dim %cst_455, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2785 = stablehlo.divide %2783, %2784 : tensor<1x8x8x8xbf16>
    %2786 = stablehlo.tanh %2785 : tensor<1x8x8x8xbf16>
    %cst_456 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2787 = stablehlo.broadcast_in_dim %cst_456, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2788 = stablehlo.multiply %2786, %2787 : tensor<1x8x8x8xbf16>
    %2789 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_457 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2790 = call @_where(%2789, %2788, %cst_457) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_458 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2791 = stablehlo.reduce(%2790 init: %cst_458) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_459 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2792 = stablehlo.broadcast_in_dim %cst_459, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %2793 = stablehlo.maximum %2792, %2791 : tensor<1x8x8xbf16>
    %2794 = stablehlo.broadcast_in_dim %2793, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %2795 = stablehlo.broadcast_in_dim %2794, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2796 = stablehlo.subtract %2790, %2795 : tensor<1x8x8x8xbf16>
    %2797 = stablehlo.exponential %2796 : tensor<1x8x8x8xbf16>
    %2798 = stablehlo.convert %2797 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_460 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2799 = stablehlo.reduce(%2798 init: %cst_460) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2800 = stablehlo.broadcast_in_dim %2799, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2801 = stablehlo.convert %2800 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %2802 = stablehlo.broadcast_in_dim %2801, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %2803 = stablehlo.divide %2797, %2802 : tensor<1x8x8x8xbf16>
    %2804 = stablehlo.reshape %2803 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2805 = stablehlo.dot_general %2707, %2804, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %2806 = stablehlo.transpose %2805, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2807 = stablehlo.reshape %2806 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2808 = stablehlo.dot_general %2807, %arg47, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2809 = chlo.square %2808 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2810 = stablehlo.convert %2809 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_461 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2811 = stablehlo.reduce(%2810 init: %cst_461) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2812 = stablehlo.broadcast_in_dim %2811, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_462 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2813 = stablehlo.broadcast_in_dim %cst_462, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2814 = stablehlo.divide %2812, %2813 : tensor<1x8x1xf32>
    %2815 = stablehlo.convert %2814 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_463 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2816 = stablehlo.broadcast_in_dim %cst_463, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2817 = stablehlo.add %2815, %2816 : tensor<1x8x1xbf16>
    %2818 = stablehlo.rsqrt %2817 : tensor<1x8x1xbf16>
    %2819 = stablehlo.broadcast_in_dim %2818, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2820 = stablehlo.multiply %2808, %2819 : tensor<1x8x2304xbf16>
    %2821 = stablehlo.broadcast_in_dim %arg52, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_464 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2822 = stablehlo.broadcast_in_dim %cst_464, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2823 = stablehlo.add %2822, %2821 : tensor<1x1x2304xbf16>
    %2824 = stablehlo.broadcast_in_dim %2823, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2825 = stablehlo.multiply %2820, %2824 : tensor<1x8x2304xbf16>
    %2826 = stablehlo.add %2825, %2683 : tensor<1x8x2304xbf16>
    %2827 = chlo.square %2826 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2828 = stablehlo.convert %2827 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_465 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2829 = stablehlo.reduce(%2828 init: %cst_465) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2830 = stablehlo.broadcast_in_dim %2829, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_466 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2831 = stablehlo.broadcast_in_dim %cst_466, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2832 = stablehlo.divide %2830, %2831 : tensor<1x8x1xf32>
    %2833 = stablehlo.convert %2832 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_467 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2834 = stablehlo.broadcast_in_dim %cst_467, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2835 = stablehlo.add %2833, %2834 : tensor<1x8x1xbf16>
    %2836 = stablehlo.rsqrt %2835 : tensor<1x8x1xbf16>
    %2837 = stablehlo.broadcast_in_dim %2836, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2838 = stablehlo.multiply %2826, %2837 : tensor<1x8x2304xbf16>
    %2839 = stablehlo.broadcast_in_dim %arg55, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_468 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2840 = stablehlo.broadcast_in_dim %cst_468, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2841 = stablehlo.add %2840, %2839 : tensor<1x1x2304xbf16>
    %2842 = stablehlo.broadcast_in_dim %2841, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2843 = stablehlo.multiply %2838, %2842 : tensor<1x8x2304xbf16>
    %2844 = stablehlo.dot_general %2843, %arg50, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %2845 = stablehlo.slice %2844 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2846 = stablehlo.reshape %2845 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2847 = stablehlo.multiply %2846, %2846 : tensor<1x8x9216xbf16>
    %2848 = stablehlo.multiply %2847, %2846 : tensor<1x8x9216xbf16>
    %cst_469 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %2849 = stablehlo.broadcast_in_dim %cst_469, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2850 = stablehlo.multiply %2849, %2848 : tensor<1x8x9216xbf16>
    %2851 = stablehlo.add %2846, %2850 : tensor<1x8x9216xbf16>
    %cst_470 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %2852 = stablehlo.broadcast_in_dim %cst_470, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2853 = stablehlo.multiply %2852, %2851 : tensor<1x8x9216xbf16>
    %2854 = stablehlo.tanh %2853 : tensor<1x8x9216xbf16>
    %cst_471 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2855 = stablehlo.broadcast_in_dim %cst_471, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2856 = stablehlo.add %2855, %2854 : tensor<1x8x9216xbf16>
    %cst_472 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %2857 = stablehlo.broadcast_in_dim %cst_472, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %2858 = stablehlo.multiply %2857, %2856 : tensor<1x8x9216xbf16>
    %2859 = stablehlo.multiply %2846, %2858 : tensor<1x8x9216xbf16>
    %2860 = stablehlo.slice %2844 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %2861 = stablehlo.reshape %2860 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %2862 = stablehlo.multiply %2859, %2861 : tensor<1x8x9216xbf16>
    %2863 = stablehlo.dot_general %2862, %arg51, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2864 = chlo.square %2863 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2865 = stablehlo.convert %2864 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_473 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2866 = stablehlo.reduce(%2865 init: %cst_473) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2867 = stablehlo.broadcast_in_dim %2866, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_474 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2868 = stablehlo.broadcast_in_dim %cst_474, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2869 = stablehlo.divide %2867, %2868 : tensor<1x8x1xf32>
    %2870 = stablehlo.convert %2869 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_475 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2871 = stablehlo.broadcast_in_dim %cst_475, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2872 = stablehlo.add %2870, %2871 : tensor<1x8x1xbf16>
    %2873 = stablehlo.rsqrt %2872 : tensor<1x8x1xbf16>
    %2874 = stablehlo.broadcast_in_dim %2873, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2875 = stablehlo.multiply %2863, %2874 : tensor<1x8x2304xbf16>
    %2876 = stablehlo.broadcast_in_dim %arg53, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_476 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2877 = stablehlo.broadcast_in_dim %cst_476, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2878 = stablehlo.add %2877, %2876 : tensor<1x1x2304xbf16>
    %2879 = stablehlo.broadcast_in_dim %2878, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2880 = stablehlo.multiply %2875, %2879 : tensor<1x8x2304xbf16>
    %2881 = stablehlo.add %2880, %2826 : tensor<1x8x2304xbf16>
    %2882 = chlo.square %2881 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %2883 = stablehlo.convert %2882 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_477 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2884 = stablehlo.reduce(%2883 init: %cst_477) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2885 = stablehlo.broadcast_in_dim %2884, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_478 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %2886 = stablehlo.broadcast_in_dim %cst_478, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2887 = stablehlo.divide %2885, %2886 : tensor<1x8x1xf32>
    %2888 = stablehlo.convert %2887 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_479 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %2889 = stablehlo.broadcast_in_dim %cst_479, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %2890 = stablehlo.add %2888, %2889 : tensor<1x8x1xbf16>
    %2891 = stablehlo.rsqrt %2890 : tensor<1x8x1xbf16>
    %2892 = stablehlo.broadcast_in_dim %2891, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %2893 = stablehlo.multiply %2881, %2892 : tensor<1x8x2304xbf16>
    %2894 = stablehlo.broadcast_in_dim %arg63, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_480 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2895 = stablehlo.broadcast_in_dim %cst_480, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %2896 = stablehlo.add %2895, %2894 : tensor<1x1x2304xbf16>
    %2897 = stablehlo.broadcast_in_dim %2896, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %2898 = stablehlo.multiply %2893, %2897 : tensor<1x8x2304xbf16>
    %2899 = stablehlo.dot_general %2898, %arg58, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %2900 = stablehlo.dot_general %arg57, %2898, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %2901 = stablehlo.transpose %2900, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %2902 = stablehlo.slice %2901 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2903 = stablehlo.reshape %2902 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2904 = stablehlo.slice %2901 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %2905 = stablehlo.reshape %2904 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %2906 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_481 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2907 = stablehlo.broadcast_in_dim %cst_481, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2908 = stablehlo.multiply %2907, %2906 : tensor<128xf32>
    %cst_482 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2909 = stablehlo.broadcast_in_dim %cst_482, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2910 = stablehlo.power %2909, %2908 : tensor<128xf32>
    %2911 = call @_pad(%2910) : (tensor<128xf32>) -> tensor<128xf32>
    %2912 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2913 = stablehlo.broadcast_in_dim %2911, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2914 = stablehlo.convert %2912 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2915 = stablehlo.broadcast_in_dim %2914, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2916 = stablehlo.broadcast_in_dim %2913, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2917 = stablehlo.divide %2915, %2916 : tensor<1x8x128xf32>
    %2918 = stablehlo.broadcast_in_dim %2917, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_483 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2919 = stablehlo.broadcast_in_dim %cst_483, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2920 = stablehlo.divide %2918, %2919 : tensor<1x8x1x128xf32>
    %2921 = stablehlo.sine %2920 : tensor<1x8x1x128xf32>
    %2922 = stablehlo.cosine %2920 : tensor<1x8x1x128xf32>
    %2923 = stablehlo.slice %2899 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2924 = stablehlo.slice %2899 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %2925 = stablehlo.convert %2923 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2926 = stablehlo.broadcast_in_dim %2922, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2927 = stablehlo.multiply %2925, %2926 : tensor<1x8x8x128xf32>
    %2928 = stablehlo.convert %2924 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2929 = stablehlo.broadcast_in_dim %2921, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2930 = stablehlo.multiply %2928, %2929 : tensor<1x8x8x128xf32>
    %2931 = stablehlo.subtract %2927, %2930 : tensor<1x8x8x128xf32>
    %2932 = stablehlo.convert %2924 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2933 = stablehlo.broadcast_in_dim %2922, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2934 = stablehlo.multiply %2932, %2933 : tensor<1x8x8x128xf32>
    %2935 = stablehlo.convert %2923 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2936 = stablehlo.broadcast_in_dim %2921, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %2937 = stablehlo.multiply %2935, %2936 : tensor<1x8x8x128xf32>
    %2938 = stablehlo.add %2934, %2937 : tensor<1x8x8x128xf32>
    %2939 = stablehlo.concatenate %2931, %2938, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %2940 = stablehlo.convert %2939 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_484 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %2941 = stablehlo.broadcast_in_dim %cst_484, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %2942 = stablehlo.multiply %2940, %2941 : tensor<1x8x8x256xbf16>
    %2943 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_485 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %2944 = stablehlo.broadcast_in_dim %cst_485, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2945 = stablehlo.multiply %2944, %2943 : tensor<128xf32>
    %cst_486 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %2946 = stablehlo.broadcast_in_dim %cst_486, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2947 = stablehlo.power %2946, %2945 : tensor<128xf32>
    %2948 = call @_pad(%2947) : (tensor<128xf32>) -> tensor<128xf32>
    %2949 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2950 = stablehlo.broadcast_in_dim %2948, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %2951 = stablehlo.convert %2949 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %2952 = stablehlo.broadcast_in_dim %2951, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %2953 = stablehlo.broadcast_in_dim %2950, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %2954 = stablehlo.divide %2952, %2953 : tensor<1x8x128xf32>
    %2955 = stablehlo.broadcast_in_dim %2954, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_487 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2956 = stablehlo.broadcast_in_dim %cst_487, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %2957 = stablehlo.divide %2955, %2956 : tensor<1x8x1x128xf32>
    %2958 = stablehlo.sine %2957 : tensor<1x8x1x128xf32>
    %2959 = stablehlo.cosine %2957 : tensor<1x8x1x128xf32>
    %2960 = stablehlo.slice %2903 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2961 = stablehlo.slice %2903 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %2962 = stablehlo.convert %2960 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2963 = stablehlo.broadcast_in_dim %2959, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2964 = stablehlo.multiply %2962, %2963 : tensor<1x8x4x128xf32>
    %2965 = stablehlo.convert %2961 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2966 = stablehlo.broadcast_in_dim %2958, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2967 = stablehlo.multiply %2965, %2966 : tensor<1x8x4x128xf32>
    %2968 = stablehlo.subtract %2964, %2967 : tensor<1x8x4x128xf32>
    %2969 = stablehlo.convert %2961 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2970 = stablehlo.broadcast_in_dim %2959, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2971 = stablehlo.multiply %2969, %2970 : tensor<1x8x4x128xf32>
    %2972 = stablehlo.convert %2960 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %2973 = stablehlo.broadcast_in_dim %2958, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %2974 = stablehlo.multiply %2972, %2973 : tensor<1x8x4x128xf32>
    %2975 = stablehlo.add %2971, %2974 : tensor<1x8x4x128xf32>
    %2976 = stablehlo.concatenate %2968, %2975, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %2977 = stablehlo.convert %2976 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %2978 = stablehlo.reshape %2942 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %2979 = stablehlo.dot_general %2977, %2978, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %2980 = stablehlo.transpose %2979, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %2981 = stablehlo.reshape %2980 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_488 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2982 = stablehlo.broadcast_in_dim %cst_488, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2983 = stablehlo.divide %2981, %2982 : tensor<1x8x8x8xbf16>
    %2984 = stablehlo.tanh %2983 : tensor<1x8x8x8xbf16>
    %cst_489 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %2985 = stablehlo.broadcast_in_dim %cst_489, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %2986 = stablehlo.multiply %2984, %2985 : tensor<1x8x8x8xbf16>
    %2987 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %2988 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_490 = stablehlo.constant dense<4096> : tensor<i32>
    %2989 = stablehlo.broadcast_in_dim %c_490, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %2990 = stablehlo.subtract %2988, %2989 : tensor<1x8x1xi32>
    %2991 = stablehlo.broadcast_in_dim %2987, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %2992 = stablehlo.broadcast_in_dim %2990, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %2993 = stablehlo.compare GT, %2991, %2992, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_491 = stablehlo.constant dense<4096> : tensor<i32>
    %2994 = stablehlo.broadcast_in_dim %c_491, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %2995 = stablehlo.add %2988, %2994 : tensor<1x8x1xi32>
    %2996 = stablehlo.broadcast_in_dim %2987, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %2997 = stablehlo.broadcast_in_dim %2995, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %2998 = stablehlo.compare LT, %2996, %2997, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %2999 = stablehlo.and %2993, %2998 : tensor<1x8x8xi1>
    %3000 = stablehlo.and %arg237, %2999 : tensor<1x8x8xi1>
    %3001 = stablehlo.broadcast_in_dim %3000, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_492 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %3002 = call @_where(%3001, %2986, %cst_492) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_493 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3003 = stablehlo.reduce(%3002 init: %cst_493) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_494 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3004 = stablehlo.broadcast_in_dim %cst_494, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %3005 = stablehlo.maximum %3004, %3003 : tensor<1x8x8xbf16>
    %3006 = stablehlo.broadcast_in_dim %3005, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %3007 = stablehlo.broadcast_in_dim %3006, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3008 = stablehlo.subtract %3002, %3007 : tensor<1x8x8x8xbf16>
    %3009 = stablehlo.exponential %3008 : tensor<1x8x8x8xbf16>
    %3010 = stablehlo.convert %3009 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_495 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3011 = stablehlo.reduce(%3010 init: %cst_495) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3012 = stablehlo.broadcast_in_dim %3011, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3013 = stablehlo.convert %3012 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %3014 = stablehlo.broadcast_in_dim %3013, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3015 = stablehlo.divide %3009, %3014 : tensor<1x8x8x8xbf16>
    %3016 = stablehlo.reshape %3015 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3017 = stablehlo.dot_general %2905, %3016, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %3018 = stablehlo.transpose %3017, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3019 = stablehlo.reshape %3018 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3020 = stablehlo.dot_general %3019, %arg56, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3021 = chlo.square %3020 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3022 = stablehlo.convert %3021 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_496 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3023 = stablehlo.reduce(%3022 init: %cst_496) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3024 = stablehlo.broadcast_in_dim %3023, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_497 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3025 = stablehlo.broadcast_in_dim %cst_497, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3026 = stablehlo.divide %3024, %3025 : tensor<1x8x1xf32>
    %3027 = stablehlo.convert %3026 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_498 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3028 = stablehlo.broadcast_in_dim %cst_498, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3029 = stablehlo.add %3027, %3028 : tensor<1x8x1xbf16>
    %3030 = stablehlo.rsqrt %3029 : tensor<1x8x1xbf16>
    %3031 = stablehlo.broadcast_in_dim %3030, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3032 = stablehlo.multiply %3020, %3031 : tensor<1x8x2304xbf16>
    %3033 = stablehlo.broadcast_in_dim %arg61, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_499 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3034 = stablehlo.broadcast_in_dim %cst_499, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3035 = stablehlo.add %3034, %3033 : tensor<1x1x2304xbf16>
    %3036 = stablehlo.broadcast_in_dim %3035, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3037 = stablehlo.multiply %3032, %3036 : tensor<1x8x2304xbf16>
    %3038 = stablehlo.add %3037, %2881 : tensor<1x8x2304xbf16>
    %3039 = chlo.square %3038 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3040 = stablehlo.convert %3039 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_500 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3041 = stablehlo.reduce(%3040 init: %cst_500) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3042 = stablehlo.broadcast_in_dim %3041, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_501 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3043 = stablehlo.broadcast_in_dim %cst_501, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3044 = stablehlo.divide %3042, %3043 : tensor<1x8x1xf32>
    %3045 = stablehlo.convert %3044 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_502 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3046 = stablehlo.broadcast_in_dim %cst_502, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3047 = stablehlo.add %3045, %3046 : tensor<1x8x1xbf16>
    %3048 = stablehlo.rsqrt %3047 : tensor<1x8x1xbf16>
    %3049 = stablehlo.broadcast_in_dim %3048, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3050 = stablehlo.multiply %3038, %3049 : tensor<1x8x2304xbf16>
    %3051 = stablehlo.broadcast_in_dim %arg64, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_503 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3052 = stablehlo.broadcast_in_dim %cst_503, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3053 = stablehlo.add %3052, %3051 : tensor<1x1x2304xbf16>
    %3054 = stablehlo.broadcast_in_dim %3053, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3055 = stablehlo.multiply %3050, %3054 : tensor<1x8x2304xbf16>
    %3056 = stablehlo.dot_general %3055, %arg59, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %3057 = stablehlo.slice %3056 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3058 = stablehlo.reshape %3057 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3059 = stablehlo.multiply %3058, %3058 : tensor<1x8x9216xbf16>
    %3060 = stablehlo.multiply %3059, %3058 : tensor<1x8x9216xbf16>
    %cst_504 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %3061 = stablehlo.broadcast_in_dim %cst_504, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3062 = stablehlo.multiply %3061, %3060 : tensor<1x8x9216xbf16>
    %3063 = stablehlo.add %3058, %3062 : tensor<1x8x9216xbf16>
    %cst_505 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %3064 = stablehlo.broadcast_in_dim %cst_505, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3065 = stablehlo.multiply %3064, %3063 : tensor<1x8x9216xbf16>
    %3066 = stablehlo.tanh %3065 : tensor<1x8x9216xbf16>
    %cst_506 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3067 = stablehlo.broadcast_in_dim %cst_506, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3068 = stablehlo.add %3067, %3066 : tensor<1x8x9216xbf16>
    %cst_507 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %3069 = stablehlo.broadcast_in_dim %cst_507, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3070 = stablehlo.multiply %3069, %3068 : tensor<1x8x9216xbf16>
    %3071 = stablehlo.multiply %3058, %3070 : tensor<1x8x9216xbf16>
    %3072 = stablehlo.slice %3056 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3073 = stablehlo.reshape %3072 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3074 = stablehlo.multiply %3071, %3073 : tensor<1x8x9216xbf16>
    %3075 = stablehlo.dot_general %3074, %arg60, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3076 = chlo.square %3075 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3077 = stablehlo.convert %3076 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_508 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3078 = stablehlo.reduce(%3077 init: %cst_508) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3079 = stablehlo.broadcast_in_dim %3078, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_509 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3080 = stablehlo.broadcast_in_dim %cst_509, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3081 = stablehlo.divide %3079, %3080 : tensor<1x8x1xf32>
    %3082 = stablehlo.convert %3081 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_510 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3083 = stablehlo.broadcast_in_dim %cst_510, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3084 = stablehlo.add %3082, %3083 : tensor<1x8x1xbf16>
    %3085 = stablehlo.rsqrt %3084 : tensor<1x8x1xbf16>
    %3086 = stablehlo.broadcast_in_dim %3085, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3087 = stablehlo.multiply %3075, %3086 : tensor<1x8x2304xbf16>
    %3088 = stablehlo.broadcast_in_dim %arg62, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_511 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3089 = stablehlo.broadcast_in_dim %cst_511, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3090 = stablehlo.add %3089, %3088 : tensor<1x1x2304xbf16>
    %3091 = stablehlo.broadcast_in_dim %3090, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3092 = stablehlo.multiply %3087, %3091 : tensor<1x8x2304xbf16>
    %3093 = stablehlo.add %3092, %3038 : tensor<1x8x2304xbf16>
    %3094 = chlo.square %3093 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3095 = stablehlo.convert %3094 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_512 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3096 = stablehlo.reduce(%3095 init: %cst_512) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3097 = stablehlo.broadcast_in_dim %3096, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_513 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3098 = stablehlo.broadcast_in_dim %cst_513, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3099 = stablehlo.divide %3097, %3098 : tensor<1x8x1xf32>
    %3100 = stablehlo.convert %3099 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_514 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3101 = stablehlo.broadcast_in_dim %cst_514, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3102 = stablehlo.add %3100, %3101 : tensor<1x8x1xbf16>
    %3103 = stablehlo.rsqrt %3102 : tensor<1x8x1xbf16>
    %3104 = stablehlo.broadcast_in_dim %3103, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3105 = stablehlo.multiply %3093, %3104 : tensor<1x8x2304xbf16>
    %3106 = stablehlo.broadcast_in_dim %arg72, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_515 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3107 = stablehlo.broadcast_in_dim %cst_515, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3108 = stablehlo.add %3107, %3106 : tensor<1x1x2304xbf16>
    %3109 = stablehlo.broadcast_in_dim %3108, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3110 = stablehlo.multiply %3105, %3109 : tensor<1x8x2304xbf16>
    %3111 = stablehlo.dot_general %3110, %arg67, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3112 = stablehlo.dot_general %arg66, %3110, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %3113 = stablehlo.transpose %3112, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %3114 = stablehlo.slice %3113 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3115 = stablehlo.reshape %3114 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3116 = stablehlo.slice %3113 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3117 = stablehlo.reshape %3116 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3118 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_516 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3119 = stablehlo.broadcast_in_dim %cst_516, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3120 = stablehlo.multiply %3119, %3118 : tensor<128xf32>
    %cst_517 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3121 = stablehlo.broadcast_in_dim %cst_517, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3122 = stablehlo.power %3121, %3120 : tensor<128xf32>
    %3123 = call @_pad(%3122) : (tensor<128xf32>) -> tensor<128xf32>
    %3124 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3125 = stablehlo.broadcast_in_dim %3123, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3126 = stablehlo.convert %3124 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3127 = stablehlo.broadcast_in_dim %3126, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3128 = stablehlo.broadcast_in_dim %3125, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3129 = stablehlo.divide %3127, %3128 : tensor<1x8x128xf32>
    %3130 = stablehlo.broadcast_in_dim %3129, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_518 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3131 = stablehlo.broadcast_in_dim %cst_518, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3132 = stablehlo.divide %3130, %3131 : tensor<1x8x1x128xf32>
    %3133 = stablehlo.sine %3132 : tensor<1x8x1x128xf32>
    %3134 = stablehlo.cosine %3132 : tensor<1x8x1x128xf32>
    %3135 = stablehlo.slice %3111 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3136 = stablehlo.slice %3111 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3137 = stablehlo.convert %3135 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3138 = stablehlo.broadcast_in_dim %3134, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3139 = stablehlo.multiply %3137, %3138 : tensor<1x8x8x128xf32>
    %3140 = stablehlo.convert %3136 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3141 = stablehlo.broadcast_in_dim %3133, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3142 = stablehlo.multiply %3140, %3141 : tensor<1x8x8x128xf32>
    %3143 = stablehlo.subtract %3139, %3142 : tensor<1x8x8x128xf32>
    %3144 = stablehlo.convert %3136 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3145 = stablehlo.broadcast_in_dim %3134, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3146 = stablehlo.multiply %3144, %3145 : tensor<1x8x8x128xf32>
    %3147 = stablehlo.convert %3135 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3148 = stablehlo.broadcast_in_dim %3133, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3149 = stablehlo.multiply %3147, %3148 : tensor<1x8x8x128xf32>
    %3150 = stablehlo.add %3146, %3149 : tensor<1x8x8x128xf32>
    %3151 = stablehlo.concatenate %3143, %3150, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %3152 = stablehlo.convert %3151 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_519 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %3153 = stablehlo.broadcast_in_dim %cst_519, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %3154 = stablehlo.multiply %3152, %3153 : tensor<1x8x8x256xbf16>
    %3155 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_520 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3156 = stablehlo.broadcast_in_dim %cst_520, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3157 = stablehlo.multiply %3156, %3155 : tensor<128xf32>
    %cst_521 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3158 = stablehlo.broadcast_in_dim %cst_521, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3159 = stablehlo.power %3158, %3157 : tensor<128xf32>
    %3160 = call @_pad(%3159) : (tensor<128xf32>) -> tensor<128xf32>
    %3161 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3162 = stablehlo.broadcast_in_dim %3160, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3163 = stablehlo.convert %3161 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3164 = stablehlo.broadcast_in_dim %3163, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3165 = stablehlo.broadcast_in_dim %3162, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3166 = stablehlo.divide %3164, %3165 : tensor<1x8x128xf32>
    %3167 = stablehlo.broadcast_in_dim %3166, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_522 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3168 = stablehlo.broadcast_in_dim %cst_522, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3169 = stablehlo.divide %3167, %3168 : tensor<1x8x1x128xf32>
    %3170 = stablehlo.sine %3169 : tensor<1x8x1x128xf32>
    %3171 = stablehlo.cosine %3169 : tensor<1x8x1x128xf32>
    %3172 = stablehlo.slice %3115 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3173 = stablehlo.slice %3115 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3174 = stablehlo.convert %3172 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3175 = stablehlo.broadcast_in_dim %3171, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3176 = stablehlo.multiply %3174, %3175 : tensor<1x8x4x128xf32>
    %3177 = stablehlo.convert %3173 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3178 = stablehlo.broadcast_in_dim %3170, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3179 = stablehlo.multiply %3177, %3178 : tensor<1x8x4x128xf32>
    %3180 = stablehlo.subtract %3176, %3179 : tensor<1x8x4x128xf32>
    %3181 = stablehlo.convert %3173 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3182 = stablehlo.broadcast_in_dim %3171, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3183 = stablehlo.multiply %3181, %3182 : tensor<1x8x4x128xf32>
    %3184 = stablehlo.convert %3172 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3185 = stablehlo.broadcast_in_dim %3170, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3186 = stablehlo.multiply %3184, %3185 : tensor<1x8x4x128xf32>
    %3187 = stablehlo.add %3183, %3186 : tensor<1x8x4x128xf32>
    %3188 = stablehlo.concatenate %3180, %3187, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %3189 = stablehlo.convert %3188 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %3190 = stablehlo.reshape %3154 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3191 = stablehlo.dot_general %3189, %3190, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %3192 = stablehlo.transpose %3191, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3193 = stablehlo.reshape %3192 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_523 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3194 = stablehlo.broadcast_in_dim %cst_523, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3195 = stablehlo.divide %3193, %3194 : tensor<1x8x8x8xbf16>
    %3196 = stablehlo.tanh %3195 : tensor<1x8x8x8xbf16>
    %cst_524 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3197 = stablehlo.broadcast_in_dim %cst_524, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3198 = stablehlo.multiply %3196, %3197 : tensor<1x8x8x8xbf16>
    %3199 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_525 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %3200 = call @_where(%3199, %3198, %cst_525) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_526 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3201 = stablehlo.reduce(%3200 init: %cst_526) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_527 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3202 = stablehlo.broadcast_in_dim %cst_527, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %3203 = stablehlo.maximum %3202, %3201 : tensor<1x8x8xbf16>
    %3204 = stablehlo.broadcast_in_dim %3203, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %3205 = stablehlo.broadcast_in_dim %3204, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3206 = stablehlo.subtract %3200, %3205 : tensor<1x8x8x8xbf16>
    %3207 = stablehlo.exponential %3206 : tensor<1x8x8x8xbf16>
    %3208 = stablehlo.convert %3207 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_528 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3209 = stablehlo.reduce(%3208 init: %cst_528) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3210 = stablehlo.broadcast_in_dim %3209, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3211 = stablehlo.convert %3210 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %3212 = stablehlo.broadcast_in_dim %3211, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3213 = stablehlo.divide %3207, %3212 : tensor<1x8x8x8xbf16>
    %3214 = stablehlo.reshape %3213 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3215 = stablehlo.dot_general %3117, %3214, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %3216 = stablehlo.transpose %3215, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3217 = stablehlo.reshape %3216 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3218 = stablehlo.dot_general %3217, %arg65, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3219 = chlo.square %3218 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3220 = stablehlo.convert %3219 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_529 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3221 = stablehlo.reduce(%3220 init: %cst_529) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3222 = stablehlo.broadcast_in_dim %3221, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_530 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3223 = stablehlo.broadcast_in_dim %cst_530, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3224 = stablehlo.divide %3222, %3223 : tensor<1x8x1xf32>
    %3225 = stablehlo.convert %3224 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_531 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3226 = stablehlo.broadcast_in_dim %cst_531, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3227 = stablehlo.add %3225, %3226 : tensor<1x8x1xbf16>
    %3228 = stablehlo.rsqrt %3227 : tensor<1x8x1xbf16>
    %3229 = stablehlo.broadcast_in_dim %3228, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3230 = stablehlo.multiply %3218, %3229 : tensor<1x8x2304xbf16>
    %3231 = stablehlo.broadcast_in_dim %arg70, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_532 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3232 = stablehlo.broadcast_in_dim %cst_532, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3233 = stablehlo.add %3232, %3231 : tensor<1x1x2304xbf16>
    %3234 = stablehlo.broadcast_in_dim %3233, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3235 = stablehlo.multiply %3230, %3234 : tensor<1x8x2304xbf16>
    %3236 = stablehlo.add %3235, %3093 : tensor<1x8x2304xbf16>
    %3237 = chlo.square %3236 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3238 = stablehlo.convert %3237 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_533 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3239 = stablehlo.reduce(%3238 init: %cst_533) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3240 = stablehlo.broadcast_in_dim %3239, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_534 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3241 = stablehlo.broadcast_in_dim %cst_534, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3242 = stablehlo.divide %3240, %3241 : tensor<1x8x1xf32>
    %3243 = stablehlo.convert %3242 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_535 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3244 = stablehlo.broadcast_in_dim %cst_535, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3245 = stablehlo.add %3243, %3244 : tensor<1x8x1xbf16>
    %3246 = stablehlo.rsqrt %3245 : tensor<1x8x1xbf16>
    %3247 = stablehlo.broadcast_in_dim %3246, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3248 = stablehlo.multiply %3236, %3247 : tensor<1x8x2304xbf16>
    %3249 = stablehlo.broadcast_in_dim %arg73, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_536 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3250 = stablehlo.broadcast_in_dim %cst_536, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3251 = stablehlo.add %3250, %3249 : tensor<1x1x2304xbf16>
    %3252 = stablehlo.broadcast_in_dim %3251, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3253 = stablehlo.multiply %3248, %3252 : tensor<1x8x2304xbf16>
    %3254 = stablehlo.dot_general %3253, %arg68, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %3255 = stablehlo.slice %3254 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3256 = stablehlo.reshape %3255 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3257 = stablehlo.multiply %3256, %3256 : tensor<1x8x9216xbf16>
    %3258 = stablehlo.multiply %3257, %3256 : tensor<1x8x9216xbf16>
    %cst_537 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %3259 = stablehlo.broadcast_in_dim %cst_537, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3260 = stablehlo.multiply %3259, %3258 : tensor<1x8x9216xbf16>
    %3261 = stablehlo.add %3256, %3260 : tensor<1x8x9216xbf16>
    %cst_538 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %3262 = stablehlo.broadcast_in_dim %cst_538, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3263 = stablehlo.multiply %3262, %3261 : tensor<1x8x9216xbf16>
    %3264 = stablehlo.tanh %3263 : tensor<1x8x9216xbf16>
    %cst_539 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3265 = stablehlo.broadcast_in_dim %cst_539, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3266 = stablehlo.add %3265, %3264 : tensor<1x8x9216xbf16>
    %cst_540 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %3267 = stablehlo.broadcast_in_dim %cst_540, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3268 = stablehlo.multiply %3267, %3266 : tensor<1x8x9216xbf16>
    %3269 = stablehlo.multiply %3256, %3268 : tensor<1x8x9216xbf16>
    %3270 = stablehlo.slice %3254 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3271 = stablehlo.reshape %3270 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3272 = stablehlo.multiply %3269, %3271 : tensor<1x8x9216xbf16>
    %3273 = stablehlo.dot_general %3272, %arg69, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3274 = chlo.square %3273 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3275 = stablehlo.convert %3274 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_541 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3276 = stablehlo.reduce(%3275 init: %cst_541) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3277 = stablehlo.broadcast_in_dim %3276, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_542 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3278 = stablehlo.broadcast_in_dim %cst_542, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3279 = stablehlo.divide %3277, %3278 : tensor<1x8x1xf32>
    %3280 = stablehlo.convert %3279 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_543 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3281 = stablehlo.broadcast_in_dim %cst_543, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3282 = stablehlo.add %3280, %3281 : tensor<1x8x1xbf16>
    %3283 = stablehlo.rsqrt %3282 : tensor<1x8x1xbf16>
    %3284 = stablehlo.broadcast_in_dim %3283, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3285 = stablehlo.multiply %3273, %3284 : tensor<1x8x2304xbf16>
    %3286 = stablehlo.broadcast_in_dim %arg71, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_544 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3287 = stablehlo.broadcast_in_dim %cst_544, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3288 = stablehlo.add %3287, %3286 : tensor<1x1x2304xbf16>
    %3289 = stablehlo.broadcast_in_dim %3288, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3290 = stablehlo.multiply %3285, %3289 : tensor<1x8x2304xbf16>
    %3291 = stablehlo.add %3290, %3236 : tensor<1x8x2304xbf16>
    %3292 = chlo.square %3291 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3293 = stablehlo.convert %3292 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_545 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3294 = stablehlo.reduce(%3293 init: %cst_545) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3295 = stablehlo.broadcast_in_dim %3294, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_546 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3296 = stablehlo.broadcast_in_dim %cst_546, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3297 = stablehlo.divide %3295, %3296 : tensor<1x8x1xf32>
    %3298 = stablehlo.convert %3297 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_547 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3299 = stablehlo.broadcast_in_dim %cst_547, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3300 = stablehlo.add %3298, %3299 : tensor<1x8x1xbf16>
    %3301 = stablehlo.rsqrt %3300 : tensor<1x8x1xbf16>
    %3302 = stablehlo.broadcast_in_dim %3301, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3303 = stablehlo.multiply %3291, %3302 : tensor<1x8x2304xbf16>
    %3304 = stablehlo.broadcast_in_dim %arg81, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_548 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3305 = stablehlo.broadcast_in_dim %cst_548, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3306 = stablehlo.add %3305, %3304 : tensor<1x1x2304xbf16>
    %3307 = stablehlo.broadcast_in_dim %3306, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3308 = stablehlo.multiply %3303, %3307 : tensor<1x8x2304xbf16>
    %3309 = stablehlo.dot_general %3308, %arg76, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3310 = stablehlo.dot_general %arg75, %3308, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %3311 = stablehlo.transpose %3310, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %3312 = stablehlo.slice %3311 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3313 = stablehlo.reshape %3312 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3314 = stablehlo.slice %3311 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3315 = stablehlo.reshape %3314 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3316 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_549 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3317 = stablehlo.broadcast_in_dim %cst_549, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3318 = stablehlo.multiply %3317, %3316 : tensor<128xf32>
    %cst_550 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3319 = stablehlo.broadcast_in_dim %cst_550, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3320 = stablehlo.power %3319, %3318 : tensor<128xf32>
    %3321 = call @_pad(%3320) : (tensor<128xf32>) -> tensor<128xf32>
    %3322 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3323 = stablehlo.broadcast_in_dim %3321, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3324 = stablehlo.convert %3322 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3325 = stablehlo.broadcast_in_dim %3324, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3326 = stablehlo.broadcast_in_dim %3323, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3327 = stablehlo.divide %3325, %3326 : tensor<1x8x128xf32>
    %3328 = stablehlo.broadcast_in_dim %3327, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_551 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3329 = stablehlo.broadcast_in_dim %cst_551, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3330 = stablehlo.divide %3328, %3329 : tensor<1x8x1x128xf32>
    %3331 = stablehlo.sine %3330 : tensor<1x8x1x128xf32>
    %3332 = stablehlo.cosine %3330 : tensor<1x8x1x128xf32>
    %3333 = stablehlo.slice %3309 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3334 = stablehlo.slice %3309 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3335 = stablehlo.convert %3333 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3336 = stablehlo.broadcast_in_dim %3332, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3337 = stablehlo.multiply %3335, %3336 : tensor<1x8x8x128xf32>
    %3338 = stablehlo.convert %3334 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3339 = stablehlo.broadcast_in_dim %3331, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3340 = stablehlo.multiply %3338, %3339 : tensor<1x8x8x128xf32>
    %3341 = stablehlo.subtract %3337, %3340 : tensor<1x8x8x128xf32>
    %3342 = stablehlo.convert %3334 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3343 = stablehlo.broadcast_in_dim %3332, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3344 = stablehlo.multiply %3342, %3343 : tensor<1x8x8x128xf32>
    %3345 = stablehlo.convert %3333 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3346 = stablehlo.broadcast_in_dim %3331, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3347 = stablehlo.multiply %3345, %3346 : tensor<1x8x8x128xf32>
    %3348 = stablehlo.add %3344, %3347 : tensor<1x8x8x128xf32>
    %3349 = stablehlo.concatenate %3341, %3348, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %3350 = stablehlo.convert %3349 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_552 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %3351 = stablehlo.broadcast_in_dim %cst_552, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %3352 = stablehlo.multiply %3350, %3351 : tensor<1x8x8x256xbf16>
    %3353 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_553 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3354 = stablehlo.broadcast_in_dim %cst_553, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3355 = stablehlo.multiply %3354, %3353 : tensor<128xf32>
    %cst_554 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3356 = stablehlo.broadcast_in_dim %cst_554, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3357 = stablehlo.power %3356, %3355 : tensor<128xf32>
    %3358 = call @_pad(%3357) : (tensor<128xf32>) -> tensor<128xf32>
    %3359 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3360 = stablehlo.broadcast_in_dim %3358, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3361 = stablehlo.convert %3359 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3362 = stablehlo.broadcast_in_dim %3361, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3363 = stablehlo.broadcast_in_dim %3360, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3364 = stablehlo.divide %3362, %3363 : tensor<1x8x128xf32>
    %3365 = stablehlo.broadcast_in_dim %3364, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_555 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3366 = stablehlo.broadcast_in_dim %cst_555, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3367 = stablehlo.divide %3365, %3366 : tensor<1x8x1x128xf32>
    %3368 = stablehlo.sine %3367 : tensor<1x8x1x128xf32>
    %3369 = stablehlo.cosine %3367 : tensor<1x8x1x128xf32>
    %3370 = stablehlo.slice %3313 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3371 = stablehlo.slice %3313 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3372 = stablehlo.convert %3370 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3373 = stablehlo.broadcast_in_dim %3369, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3374 = stablehlo.multiply %3372, %3373 : tensor<1x8x4x128xf32>
    %3375 = stablehlo.convert %3371 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3376 = stablehlo.broadcast_in_dim %3368, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3377 = stablehlo.multiply %3375, %3376 : tensor<1x8x4x128xf32>
    %3378 = stablehlo.subtract %3374, %3377 : tensor<1x8x4x128xf32>
    %3379 = stablehlo.convert %3371 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3380 = stablehlo.broadcast_in_dim %3369, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3381 = stablehlo.multiply %3379, %3380 : tensor<1x8x4x128xf32>
    %3382 = stablehlo.convert %3370 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3383 = stablehlo.broadcast_in_dim %3368, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3384 = stablehlo.multiply %3382, %3383 : tensor<1x8x4x128xf32>
    %3385 = stablehlo.add %3381, %3384 : tensor<1x8x4x128xf32>
    %3386 = stablehlo.concatenate %3378, %3385, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %3387 = stablehlo.convert %3386 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %3388 = stablehlo.reshape %3352 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3389 = stablehlo.dot_general %3387, %3388, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %3390 = stablehlo.transpose %3389, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3391 = stablehlo.reshape %3390 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_556 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3392 = stablehlo.broadcast_in_dim %cst_556, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3393 = stablehlo.divide %3391, %3392 : tensor<1x8x8x8xbf16>
    %3394 = stablehlo.tanh %3393 : tensor<1x8x8x8xbf16>
    %cst_557 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3395 = stablehlo.broadcast_in_dim %cst_557, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3396 = stablehlo.multiply %3394, %3395 : tensor<1x8x8x8xbf16>
    %3397 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %3398 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_558 = stablehlo.constant dense<4096> : tensor<i32>
    %3399 = stablehlo.broadcast_in_dim %c_558, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %3400 = stablehlo.subtract %3398, %3399 : tensor<1x8x1xi32>
    %3401 = stablehlo.broadcast_in_dim %3397, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %3402 = stablehlo.broadcast_in_dim %3400, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %3403 = stablehlo.compare GT, %3401, %3402, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_559 = stablehlo.constant dense<4096> : tensor<i32>
    %3404 = stablehlo.broadcast_in_dim %c_559, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %3405 = stablehlo.add %3398, %3404 : tensor<1x8x1xi32>
    %3406 = stablehlo.broadcast_in_dim %3397, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %3407 = stablehlo.broadcast_in_dim %3405, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %3408 = stablehlo.compare LT, %3406, %3407, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %3409 = stablehlo.and %3403, %3408 : tensor<1x8x8xi1>
    %3410 = stablehlo.and %arg237, %3409 : tensor<1x8x8xi1>
    %3411 = stablehlo.broadcast_in_dim %3410, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_560 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %3412 = call @_where(%3411, %3396, %cst_560) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_561 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3413 = stablehlo.reduce(%3412 init: %cst_561) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_562 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3414 = stablehlo.broadcast_in_dim %cst_562, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %3415 = stablehlo.maximum %3414, %3413 : tensor<1x8x8xbf16>
    %3416 = stablehlo.broadcast_in_dim %3415, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %3417 = stablehlo.broadcast_in_dim %3416, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3418 = stablehlo.subtract %3412, %3417 : tensor<1x8x8x8xbf16>
    %3419 = stablehlo.exponential %3418 : tensor<1x8x8x8xbf16>
    %3420 = stablehlo.convert %3419 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_563 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3421 = stablehlo.reduce(%3420 init: %cst_563) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3422 = stablehlo.broadcast_in_dim %3421, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3423 = stablehlo.convert %3422 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %3424 = stablehlo.broadcast_in_dim %3423, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3425 = stablehlo.divide %3419, %3424 : tensor<1x8x8x8xbf16>
    %3426 = stablehlo.reshape %3425 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3427 = stablehlo.dot_general %3315, %3426, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %3428 = stablehlo.transpose %3427, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3429 = stablehlo.reshape %3428 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3430 = stablehlo.dot_general %3429, %arg74, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3431 = chlo.square %3430 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3432 = stablehlo.convert %3431 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_564 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3433 = stablehlo.reduce(%3432 init: %cst_564) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3434 = stablehlo.broadcast_in_dim %3433, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_565 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3435 = stablehlo.broadcast_in_dim %cst_565, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3436 = stablehlo.divide %3434, %3435 : tensor<1x8x1xf32>
    %3437 = stablehlo.convert %3436 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_566 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3438 = stablehlo.broadcast_in_dim %cst_566, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3439 = stablehlo.add %3437, %3438 : tensor<1x8x1xbf16>
    %3440 = stablehlo.rsqrt %3439 : tensor<1x8x1xbf16>
    %3441 = stablehlo.broadcast_in_dim %3440, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3442 = stablehlo.multiply %3430, %3441 : tensor<1x8x2304xbf16>
    %3443 = stablehlo.broadcast_in_dim %arg79, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_567 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3444 = stablehlo.broadcast_in_dim %cst_567, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3445 = stablehlo.add %3444, %3443 : tensor<1x1x2304xbf16>
    %3446 = stablehlo.broadcast_in_dim %3445, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3447 = stablehlo.multiply %3442, %3446 : tensor<1x8x2304xbf16>
    %3448 = stablehlo.add %3447, %3291 : tensor<1x8x2304xbf16>
    %3449 = chlo.square %3448 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3450 = stablehlo.convert %3449 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_568 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3451 = stablehlo.reduce(%3450 init: %cst_568) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3452 = stablehlo.broadcast_in_dim %3451, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_569 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3453 = stablehlo.broadcast_in_dim %cst_569, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3454 = stablehlo.divide %3452, %3453 : tensor<1x8x1xf32>
    %3455 = stablehlo.convert %3454 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_570 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3456 = stablehlo.broadcast_in_dim %cst_570, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3457 = stablehlo.add %3455, %3456 : tensor<1x8x1xbf16>
    %3458 = stablehlo.rsqrt %3457 : tensor<1x8x1xbf16>
    %3459 = stablehlo.broadcast_in_dim %3458, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3460 = stablehlo.multiply %3448, %3459 : tensor<1x8x2304xbf16>
    %3461 = stablehlo.broadcast_in_dim %arg82, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_571 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3462 = stablehlo.broadcast_in_dim %cst_571, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3463 = stablehlo.add %3462, %3461 : tensor<1x1x2304xbf16>
    %3464 = stablehlo.broadcast_in_dim %3463, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3465 = stablehlo.multiply %3460, %3464 : tensor<1x8x2304xbf16>
    %3466 = stablehlo.dot_general %3465, %arg77, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %3467 = stablehlo.slice %3466 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3468 = stablehlo.reshape %3467 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3469 = stablehlo.multiply %3468, %3468 : tensor<1x8x9216xbf16>
    %3470 = stablehlo.multiply %3469, %3468 : tensor<1x8x9216xbf16>
    %cst_572 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %3471 = stablehlo.broadcast_in_dim %cst_572, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3472 = stablehlo.multiply %3471, %3470 : tensor<1x8x9216xbf16>
    %3473 = stablehlo.add %3468, %3472 : tensor<1x8x9216xbf16>
    %cst_573 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %3474 = stablehlo.broadcast_in_dim %cst_573, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3475 = stablehlo.multiply %3474, %3473 : tensor<1x8x9216xbf16>
    %3476 = stablehlo.tanh %3475 : tensor<1x8x9216xbf16>
    %cst_574 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3477 = stablehlo.broadcast_in_dim %cst_574, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3478 = stablehlo.add %3477, %3476 : tensor<1x8x9216xbf16>
    %cst_575 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %3479 = stablehlo.broadcast_in_dim %cst_575, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3480 = stablehlo.multiply %3479, %3478 : tensor<1x8x9216xbf16>
    %3481 = stablehlo.multiply %3468, %3480 : tensor<1x8x9216xbf16>
    %3482 = stablehlo.slice %3466 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3483 = stablehlo.reshape %3482 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3484 = stablehlo.multiply %3481, %3483 : tensor<1x8x9216xbf16>
    %3485 = stablehlo.dot_general %3484, %arg78, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3486 = chlo.square %3485 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3487 = stablehlo.convert %3486 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_576 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3488 = stablehlo.reduce(%3487 init: %cst_576) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3489 = stablehlo.broadcast_in_dim %3488, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_577 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3490 = stablehlo.broadcast_in_dim %cst_577, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3491 = stablehlo.divide %3489, %3490 : tensor<1x8x1xf32>
    %3492 = stablehlo.convert %3491 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_578 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3493 = stablehlo.broadcast_in_dim %cst_578, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3494 = stablehlo.add %3492, %3493 : tensor<1x8x1xbf16>
    %3495 = stablehlo.rsqrt %3494 : tensor<1x8x1xbf16>
    %3496 = stablehlo.broadcast_in_dim %3495, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3497 = stablehlo.multiply %3485, %3496 : tensor<1x8x2304xbf16>
    %3498 = stablehlo.broadcast_in_dim %arg80, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_579 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3499 = stablehlo.broadcast_in_dim %cst_579, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3500 = stablehlo.add %3499, %3498 : tensor<1x1x2304xbf16>
    %3501 = stablehlo.broadcast_in_dim %3500, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3502 = stablehlo.multiply %3497, %3501 : tensor<1x8x2304xbf16>
    %3503 = stablehlo.add %3502, %3448 : tensor<1x8x2304xbf16>
    %3504 = chlo.square %3503 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3505 = stablehlo.convert %3504 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_580 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3506 = stablehlo.reduce(%3505 init: %cst_580) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3507 = stablehlo.broadcast_in_dim %3506, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_581 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3508 = stablehlo.broadcast_in_dim %cst_581, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3509 = stablehlo.divide %3507, %3508 : tensor<1x8x1xf32>
    %3510 = stablehlo.convert %3509 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_582 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3511 = stablehlo.broadcast_in_dim %cst_582, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3512 = stablehlo.add %3510, %3511 : tensor<1x8x1xbf16>
    %3513 = stablehlo.rsqrt %3512 : tensor<1x8x1xbf16>
    %3514 = stablehlo.broadcast_in_dim %3513, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3515 = stablehlo.multiply %3503, %3514 : tensor<1x8x2304xbf16>
    %3516 = stablehlo.broadcast_in_dim %arg90, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_583 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3517 = stablehlo.broadcast_in_dim %cst_583, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3518 = stablehlo.add %3517, %3516 : tensor<1x1x2304xbf16>
    %3519 = stablehlo.broadcast_in_dim %3518, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3520 = stablehlo.multiply %3515, %3519 : tensor<1x8x2304xbf16>
    %3521 = stablehlo.dot_general %3520, %arg85, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3522 = stablehlo.dot_general %arg84, %3520, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %3523 = stablehlo.transpose %3522, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %3524 = stablehlo.slice %3523 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3525 = stablehlo.reshape %3524 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3526 = stablehlo.slice %3523 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3527 = stablehlo.reshape %3526 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3528 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_584 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3529 = stablehlo.broadcast_in_dim %cst_584, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3530 = stablehlo.multiply %3529, %3528 : tensor<128xf32>
    %cst_585 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3531 = stablehlo.broadcast_in_dim %cst_585, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3532 = stablehlo.power %3531, %3530 : tensor<128xf32>
    %3533 = call @_pad(%3532) : (tensor<128xf32>) -> tensor<128xf32>
    %3534 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3535 = stablehlo.broadcast_in_dim %3533, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3536 = stablehlo.convert %3534 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3537 = stablehlo.broadcast_in_dim %3536, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3538 = stablehlo.broadcast_in_dim %3535, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3539 = stablehlo.divide %3537, %3538 : tensor<1x8x128xf32>
    %3540 = stablehlo.broadcast_in_dim %3539, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_586 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3541 = stablehlo.broadcast_in_dim %cst_586, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3542 = stablehlo.divide %3540, %3541 : tensor<1x8x1x128xf32>
    %3543 = stablehlo.sine %3542 : tensor<1x8x1x128xf32>
    %3544 = stablehlo.cosine %3542 : tensor<1x8x1x128xf32>
    %3545 = stablehlo.slice %3521 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3546 = stablehlo.slice %3521 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3547 = stablehlo.convert %3545 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3548 = stablehlo.broadcast_in_dim %3544, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3549 = stablehlo.multiply %3547, %3548 : tensor<1x8x8x128xf32>
    %3550 = stablehlo.convert %3546 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3551 = stablehlo.broadcast_in_dim %3543, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3552 = stablehlo.multiply %3550, %3551 : tensor<1x8x8x128xf32>
    %3553 = stablehlo.subtract %3549, %3552 : tensor<1x8x8x128xf32>
    %3554 = stablehlo.convert %3546 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3555 = stablehlo.broadcast_in_dim %3544, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3556 = stablehlo.multiply %3554, %3555 : tensor<1x8x8x128xf32>
    %3557 = stablehlo.convert %3545 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3558 = stablehlo.broadcast_in_dim %3543, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3559 = stablehlo.multiply %3557, %3558 : tensor<1x8x8x128xf32>
    %3560 = stablehlo.add %3556, %3559 : tensor<1x8x8x128xf32>
    %3561 = stablehlo.concatenate %3553, %3560, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %3562 = stablehlo.convert %3561 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_587 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %3563 = stablehlo.broadcast_in_dim %cst_587, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %3564 = stablehlo.multiply %3562, %3563 : tensor<1x8x8x256xbf16>
    %3565 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_588 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3566 = stablehlo.broadcast_in_dim %cst_588, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3567 = stablehlo.multiply %3566, %3565 : tensor<128xf32>
    %cst_589 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3568 = stablehlo.broadcast_in_dim %cst_589, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3569 = stablehlo.power %3568, %3567 : tensor<128xf32>
    %3570 = call @_pad(%3569) : (tensor<128xf32>) -> tensor<128xf32>
    %3571 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3572 = stablehlo.broadcast_in_dim %3570, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3573 = stablehlo.convert %3571 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3574 = stablehlo.broadcast_in_dim %3573, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3575 = stablehlo.broadcast_in_dim %3572, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3576 = stablehlo.divide %3574, %3575 : tensor<1x8x128xf32>
    %3577 = stablehlo.broadcast_in_dim %3576, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_590 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3578 = stablehlo.broadcast_in_dim %cst_590, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3579 = stablehlo.divide %3577, %3578 : tensor<1x8x1x128xf32>
    %3580 = stablehlo.sine %3579 : tensor<1x8x1x128xf32>
    %3581 = stablehlo.cosine %3579 : tensor<1x8x1x128xf32>
    %3582 = stablehlo.slice %3525 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3583 = stablehlo.slice %3525 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3584 = stablehlo.convert %3582 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3585 = stablehlo.broadcast_in_dim %3581, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3586 = stablehlo.multiply %3584, %3585 : tensor<1x8x4x128xf32>
    %3587 = stablehlo.convert %3583 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3588 = stablehlo.broadcast_in_dim %3580, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3589 = stablehlo.multiply %3587, %3588 : tensor<1x8x4x128xf32>
    %3590 = stablehlo.subtract %3586, %3589 : tensor<1x8x4x128xf32>
    %3591 = stablehlo.convert %3583 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3592 = stablehlo.broadcast_in_dim %3581, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3593 = stablehlo.multiply %3591, %3592 : tensor<1x8x4x128xf32>
    %3594 = stablehlo.convert %3582 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3595 = stablehlo.broadcast_in_dim %3580, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3596 = stablehlo.multiply %3594, %3595 : tensor<1x8x4x128xf32>
    %3597 = stablehlo.add %3593, %3596 : tensor<1x8x4x128xf32>
    %3598 = stablehlo.concatenate %3590, %3597, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %3599 = stablehlo.convert %3598 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %3600 = stablehlo.reshape %3564 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3601 = stablehlo.dot_general %3599, %3600, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %3602 = stablehlo.transpose %3601, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3603 = stablehlo.reshape %3602 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_591 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3604 = stablehlo.broadcast_in_dim %cst_591, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3605 = stablehlo.divide %3603, %3604 : tensor<1x8x8x8xbf16>
    %3606 = stablehlo.tanh %3605 : tensor<1x8x8x8xbf16>
    %cst_592 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3607 = stablehlo.broadcast_in_dim %cst_592, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3608 = stablehlo.multiply %3606, %3607 : tensor<1x8x8x8xbf16>
    %3609 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_593 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %3610 = call @_where(%3609, %3608, %cst_593) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_594 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3611 = stablehlo.reduce(%3610 init: %cst_594) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_595 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3612 = stablehlo.broadcast_in_dim %cst_595, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %3613 = stablehlo.maximum %3612, %3611 : tensor<1x8x8xbf16>
    %3614 = stablehlo.broadcast_in_dim %3613, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %3615 = stablehlo.broadcast_in_dim %3614, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3616 = stablehlo.subtract %3610, %3615 : tensor<1x8x8x8xbf16>
    %3617 = stablehlo.exponential %3616 : tensor<1x8x8x8xbf16>
    %3618 = stablehlo.convert %3617 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_596 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3619 = stablehlo.reduce(%3618 init: %cst_596) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3620 = stablehlo.broadcast_in_dim %3619, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3621 = stablehlo.convert %3620 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %3622 = stablehlo.broadcast_in_dim %3621, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3623 = stablehlo.divide %3617, %3622 : tensor<1x8x8x8xbf16>
    %3624 = stablehlo.reshape %3623 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3625 = stablehlo.dot_general %3527, %3624, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %3626 = stablehlo.transpose %3625, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3627 = stablehlo.reshape %3626 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3628 = stablehlo.dot_general %3627, %arg83, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3629 = chlo.square %3628 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3630 = stablehlo.convert %3629 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_597 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3631 = stablehlo.reduce(%3630 init: %cst_597) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3632 = stablehlo.broadcast_in_dim %3631, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_598 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3633 = stablehlo.broadcast_in_dim %cst_598, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3634 = stablehlo.divide %3632, %3633 : tensor<1x8x1xf32>
    %3635 = stablehlo.convert %3634 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_599 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3636 = stablehlo.broadcast_in_dim %cst_599, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3637 = stablehlo.add %3635, %3636 : tensor<1x8x1xbf16>
    %3638 = stablehlo.rsqrt %3637 : tensor<1x8x1xbf16>
    %3639 = stablehlo.broadcast_in_dim %3638, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3640 = stablehlo.multiply %3628, %3639 : tensor<1x8x2304xbf16>
    %3641 = stablehlo.broadcast_in_dim %arg88, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_600 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3642 = stablehlo.broadcast_in_dim %cst_600, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3643 = stablehlo.add %3642, %3641 : tensor<1x1x2304xbf16>
    %3644 = stablehlo.broadcast_in_dim %3643, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3645 = stablehlo.multiply %3640, %3644 : tensor<1x8x2304xbf16>
    %3646 = stablehlo.add %3645, %3503 : tensor<1x8x2304xbf16>
    %3647 = chlo.square %3646 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3648 = stablehlo.convert %3647 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_601 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3649 = stablehlo.reduce(%3648 init: %cst_601) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3650 = stablehlo.broadcast_in_dim %3649, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_602 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3651 = stablehlo.broadcast_in_dim %cst_602, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3652 = stablehlo.divide %3650, %3651 : tensor<1x8x1xf32>
    %3653 = stablehlo.convert %3652 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_603 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3654 = stablehlo.broadcast_in_dim %cst_603, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3655 = stablehlo.add %3653, %3654 : tensor<1x8x1xbf16>
    %3656 = stablehlo.rsqrt %3655 : tensor<1x8x1xbf16>
    %3657 = stablehlo.broadcast_in_dim %3656, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3658 = stablehlo.multiply %3646, %3657 : tensor<1x8x2304xbf16>
    %3659 = stablehlo.broadcast_in_dim %arg91, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_604 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3660 = stablehlo.broadcast_in_dim %cst_604, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3661 = stablehlo.add %3660, %3659 : tensor<1x1x2304xbf16>
    %3662 = stablehlo.broadcast_in_dim %3661, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3663 = stablehlo.multiply %3658, %3662 : tensor<1x8x2304xbf16>
    %3664 = stablehlo.dot_general %3663, %arg86, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %3665 = stablehlo.slice %3664 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3666 = stablehlo.reshape %3665 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3667 = stablehlo.multiply %3666, %3666 : tensor<1x8x9216xbf16>
    %3668 = stablehlo.multiply %3667, %3666 : tensor<1x8x9216xbf16>
    %cst_605 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %3669 = stablehlo.broadcast_in_dim %cst_605, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3670 = stablehlo.multiply %3669, %3668 : tensor<1x8x9216xbf16>
    %3671 = stablehlo.add %3666, %3670 : tensor<1x8x9216xbf16>
    %cst_606 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %3672 = stablehlo.broadcast_in_dim %cst_606, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3673 = stablehlo.multiply %3672, %3671 : tensor<1x8x9216xbf16>
    %3674 = stablehlo.tanh %3673 : tensor<1x8x9216xbf16>
    %cst_607 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3675 = stablehlo.broadcast_in_dim %cst_607, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3676 = stablehlo.add %3675, %3674 : tensor<1x8x9216xbf16>
    %cst_608 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %3677 = stablehlo.broadcast_in_dim %cst_608, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3678 = stablehlo.multiply %3677, %3676 : tensor<1x8x9216xbf16>
    %3679 = stablehlo.multiply %3666, %3678 : tensor<1x8x9216xbf16>
    %3680 = stablehlo.slice %3664 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3681 = stablehlo.reshape %3680 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3682 = stablehlo.multiply %3679, %3681 : tensor<1x8x9216xbf16>
    %3683 = stablehlo.dot_general %3682, %arg87, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3684 = chlo.square %3683 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3685 = stablehlo.convert %3684 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_609 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3686 = stablehlo.reduce(%3685 init: %cst_609) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3687 = stablehlo.broadcast_in_dim %3686, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_610 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3688 = stablehlo.broadcast_in_dim %cst_610, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3689 = stablehlo.divide %3687, %3688 : tensor<1x8x1xf32>
    %3690 = stablehlo.convert %3689 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_611 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3691 = stablehlo.broadcast_in_dim %cst_611, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3692 = stablehlo.add %3690, %3691 : tensor<1x8x1xbf16>
    %3693 = stablehlo.rsqrt %3692 : tensor<1x8x1xbf16>
    %3694 = stablehlo.broadcast_in_dim %3693, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3695 = stablehlo.multiply %3683, %3694 : tensor<1x8x2304xbf16>
    %3696 = stablehlo.broadcast_in_dim %arg89, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_612 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3697 = stablehlo.broadcast_in_dim %cst_612, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3698 = stablehlo.add %3697, %3696 : tensor<1x1x2304xbf16>
    %3699 = stablehlo.broadcast_in_dim %3698, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3700 = stablehlo.multiply %3695, %3699 : tensor<1x8x2304xbf16>
    %3701 = stablehlo.add %3700, %3646 : tensor<1x8x2304xbf16>
    %3702 = chlo.square %3701 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3703 = stablehlo.convert %3702 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_613 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3704 = stablehlo.reduce(%3703 init: %cst_613) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3705 = stablehlo.broadcast_in_dim %3704, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_614 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3706 = stablehlo.broadcast_in_dim %cst_614, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3707 = stablehlo.divide %3705, %3706 : tensor<1x8x1xf32>
    %3708 = stablehlo.convert %3707 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_615 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3709 = stablehlo.broadcast_in_dim %cst_615, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3710 = stablehlo.add %3708, %3709 : tensor<1x8x1xbf16>
    %3711 = stablehlo.rsqrt %3710 : tensor<1x8x1xbf16>
    %3712 = stablehlo.broadcast_in_dim %3711, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3713 = stablehlo.multiply %3701, %3712 : tensor<1x8x2304xbf16>
    %3714 = stablehlo.broadcast_in_dim %arg99, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_616 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3715 = stablehlo.broadcast_in_dim %cst_616, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3716 = stablehlo.add %3715, %3714 : tensor<1x1x2304xbf16>
    %3717 = stablehlo.broadcast_in_dim %3716, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3718 = stablehlo.multiply %3713, %3717 : tensor<1x8x2304xbf16>
    %3719 = stablehlo.dot_general %3718, %arg94, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3720 = stablehlo.dot_general %arg93, %3718, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %3721 = stablehlo.transpose %3720, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %3722 = stablehlo.slice %3721 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3723 = stablehlo.reshape %3722 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3724 = stablehlo.slice %3721 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3725 = stablehlo.reshape %3724 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3726 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_617 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3727 = stablehlo.broadcast_in_dim %cst_617, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3728 = stablehlo.multiply %3727, %3726 : tensor<128xf32>
    %cst_618 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3729 = stablehlo.broadcast_in_dim %cst_618, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3730 = stablehlo.power %3729, %3728 : tensor<128xf32>
    %3731 = call @_pad(%3730) : (tensor<128xf32>) -> tensor<128xf32>
    %3732 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3733 = stablehlo.broadcast_in_dim %3731, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3734 = stablehlo.convert %3732 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3735 = stablehlo.broadcast_in_dim %3734, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3736 = stablehlo.broadcast_in_dim %3733, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3737 = stablehlo.divide %3735, %3736 : tensor<1x8x128xf32>
    %3738 = stablehlo.broadcast_in_dim %3737, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_619 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3739 = stablehlo.broadcast_in_dim %cst_619, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3740 = stablehlo.divide %3738, %3739 : tensor<1x8x1x128xf32>
    %3741 = stablehlo.sine %3740 : tensor<1x8x1x128xf32>
    %3742 = stablehlo.cosine %3740 : tensor<1x8x1x128xf32>
    %3743 = stablehlo.slice %3719 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3744 = stablehlo.slice %3719 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3745 = stablehlo.convert %3743 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3746 = stablehlo.broadcast_in_dim %3742, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3747 = stablehlo.multiply %3745, %3746 : tensor<1x8x8x128xf32>
    %3748 = stablehlo.convert %3744 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3749 = stablehlo.broadcast_in_dim %3741, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3750 = stablehlo.multiply %3748, %3749 : tensor<1x8x8x128xf32>
    %3751 = stablehlo.subtract %3747, %3750 : tensor<1x8x8x128xf32>
    %3752 = stablehlo.convert %3744 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3753 = stablehlo.broadcast_in_dim %3742, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3754 = stablehlo.multiply %3752, %3753 : tensor<1x8x8x128xf32>
    %3755 = stablehlo.convert %3743 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3756 = stablehlo.broadcast_in_dim %3741, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3757 = stablehlo.multiply %3755, %3756 : tensor<1x8x8x128xf32>
    %3758 = stablehlo.add %3754, %3757 : tensor<1x8x8x128xf32>
    %3759 = stablehlo.concatenate %3751, %3758, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %3760 = stablehlo.convert %3759 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_620 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %3761 = stablehlo.broadcast_in_dim %cst_620, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %3762 = stablehlo.multiply %3760, %3761 : tensor<1x8x8x256xbf16>
    %3763 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_621 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3764 = stablehlo.broadcast_in_dim %cst_621, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3765 = stablehlo.multiply %3764, %3763 : tensor<128xf32>
    %cst_622 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3766 = stablehlo.broadcast_in_dim %cst_622, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3767 = stablehlo.power %3766, %3765 : tensor<128xf32>
    %3768 = call @_pad(%3767) : (tensor<128xf32>) -> tensor<128xf32>
    %3769 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3770 = stablehlo.broadcast_in_dim %3768, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3771 = stablehlo.convert %3769 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3772 = stablehlo.broadcast_in_dim %3771, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3773 = stablehlo.broadcast_in_dim %3770, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3774 = stablehlo.divide %3772, %3773 : tensor<1x8x128xf32>
    %3775 = stablehlo.broadcast_in_dim %3774, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_623 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3776 = stablehlo.broadcast_in_dim %cst_623, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3777 = stablehlo.divide %3775, %3776 : tensor<1x8x1x128xf32>
    %3778 = stablehlo.sine %3777 : tensor<1x8x1x128xf32>
    %3779 = stablehlo.cosine %3777 : tensor<1x8x1x128xf32>
    %3780 = stablehlo.slice %3723 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3781 = stablehlo.slice %3723 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3782 = stablehlo.convert %3780 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3783 = stablehlo.broadcast_in_dim %3779, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3784 = stablehlo.multiply %3782, %3783 : tensor<1x8x4x128xf32>
    %3785 = stablehlo.convert %3781 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3786 = stablehlo.broadcast_in_dim %3778, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3787 = stablehlo.multiply %3785, %3786 : tensor<1x8x4x128xf32>
    %3788 = stablehlo.subtract %3784, %3787 : tensor<1x8x4x128xf32>
    %3789 = stablehlo.convert %3781 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3790 = stablehlo.broadcast_in_dim %3779, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3791 = stablehlo.multiply %3789, %3790 : tensor<1x8x4x128xf32>
    %3792 = stablehlo.convert %3780 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3793 = stablehlo.broadcast_in_dim %3778, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3794 = stablehlo.multiply %3792, %3793 : tensor<1x8x4x128xf32>
    %3795 = stablehlo.add %3791, %3794 : tensor<1x8x4x128xf32>
    %3796 = stablehlo.concatenate %3788, %3795, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %3797 = stablehlo.convert %3796 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %3798 = stablehlo.reshape %3762 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3799 = stablehlo.dot_general %3797, %3798, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %3800 = stablehlo.transpose %3799, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3801 = stablehlo.reshape %3800 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_624 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3802 = stablehlo.broadcast_in_dim %cst_624, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3803 = stablehlo.divide %3801, %3802 : tensor<1x8x8x8xbf16>
    %3804 = stablehlo.tanh %3803 : tensor<1x8x8x8xbf16>
    %cst_625 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %3805 = stablehlo.broadcast_in_dim %cst_625, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3806 = stablehlo.multiply %3804, %3805 : tensor<1x8x8x8xbf16>
    %3807 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %3808 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_626 = stablehlo.constant dense<4096> : tensor<i32>
    %3809 = stablehlo.broadcast_in_dim %c_626, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %3810 = stablehlo.subtract %3808, %3809 : tensor<1x8x1xi32>
    %3811 = stablehlo.broadcast_in_dim %3807, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %3812 = stablehlo.broadcast_in_dim %3810, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %3813 = stablehlo.compare GT, %3811, %3812, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_627 = stablehlo.constant dense<4096> : tensor<i32>
    %3814 = stablehlo.broadcast_in_dim %c_627, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %3815 = stablehlo.add %3808, %3814 : tensor<1x8x1xi32>
    %3816 = stablehlo.broadcast_in_dim %3807, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %3817 = stablehlo.broadcast_in_dim %3815, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %3818 = stablehlo.compare LT, %3816, %3817, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %3819 = stablehlo.and %3813, %3818 : tensor<1x8x8xi1>
    %3820 = stablehlo.and %arg237, %3819 : tensor<1x8x8xi1>
    %3821 = stablehlo.broadcast_in_dim %3820, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_628 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %3822 = call @_where(%3821, %3806, %cst_628) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_629 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3823 = stablehlo.reduce(%3822 init: %cst_629) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_630 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3824 = stablehlo.broadcast_in_dim %cst_630, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %3825 = stablehlo.maximum %3824, %3823 : tensor<1x8x8xbf16>
    %3826 = stablehlo.broadcast_in_dim %3825, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %3827 = stablehlo.broadcast_in_dim %3826, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3828 = stablehlo.subtract %3822, %3827 : tensor<1x8x8x8xbf16>
    %3829 = stablehlo.exponential %3828 : tensor<1x8x8x8xbf16>
    %3830 = stablehlo.convert %3829 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_631 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3831 = stablehlo.reduce(%3830 init: %cst_631) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3832 = stablehlo.broadcast_in_dim %3831, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3833 = stablehlo.convert %3832 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %3834 = stablehlo.broadcast_in_dim %3833, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %3835 = stablehlo.divide %3829, %3834 : tensor<1x8x8x8xbf16>
    %3836 = stablehlo.reshape %3835 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %3837 = stablehlo.dot_general %3725, %3836, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %3838 = stablehlo.transpose %3837, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %3839 = stablehlo.reshape %3838 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3840 = stablehlo.dot_general %3839, %arg92, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3841 = chlo.square %3840 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3842 = stablehlo.convert %3841 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_632 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3843 = stablehlo.reduce(%3842 init: %cst_632) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3844 = stablehlo.broadcast_in_dim %3843, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_633 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3845 = stablehlo.broadcast_in_dim %cst_633, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3846 = stablehlo.divide %3844, %3845 : tensor<1x8x1xf32>
    %3847 = stablehlo.convert %3846 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_634 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3848 = stablehlo.broadcast_in_dim %cst_634, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3849 = stablehlo.add %3847, %3848 : tensor<1x8x1xbf16>
    %3850 = stablehlo.rsqrt %3849 : tensor<1x8x1xbf16>
    %3851 = stablehlo.broadcast_in_dim %3850, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3852 = stablehlo.multiply %3840, %3851 : tensor<1x8x2304xbf16>
    %3853 = stablehlo.broadcast_in_dim %arg97, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_635 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3854 = stablehlo.broadcast_in_dim %cst_635, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3855 = stablehlo.add %3854, %3853 : tensor<1x1x2304xbf16>
    %3856 = stablehlo.broadcast_in_dim %3855, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3857 = stablehlo.multiply %3852, %3856 : tensor<1x8x2304xbf16>
    %3858 = stablehlo.add %3857, %3701 : tensor<1x8x2304xbf16>
    %3859 = chlo.square %3858 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3860 = stablehlo.convert %3859 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_636 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3861 = stablehlo.reduce(%3860 init: %cst_636) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3862 = stablehlo.broadcast_in_dim %3861, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_637 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3863 = stablehlo.broadcast_in_dim %cst_637, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3864 = stablehlo.divide %3862, %3863 : tensor<1x8x1xf32>
    %3865 = stablehlo.convert %3864 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_638 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3866 = stablehlo.broadcast_in_dim %cst_638, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3867 = stablehlo.add %3865, %3866 : tensor<1x8x1xbf16>
    %3868 = stablehlo.rsqrt %3867 : tensor<1x8x1xbf16>
    %3869 = stablehlo.broadcast_in_dim %3868, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3870 = stablehlo.multiply %3858, %3869 : tensor<1x8x2304xbf16>
    %3871 = stablehlo.broadcast_in_dim %arg100, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_639 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3872 = stablehlo.broadcast_in_dim %cst_639, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3873 = stablehlo.add %3872, %3871 : tensor<1x1x2304xbf16>
    %3874 = stablehlo.broadcast_in_dim %3873, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3875 = stablehlo.multiply %3870, %3874 : tensor<1x8x2304xbf16>
    %3876 = stablehlo.dot_general %3875, %arg95, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %3877 = stablehlo.slice %3876 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3878 = stablehlo.reshape %3877 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3879 = stablehlo.multiply %3878, %3878 : tensor<1x8x9216xbf16>
    %3880 = stablehlo.multiply %3879, %3878 : tensor<1x8x9216xbf16>
    %cst_640 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %3881 = stablehlo.broadcast_in_dim %cst_640, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3882 = stablehlo.multiply %3881, %3880 : tensor<1x8x9216xbf16>
    %3883 = stablehlo.add %3878, %3882 : tensor<1x8x9216xbf16>
    %cst_641 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %3884 = stablehlo.broadcast_in_dim %cst_641, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3885 = stablehlo.multiply %3884, %3883 : tensor<1x8x9216xbf16>
    %3886 = stablehlo.tanh %3885 : tensor<1x8x9216xbf16>
    %cst_642 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3887 = stablehlo.broadcast_in_dim %cst_642, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3888 = stablehlo.add %3887, %3886 : tensor<1x8x9216xbf16>
    %cst_643 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %3889 = stablehlo.broadcast_in_dim %cst_643, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %3890 = stablehlo.multiply %3889, %3888 : tensor<1x8x9216xbf16>
    %3891 = stablehlo.multiply %3878, %3890 : tensor<1x8x9216xbf16>
    %3892 = stablehlo.slice %3876 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %3893 = stablehlo.reshape %3892 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %3894 = stablehlo.multiply %3891, %3893 : tensor<1x8x9216xbf16>
    %3895 = stablehlo.dot_general %3894, %arg96, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3896 = chlo.square %3895 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3897 = stablehlo.convert %3896 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_644 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3898 = stablehlo.reduce(%3897 init: %cst_644) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3899 = stablehlo.broadcast_in_dim %3898, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_645 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3900 = stablehlo.broadcast_in_dim %cst_645, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3901 = stablehlo.divide %3899, %3900 : tensor<1x8x1xf32>
    %3902 = stablehlo.convert %3901 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_646 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3903 = stablehlo.broadcast_in_dim %cst_646, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3904 = stablehlo.add %3902, %3903 : tensor<1x8x1xbf16>
    %3905 = stablehlo.rsqrt %3904 : tensor<1x8x1xbf16>
    %3906 = stablehlo.broadcast_in_dim %3905, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3907 = stablehlo.multiply %3895, %3906 : tensor<1x8x2304xbf16>
    %3908 = stablehlo.broadcast_in_dim %arg98, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_647 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3909 = stablehlo.broadcast_in_dim %cst_647, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3910 = stablehlo.add %3909, %3908 : tensor<1x1x2304xbf16>
    %3911 = stablehlo.broadcast_in_dim %3910, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3912 = stablehlo.multiply %3907, %3911 : tensor<1x8x2304xbf16>
    %3913 = stablehlo.add %3912, %3858 : tensor<1x8x2304xbf16>
    %3914 = chlo.square %3913 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %3915 = stablehlo.convert %3914 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_648 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3916 = stablehlo.reduce(%3915 init: %cst_648) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3917 = stablehlo.broadcast_in_dim %3916, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_649 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %3918 = stablehlo.broadcast_in_dim %cst_649, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3919 = stablehlo.divide %3917, %3918 : tensor<1x8x1xf32>
    %3920 = stablehlo.convert %3919 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_650 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %3921 = stablehlo.broadcast_in_dim %cst_650, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %3922 = stablehlo.add %3920, %3921 : tensor<1x8x1xbf16>
    %3923 = stablehlo.rsqrt %3922 : tensor<1x8x1xbf16>
    %3924 = stablehlo.broadcast_in_dim %3923, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %3925 = stablehlo.multiply %3913, %3924 : tensor<1x8x2304xbf16>
    %3926 = stablehlo.broadcast_in_dim %arg108, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_651 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3927 = stablehlo.broadcast_in_dim %cst_651, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %3928 = stablehlo.add %3927, %3926 : tensor<1x1x2304xbf16>
    %3929 = stablehlo.broadcast_in_dim %3928, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %3930 = stablehlo.multiply %3925, %3929 : tensor<1x8x2304xbf16>
    %3931 = stablehlo.dot_general %3930, %arg103, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %3932 = stablehlo.dot_general %arg102, %3930, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %3933 = stablehlo.transpose %3932, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %3934 = stablehlo.slice %3933 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3935 = stablehlo.reshape %3934 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3936 = stablehlo.slice %3933 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %3937 = stablehlo.reshape %3936 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %3938 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_652 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3939 = stablehlo.broadcast_in_dim %cst_652, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3940 = stablehlo.multiply %3939, %3938 : tensor<128xf32>
    %cst_653 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3941 = stablehlo.broadcast_in_dim %cst_653, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3942 = stablehlo.power %3941, %3940 : tensor<128xf32>
    %3943 = call @_pad(%3942) : (tensor<128xf32>) -> tensor<128xf32>
    %3944 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3945 = stablehlo.broadcast_in_dim %3943, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3946 = stablehlo.convert %3944 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3947 = stablehlo.broadcast_in_dim %3946, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3948 = stablehlo.broadcast_in_dim %3945, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3949 = stablehlo.divide %3947, %3948 : tensor<1x8x128xf32>
    %3950 = stablehlo.broadcast_in_dim %3949, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_654 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3951 = stablehlo.broadcast_in_dim %cst_654, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3952 = stablehlo.divide %3950, %3951 : tensor<1x8x1x128xf32>
    %3953 = stablehlo.sine %3952 : tensor<1x8x1x128xf32>
    %3954 = stablehlo.cosine %3952 : tensor<1x8x1x128xf32>
    %3955 = stablehlo.slice %3931 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3956 = stablehlo.slice %3931 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %3957 = stablehlo.convert %3955 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3958 = stablehlo.broadcast_in_dim %3954, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3959 = stablehlo.multiply %3957, %3958 : tensor<1x8x8x128xf32>
    %3960 = stablehlo.convert %3956 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3961 = stablehlo.broadcast_in_dim %3953, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3962 = stablehlo.multiply %3960, %3961 : tensor<1x8x8x128xf32>
    %3963 = stablehlo.subtract %3959, %3962 : tensor<1x8x8x128xf32>
    %3964 = stablehlo.convert %3956 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3965 = stablehlo.broadcast_in_dim %3954, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3966 = stablehlo.multiply %3964, %3965 : tensor<1x8x8x128xf32>
    %3967 = stablehlo.convert %3955 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3968 = stablehlo.broadcast_in_dim %3953, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %3969 = stablehlo.multiply %3967, %3968 : tensor<1x8x8x128xf32>
    %3970 = stablehlo.add %3966, %3969 : tensor<1x8x8x128xf32>
    %3971 = stablehlo.concatenate %3963, %3970, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %3972 = stablehlo.convert %3971 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_655 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %3973 = stablehlo.broadcast_in_dim %cst_655, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %3974 = stablehlo.multiply %3972, %3973 : tensor<1x8x8x256xbf16>
    %3975 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_656 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3976 = stablehlo.broadcast_in_dim %cst_656, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3977 = stablehlo.multiply %3976, %3975 : tensor<128xf32>
    %cst_657 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %3978 = stablehlo.broadcast_in_dim %cst_657, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3979 = stablehlo.power %3978, %3977 : tensor<128xf32>
    %3980 = call @_pad(%3979) : (tensor<128xf32>) -> tensor<128xf32>
    %3981 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3982 = stablehlo.broadcast_in_dim %3980, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %3983 = stablehlo.convert %3981 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %3984 = stablehlo.broadcast_in_dim %3983, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %3985 = stablehlo.broadcast_in_dim %3982, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %3986 = stablehlo.divide %3984, %3985 : tensor<1x8x128xf32>
    %3987 = stablehlo.broadcast_in_dim %3986, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_658 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3988 = stablehlo.broadcast_in_dim %cst_658, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %3989 = stablehlo.divide %3987, %3988 : tensor<1x8x1x128xf32>
    %3990 = stablehlo.sine %3989 : tensor<1x8x1x128xf32>
    %3991 = stablehlo.cosine %3989 : tensor<1x8x1x128xf32>
    %3992 = stablehlo.slice %3935 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3993 = stablehlo.slice %3935 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %3994 = stablehlo.convert %3992 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3995 = stablehlo.broadcast_in_dim %3991, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3996 = stablehlo.multiply %3994, %3995 : tensor<1x8x4x128xf32>
    %3997 = stablehlo.convert %3993 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %3998 = stablehlo.broadcast_in_dim %3990, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %3999 = stablehlo.multiply %3997, %3998 : tensor<1x8x4x128xf32>
    %4000 = stablehlo.subtract %3996, %3999 : tensor<1x8x4x128xf32>
    %4001 = stablehlo.convert %3993 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4002 = stablehlo.broadcast_in_dim %3991, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4003 = stablehlo.multiply %4001, %4002 : tensor<1x8x4x128xf32>
    %4004 = stablehlo.convert %3992 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4005 = stablehlo.broadcast_in_dim %3990, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4006 = stablehlo.multiply %4004, %4005 : tensor<1x8x4x128xf32>
    %4007 = stablehlo.add %4003, %4006 : tensor<1x8x4x128xf32>
    %4008 = stablehlo.concatenate %4000, %4007, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %4009 = stablehlo.convert %4008 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %4010 = stablehlo.reshape %3974 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4011 = stablehlo.dot_general %4009, %4010, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %4012 = stablehlo.transpose %4011, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4013 = stablehlo.reshape %4012 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_659 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4014 = stablehlo.broadcast_in_dim %cst_659, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4015 = stablehlo.divide %4013, %4014 : tensor<1x8x8x8xbf16>
    %4016 = stablehlo.tanh %4015 : tensor<1x8x8x8xbf16>
    %cst_660 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4017 = stablehlo.broadcast_in_dim %cst_660, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4018 = stablehlo.multiply %4016, %4017 : tensor<1x8x8x8xbf16>
    %4019 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_661 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %4020 = call @_where(%4019, %4018, %cst_661) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_662 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4021 = stablehlo.reduce(%4020 init: %cst_662) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_663 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4022 = stablehlo.broadcast_in_dim %cst_663, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %4023 = stablehlo.maximum %4022, %4021 : tensor<1x8x8xbf16>
    %4024 = stablehlo.broadcast_in_dim %4023, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %4025 = stablehlo.broadcast_in_dim %4024, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4026 = stablehlo.subtract %4020, %4025 : tensor<1x8x8x8xbf16>
    %4027 = stablehlo.exponential %4026 : tensor<1x8x8x8xbf16>
    %4028 = stablehlo.convert %4027 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_664 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4029 = stablehlo.reduce(%4028 init: %cst_664) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4030 = stablehlo.broadcast_in_dim %4029, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4031 = stablehlo.convert %4030 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %4032 = stablehlo.broadcast_in_dim %4031, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4033 = stablehlo.divide %4027, %4032 : tensor<1x8x8x8xbf16>
    %4034 = stablehlo.reshape %4033 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4035 = stablehlo.dot_general %3937, %4034, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %4036 = stablehlo.transpose %4035, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4037 = stablehlo.reshape %4036 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4038 = stablehlo.dot_general %4037, %arg101, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4039 = chlo.square %4038 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4040 = stablehlo.convert %4039 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_665 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4041 = stablehlo.reduce(%4040 init: %cst_665) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4042 = stablehlo.broadcast_in_dim %4041, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_666 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4043 = stablehlo.broadcast_in_dim %cst_666, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4044 = stablehlo.divide %4042, %4043 : tensor<1x8x1xf32>
    %4045 = stablehlo.convert %4044 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_667 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4046 = stablehlo.broadcast_in_dim %cst_667, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4047 = stablehlo.add %4045, %4046 : tensor<1x8x1xbf16>
    %4048 = stablehlo.rsqrt %4047 : tensor<1x8x1xbf16>
    %4049 = stablehlo.broadcast_in_dim %4048, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4050 = stablehlo.multiply %4038, %4049 : tensor<1x8x2304xbf16>
    %4051 = stablehlo.broadcast_in_dim %arg106, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_668 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4052 = stablehlo.broadcast_in_dim %cst_668, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4053 = stablehlo.add %4052, %4051 : tensor<1x1x2304xbf16>
    %4054 = stablehlo.broadcast_in_dim %4053, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4055 = stablehlo.multiply %4050, %4054 : tensor<1x8x2304xbf16>
    %4056 = stablehlo.add %4055, %3913 : tensor<1x8x2304xbf16>
    %4057 = chlo.square %4056 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4058 = stablehlo.convert %4057 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_669 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4059 = stablehlo.reduce(%4058 init: %cst_669) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4060 = stablehlo.broadcast_in_dim %4059, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_670 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4061 = stablehlo.broadcast_in_dim %cst_670, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4062 = stablehlo.divide %4060, %4061 : tensor<1x8x1xf32>
    %4063 = stablehlo.convert %4062 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_671 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4064 = stablehlo.broadcast_in_dim %cst_671, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4065 = stablehlo.add %4063, %4064 : tensor<1x8x1xbf16>
    %4066 = stablehlo.rsqrt %4065 : tensor<1x8x1xbf16>
    %4067 = stablehlo.broadcast_in_dim %4066, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4068 = stablehlo.multiply %4056, %4067 : tensor<1x8x2304xbf16>
    %4069 = stablehlo.broadcast_in_dim %arg109, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_672 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4070 = stablehlo.broadcast_in_dim %cst_672, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4071 = stablehlo.add %4070, %4069 : tensor<1x1x2304xbf16>
    %4072 = stablehlo.broadcast_in_dim %4071, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4073 = stablehlo.multiply %4068, %4072 : tensor<1x8x2304xbf16>
    %4074 = stablehlo.dot_general %4073, %arg104, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %4075 = stablehlo.slice %4074 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4076 = stablehlo.reshape %4075 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4077 = stablehlo.multiply %4076, %4076 : tensor<1x8x9216xbf16>
    %4078 = stablehlo.multiply %4077, %4076 : tensor<1x8x9216xbf16>
    %cst_673 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %4079 = stablehlo.broadcast_in_dim %cst_673, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4080 = stablehlo.multiply %4079, %4078 : tensor<1x8x9216xbf16>
    %4081 = stablehlo.add %4076, %4080 : tensor<1x8x9216xbf16>
    %cst_674 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %4082 = stablehlo.broadcast_in_dim %cst_674, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4083 = stablehlo.multiply %4082, %4081 : tensor<1x8x9216xbf16>
    %4084 = stablehlo.tanh %4083 : tensor<1x8x9216xbf16>
    %cst_675 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4085 = stablehlo.broadcast_in_dim %cst_675, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4086 = stablehlo.add %4085, %4084 : tensor<1x8x9216xbf16>
    %cst_676 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %4087 = stablehlo.broadcast_in_dim %cst_676, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4088 = stablehlo.multiply %4087, %4086 : tensor<1x8x9216xbf16>
    %4089 = stablehlo.multiply %4076, %4088 : tensor<1x8x9216xbf16>
    %4090 = stablehlo.slice %4074 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4091 = stablehlo.reshape %4090 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4092 = stablehlo.multiply %4089, %4091 : tensor<1x8x9216xbf16>
    %4093 = stablehlo.dot_general %4092, %arg105, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4094 = chlo.square %4093 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4095 = stablehlo.convert %4094 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_677 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4096 = stablehlo.reduce(%4095 init: %cst_677) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4097 = stablehlo.broadcast_in_dim %4096, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_678 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4098 = stablehlo.broadcast_in_dim %cst_678, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4099 = stablehlo.divide %4097, %4098 : tensor<1x8x1xf32>
    %4100 = stablehlo.convert %4099 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_679 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4101 = stablehlo.broadcast_in_dim %cst_679, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4102 = stablehlo.add %4100, %4101 : tensor<1x8x1xbf16>
    %4103 = stablehlo.rsqrt %4102 : tensor<1x8x1xbf16>
    %4104 = stablehlo.broadcast_in_dim %4103, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4105 = stablehlo.multiply %4093, %4104 : tensor<1x8x2304xbf16>
    %4106 = stablehlo.broadcast_in_dim %arg107, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_680 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4107 = stablehlo.broadcast_in_dim %cst_680, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4108 = stablehlo.add %4107, %4106 : tensor<1x1x2304xbf16>
    %4109 = stablehlo.broadcast_in_dim %4108, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4110 = stablehlo.multiply %4105, %4109 : tensor<1x8x2304xbf16>
    %4111 = stablehlo.add %4110, %4056 : tensor<1x8x2304xbf16>
    %4112 = chlo.square %4111 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4113 = stablehlo.convert %4112 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_681 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4114 = stablehlo.reduce(%4113 init: %cst_681) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4115 = stablehlo.broadcast_in_dim %4114, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_682 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4116 = stablehlo.broadcast_in_dim %cst_682, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4117 = stablehlo.divide %4115, %4116 : tensor<1x8x1xf32>
    %4118 = stablehlo.convert %4117 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_683 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4119 = stablehlo.broadcast_in_dim %cst_683, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4120 = stablehlo.add %4118, %4119 : tensor<1x8x1xbf16>
    %4121 = stablehlo.rsqrt %4120 : tensor<1x8x1xbf16>
    %4122 = stablehlo.broadcast_in_dim %4121, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4123 = stablehlo.multiply %4111, %4122 : tensor<1x8x2304xbf16>
    %4124 = stablehlo.broadcast_in_dim %arg126, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_684 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4125 = stablehlo.broadcast_in_dim %cst_684, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4126 = stablehlo.add %4125, %4124 : tensor<1x1x2304xbf16>
    %4127 = stablehlo.broadcast_in_dim %4126, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4128 = stablehlo.multiply %4123, %4127 : tensor<1x8x2304xbf16>
    %4129 = stablehlo.dot_general %4128, %arg121, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4130 = stablehlo.dot_general %arg120, %4128, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %4131 = stablehlo.transpose %4130, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %4132 = stablehlo.slice %4131 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4133 = stablehlo.reshape %4132 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4134 = stablehlo.slice %4131 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4135 = stablehlo.reshape %4134 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4136 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_685 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4137 = stablehlo.broadcast_in_dim %cst_685, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4138 = stablehlo.multiply %4137, %4136 : tensor<128xf32>
    %cst_686 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4139 = stablehlo.broadcast_in_dim %cst_686, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4140 = stablehlo.power %4139, %4138 : tensor<128xf32>
    %4141 = call @_pad(%4140) : (tensor<128xf32>) -> tensor<128xf32>
    %4142 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4143 = stablehlo.broadcast_in_dim %4141, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4144 = stablehlo.convert %4142 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4145 = stablehlo.broadcast_in_dim %4144, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4146 = stablehlo.broadcast_in_dim %4143, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4147 = stablehlo.divide %4145, %4146 : tensor<1x8x128xf32>
    %4148 = stablehlo.broadcast_in_dim %4147, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_687 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4149 = stablehlo.broadcast_in_dim %cst_687, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4150 = stablehlo.divide %4148, %4149 : tensor<1x8x1x128xf32>
    %4151 = stablehlo.sine %4150 : tensor<1x8x1x128xf32>
    %4152 = stablehlo.cosine %4150 : tensor<1x8x1x128xf32>
    %4153 = stablehlo.slice %4129 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4154 = stablehlo.slice %4129 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4155 = stablehlo.convert %4153 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4156 = stablehlo.broadcast_in_dim %4152, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4157 = stablehlo.multiply %4155, %4156 : tensor<1x8x8x128xf32>
    %4158 = stablehlo.convert %4154 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4159 = stablehlo.broadcast_in_dim %4151, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4160 = stablehlo.multiply %4158, %4159 : tensor<1x8x8x128xf32>
    %4161 = stablehlo.subtract %4157, %4160 : tensor<1x8x8x128xf32>
    %4162 = stablehlo.convert %4154 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4163 = stablehlo.broadcast_in_dim %4152, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4164 = stablehlo.multiply %4162, %4163 : tensor<1x8x8x128xf32>
    %4165 = stablehlo.convert %4153 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4166 = stablehlo.broadcast_in_dim %4151, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4167 = stablehlo.multiply %4165, %4166 : tensor<1x8x8x128xf32>
    %4168 = stablehlo.add %4164, %4167 : tensor<1x8x8x128xf32>
    %4169 = stablehlo.concatenate %4161, %4168, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %4170 = stablehlo.convert %4169 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_688 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %4171 = stablehlo.broadcast_in_dim %cst_688, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %4172 = stablehlo.multiply %4170, %4171 : tensor<1x8x8x256xbf16>
    %4173 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_689 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4174 = stablehlo.broadcast_in_dim %cst_689, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4175 = stablehlo.multiply %4174, %4173 : tensor<128xf32>
    %cst_690 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4176 = stablehlo.broadcast_in_dim %cst_690, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4177 = stablehlo.power %4176, %4175 : tensor<128xf32>
    %4178 = call @_pad(%4177) : (tensor<128xf32>) -> tensor<128xf32>
    %4179 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4180 = stablehlo.broadcast_in_dim %4178, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4181 = stablehlo.convert %4179 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4182 = stablehlo.broadcast_in_dim %4181, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4183 = stablehlo.broadcast_in_dim %4180, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4184 = stablehlo.divide %4182, %4183 : tensor<1x8x128xf32>
    %4185 = stablehlo.broadcast_in_dim %4184, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_691 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4186 = stablehlo.broadcast_in_dim %cst_691, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4187 = stablehlo.divide %4185, %4186 : tensor<1x8x1x128xf32>
    %4188 = stablehlo.sine %4187 : tensor<1x8x1x128xf32>
    %4189 = stablehlo.cosine %4187 : tensor<1x8x1x128xf32>
    %4190 = stablehlo.slice %4133 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4191 = stablehlo.slice %4133 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4192 = stablehlo.convert %4190 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4193 = stablehlo.broadcast_in_dim %4189, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4194 = stablehlo.multiply %4192, %4193 : tensor<1x8x4x128xf32>
    %4195 = stablehlo.convert %4191 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4196 = stablehlo.broadcast_in_dim %4188, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4197 = stablehlo.multiply %4195, %4196 : tensor<1x8x4x128xf32>
    %4198 = stablehlo.subtract %4194, %4197 : tensor<1x8x4x128xf32>
    %4199 = stablehlo.convert %4191 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4200 = stablehlo.broadcast_in_dim %4189, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4201 = stablehlo.multiply %4199, %4200 : tensor<1x8x4x128xf32>
    %4202 = stablehlo.convert %4190 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4203 = stablehlo.broadcast_in_dim %4188, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4204 = stablehlo.multiply %4202, %4203 : tensor<1x8x4x128xf32>
    %4205 = stablehlo.add %4201, %4204 : tensor<1x8x4x128xf32>
    %4206 = stablehlo.concatenate %4198, %4205, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %4207 = stablehlo.convert %4206 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %4208 = stablehlo.reshape %4172 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4209 = stablehlo.dot_general %4207, %4208, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %4210 = stablehlo.transpose %4209, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4211 = stablehlo.reshape %4210 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_692 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4212 = stablehlo.broadcast_in_dim %cst_692, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4213 = stablehlo.divide %4211, %4212 : tensor<1x8x8x8xbf16>
    %4214 = stablehlo.tanh %4213 : tensor<1x8x8x8xbf16>
    %cst_693 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4215 = stablehlo.broadcast_in_dim %cst_693, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4216 = stablehlo.multiply %4214, %4215 : tensor<1x8x8x8xbf16>
    %4217 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %4218 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_694 = stablehlo.constant dense<4096> : tensor<i32>
    %4219 = stablehlo.broadcast_in_dim %c_694, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %4220 = stablehlo.subtract %4218, %4219 : tensor<1x8x1xi32>
    %4221 = stablehlo.broadcast_in_dim %4217, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %4222 = stablehlo.broadcast_in_dim %4220, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %4223 = stablehlo.compare GT, %4221, %4222, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_695 = stablehlo.constant dense<4096> : tensor<i32>
    %4224 = stablehlo.broadcast_in_dim %c_695, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %4225 = stablehlo.add %4218, %4224 : tensor<1x8x1xi32>
    %4226 = stablehlo.broadcast_in_dim %4217, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %4227 = stablehlo.broadcast_in_dim %4225, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %4228 = stablehlo.compare LT, %4226, %4227, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %4229 = stablehlo.and %4223, %4228 : tensor<1x8x8xi1>
    %4230 = stablehlo.and %arg237, %4229 : tensor<1x8x8xi1>
    %4231 = stablehlo.broadcast_in_dim %4230, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_696 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %4232 = call @_where(%4231, %4216, %cst_696) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_697 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4233 = stablehlo.reduce(%4232 init: %cst_697) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_698 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4234 = stablehlo.broadcast_in_dim %cst_698, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %4235 = stablehlo.maximum %4234, %4233 : tensor<1x8x8xbf16>
    %4236 = stablehlo.broadcast_in_dim %4235, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %4237 = stablehlo.broadcast_in_dim %4236, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4238 = stablehlo.subtract %4232, %4237 : tensor<1x8x8x8xbf16>
    %4239 = stablehlo.exponential %4238 : tensor<1x8x8x8xbf16>
    %4240 = stablehlo.convert %4239 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_699 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4241 = stablehlo.reduce(%4240 init: %cst_699) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4242 = stablehlo.broadcast_in_dim %4241, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4243 = stablehlo.convert %4242 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %4244 = stablehlo.broadcast_in_dim %4243, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4245 = stablehlo.divide %4239, %4244 : tensor<1x8x8x8xbf16>
    %4246 = stablehlo.reshape %4245 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4247 = stablehlo.dot_general %4135, %4246, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %4248 = stablehlo.transpose %4247, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4249 = stablehlo.reshape %4248 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4250 = stablehlo.dot_general %4249, %arg119, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4251 = chlo.square %4250 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4252 = stablehlo.convert %4251 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_700 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4253 = stablehlo.reduce(%4252 init: %cst_700) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4254 = stablehlo.broadcast_in_dim %4253, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_701 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4255 = stablehlo.broadcast_in_dim %cst_701, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4256 = stablehlo.divide %4254, %4255 : tensor<1x8x1xf32>
    %4257 = stablehlo.convert %4256 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_702 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4258 = stablehlo.broadcast_in_dim %cst_702, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4259 = stablehlo.add %4257, %4258 : tensor<1x8x1xbf16>
    %4260 = stablehlo.rsqrt %4259 : tensor<1x8x1xbf16>
    %4261 = stablehlo.broadcast_in_dim %4260, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4262 = stablehlo.multiply %4250, %4261 : tensor<1x8x2304xbf16>
    %4263 = stablehlo.broadcast_in_dim %arg124, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_703 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4264 = stablehlo.broadcast_in_dim %cst_703, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4265 = stablehlo.add %4264, %4263 : tensor<1x1x2304xbf16>
    %4266 = stablehlo.broadcast_in_dim %4265, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4267 = stablehlo.multiply %4262, %4266 : tensor<1x8x2304xbf16>
    %4268 = stablehlo.add %4267, %4111 : tensor<1x8x2304xbf16>
    %4269 = chlo.square %4268 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4270 = stablehlo.convert %4269 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_704 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4271 = stablehlo.reduce(%4270 init: %cst_704) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4272 = stablehlo.broadcast_in_dim %4271, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_705 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4273 = stablehlo.broadcast_in_dim %cst_705, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4274 = stablehlo.divide %4272, %4273 : tensor<1x8x1xf32>
    %4275 = stablehlo.convert %4274 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_706 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4276 = stablehlo.broadcast_in_dim %cst_706, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4277 = stablehlo.add %4275, %4276 : tensor<1x8x1xbf16>
    %4278 = stablehlo.rsqrt %4277 : tensor<1x8x1xbf16>
    %4279 = stablehlo.broadcast_in_dim %4278, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4280 = stablehlo.multiply %4268, %4279 : tensor<1x8x2304xbf16>
    %4281 = stablehlo.broadcast_in_dim %arg127, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_707 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4282 = stablehlo.broadcast_in_dim %cst_707, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4283 = stablehlo.add %4282, %4281 : tensor<1x1x2304xbf16>
    %4284 = stablehlo.broadcast_in_dim %4283, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4285 = stablehlo.multiply %4280, %4284 : tensor<1x8x2304xbf16>
    %4286 = stablehlo.dot_general %4285, %arg122, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %4287 = stablehlo.slice %4286 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4288 = stablehlo.reshape %4287 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4289 = stablehlo.multiply %4288, %4288 : tensor<1x8x9216xbf16>
    %4290 = stablehlo.multiply %4289, %4288 : tensor<1x8x9216xbf16>
    %cst_708 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %4291 = stablehlo.broadcast_in_dim %cst_708, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4292 = stablehlo.multiply %4291, %4290 : tensor<1x8x9216xbf16>
    %4293 = stablehlo.add %4288, %4292 : tensor<1x8x9216xbf16>
    %cst_709 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %4294 = stablehlo.broadcast_in_dim %cst_709, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4295 = stablehlo.multiply %4294, %4293 : tensor<1x8x9216xbf16>
    %4296 = stablehlo.tanh %4295 : tensor<1x8x9216xbf16>
    %cst_710 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4297 = stablehlo.broadcast_in_dim %cst_710, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4298 = stablehlo.add %4297, %4296 : tensor<1x8x9216xbf16>
    %cst_711 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %4299 = stablehlo.broadcast_in_dim %cst_711, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4300 = stablehlo.multiply %4299, %4298 : tensor<1x8x9216xbf16>
    %4301 = stablehlo.multiply %4288, %4300 : tensor<1x8x9216xbf16>
    %4302 = stablehlo.slice %4286 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4303 = stablehlo.reshape %4302 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4304 = stablehlo.multiply %4301, %4303 : tensor<1x8x9216xbf16>
    %4305 = stablehlo.dot_general %4304, %arg123, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4306 = chlo.square %4305 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4307 = stablehlo.convert %4306 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_712 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4308 = stablehlo.reduce(%4307 init: %cst_712) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4309 = stablehlo.broadcast_in_dim %4308, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_713 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4310 = stablehlo.broadcast_in_dim %cst_713, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4311 = stablehlo.divide %4309, %4310 : tensor<1x8x1xf32>
    %4312 = stablehlo.convert %4311 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_714 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4313 = stablehlo.broadcast_in_dim %cst_714, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4314 = stablehlo.add %4312, %4313 : tensor<1x8x1xbf16>
    %4315 = stablehlo.rsqrt %4314 : tensor<1x8x1xbf16>
    %4316 = stablehlo.broadcast_in_dim %4315, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4317 = stablehlo.multiply %4305, %4316 : tensor<1x8x2304xbf16>
    %4318 = stablehlo.broadcast_in_dim %arg125, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_715 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4319 = stablehlo.broadcast_in_dim %cst_715, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4320 = stablehlo.add %4319, %4318 : tensor<1x1x2304xbf16>
    %4321 = stablehlo.broadcast_in_dim %4320, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4322 = stablehlo.multiply %4317, %4321 : tensor<1x8x2304xbf16>
    %4323 = stablehlo.add %4322, %4268 : tensor<1x8x2304xbf16>
    %4324 = chlo.square %4323 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4325 = stablehlo.convert %4324 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_716 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4326 = stablehlo.reduce(%4325 init: %cst_716) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4327 = stablehlo.broadcast_in_dim %4326, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_717 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4328 = stablehlo.broadcast_in_dim %cst_717, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4329 = stablehlo.divide %4327, %4328 : tensor<1x8x1xf32>
    %4330 = stablehlo.convert %4329 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_718 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4331 = stablehlo.broadcast_in_dim %cst_718, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4332 = stablehlo.add %4330, %4331 : tensor<1x8x1xbf16>
    %4333 = stablehlo.rsqrt %4332 : tensor<1x8x1xbf16>
    %4334 = stablehlo.broadcast_in_dim %4333, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4335 = stablehlo.multiply %4323, %4334 : tensor<1x8x2304xbf16>
    %4336 = stablehlo.broadcast_in_dim %arg135, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_719 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4337 = stablehlo.broadcast_in_dim %cst_719, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4338 = stablehlo.add %4337, %4336 : tensor<1x1x2304xbf16>
    %4339 = stablehlo.broadcast_in_dim %4338, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4340 = stablehlo.multiply %4335, %4339 : tensor<1x8x2304xbf16>
    %4341 = stablehlo.dot_general %4340, %arg130, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4342 = stablehlo.dot_general %arg129, %4340, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %4343 = stablehlo.transpose %4342, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %4344 = stablehlo.slice %4343 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4345 = stablehlo.reshape %4344 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4346 = stablehlo.slice %4343 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4347 = stablehlo.reshape %4346 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4348 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_720 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4349 = stablehlo.broadcast_in_dim %cst_720, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4350 = stablehlo.multiply %4349, %4348 : tensor<128xf32>
    %cst_721 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4351 = stablehlo.broadcast_in_dim %cst_721, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4352 = stablehlo.power %4351, %4350 : tensor<128xf32>
    %4353 = call @_pad(%4352) : (tensor<128xf32>) -> tensor<128xf32>
    %4354 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4355 = stablehlo.broadcast_in_dim %4353, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4356 = stablehlo.convert %4354 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4357 = stablehlo.broadcast_in_dim %4356, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4358 = stablehlo.broadcast_in_dim %4355, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4359 = stablehlo.divide %4357, %4358 : tensor<1x8x128xf32>
    %4360 = stablehlo.broadcast_in_dim %4359, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_722 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4361 = stablehlo.broadcast_in_dim %cst_722, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4362 = stablehlo.divide %4360, %4361 : tensor<1x8x1x128xf32>
    %4363 = stablehlo.sine %4362 : tensor<1x8x1x128xf32>
    %4364 = stablehlo.cosine %4362 : tensor<1x8x1x128xf32>
    %4365 = stablehlo.slice %4341 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4366 = stablehlo.slice %4341 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4367 = stablehlo.convert %4365 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4368 = stablehlo.broadcast_in_dim %4364, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4369 = stablehlo.multiply %4367, %4368 : tensor<1x8x8x128xf32>
    %4370 = stablehlo.convert %4366 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4371 = stablehlo.broadcast_in_dim %4363, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4372 = stablehlo.multiply %4370, %4371 : tensor<1x8x8x128xf32>
    %4373 = stablehlo.subtract %4369, %4372 : tensor<1x8x8x128xf32>
    %4374 = stablehlo.convert %4366 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4375 = stablehlo.broadcast_in_dim %4364, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4376 = stablehlo.multiply %4374, %4375 : tensor<1x8x8x128xf32>
    %4377 = stablehlo.convert %4365 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4378 = stablehlo.broadcast_in_dim %4363, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4379 = stablehlo.multiply %4377, %4378 : tensor<1x8x8x128xf32>
    %4380 = stablehlo.add %4376, %4379 : tensor<1x8x8x128xf32>
    %4381 = stablehlo.concatenate %4373, %4380, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %4382 = stablehlo.convert %4381 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_723 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %4383 = stablehlo.broadcast_in_dim %cst_723, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %4384 = stablehlo.multiply %4382, %4383 : tensor<1x8x8x256xbf16>
    %4385 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_724 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4386 = stablehlo.broadcast_in_dim %cst_724, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4387 = stablehlo.multiply %4386, %4385 : tensor<128xf32>
    %cst_725 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4388 = stablehlo.broadcast_in_dim %cst_725, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4389 = stablehlo.power %4388, %4387 : tensor<128xf32>
    %4390 = call @_pad(%4389) : (tensor<128xf32>) -> tensor<128xf32>
    %4391 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4392 = stablehlo.broadcast_in_dim %4390, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4393 = stablehlo.convert %4391 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4394 = stablehlo.broadcast_in_dim %4393, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4395 = stablehlo.broadcast_in_dim %4392, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4396 = stablehlo.divide %4394, %4395 : tensor<1x8x128xf32>
    %4397 = stablehlo.broadcast_in_dim %4396, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_726 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4398 = stablehlo.broadcast_in_dim %cst_726, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4399 = stablehlo.divide %4397, %4398 : tensor<1x8x1x128xf32>
    %4400 = stablehlo.sine %4399 : tensor<1x8x1x128xf32>
    %4401 = stablehlo.cosine %4399 : tensor<1x8x1x128xf32>
    %4402 = stablehlo.slice %4345 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4403 = stablehlo.slice %4345 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4404 = stablehlo.convert %4402 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4405 = stablehlo.broadcast_in_dim %4401, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4406 = stablehlo.multiply %4404, %4405 : tensor<1x8x4x128xf32>
    %4407 = stablehlo.convert %4403 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4408 = stablehlo.broadcast_in_dim %4400, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4409 = stablehlo.multiply %4407, %4408 : tensor<1x8x4x128xf32>
    %4410 = stablehlo.subtract %4406, %4409 : tensor<1x8x4x128xf32>
    %4411 = stablehlo.convert %4403 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4412 = stablehlo.broadcast_in_dim %4401, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4413 = stablehlo.multiply %4411, %4412 : tensor<1x8x4x128xf32>
    %4414 = stablehlo.convert %4402 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4415 = stablehlo.broadcast_in_dim %4400, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4416 = stablehlo.multiply %4414, %4415 : tensor<1x8x4x128xf32>
    %4417 = stablehlo.add %4413, %4416 : tensor<1x8x4x128xf32>
    %4418 = stablehlo.concatenate %4410, %4417, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %4419 = stablehlo.convert %4418 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %4420 = stablehlo.reshape %4384 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4421 = stablehlo.dot_general %4419, %4420, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %4422 = stablehlo.transpose %4421, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4423 = stablehlo.reshape %4422 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_727 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4424 = stablehlo.broadcast_in_dim %cst_727, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4425 = stablehlo.divide %4423, %4424 : tensor<1x8x8x8xbf16>
    %4426 = stablehlo.tanh %4425 : tensor<1x8x8x8xbf16>
    %cst_728 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4427 = stablehlo.broadcast_in_dim %cst_728, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4428 = stablehlo.multiply %4426, %4427 : tensor<1x8x8x8xbf16>
    %4429 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_729 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %4430 = call @_where(%4429, %4428, %cst_729) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_730 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4431 = stablehlo.reduce(%4430 init: %cst_730) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_731 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4432 = stablehlo.broadcast_in_dim %cst_731, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %4433 = stablehlo.maximum %4432, %4431 : tensor<1x8x8xbf16>
    %4434 = stablehlo.broadcast_in_dim %4433, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %4435 = stablehlo.broadcast_in_dim %4434, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4436 = stablehlo.subtract %4430, %4435 : tensor<1x8x8x8xbf16>
    %4437 = stablehlo.exponential %4436 : tensor<1x8x8x8xbf16>
    %4438 = stablehlo.convert %4437 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_732 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4439 = stablehlo.reduce(%4438 init: %cst_732) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4440 = stablehlo.broadcast_in_dim %4439, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4441 = stablehlo.convert %4440 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %4442 = stablehlo.broadcast_in_dim %4441, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4443 = stablehlo.divide %4437, %4442 : tensor<1x8x8x8xbf16>
    %4444 = stablehlo.reshape %4443 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4445 = stablehlo.dot_general %4347, %4444, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %4446 = stablehlo.transpose %4445, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4447 = stablehlo.reshape %4446 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4448 = stablehlo.dot_general %4447, %arg128, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4449 = chlo.square %4448 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4450 = stablehlo.convert %4449 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_733 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4451 = stablehlo.reduce(%4450 init: %cst_733) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4452 = stablehlo.broadcast_in_dim %4451, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_734 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4453 = stablehlo.broadcast_in_dim %cst_734, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4454 = stablehlo.divide %4452, %4453 : tensor<1x8x1xf32>
    %4455 = stablehlo.convert %4454 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_735 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4456 = stablehlo.broadcast_in_dim %cst_735, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4457 = stablehlo.add %4455, %4456 : tensor<1x8x1xbf16>
    %4458 = stablehlo.rsqrt %4457 : tensor<1x8x1xbf16>
    %4459 = stablehlo.broadcast_in_dim %4458, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4460 = stablehlo.multiply %4448, %4459 : tensor<1x8x2304xbf16>
    %4461 = stablehlo.broadcast_in_dim %arg133, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_736 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4462 = stablehlo.broadcast_in_dim %cst_736, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4463 = stablehlo.add %4462, %4461 : tensor<1x1x2304xbf16>
    %4464 = stablehlo.broadcast_in_dim %4463, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4465 = stablehlo.multiply %4460, %4464 : tensor<1x8x2304xbf16>
    %4466 = stablehlo.add %4465, %4323 : tensor<1x8x2304xbf16>
    %4467 = chlo.square %4466 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4468 = stablehlo.convert %4467 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_737 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4469 = stablehlo.reduce(%4468 init: %cst_737) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4470 = stablehlo.broadcast_in_dim %4469, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_738 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4471 = stablehlo.broadcast_in_dim %cst_738, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4472 = stablehlo.divide %4470, %4471 : tensor<1x8x1xf32>
    %4473 = stablehlo.convert %4472 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_739 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4474 = stablehlo.broadcast_in_dim %cst_739, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4475 = stablehlo.add %4473, %4474 : tensor<1x8x1xbf16>
    %4476 = stablehlo.rsqrt %4475 : tensor<1x8x1xbf16>
    %4477 = stablehlo.broadcast_in_dim %4476, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4478 = stablehlo.multiply %4466, %4477 : tensor<1x8x2304xbf16>
    %4479 = stablehlo.broadcast_in_dim %arg136, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_740 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4480 = stablehlo.broadcast_in_dim %cst_740, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4481 = stablehlo.add %4480, %4479 : tensor<1x1x2304xbf16>
    %4482 = stablehlo.broadcast_in_dim %4481, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4483 = stablehlo.multiply %4478, %4482 : tensor<1x8x2304xbf16>
    %4484 = stablehlo.dot_general %4483, %arg131, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %4485 = stablehlo.slice %4484 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4486 = stablehlo.reshape %4485 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4487 = stablehlo.multiply %4486, %4486 : tensor<1x8x9216xbf16>
    %4488 = stablehlo.multiply %4487, %4486 : tensor<1x8x9216xbf16>
    %cst_741 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %4489 = stablehlo.broadcast_in_dim %cst_741, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4490 = stablehlo.multiply %4489, %4488 : tensor<1x8x9216xbf16>
    %4491 = stablehlo.add %4486, %4490 : tensor<1x8x9216xbf16>
    %cst_742 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %4492 = stablehlo.broadcast_in_dim %cst_742, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4493 = stablehlo.multiply %4492, %4491 : tensor<1x8x9216xbf16>
    %4494 = stablehlo.tanh %4493 : tensor<1x8x9216xbf16>
    %cst_743 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4495 = stablehlo.broadcast_in_dim %cst_743, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4496 = stablehlo.add %4495, %4494 : tensor<1x8x9216xbf16>
    %cst_744 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %4497 = stablehlo.broadcast_in_dim %cst_744, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4498 = stablehlo.multiply %4497, %4496 : tensor<1x8x9216xbf16>
    %4499 = stablehlo.multiply %4486, %4498 : tensor<1x8x9216xbf16>
    %4500 = stablehlo.slice %4484 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4501 = stablehlo.reshape %4500 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4502 = stablehlo.multiply %4499, %4501 : tensor<1x8x9216xbf16>
    %4503 = stablehlo.dot_general %4502, %arg132, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4504 = chlo.square %4503 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4505 = stablehlo.convert %4504 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_745 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4506 = stablehlo.reduce(%4505 init: %cst_745) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4507 = stablehlo.broadcast_in_dim %4506, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_746 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4508 = stablehlo.broadcast_in_dim %cst_746, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4509 = stablehlo.divide %4507, %4508 : tensor<1x8x1xf32>
    %4510 = stablehlo.convert %4509 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_747 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4511 = stablehlo.broadcast_in_dim %cst_747, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4512 = stablehlo.add %4510, %4511 : tensor<1x8x1xbf16>
    %4513 = stablehlo.rsqrt %4512 : tensor<1x8x1xbf16>
    %4514 = stablehlo.broadcast_in_dim %4513, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4515 = stablehlo.multiply %4503, %4514 : tensor<1x8x2304xbf16>
    %4516 = stablehlo.broadcast_in_dim %arg134, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_748 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4517 = stablehlo.broadcast_in_dim %cst_748, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4518 = stablehlo.add %4517, %4516 : tensor<1x1x2304xbf16>
    %4519 = stablehlo.broadcast_in_dim %4518, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4520 = stablehlo.multiply %4515, %4519 : tensor<1x8x2304xbf16>
    %4521 = stablehlo.add %4520, %4466 : tensor<1x8x2304xbf16>
    %4522 = chlo.square %4521 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4523 = stablehlo.convert %4522 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_749 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4524 = stablehlo.reduce(%4523 init: %cst_749) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4525 = stablehlo.broadcast_in_dim %4524, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_750 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4526 = stablehlo.broadcast_in_dim %cst_750, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4527 = stablehlo.divide %4525, %4526 : tensor<1x8x1xf32>
    %4528 = stablehlo.convert %4527 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_751 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4529 = stablehlo.broadcast_in_dim %cst_751, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4530 = stablehlo.add %4528, %4529 : tensor<1x8x1xbf16>
    %4531 = stablehlo.rsqrt %4530 : tensor<1x8x1xbf16>
    %4532 = stablehlo.broadcast_in_dim %4531, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4533 = stablehlo.multiply %4521, %4532 : tensor<1x8x2304xbf16>
    %4534 = stablehlo.broadcast_in_dim %arg144, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_752 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4535 = stablehlo.broadcast_in_dim %cst_752, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4536 = stablehlo.add %4535, %4534 : tensor<1x1x2304xbf16>
    %4537 = stablehlo.broadcast_in_dim %4536, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4538 = stablehlo.multiply %4533, %4537 : tensor<1x8x2304xbf16>
    %4539 = stablehlo.dot_general %4538, %arg139, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4540 = stablehlo.dot_general %arg138, %4538, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %4541 = stablehlo.transpose %4540, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %4542 = stablehlo.slice %4541 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4543 = stablehlo.reshape %4542 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4544 = stablehlo.slice %4541 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4545 = stablehlo.reshape %4544 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4546 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_753 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4547 = stablehlo.broadcast_in_dim %cst_753, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4548 = stablehlo.multiply %4547, %4546 : tensor<128xf32>
    %cst_754 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4549 = stablehlo.broadcast_in_dim %cst_754, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4550 = stablehlo.power %4549, %4548 : tensor<128xf32>
    %4551 = call @_pad(%4550) : (tensor<128xf32>) -> tensor<128xf32>
    %4552 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4553 = stablehlo.broadcast_in_dim %4551, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4554 = stablehlo.convert %4552 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4555 = stablehlo.broadcast_in_dim %4554, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4556 = stablehlo.broadcast_in_dim %4553, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4557 = stablehlo.divide %4555, %4556 : tensor<1x8x128xf32>
    %4558 = stablehlo.broadcast_in_dim %4557, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_755 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4559 = stablehlo.broadcast_in_dim %cst_755, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4560 = stablehlo.divide %4558, %4559 : tensor<1x8x1x128xf32>
    %4561 = stablehlo.sine %4560 : tensor<1x8x1x128xf32>
    %4562 = stablehlo.cosine %4560 : tensor<1x8x1x128xf32>
    %4563 = stablehlo.slice %4539 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4564 = stablehlo.slice %4539 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4565 = stablehlo.convert %4563 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4566 = stablehlo.broadcast_in_dim %4562, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4567 = stablehlo.multiply %4565, %4566 : tensor<1x8x8x128xf32>
    %4568 = stablehlo.convert %4564 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4569 = stablehlo.broadcast_in_dim %4561, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4570 = stablehlo.multiply %4568, %4569 : tensor<1x8x8x128xf32>
    %4571 = stablehlo.subtract %4567, %4570 : tensor<1x8x8x128xf32>
    %4572 = stablehlo.convert %4564 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4573 = stablehlo.broadcast_in_dim %4562, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4574 = stablehlo.multiply %4572, %4573 : tensor<1x8x8x128xf32>
    %4575 = stablehlo.convert %4563 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4576 = stablehlo.broadcast_in_dim %4561, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4577 = stablehlo.multiply %4575, %4576 : tensor<1x8x8x128xf32>
    %4578 = stablehlo.add %4574, %4577 : tensor<1x8x8x128xf32>
    %4579 = stablehlo.concatenate %4571, %4578, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %4580 = stablehlo.convert %4579 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_756 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %4581 = stablehlo.broadcast_in_dim %cst_756, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %4582 = stablehlo.multiply %4580, %4581 : tensor<1x8x8x256xbf16>
    %4583 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_757 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4584 = stablehlo.broadcast_in_dim %cst_757, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4585 = stablehlo.multiply %4584, %4583 : tensor<128xf32>
    %cst_758 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4586 = stablehlo.broadcast_in_dim %cst_758, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4587 = stablehlo.power %4586, %4585 : tensor<128xf32>
    %4588 = call @_pad(%4587) : (tensor<128xf32>) -> tensor<128xf32>
    %4589 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4590 = stablehlo.broadcast_in_dim %4588, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4591 = stablehlo.convert %4589 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4592 = stablehlo.broadcast_in_dim %4591, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4593 = stablehlo.broadcast_in_dim %4590, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4594 = stablehlo.divide %4592, %4593 : tensor<1x8x128xf32>
    %4595 = stablehlo.broadcast_in_dim %4594, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_759 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4596 = stablehlo.broadcast_in_dim %cst_759, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4597 = stablehlo.divide %4595, %4596 : tensor<1x8x1x128xf32>
    %4598 = stablehlo.sine %4597 : tensor<1x8x1x128xf32>
    %4599 = stablehlo.cosine %4597 : tensor<1x8x1x128xf32>
    %4600 = stablehlo.slice %4543 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4601 = stablehlo.slice %4543 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4602 = stablehlo.convert %4600 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4603 = stablehlo.broadcast_in_dim %4599, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4604 = stablehlo.multiply %4602, %4603 : tensor<1x8x4x128xf32>
    %4605 = stablehlo.convert %4601 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4606 = stablehlo.broadcast_in_dim %4598, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4607 = stablehlo.multiply %4605, %4606 : tensor<1x8x4x128xf32>
    %4608 = stablehlo.subtract %4604, %4607 : tensor<1x8x4x128xf32>
    %4609 = stablehlo.convert %4601 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4610 = stablehlo.broadcast_in_dim %4599, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4611 = stablehlo.multiply %4609, %4610 : tensor<1x8x4x128xf32>
    %4612 = stablehlo.convert %4600 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4613 = stablehlo.broadcast_in_dim %4598, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4614 = stablehlo.multiply %4612, %4613 : tensor<1x8x4x128xf32>
    %4615 = stablehlo.add %4611, %4614 : tensor<1x8x4x128xf32>
    %4616 = stablehlo.concatenate %4608, %4615, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %4617 = stablehlo.convert %4616 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %4618 = stablehlo.reshape %4582 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4619 = stablehlo.dot_general %4617, %4618, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %4620 = stablehlo.transpose %4619, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4621 = stablehlo.reshape %4620 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_760 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4622 = stablehlo.broadcast_in_dim %cst_760, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4623 = stablehlo.divide %4621, %4622 : tensor<1x8x8x8xbf16>
    %4624 = stablehlo.tanh %4623 : tensor<1x8x8x8xbf16>
    %cst_761 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4625 = stablehlo.broadcast_in_dim %cst_761, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4626 = stablehlo.multiply %4624, %4625 : tensor<1x8x8x8xbf16>
    %4627 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %4628 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_762 = stablehlo.constant dense<4096> : tensor<i32>
    %4629 = stablehlo.broadcast_in_dim %c_762, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %4630 = stablehlo.subtract %4628, %4629 : tensor<1x8x1xi32>
    %4631 = stablehlo.broadcast_in_dim %4627, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %4632 = stablehlo.broadcast_in_dim %4630, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %4633 = stablehlo.compare GT, %4631, %4632, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_763 = stablehlo.constant dense<4096> : tensor<i32>
    %4634 = stablehlo.broadcast_in_dim %c_763, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %4635 = stablehlo.add %4628, %4634 : tensor<1x8x1xi32>
    %4636 = stablehlo.broadcast_in_dim %4627, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %4637 = stablehlo.broadcast_in_dim %4635, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %4638 = stablehlo.compare LT, %4636, %4637, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %4639 = stablehlo.and %4633, %4638 : tensor<1x8x8xi1>
    %4640 = stablehlo.and %arg237, %4639 : tensor<1x8x8xi1>
    %4641 = stablehlo.broadcast_in_dim %4640, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_764 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %4642 = call @_where(%4641, %4626, %cst_764) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_765 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4643 = stablehlo.reduce(%4642 init: %cst_765) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_766 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4644 = stablehlo.broadcast_in_dim %cst_766, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %4645 = stablehlo.maximum %4644, %4643 : tensor<1x8x8xbf16>
    %4646 = stablehlo.broadcast_in_dim %4645, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %4647 = stablehlo.broadcast_in_dim %4646, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4648 = stablehlo.subtract %4642, %4647 : tensor<1x8x8x8xbf16>
    %4649 = stablehlo.exponential %4648 : tensor<1x8x8x8xbf16>
    %4650 = stablehlo.convert %4649 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_767 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4651 = stablehlo.reduce(%4650 init: %cst_767) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4652 = stablehlo.broadcast_in_dim %4651, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4653 = stablehlo.convert %4652 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %4654 = stablehlo.broadcast_in_dim %4653, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4655 = stablehlo.divide %4649, %4654 : tensor<1x8x8x8xbf16>
    %4656 = stablehlo.reshape %4655 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4657 = stablehlo.dot_general %4545, %4656, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %4658 = stablehlo.transpose %4657, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4659 = stablehlo.reshape %4658 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4660 = stablehlo.dot_general %4659, %arg137, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4661 = chlo.square %4660 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4662 = stablehlo.convert %4661 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_768 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4663 = stablehlo.reduce(%4662 init: %cst_768) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4664 = stablehlo.broadcast_in_dim %4663, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_769 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4665 = stablehlo.broadcast_in_dim %cst_769, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4666 = stablehlo.divide %4664, %4665 : tensor<1x8x1xf32>
    %4667 = stablehlo.convert %4666 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_770 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4668 = stablehlo.broadcast_in_dim %cst_770, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4669 = stablehlo.add %4667, %4668 : tensor<1x8x1xbf16>
    %4670 = stablehlo.rsqrt %4669 : tensor<1x8x1xbf16>
    %4671 = stablehlo.broadcast_in_dim %4670, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4672 = stablehlo.multiply %4660, %4671 : tensor<1x8x2304xbf16>
    %4673 = stablehlo.broadcast_in_dim %arg142, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_771 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4674 = stablehlo.broadcast_in_dim %cst_771, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4675 = stablehlo.add %4674, %4673 : tensor<1x1x2304xbf16>
    %4676 = stablehlo.broadcast_in_dim %4675, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4677 = stablehlo.multiply %4672, %4676 : tensor<1x8x2304xbf16>
    %4678 = stablehlo.add %4677, %4521 : tensor<1x8x2304xbf16>
    %4679 = chlo.square %4678 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4680 = stablehlo.convert %4679 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_772 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4681 = stablehlo.reduce(%4680 init: %cst_772) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4682 = stablehlo.broadcast_in_dim %4681, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_773 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4683 = stablehlo.broadcast_in_dim %cst_773, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4684 = stablehlo.divide %4682, %4683 : tensor<1x8x1xf32>
    %4685 = stablehlo.convert %4684 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_774 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4686 = stablehlo.broadcast_in_dim %cst_774, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4687 = stablehlo.add %4685, %4686 : tensor<1x8x1xbf16>
    %4688 = stablehlo.rsqrt %4687 : tensor<1x8x1xbf16>
    %4689 = stablehlo.broadcast_in_dim %4688, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4690 = stablehlo.multiply %4678, %4689 : tensor<1x8x2304xbf16>
    %4691 = stablehlo.broadcast_in_dim %arg145, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_775 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4692 = stablehlo.broadcast_in_dim %cst_775, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4693 = stablehlo.add %4692, %4691 : tensor<1x1x2304xbf16>
    %4694 = stablehlo.broadcast_in_dim %4693, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4695 = stablehlo.multiply %4690, %4694 : tensor<1x8x2304xbf16>
    %4696 = stablehlo.dot_general %4695, %arg140, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %4697 = stablehlo.slice %4696 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4698 = stablehlo.reshape %4697 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4699 = stablehlo.multiply %4698, %4698 : tensor<1x8x9216xbf16>
    %4700 = stablehlo.multiply %4699, %4698 : tensor<1x8x9216xbf16>
    %cst_776 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %4701 = stablehlo.broadcast_in_dim %cst_776, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4702 = stablehlo.multiply %4701, %4700 : tensor<1x8x9216xbf16>
    %4703 = stablehlo.add %4698, %4702 : tensor<1x8x9216xbf16>
    %cst_777 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %4704 = stablehlo.broadcast_in_dim %cst_777, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4705 = stablehlo.multiply %4704, %4703 : tensor<1x8x9216xbf16>
    %4706 = stablehlo.tanh %4705 : tensor<1x8x9216xbf16>
    %cst_778 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4707 = stablehlo.broadcast_in_dim %cst_778, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4708 = stablehlo.add %4707, %4706 : tensor<1x8x9216xbf16>
    %cst_779 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %4709 = stablehlo.broadcast_in_dim %cst_779, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4710 = stablehlo.multiply %4709, %4708 : tensor<1x8x9216xbf16>
    %4711 = stablehlo.multiply %4698, %4710 : tensor<1x8x9216xbf16>
    %4712 = stablehlo.slice %4696 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4713 = stablehlo.reshape %4712 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4714 = stablehlo.multiply %4711, %4713 : tensor<1x8x9216xbf16>
    %4715 = stablehlo.dot_general %4714, %arg141, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4716 = chlo.square %4715 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4717 = stablehlo.convert %4716 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_780 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4718 = stablehlo.reduce(%4717 init: %cst_780) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4719 = stablehlo.broadcast_in_dim %4718, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_781 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4720 = stablehlo.broadcast_in_dim %cst_781, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4721 = stablehlo.divide %4719, %4720 : tensor<1x8x1xf32>
    %4722 = stablehlo.convert %4721 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_782 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4723 = stablehlo.broadcast_in_dim %cst_782, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4724 = stablehlo.add %4722, %4723 : tensor<1x8x1xbf16>
    %4725 = stablehlo.rsqrt %4724 : tensor<1x8x1xbf16>
    %4726 = stablehlo.broadcast_in_dim %4725, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4727 = stablehlo.multiply %4715, %4726 : tensor<1x8x2304xbf16>
    %4728 = stablehlo.broadcast_in_dim %arg143, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_783 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4729 = stablehlo.broadcast_in_dim %cst_783, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4730 = stablehlo.add %4729, %4728 : tensor<1x1x2304xbf16>
    %4731 = stablehlo.broadcast_in_dim %4730, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4732 = stablehlo.multiply %4727, %4731 : tensor<1x8x2304xbf16>
    %4733 = stablehlo.add %4732, %4678 : tensor<1x8x2304xbf16>
    %4734 = chlo.square %4733 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4735 = stablehlo.convert %4734 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_784 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4736 = stablehlo.reduce(%4735 init: %cst_784) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4737 = stablehlo.broadcast_in_dim %4736, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_785 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4738 = stablehlo.broadcast_in_dim %cst_785, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4739 = stablehlo.divide %4737, %4738 : tensor<1x8x1xf32>
    %4740 = stablehlo.convert %4739 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_786 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4741 = stablehlo.broadcast_in_dim %cst_786, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4742 = stablehlo.add %4740, %4741 : tensor<1x8x1xbf16>
    %4743 = stablehlo.rsqrt %4742 : tensor<1x8x1xbf16>
    %4744 = stablehlo.broadcast_in_dim %4743, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4745 = stablehlo.multiply %4733, %4744 : tensor<1x8x2304xbf16>
    %4746 = stablehlo.broadcast_in_dim %arg153, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_787 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4747 = stablehlo.broadcast_in_dim %cst_787, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4748 = stablehlo.add %4747, %4746 : tensor<1x1x2304xbf16>
    %4749 = stablehlo.broadcast_in_dim %4748, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4750 = stablehlo.multiply %4745, %4749 : tensor<1x8x2304xbf16>
    %4751 = stablehlo.dot_general %4750, %arg148, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4752 = stablehlo.dot_general %arg147, %4750, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %4753 = stablehlo.transpose %4752, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %4754 = stablehlo.slice %4753 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4755 = stablehlo.reshape %4754 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4756 = stablehlo.slice %4753 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4757 = stablehlo.reshape %4756 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4758 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_788 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4759 = stablehlo.broadcast_in_dim %cst_788, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4760 = stablehlo.multiply %4759, %4758 : tensor<128xf32>
    %cst_789 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4761 = stablehlo.broadcast_in_dim %cst_789, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4762 = stablehlo.power %4761, %4760 : tensor<128xf32>
    %4763 = call @_pad(%4762) : (tensor<128xf32>) -> tensor<128xf32>
    %4764 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4765 = stablehlo.broadcast_in_dim %4763, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4766 = stablehlo.convert %4764 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4767 = stablehlo.broadcast_in_dim %4766, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4768 = stablehlo.broadcast_in_dim %4765, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4769 = stablehlo.divide %4767, %4768 : tensor<1x8x128xf32>
    %4770 = stablehlo.broadcast_in_dim %4769, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_790 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4771 = stablehlo.broadcast_in_dim %cst_790, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4772 = stablehlo.divide %4770, %4771 : tensor<1x8x1x128xf32>
    %4773 = stablehlo.sine %4772 : tensor<1x8x1x128xf32>
    %4774 = stablehlo.cosine %4772 : tensor<1x8x1x128xf32>
    %4775 = stablehlo.slice %4751 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4776 = stablehlo.slice %4751 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4777 = stablehlo.convert %4775 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4778 = stablehlo.broadcast_in_dim %4774, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4779 = stablehlo.multiply %4777, %4778 : tensor<1x8x8x128xf32>
    %4780 = stablehlo.convert %4776 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4781 = stablehlo.broadcast_in_dim %4773, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4782 = stablehlo.multiply %4780, %4781 : tensor<1x8x8x128xf32>
    %4783 = stablehlo.subtract %4779, %4782 : tensor<1x8x8x128xf32>
    %4784 = stablehlo.convert %4776 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4785 = stablehlo.broadcast_in_dim %4774, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4786 = stablehlo.multiply %4784, %4785 : tensor<1x8x8x128xf32>
    %4787 = stablehlo.convert %4775 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4788 = stablehlo.broadcast_in_dim %4773, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4789 = stablehlo.multiply %4787, %4788 : tensor<1x8x8x128xf32>
    %4790 = stablehlo.add %4786, %4789 : tensor<1x8x8x128xf32>
    %4791 = stablehlo.concatenate %4783, %4790, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %4792 = stablehlo.convert %4791 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_791 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %4793 = stablehlo.broadcast_in_dim %cst_791, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %4794 = stablehlo.multiply %4792, %4793 : tensor<1x8x8x256xbf16>
    %4795 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_792 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4796 = stablehlo.broadcast_in_dim %cst_792, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4797 = stablehlo.multiply %4796, %4795 : tensor<128xf32>
    %cst_793 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4798 = stablehlo.broadcast_in_dim %cst_793, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4799 = stablehlo.power %4798, %4797 : tensor<128xf32>
    %4800 = call @_pad(%4799) : (tensor<128xf32>) -> tensor<128xf32>
    %4801 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4802 = stablehlo.broadcast_in_dim %4800, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4803 = stablehlo.convert %4801 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4804 = stablehlo.broadcast_in_dim %4803, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4805 = stablehlo.broadcast_in_dim %4802, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4806 = stablehlo.divide %4804, %4805 : tensor<1x8x128xf32>
    %4807 = stablehlo.broadcast_in_dim %4806, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_794 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4808 = stablehlo.broadcast_in_dim %cst_794, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4809 = stablehlo.divide %4807, %4808 : tensor<1x8x1x128xf32>
    %4810 = stablehlo.sine %4809 : tensor<1x8x1x128xf32>
    %4811 = stablehlo.cosine %4809 : tensor<1x8x1x128xf32>
    %4812 = stablehlo.slice %4755 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4813 = stablehlo.slice %4755 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %4814 = stablehlo.convert %4812 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4815 = stablehlo.broadcast_in_dim %4811, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4816 = stablehlo.multiply %4814, %4815 : tensor<1x8x4x128xf32>
    %4817 = stablehlo.convert %4813 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4818 = stablehlo.broadcast_in_dim %4810, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4819 = stablehlo.multiply %4817, %4818 : tensor<1x8x4x128xf32>
    %4820 = stablehlo.subtract %4816, %4819 : tensor<1x8x4x128xf32>
    %4821 = stablehlo.convert %4813 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4822 = stablehlo.broadcast_in_dim %4811, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4823 = stablehlo.multiply %4821, %4822 : tensor<1x8x4x128xf32>
    %4824 = stablehlo.convert %4812 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %4825 = stablehlo.broadcast_in_dim %4810, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %4826 = stablehlo.multiply %4824, %4825 : tensor<1x8x4x128xf32>
    %4827 = stablehlo.add %4823, %4826 : tensor<1x8x4x128xf32>
    %4828 = stablehlo.concatenate %4820, %4827, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %4829 = stablehlo.convert %4828 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %4830 = stablehlo.reshape %4794 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4831 = stablehlo.dot_general %4829, %4830, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %4832 = stablehlo.transpose %4831, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4833 = stablehlo.reshape %4832 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_795 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4834 = stablehlo.broadcast_in_dim %cst_795, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4835 = stablehlo.divide %4833, %4834 : tensor<1x8x8x8xbf16>
    %4836 = stablehlo.tanh %4835 : tensor<1x8x8x8xbf16>
    %cst_796 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %4837 = stablehlo.broadcast_in_dim %cst_796, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %4838 = stablehlo.multiply %4836, %4837 : tensor<1x8x8x8xbf16>
    %4839 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_797 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %4840 = call @_where(%4839, %4838, %cst_797) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_798 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4841 = stablehlo.reduce(%4840 init: %cst_798) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_799 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4842 = stablehlo.broadcast_in_dim %cst_799, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %4843 = stablehlo.maximum %4842, %4841 : tensor<1x8x8xbf16>
    %4844 = stablehlo.broadcast_in_dim %4843, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %4845 = stablehlo.broadcast_in_dim %4844, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4846 = stablehlo.subtract %4840, %4845 : tensor<1x8x8x8xbf16>
    %4847 = stablehlo.exponential %4846 : tensor<1x8x8x8xbf16>
    %4848 = stablehlo.convert %4847 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_800 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4849 = stablehlo.reduce(%4848 init: %cst_800) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4850 = stablehlo.broadcast_in_dim %4849, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4851 = stablehlo.convert %4850 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %4852 = stablehlo.broadcast_in_dim %4851, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %4853 = stablehlo.divide %4847, %4852 : tensor<1x8x8x8xbf16>
    %4854 = stablehlo.reshape %4853 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %4855 = stablehlo.dot_general %4757, %4854, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %4856 = stablehlo.transpose %4855, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %4857 = stablehlo.reshape %4856 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4858 = stablehlo.dot_general %4857, %arg146, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4859 = chlo.square %4858 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4860 = stablehlo.convert %4859 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_801 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4861 = stablehlo.reduce(%4860 init: %cst_801) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4862 = stablehlo.broadcast_in_dim %4861, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_802 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4863 = stablehlo.broadcast_in_dim %cst_802, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4864 = stablehlo.divide %4862, %4863 : tensor<1x8x1xf32>
    %4865 = stablehlo.convert %4864 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_803 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4866 = stablehlo.broadcast_in_dim %cst_803, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4867 = stablehlo.add %4865, %4866 : tensor<1x8x1xbf16>
    %4868 = stablehlo.rsqrt %4867 : tensor<1x8x1xbf16>
    %4869 = stablehlo.broadcast_in_dim %4868, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4870 = stablehlo.multiply %4858, %4869 : tensor<1x8x2304xbf16>
    %4871 = stablehlo.broadcast_in_dim %arg151, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_804 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4872 = stablehlo.broadcast_in_dim %cst_804, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4873 = stablehlo.add %4872, %4871 : tensor<1x1x2304xbf16>
    %4874 = stablehlo.broadcast_in_dim %4873, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4875 = stablehlo.multiply %4870, %4874 : tensor<1x8x2304xbf16>
    %4876 = stablehlo.add %4875, %4733 : tensor<1x8x2304xbf16>
    %4877 = chlo.square %4876 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4878 = stablehlo.convert %4877 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_805 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4879 = stablehlo.reduce(%4878 init: %cst_805) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4880 = stablehlo.broadcast_in_dim %4879, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_806 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4881 = stablehlo.broadcast_in_dim %cst_806, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4882 = stablehlo.divide %4880, %4881 : tensor<1x8x1xf32>
    %4883 = stablehlo.convert %4882 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_807 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4884 = stablehlo.broadcast_in_dim %cst_807, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4885 = stablehlo.add %4883, %4884 : tensor<1x8x1xbf16>
    %4886 = stablehlo.rsqrt %4885 : tensor<1x8x1xbf16>
    %4887 = stablehlo.broadcast_in_dim %4886, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4888 = stablehlo.multiply %4876, %4887 : tensor<1x8x2304xbf16>
    %4889 = stablehlo.broadcast_in_dim %arg154, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_808 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4890 = stablehlo.broadcast_in_dim %cst_808, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4891 = stablehlo.add %4890, %4889 : tensor<1x1x2304xbf16>
    %4892 = stablehlo.broadcast_in_dim %4891, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4893 = stablehlo.multiply %4888, %4892 : tensor<1x8x2304xbf16>
    %4894 = stablehlo.dot_general %4893, %arg149, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %4895 = stablehlo.slice %4894 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4896 = stablehlo.reshape %4895 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4897 = stablehlo.multiply %4896, %4896 : tensor<1x8x9216xbf16>
    %4898 = stablehlo.multiply %4897, %4896 : tensor<1x8x9216xbf16>
    %cst_809 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %4899 = stablehlo.broadcast_in_dim %cst_809, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4900 = stablehlo.multiply %4899, %4898 : tensor<1x8x9216xbf16>
    %4901 = stablehlo.add %4896, %4900 : tensor<1x8x9216xbf16>
    %cst_810 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %4902 = stablehlo.broadcast_in_dim %cst_810, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4903 = stablehlo.multiply %4902, %4901 : tensor<1x8x9216xbf16>
    %4904 = stablehlo.tanh %4903 : tensor<1x8x9216xbf16>
    %cst_811 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4905 = stablehlo.broadcast_in_dim %cst_811, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4906 = stablehlo.add %4905, %4904 : tensor<1x8x9216xbf16>
    %cst_812 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %4907 = stablehlo.broadcast_in_dim %cst_812, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %4908 = stablehlo.multiply %4907, %4906 : tensor<1x8x9216xbf16>
    %4909 = stablehlo.multiply %4896, %4908 : tensor<1x8x9216xbf16>
    %4910 = stablehlo.slice %4894 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %4911 = stablehlo.reshape %4910 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %4912 = stablehlo.multiply %4909, %4911 : tensor<1x8x9216xbf16>
    %4913 = stablehlo.dot_general %4912, %arg150, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4914 = chlo.square %4913 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4915 = stablehlo.convert %4914 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_813 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4916 = stablehlo.reduce(%4915 init: %cst_813) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4917 = stablehlo.broadcast_in_dim %4916, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_814 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4918 = stablehlo.broadcast_in_dim %cst_814, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4919 = stablehlo.divide %4917, %4918 : tensor<1x8x1xf32>
    %4920 = stablehlo.convert %4919 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_815 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4921 = stablehlo.broadcast_in_dim %cst_815, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4922 = stablehlo.add %4920, %4921 : tensor<1x8x1xbf16>
    %4923 = stablehlo.rsqrt %4922 : tensor<1x8x1xbf16>
    %4924 = stablehlo.broadcast_in_dim %4923, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4925 = stablehlo.multiply %4913, %4924 : tensor<1x8x2304xbf16>
    %4926 = stablehlo.broadcast_in_dim %arg152, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_816 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4927 = stablehlo.broadcast_in_dim %cst_816, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4928 = stablehlo.add %4927, %4926 : tensor<1x1x2304xbf16>
    %4929 = stablehlo.broadcast_in_dim %4928, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4930 = stablehlo.multiply %4925, %4929 : tensor<1x8x2304xbf16>
    %4931 = stablehlo.add %4930, %4876 : tensor<1x8x2304xbf16>
    %4932 = chlo.square %4931 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %4933 = stablehlo.convert %4932 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_817 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4934 = stablehlo.reduce(%4933 init: %cst_817) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4935 = stablehlo.broadcast_in_dim %4934, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_818 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %4936 = stablehlo.broadcast_in_dim %cst_818, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4937 = stablehlo.divide %4935, %4936 : tensor<1x8x1xf32>
    %4938 = stablehlo.convert %4937 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_819 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %4939 = stablehlo.broadcast_in_dim %cst_819, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %4940 = stablehlo.add %4938, %4939 : tensor<1x8x1xbf16>
    %4941 = stablehlo.rsqrt %4940 : tensor<1x8x1xbf16>
    %4942 = stablehlo.broadcast_in_dim %4941, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %4943 = stablehlo.multiply %4931, %4942 : tensor<1x8x2304xbf16>
    %4944 = stablehlo.broadcast_in_dim %arg162, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_820 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4945 = stablehlo.broadcast_in_dim %cst_820, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %4946 = stablehlo.add %4945, %4944 : tensor<1x1x2304xbf16>
    %4947 = stablehlo.broadcast_in_dim %4946, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %4948 = stablehlo.multiply %4943, %4947 : tensor<1x8x2304xbf16>
    %4949 = stablehlo.dot_general %4948, %arg157, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %4950 = stablehlo.dot_general %arg156, %4948, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %4951 = stablehlo.transpose %4950, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %4952 = stablehlo.slice %4951 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4953 = stablehlo.reshape %4952 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4954 = stablehlo.slice %4951 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %4955 = stablehlo.reshape %4954 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %4956 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_821 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4957 = stablehlo.broadcast_in_dim %cst_821, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4958 = stablehlo.multiply %4957, %4956 : tensor<128xf32>
    %cst_822 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4959 = stablehlo.broadcast_in_dim %cst_822, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4960 = stablehlo.power %4959, %4958 : tensor<128xf32>
    %4961 = call @_pad(%4960) : (tensor<128xf32>) -> tensor<128xf32>
    %4962 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4963 = stablehlo.broadcast_in_dim %4961, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %4964 = stablehlo.convert %4962 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %4965 = stablehlo.broadcast_in_dim %4964, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %4966 = stablehlo.broadcast_in_dim %4963, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %4967 = stablehlo.divide %4965, %4966 : tensor<1x8x128xf32>
    %4968 = stablehlo.broadcast_in_dim %4967, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_823 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4969 = stablehlo.broadcast_in_dim %cst_823, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %4970 = stablehlo.divide %4968, %4969 : tensor<1x8x1x128xf32>
    %4971 = stablehlo.sine %4970 : tensor<1x8x1x128xf32>
    %4972 = stablehlo.cosine %4970 : tensor<1x8x1x128xf32>
    %4973 = stablehlo.slice %4949 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4974 = stablehlo.slice %4949 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %4975 = stablehlo.convert %4973 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4976 = stablehlo.broadcast_in_dim %4972, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4977 = stablehlo.multiply %4975, %4976 : tensor<1x8x8x128xf32>
    %4978 = stablehlo.convert %4974 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4979 = stablehlo.broadcast_in_dim %4971, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4980 = stablehlo.multiply %4978, %4979 : tensor<1x8x8x128xf32>
    %4981 = stablehlo.subtract %4977, %4980 : tensor<1x8x8x128xf32>
    %4982 = stablehlo.convert %4974 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4983 = stablehlo.broadcast_in_dim %4972, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4984 = stablehlo.multiply %4982, %4983 : tensor<1x8x8x128xf32>
    %4985 = stablehlo.convert %4973 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4986 = stablehlo.broadcast_in_dim %4971, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %4987 = stablehlo.multiply %4985, %4986 : tensor<1x8x8x128xf32>
    %4988 = stablehlo.add %4984, %4987 : tensor<1x8x8x128xf32>
    %4989 = stablehlo.concatenate %4981, %4988, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %4990 = stablehlo.convert %4989 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_824 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %4991 = stablehlo.broadcast_in_dim %cst_824, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %4992 = stablehlo.multiply %4990, %4991 : tensor<1x8x8x256xbf16>
    %4993 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_825 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4994 = stablehlo.broadcast_in_dim %cst_825, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4995 = stablehlo.multiply %4994, %4993 : tensor<128xf32>
    %cst_826 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %4996 = stablehlo.broadcast_in_dim %cst_826, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4997 = stablehlo.power %4996, %4995 : tensor<128xf32>
    %4998 = call @_pad(%4997) : (tensor<128xf32>) -> tensor<128xf32>
    %4999 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %5000 = stablehlo.broadcast_in_dim %4998, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %5001 = stablehlo.convert %4999 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %5002 = stablehlo.broadcast_in_dim %5001, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %5003 = stablehlo.broadcast_in_dim %5000, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %5004 = stablehlo.divide %5002, %5003 : tensor<1x8x128xf32>
    %5005 = stablehlo.broadcast_in_dim %5004, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_827 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5006 = stablehlo.broadcast_in_dim %cst_827, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %5007 = stablehlo.divide %5005, %5006 : tensor<1x8x1x128xf32>
    %5008 = stablehlo.sine %5007 : tensor<1x8x1x128xf32>
    %5009 = stablehlo.cosine %5007 : tensor<1x8x1x128xf32>
    %5010 = stablehlo.slice %4953 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %5011 = stablehlo.slice %4953 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %5012 = stablehlo.convert %5010 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5013 = stablehlo.broadcast_in_dim %5009, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5014 = stablehlo.multiply %5012, %5013 : tensor<1x8x4x128xf32>
    %5015 = stablehlo.convert %5011 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5016 = stablehlo.broadcast_in_dim %5008, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5017 = stablehlo.multiply %5015, %5016 : tensor<1x8x4x128xf32>
    %5018 = stablehlo.subtract %5014, %5017 : tensor<1x8x4x128xf32>
    %5019 = stablehlo.convert %5011 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5020 = stablehlo.broadcast_in_dim %5009, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5021 = stablehlo.multiply %5019, %5020 : tensor<1x8x4x128xf32>
    %5022 = stablehlo.convert %5010 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5023 = stablehlo.broadcast_in_dim %5008, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5024 = stablehlo.multiply %5022, %5023 : tensor<1x8x4x128xf32>
    %5025 = stablehlo.add %5021, %5024 : tensor<1x8x4x128xf32>
    %5026 = stablehlo.concatenate %5018, %5025, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %5027 = stablehlo.convert %5026 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %5028 = stablehlo.reshape %4992 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %5029 = stablehlo.dot_general %5027, %5028, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %5030 = stablehlo.transpose %5029, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %5031 = stablehlo.reshape %5030 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_828 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %5032 = stablehlo.broadcast_in_dim %cst_828, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %5033 = stablehlo.divide %5031, %5032 : tensor<1x8x8x8xbf16>
    %5034 = stablehlo.tanh %5033 : tensor<1x8x8x8xbf16>
    %cst_829 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %5035 = stablehlo.broadcast_in_dim %cst_829, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %5036 = stablehlo.multiply %5034, %5035 : tensor<1x8x8x8xbf16>
    %5037 = stablehlo.broadcast_in_dim %arg238, dims = [0, 2] : (tensor<1x8xi32>) -> tensor<1x1x8xi32>
    %5038 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_830 = stablehlo.constant dense<4096> : tensor<i32>
    %5039 = stablehlo.broadcast_in_dim %c_830, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %5040 = stablehlo.subtract %5038, %5039 : tensor<1x8x1xi32>
    %5041 = stablehlo.broadcast_in_dim %5037, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %5042 = stablehlo.broadcast_in_dim %5040, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %5043 = stablehlo.compare GT, %5041, %5042, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %c_831 = stablehlo.constant dense<4096> : tensor<i32>
    %5044 = stablehlo.broadcast_in_dim %c_831, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %5045 = stablehlo.add %5038, %5044 : tensor<1x8x1xi32>
    %5046 = stablehlo.broadcast_in_dim %5037, dims = [0, 1, 2] : (tensor<1x1x8xi32>) -> tensor<1x8x8xi32>
    %5047 = stablehlo.broadcast_in_dim %5045, dims = [0, 1, 2] : (tensor<1x8x1xi32>) -> tensor<1x8x8xi32>
    %5048 = stablehlo.compare LT, %5046, %5047, SIGNED : (tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi1>
    %5049 = stablehlo.and %5043, %5048 : tensor<1x8x8xi1>
    %5050 = stablehlo.and %arg237, %5049 : tensor<1x8x8xi1>
    %5051 = stablehlo.broadcast_in_dim %5050, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_832 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %5052 = call @_where(%5051, %5036, %cst_832) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_833 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %5053 = stablehlo.reduce(%5052 init: %cst_833) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_834 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %5054 = stablehlo.broadcast_in_dim %cst_834, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %5055 = stablehlo.maximum %5054, %5053 : tensor<1x8x8xbf16>
    %5056 = stablehlo.broadcast_in_dim %5055, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %5057 = stablehlo.broadcast_in_dim %5056, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %5058 = stablehlo.subtract %5052, %5057 : tensor<1x8x8x8xbf16>
    %5059 = stablehlo.exponential %5058 : tensor<1x8x8x8xbf16>
    %5060 = stablehlo.convert %5059 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_835 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5061 = stablehlo.reduce(%5060 init: %cst_835) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %5062 = stablehlo.broadcast_in_dim %5061, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %5063 = stablehlo.convert %5062 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %5064 = stablehlo.broadcast_in_dim %5063, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %5065 = stablehlo.divide %5059, %5064 : tensor<1x8x8x8xbf16>
    %5066 = stablehlo.reshape %5065 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %5067 = stablehlo.dot_general %4955, %5066, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %5068 = stablehlo.transpose %5067, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %5069 = stablehlo.reshape %5068 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %5070 = stablehlo.dot_general %5069, %arg155, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5071 = chlo.square %5070 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5072 = stablehlo.convert %5071 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_836 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5073 = stablehlo.reduce(%5072 init: %cst_836) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5074 = stablehlo.broadcast_in_dim %5073, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_837 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5075 = stablehlo.broadcast_in_dim %cst_837, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5076 = stablehlo.divide %5074, %5075 : tensor<1x8x1xf32>
    %5077 = stablehlo.convert %5076 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_838 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5078 = stablehlo.broadcast_in_dim %cst_838, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5079 = stablehlo.add %5077, %5078 : tensor<1x8x1xbf16>
    %5080 = stablehlo.rsqrt %5079 : tensor<1x8x1xbf16>
    %5081 = stablehlo.broadcast_in_dim %5080, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5082 = stablehlo.multiply %5070, %5081 : tensor<1x8x2304xbf16>
    %5083 = stablehlo.broadcast_in_dim %arg160, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_839 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5084 = stablehlo.broadcast_in_dim %cst_839, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5085 = stablehlo.add %5084, %5083 : tensor<1x1x2304xbf16>
    %5086 = stablehlo.broadcast_in_dim %5085, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5087 = stablehlo.multiply %5082, %5086 : tensor<1x8x2304xbf16>
    %5088 = stablehlo.add %5087, %4931 : tensor<1x8x2304xbf16>
    %5089 = chlo.square %5088 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5090 = stablehlo.convert %5089 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_840 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5091 = stablehlo.reduce(%5090 init: %cst_840) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5092 = stablehlo.broadcast_in_dim %5091, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_841 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5093 = stablehlo.broadcast_in_dim %cst_841, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5094 = stablehlo.divide %5092, %5093 : tensor<1x8x1xf32>
    %5095 = stablehlo.convert %5094 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_842 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5096 = stablehlo.broadcast_in_dim %cst_842, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5097 = stablehlo.add %5095, %5096 : tensor<1x8x1xbf16>
    %5098 = stablehlo.rsqrt %5097 : tensor<1x8x1xbf16>
    %5099 = stablehlo.broadcast_in_dim %5098, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5100 = stablehlo.multiply %5088, %5099 : tensor<1x8x2304xbf16>
    %5101 = stablehlo.broadcast_in_dim %arg163, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_843 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5102 = stablehlo.broadcast_in_dim %cst_843, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5103 = stablehlo.add %5102, %5101 : tensor<1x1x2304xbf16>
    %5104 = stablehlo.broadcast_in_dim %5103, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5105 = stablehlo.multiply %5100, %5104 : tensor<1x8x2304xbf16>
    %5106 = stablehlo.dot_general %5105, %arg158, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %5107 = stablehlo.slice %5106 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %5108 = stablehlo.reshape %5107 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %5109 = stablehlo.multiply %5108, %5108 : tensor<1x8x9216xbf16>
    %5110 = stablehlo.multiply %5109, %5108 : tensor<1x8x9216xbf16>
    %cst_844 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %5111 = stablehlo.broadcast_in_dim %cst_844, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5112 = stablehlo.multiply %5111, %5110 : tensor<1x8x9216xbf16>
    %5113 = stablehlo.add %5108, %5112 : tensor<1x8x9216xbf16>
    %cst_845 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %5114 = stablehlo.broadcast_in_dim %cst_845, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5115 = stablehlo.multiply %5114, %5113 : tensor<1x8x9216xbf16>
    %5116 = stablehlo.tanh %5115 : tensor<1x8x9216xbf16>
    %cst_846 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5117 = stablehlo.broadcast_in_dim %cst_846, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5118 = stablehlo.add %5117, %5116 : tensor<1x8x9216xbf16>
    %cst_847 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %5119 = stablehlo.broadcast_in_dim %cst_847, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5120 = stablehlo.multiply %5119, %5118 : tensor<1x8x9216xbf16>
    %5121 = stablehlo.multiply %5108, %5120 : tensor<1x8x9216xbf16>
    %5122 = stablehlo.slice %5106 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %5123 = stablehlo.reshape %5122 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %5124 = stablehlo.multiply %5121, %5123 : tensor<1x8x9216xbf16>
    %5125 = stablehlo.dot_general %5124, %arg159, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5126 = chlo.square %5125 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5127 = stablehlo.convert %5126 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_848 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5128 = stablehlo.reduce(%5127 init: %cst_848) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5129 = stablehlo.broadcast_in_dim %5128, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_849 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5130 = stablehlo.broadcast_in_dim %cst_849, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5131 = stablehlo.divide %5129, %5130 : tensor<1x8x1xf32>
    %5132 = stablehlo.convert %5131 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_850 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5133 = stablehlo.broadcast_in_dim %cst_850, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5134 = stablehlo.add %5132, %5133 : tensor<1x8x1xbf16>
    %5135 = stablehlo.rsqrt %5134 : tensor<1x8x1xbf16>
    %5136 = stablehlo.broadcast_in_dim %5135, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5137 = stablehlo.multiply %5125, %5136 : tensor<1x8x2304xbf16>
    %5138 = stablehlo.broadcast_in_dim %arg161, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_851 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5139 = stablehlo.broadcast_in_dim %cst_851, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5140 = stablehlo.add %5139, %5138 : tensor<1x1x2304xbf16>
    %5141 = stablehlo.broadcast_in_dim %5140, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5142 = stablehlo.multiply %5137, %5141 : tensor<1x8x2304xbf16>
    %5143 = stablehlo.add %5142, %5088 : tensor<1x8x2304xbf16>
    %5144 = chlo.square %5143 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5145 = stablehlo.convert %5144 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_852 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5146 = stablehlo.reduce(%5145 init: %cst_852) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5147 = stablehlo.broadcast_in_dim %5146, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_853 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5148 = stablehlo.broadcast_in_dim %cst_853, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5149 = stablehlo.divide %5147, %5148 : tensor<1x8x1xf32>
    %5150 = stablehlo.convert %5149 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_854 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5151 = stablehlo.broadcast_in_dim %cst_854, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5152 = stablehlo.add %5150, %5151 : tensor<1x8x1xbf16>
    %5153 = stablehlo.rsqrt %5152 : tensor<1x8x1xbf16>
    %5154 = stablehlo.broadcast_in_dim %5153, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5155 = stablehlo.multiply %5143, %5154 : tensor<1x8x2304xbf16>
    %5156 = stablehlo.broadcast_in_dim %arg171, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_855 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5157 = stablehlo.broadcast_in_dim %cst_855, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5158 = stablehlo.add %5157, %5156 : tensor<1x1x2304xbf16>
    %5159 = stablehlo.broadcast_in_dim %5158, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5160 = stablehlo.multiply %5155, %5159 : tensor<1x8x2304xbf16>
    %5161 = stablehlo.dot_general %5160, %arg166, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<8x2304x256xbf16>) -> tensor<1x8x8x256xbf16>
    %5162 = stablehlo.dot_general %arg165, %5160, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2304x256xbf16>, tensor<1x8x2304xbf16>) -> tensor<2x4x256x1x8xbf16>
    %5163 = stablehlo.transpose %5162, dims = [0, 3, 4, 1, 2] : (tensor<2x4x256x1x8xbf16>) -> tensor<2x1x8x4x256xbf16>
    %5164 = stablehlo.slice %5163 [0:1, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %5165 = stablehlo.reshape %5164 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %5166 = stablehlo.slice %5163 [1:2, 0:1, 0:8, 0:4, 0:256] : (tensor<2x1x8x4x256xbf16>) -> tensor<1x1x8x4x256xbf16>
    %5167 = stablehlo.reshape %5166 : (tensor<1x1x8x4x256xbf16>) -> tensor<1x8x4x256xbf16>
    %5168 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_856 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %5169 = stablehlo.broadcast_in_dim %cst_856, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5170 = stablehlo.multiply %5169, %5168 : tensor<128xf32>
    %cst_857 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %5171 = stablehlo.broadcast_in_dim %cst_857, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5172 = stablehlo.power %5171, %5170 : tensor<128xf32>
    %5173 = call @_pad(%5172) : (tensor<128xf32>) -> tensor<128xf32>
    %5174 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %5175 = stablehlo.broadcast_in_dim %5173, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %5176 = stablehlo.convert %5174 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %5177 = stablehlo.broadcast_in_dim %5176, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %5178 = stablehlo.broadcast_in_dim %5175, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %5179 = stablehlo.divide %5177, %5178 : tensor<1x8x128xf32>
    %5180 = stablehlo.broadcast_in_dim %5179, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_858 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5181 = stablehlo.broadcast_in_dim %cst_858, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %5182 = stablehlo.divide %5180, %5181 : tensor<1x8x1x128xf32>
    %5183 = stablehlo.sine %5182 : tensor<1x8x1x128xf32>
    %5184 = stablehlo.cosine %5182 : tensor<1x8x1x128xf32>
    %5185 = stablehlo.slice %5161 [0:1, 0:8, 0:8, 0:128] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %5186 = stablehlo.slice %5161 [0:1, 0:8, 0:8, 128:256] : (tensor<1x8x8x256xbf16>) -> tensor<1x8x8x128xbf16>
    %5187 = stablehlo.convert %5185 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %5188 = stablehlo.broadcast_in_dim %5184, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %5189 = stablehlo.multiply %5187, %5188 : tensor<1x8x8x128xf32>
    %5190 = stablehlo.convert %5186 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %5191 = stablehlo.broadcast_in_dim %5183, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %5192 = stablehlo.multiply %5190, %5191 : tensor<1x8x8x128xf32>
    %5193 = stablehlo.subtract %5189, %5192 : tensor<1x8x8x128xf32>
    %5194 = stablehlo.convert %5186 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %5195 = stablehlo.broadcast_in_dim %5184, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %5196 = stablehlo.multiply %5194, %5195 : tensor<1x8x8x128xf32>
    %5197 = stablehlo.convert %5185 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %5198 = stablehlo.broadcast_in_dim %5183, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x8x128xf32>
    %5199 = stablehlo.multiply %5197, %5198 : tensor<1x8x8x128xf32>
    %5200 = stablehlo.add %5196, %5199 : tensor<1x8x8x128xf32>
    %5201 = stablehlo.concatenate %5193, %5200, dim = 3 : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x256xf32>
    %5202 = stablehlo.convert %5201 : (tensor<1x8x8x256xf32>) -> tensor<1x8x8x256xbf16>
    %cst_859 = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %5203 = stablehlo.broadcast_in_dim %cst_859, dims = [] : (tensor<bf16>) -> tensor<1x8x8x256xbf16>
    %5204 = stablehlo.multiply %5202, %5203 : tensor<1x8x8x256xbf16>
    %5205 = stablehlo.iota dim = 0 : tensor<128xf32>
    %cst_860 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %5206 = stablehlo.broadcast_in_dim %cst_860, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5207 = stablehlo.multiply %5206, %5205 : tensor<128xf32>
    %cst_861 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %5208 = stablehlo.broadcast_in_dim %cst_861, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5209 = stablehlo.power %5208, %5207 : tensor<128xf32>
    %5210 = call @_pad(%5209) : (tensor<128xf32>) -> tensor<128xf32>
    %5211 = stablehlo.broadcast_in_dim %arg238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %5212 = stablehlo.broadcast_in_dim %5210, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %5213 = stablehlo.convert %5211 : (tensor<1x8x1xi32>) -> tensor<1x8x1xf32>
    %5214 = stablehlo.broadcast_in_dim %5213, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x128xf32>
    %5215 = stablehlo.broadcast_in_dim %5212, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x8x128xf32>
    %5216 = stablehlo.divide %5214, %5215 : tensor<1x8x128xf32>
    %5217 = stablehlo.broadcast_in_dim %5216, dims = [0, 1, 3] : (tensor<1x8x128xf32>) -> tensor<1x8x1x128xf32>
    %cst_862 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5218 = stablehlo.broadcast_in_dim %cst_862, dims = [] : (tensor<f32>) -> tensor<1x8x1x128xf32>
    %5219 = stablehlo.divide %5217, %5218 : tensor<1x8x1x128xf32>
    %5220 = stablehlo.sine %5219 : tensor<1x8x1x128xf32>
    %5221 = stablehlo.cosine %5219 : tensor<1x8x1x128xf32>
    %5222 = stablehlo.slice %5165 [0:1, 0:8, 0:4, 0:128] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %5223 = stablehlo.slice %5165 [0:1, 0:8, 0:4, 128:256] : (tensor<1x8x4x256xbf16>) -> tensor<1x8x4x128xbf16>
    %5224 = stablehlo.convert %5222 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5225 = stablehlo.broadcast_in_dim %5221, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5226 = stablehlo.multiply %5224, %5225 : tensor<1x8x4x128xf32>
    %5227 = stablehlo.convert %5223 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5228 = stablehlo.broadcast_in_dim %5220, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5229 = stablehlo.multiply %5227, %5228 : tensor<1x8x4x128xf32>
    %5230 = stablehlo.subtract %5226, %5229 : tensor<1x8x4x128xf32>
    %5231 = stablehlo.convert %5223 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5232 = stablehlo.broadcast_in_dim %5221, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5233 = stablehlo.multiply %5231, %5232 : tensor<1x8x4x128xf32>
    %5234 = stablehlo.convert %5222 : (tensor<1x8x4x128xbf16>) -> tensor<1x8x4x128xf32>
    %5235 = stablehlo.broadcast_in_dim %5220, dims = [0, 1, 2, 3] : (tensor<1x8x1x128xf32>) -> tensor<1x8x4x128xf32>
    %5236 = stablehlo.multiply %5234, %5235 : tensor<1x8x4x128xf32>
    %5237 = stablehlo.add %5233, %5236 : tensor<1x8x4x128xf32>
    %5238 = stablehlo.concatenate %5230, %5237, dim = 3 : (tensor<1x8x4x128xf32>, tensor<1x8x4x128xf32>) -> tensor<1x8x4x256xf32>
    %5239 = stablehlo.convert %5238 : (tensor<1x8x4x256xf32>) -> tensor<1x8x4x256xbf16>
    %5240 = stablehlo.reshape %5204 : (tensor<1x8x8x256xbf16>) -> tensor<1x8x4x2x256xbf16>
    %5241 = stablehlo.dot_general %5239, %5240, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x256xbf16>) -> tensor<1x4x8x8x2xbf16>
    %5242 = stablehlo.transpose %5241, dims = [0, 3, 1, 4, 2] : (tensor<1x4x8x8x2xbf16>) -> tensor<1x8x4x2x8xbf16>
    %5243 = stablehlo.reshape %5242 : (tensor<1x8x4x2x8xbf16>) -> tensor<1x8x8x8xbf16>
    %cst_863 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %5244 = stablehlo.broadcast_in_dim %cst_863, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %5245 = stablehlo.divide %5243, %5244 : tensor<1x8x8x8xbf16>
    %5246 = stablehlo.tanh %5245 : tensor<1x8x8x8xbf16>
    %cst_864 = stablehlo.constant dense<5.000000e+01> : tensor<bf16>
    %5247 = stablehlo.broadcast_in_dim %cst_864, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %5248 = stablehlo.multiply %5246, %5247 : tensor<1x8x8x8xbf16>
    %5249 = stablehlo.broadcast_in_dim %arg237, dims = [0, 1, 3] : (tensor<1x8x8xi1>) -> tensor<1x8x1x8xi1>
    %cst_865 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %5250 = call @_where(%5249, %5248, %cst_865) : (tensor<1x8x1x8xi1>, tensor<1x8x8x8xbf16>, tensor<f32>) -> tensor<1x8x8x8xbf16>
    %cst_866 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %5251 = stablehlo.reduce(%5250 init: %cst_866) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x8x8xbf16>, tensor<bf16>) -> tensor<1x8x8xbf16>
    %cst_867 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %5252 = stablehlo.broadcast_in_dim %cst_867, dims = [] : (tensor<bf16>) -> tensor<1x8x8xbf16>
    %5253 = stablehlo.maximum %5252, %5251 : tensor<1x8x8xbf16>
    %5254 = stablehlo.broadcast_in_dim %5253, dims = [0, 1, 2] : (tensor<1x8x8xbf16>) -> tensor<1x8x8x1xbf16>
    %5255 = stablehlo.broadcast_in_dim %5254, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %5256 = stablehlo.subtract %5250, %5255 : tensor<1x8x8x8xbf16>
    %5257 = stablehlo.exponential %5256 : tensor<1x8x8x8xbf16>
    %5258 = stablehlo.convert %5257 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x8x8xf32>
    %cst_868 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5259 = stablehlo.reduce(%5258 init: %cst_868) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %5260 = stablehlo.broadcast_in_dim %5259, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %5261 = stablehlo.convert %5260 : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x1xbf16>
    %5262 = stablehlo.broadcast_in_dim %5261, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xbf16>) -> tensor<1x8x8x8xbf16>
    %5263 = stablehlo.divide %5257, %5262 : tensor<1x8x8x8xbf16>
    %5264 = stablehlo.reshape %5263 : (tensor<1x8x8x8xbf16>) -> tensor<1x8x4x2x8xbf16>
    %5265 = stablehlo.dot_general %5167, %5264, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x4x256xbf16>, tensor<1x8x4x2x8xbf16>) -> tensor<1x4x256x8x2xbf16>
    %5266 = stablehlo.transpose %5265, dims = [0, 3, 1, 4, 2] : (tensor<1x4x256x8x2xbf16>) -> tensor<1x8x4x2x256xbf16>
    %5267 = stablehlo.reshape %5266 : (tensor<1x8x4x2x256xbf16>) -> tensor<1x8x8x256xbf16>
    %5268 = stablehlo.dot_general %5267, %arg164, contracting_dims = [3, 2] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x256xbf16>, tensor<8x256x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5269 = chlo.square %5268 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5270 = stablehlo.convert %5269 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_869 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5271 = stablehlo.reduce(%5270 init: %cst_869) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5272 = stablehlo.broadcast_in_dim %5271, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_870 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5273 = stablehlo.broadcast_in_dim %cst_870, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5274 = stablehlo.divide %5272, %5273 : tensor<1x8x1xf32>
    %5275 = stablehlo.convert %5274 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_871 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5276 = stablehlo.broadcast_in_dim %cst_871, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5277 = stablehlo.add %5275, %5276 : tensor<1x8x1xbf16>
    %5278 = stablehlo.rsqrt %5277 : tensor<1x8x1xbf16>
    %5279 = stablehlo.broadcast_in_dim %5278, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5280 = stablehlo.multiply %5268, %5279 : tensor<1x8x2304xbf16>
    %5281 = stablehlo.broadcast_in_dim %arg169, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_872 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5282 = stablehlo.broadcast_in_dim %cst_872, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5283 = stablehlo.add %5282, %5281 : tensor<1x1x2304xbf16>
    %5284 = stablehlo.broadcast_in_dim %5283, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5285 = stablehlo.multiply %5280, %5284 : tensor<1x8x2304xbf16>
    %5286 = stablehlo.add %5285, %5143 : tensor<1x8x2304xbf16>
    %5287 = chlo.square %5286 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5288 = stablehlo.convert %5287 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_873 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5289 = stablehlo.reduce(%5288 init: %cst_873) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5290 = stablehlo.broadcast_in_dim %5289, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_874 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5291 = stablehlo.broadcast_in_dim %cst_874, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5292 = stablehlo.divide %5290, %5291 : tensor<1x8x1xf32>
    %5293 = stablehlo.convert %5292 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_875 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5294 = stablehlo.broadcast_in_dim %cst_875, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5295 = stablehlo.add %5293, %5294 : tensor<1x8x1xbf16>
    %5296 = stablehlo.rsqrt %5295 : tensor<1x8x1xbf16>
    %5297 = stablehlo.broadcast_in_dim %5296, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5298 = stablehlo.multiply %5286, %5297 : tensor<1x8x2304xbf16>
    %5299 = stablehlo.broadcast_in_dim %arg172, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_876 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5300 = stablehlo.broadcast_in_dim %cst_876, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5301 = stablehlo.add %5300, %5299 : tensor<1x1x2304xbf16>
    %5302 = stablehlo.broadcast_in_dim %5301, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5303 = stablehlo.multiply %5298, %5302 : tensor<1x8x2304xbf16>
    %5304 = stablehlo.dot_general %5303, %arg167, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2x2304x9216xbf16>) -> tensor<1x8x2x9216xbf16>
    %5305 = stablehlo.slice %5304 [0:1, 0:8, 0:1, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %5306 = stablehlo.reshape %5305 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %5307 = stablehlo.multiply %5306, %5306 : tensor<1x8x9216xbf16>
    %5308 = stablehlo.multiply %5307, %5306 : tensor<1x8x9216xbf16>
    %cst_877 = stablehlo.constant dense<4.467770e-02> : tensor<bf16>
    %5309 = stablehlo.broadcast_in_dim %cst_877, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5310 = stablehlo.multiply %5309, %5308 : tensor<1x8x9216xbf16>
    %5311 = stablehlo.add %5306, %5310 : tensor<1x8x9216xbf16>
    %cst_878 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %5312 = stablehlo.broadcast_in_dim %cst_878, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5313 = stablehlo.multiply %5312, %5311 : tensor<1x8x9216xbf16>
    %5314 = stablehlo.tanh %5313 : tensor<1x8x9216xbf16>
    %cst_879 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5315 = stablehlo.broadcast_in_dim %cst_879, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5316 = stablehlo.add %5315, %5314 : tensor<1x8x9216xbf16>
    %cst_880 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %5317 = stablehlo.broadcast_in_dim %cst_880, dims = [] : (tensor<bf16>) -> tensor<1x8x9216xbf16>
    %5318 = stablehlo.multiply %5317, %5316 : tensor<1x8x9216xbf16>
    %5319 = stablehlo.multiply %5306, %5318 : tensor<1x8x9216xbf16>
    %5320 = stablehlo.slice %5304 [0:1, 0:8, 1:2, 0:9216] : (tensor<1x8x2x9216xbf16>) -> tensor<1x8x1x9216xbf16>
    %5321 = stablehlo.reshape %5320 : (tensor<1x8x1x9216xbf16>) -> tensor<1x8x9216xbf16>
    %5322 = stablehlo.multiply %5319, %5321 : tensor<1x8x9216xbf16>
    %5323 = stablehlo.dot_general %5322, %arg168, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x9216xbf16>, tensor<9216x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5324 = chlo.square %5323 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5325 = stablehlo.convert %5324 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_881 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5326 = stablehlo.reduce(%5325 init: %cst_881) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5327 = stablehlo.broadcast_in_dim %5326, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_882 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5328 = stablehlo.broadcast_in_dim %cst_882, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5329 = stablehlo.divide %5327, %5328 : tensor<1x8x1xf32>
    %5330 = stablehlo.convert %5329 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_883 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5331 = stablehlo.broadcast_in_dim %cst_883, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5332 = stablehlo.add %5330, %5331 : tensor<1x8x1xbf16>
    %5333 = stablehlo.rsqrt %5332 : tensor<1x8x1xbf16>
    %5334 = stablehlo.broadcast_in_dim %5333, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5335 = stablehlo.multiply %5323, %5334 : tensor<1x8x2304xbf16>
    %5336 = stablehlo.broadcast_in_dim %arg170, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_884 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5337 = stablehlo.broadcast_in_dim %cst_884, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5338 = stablehlo.add %5337, %5336 : tensor<1x1x2304xbf16>
    %5339 = stablehlo.broadcast_in_dim %5338, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5340 = stablehlo.multiply %5335, %5339 : tensor<1x8x2304xbf16>
    %5341 = stablehlo.add %5340, %5286 : tensor<1x8x2304xbf16>
    %5342 = chlo.square %5341 : tensor<1x8x2304xbf16> -> tensor<1x8x2304xbf16>
    %5343 = stablehlo.convert %5342 : (tensor<1x8x2304xbf16>) -> tensor<1x8x2304xf32>
    %cst_885 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5344 = stablehlo.reduce(%5343 init: %cst_885) applies stablehlo.add across dimensions = [2] : (tensor<1x8x2304xf32>, tensor<f32>) -> tensor<1x8xf32>
    %5345 = stablehlo.broadcast_in_dim %5344, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_886 = stablehlo.constant dense<2.304000e+03> : tensor<f32>
    %5346 = stablehlo.broadcast_in_dim %cst_886, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %5347 = stablehlo.divide %5345, %5346 : tensor<1x8x1xf32>
    %5348 = stablehlo.convert %5347 : (tensor<1x8x1xf32>) -> tensor<1x8x1xbf16>
    %cst_887 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %5349 = stablehlo.broadcast_in_dim %cst_887, dims = [] : (tensor<bf16>) -> tensor<1x8x1xbf16>
    %5350 = stablehlo.add %5348, %5349 : tensor<1x8x1xbf16>
    %5351 = stablehlo.rsqrt %5350 : tensor<1x8x1xbf16>
    %5352 = stablehlo.broadcast_in_dim %5351, dims = [0, 1, 2] : (tensor<1x8x1xbf16>) -> tensor<1x8x2304xbf16>
    %5353 = stablehlo.multiply %5341, %5352 : tensor<1x8x2304xbf16>
    %5354 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2304xbf16>) -> tensor<1x1x2304xbf16>
    %cst_888 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5355 = stablehlo.broadcast_in_dim %cst_888, dims = [] : (tensor<bf16>) -> tensor<1x1x2304xbf16>
    %5356 = stablehlo.add %5355, %5354 : tensor<1x1x2304xbf16>
    %5357 = stablehlo.broadcast_in_dim %5356, dims = [0, 1, 2] : (tensor<1x1x2304xbf16>) -> tensor<1x8x2304xbf16>
    %5358 = stablehlo.multiply %5353, %5357 : tensor<1x8x2304xbf16>
    %5359 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<256128x2304xbf16>) -> tensor<2304x256128xbf16>
    %5360 = stablehlo.dot_general %5358, %5359, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2304xbf16>, tensor<2304x256128xbf16>) -> tensor<1x8x256128xbf16>
    %cst_889 = stablehlo.constant dense<3.000000e+01> : tensor<bf16>
    %5361 = stablehlo.broadcast_in_dim %cst_889, dims = [] : (tensor<bf16>) -> tensor<1x8x256128xbf16>
    %5362 = stablehlo.divide %5360, %5361 : tensor<1x8x256128xbf16>
    %5363 = stablehlo.tanh %5362 : tensor<1x8x256128xbf16>
    %cst_890 = stablehlo.constant dense<3.000000e+01> : tensor<bf16>
    %5364 = stablehlo.broadcast_in_dim %cst_890, dims = [] : (tensor<bf16>) -> tensor<1x8x256128xbf16>
    %5365 = stablehlo.multiply %5363, %5364 : tensor<1x8x256128xbf16>
    return %5365 : tensor<1x8x256128xbf16>
  }
  func.func private @tokens_with_mm(%arg0: tensor<1x8xi32>) -> tensor<1x8xi32> {
    return %arg0 : tensor<1x8xi32>
  }
  func.func private @_pad(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }
  func.func private @_where(%arg0: tensor<1x8x1x8xi1>, %arg1: tensor<1x8x8x8xbf16>, %arg2: tensor<f32>) -> tensor<1x8x8x8xbf16> {
    %0 = stablehlo.convert %arg2 : (tensor<f32>) -> tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x8x1x8xi1>) -> tensor<1x8x8x8xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<1x8x8x8xbf16>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<1x8x8x8xi1>, tensor<1x8x8x8xbf16>
    return %3 : tensor<1x8x8x8xbf16>
  }
}
