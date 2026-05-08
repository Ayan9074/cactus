module @jit_forward_flat attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x256xbf16>, %arg1: tensor<2048x2048xbf16>, %arg2: tensor<2048x2048xbf16>, %arg3: tensor<2048x256xbf16>, %arg4: tensor<2048xbf16>, %arg5: tensor<5632x2048xbf16>, %arg6: tensor<2048x5632xbf16>, %arg7: tensor<2048x5632xbf16>, %arg8: tensor<2048xbf16>, %arg9: tensor<2048x256xbf16>, %arg10: tensor<2048x2048xbf16>, %arg11: tensor<2048x2048xbf16>, %arg12: tensor<2048x256xbf16>, %arg13: tensor<2048xbf16>, %arg14: tensor<5632x2048xbf16>, %arg15: tensor<2048x5632xbf16>, %arg16: tensor<2048x5632xbf16>, %arg17: tensor<2048xbf16>, %arg18: tensor<2048x256xbf16>, %arg19: tensor<2048x2048xbf16>, %arg20: tensor<2048x2048xbf16>, %arg21: tensor<2048x256xbf16>, %arg22: tensor<2048xbf16>, %arg23: tensor<5632x2048xbf16>, %arg24: tensor<2048x5632xbf16>, %arg25: tensor<2048x5632xbf16>, %arg26: tensor<2048xbf16>, %arg27: tensor<2048x256xbf16>, %arg28: tensor<2048x2048xbf16>, %arg29: tensor<2048x2048xbf16>, %arg30: tensor<2048x256xbf16>, %arg31: tensor<2048xbf16>, %arg32: tensor<5632x2048xbf16>, %arg33: tensor<2048x5632xbf16>, %arg34: tensor<2048x5632xbf16>, %arg35: tensor<2048xbf16>, %arg36: tensor<2048x256xbf16>, %arg37: tensor<2048x2048xbf16>, %arg38: tensor<2048x2048xbf16>, %arg39: tensor<2048x256xbf16>, %arg40: tensor<2048xbf16>, %arg41: tensor<5632x2048xbf16>, %arg42: tensor<2048x5632xbf16>, %arg43: tensor<2048x5632xbf16>, %arg44: tensor<2048xbf16>, %arg45: tensor<2048x256xbf16>, %arg46: tensor<2048x2048xbf16>, %arg47: tensor<2048x2048xbf16>, %arg48: tensor<2048x256xbf16>, %arg49: tensor<2048xbf16>, %arg50: tensor<5632x2048xbf16>, %arg51: tensor<2048x5632xbf16>, %arg52: tensor<2048x5632xbf16>, %arg53: tensor<2048xbf16>, %arg54: tensor<2048x256xbf16>, %arg55: tensor<2048x2048xbf16>, %arg56: tensor<2048x2048xbf16>, %arg57: tensor<2048x256xbf16>, %arg58: tensor<2048xbf16>, %arg59: tensor<5632x2048xbf16>, %arg60: tensor<2048x5632xbf16>, %arg61: tensor<2048x5632xbf16>, %arg62: tensor<2048xbf16>, %arg63: tensor<2048x256xbf16>, %arg64: tensor<2048x2048xbf16>, %arg65: tensor<2048x2048xbf16>, %arg66: tensor<2048x256xbf16>, %arg67: tensor<2048xbf16>, %arg68: tensor<5632x2048xbf16>, %arg69: tensor<2048x5632xbf16>, %arg70: tensor<2048x5632xbf16>, %arg71: tensor<2048xbf16>, %arg72: tensor<2048x256xbf16>, %arg73: tensor<2048x2048xbf16>, %arg74: tensor<2048x2048xbf16>, %arg75: tensor<2048x256xbf16>, %arg76: tensor<2048xbf16>, %arg77: tensor<5632x2048xbf16>, %arg78: tensor<2048x5632xbf16>, %arg79: tensor<2048x5632xbf16>, %arg80: tensor<2048xbf16>, %arg81: tensor<2048x256xbf16>, %arg82: tensor<2048x2048xbf16>, %arg83: tensor<2048x2048xbf16>, %arg84: tensor<2048x256xbf16>, %arg85: tensor<2048xbf16>, %arg86: tensor<5632x2048xbf16>, %arg87: tensor<2048x5632xbf16>, %arg88: tensor<2048x5632xbf16>, %arg89: tensor<2048xbf16>, %arg90: tensor<2048x256xbf16>, %arg91: tensor<2048x2048xbf16>, %arg92: tensor<2048x2048xbf16>, %arg93: tensor<2048x256xbf16>, %arg94: tensor<2048xbf16>, %arg95: tensor<5632x2048xbf16>, %arg96: tensor<2048x5632xbf16>, %arg97: tensor<2048x5632xbf16>, %arg98: tensor<2048xbf16>, %arg99: tensor<2048x256xbf16>, %arg100: tensor<2048x2048xbf16>, %arg101: tensor<2048x2048xbf16>, %arg102: tensor<2048x256xbf16>, %arg103: tensor<2048xbf16>, %arg104: tensor<5632x2048xbf16>, %arg105: tensor<2048x5632xbf16>, %arg106: tensor<2048x5632xbf16>, %arg107: tensor<2048xbf16>, %arg108: tensor<2048x256xbf16>, %arg109: tensor<2048x2048xbf16>, %arg110: tensor<2048x2048xbf16>, %arg111: tensor<2048x256xbf16>, %arg112: tensor<2048xbf16>, %arg113: tensor<5632x2048xbf16>, %arg114: tensor<2048x5632xbf16>, %arg115: tensor<2048x5632xbf16>, %arg116: tensor<2048xbf16>, %arg117: tensor<2048x256xbf16>, %arg118: tensor<2048x2048xbf16>, %arg119: tensor<2048x2048xbf16>, %arg120: tensor<2048x256xbf16>, %arg121: tensor<2048xbf16>, %arg122: tensor<5632x2048xbf16>, %arg123: tensor<2048x5632xbf16>, %arg124: tensor<2048x5632xbf16>, %arg125: tensor<2048xbf16>, %arg126: tensor<2048x256xbf16>, %arg127: tensor<2048x2048xbf16>, %arg128: tensor<2048x2048xbf16>, %arg129: tensor<2048x256xbf16>, %arg130: tensor<2048xbf16>, %arg131: tensor<5632x2048xbf16>, %arg132: tensor<2048x5632xbf16>, %arg133: tensor<2048x5632xbf16>, %arg134: tensor<2048xbf16>, %arg135: tensor<2048x256xbf16>, %arg136: tensor<2048x2048xbf16>, %arg137: tensor<2048x2048xbf16>, %arg138: tensor<2048x256xbf16>, %arg139: tensor<2048xbf16>, %arg140: tensor<5632x2048xbf16>, %arg141: tensor<2048x5632xbf16>, %arg142: tensor<2048x5632xbf16>, %arg143: tensor<2048xbf16>, %arg144: tensor<2048x256xbf16>, %arg145: tensor<2048x2048xbf16>, %arg146: tensor<2048x2048xbf16>, %arg147: tensor<2048x256xbf16>, %arg148: tensor<2048xbf16>, %arg149: tensor<5632x2048xbf16>, %arg150: tensor<2048x5632xbf16>, %arg151: tensor<2048x5632xbf16>, %arg152: tensor<2048xbf16>, %arg153: tensor<2048x256xbf16>, %arg154: tensor<2048x2048xbf16>, %arg155: tensor<2048x2048xbf16>, %arg156: tensor<2048x256xbf16>, %arg157: tensor<2048xbf16>, %arg158: tensor<5632x2048xbf16>, %arg159: tensor<2048x5632xbf16>, %arg160: tensor<2048x5632xbf16>, %arg161: tensor<2048xbf16>, %arg162: tensor<2048x256xbf16>, %arg163: tensor<2048x2048xbf16>, %arg164: tensor<2048x2048xbf16>, %arg165: tensor<2048x256xbf16>, %arg166: tensor<2048xbf16>, %arg167: tensor<5632x2048xbf16>, %arg168: tensor<2048x5632xbf16>, %arg169: tensor<2048x5632xbf16>, %arg170: tensor<2048xbf16>, %arg171: tensor<2048x256xbf16>, %arg172: tensor<2048x2048xbf16>, %arg173: tensor<2048x2048xbf16>, %arg174: tensor<2048x256xbf16>, %arg175: tensor<2048xbf16>, %arg176: tensor<5632x2048xbf16>, %arg177: tensor<2048x5632xbf16>, %arg178: tensor<2048x5632xbf16>, %arg179: tensor<2048xbf16>, %arg180: tensor<2048x256xbf16>, %arg181: tensor<2048x2048xbf16>, %arg182: tensor<2048x2048xbf16>, %arg183: tensor<2048x256xbf16>, %arg184: tensor<2048xbf16>, %arg185: tensor<5632x2048xbf16>, %arg186: tensor<2048x5632xbf16>, %arg187: tensor<2048x5632xbf16>, %arg188: tensor<2048xbf16>, %arg189: tensor<2048x256xbf16>, %arg190: tensor<2048x2048xbf16>, %arg191: tensor<2048x2048xbf16>, %arg192: tensor<2048x256xbf16>, %arg193: tensor<2048xbf16>, %arg194: tensor<5632x2048xbf16>, %arg195: tensor<2048x5632xbf16>, %arg196: tensor<2048x5632xbf16>, %arg197: tensor<2048xbf16>, %arg198: tensor<2048x32000xbf16>, %arg199: tensor<2048xbf16>, %arg200: tensor<32000x2048xbf16>, %arg201: tensor<1x16xi32>, %arg202: tensor<16x16xi1>) -> (tensor<1x16x32000xf32> {jax.result_info = "result"}) {
    %0 = call @__call__(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<2048x32000xbf16>, tensor<2048xbf16>, tensor<32000x2048xbf16>, tensor<1x16xi32>, tensor<16x16xi1>) -> tensor<1x16x32000xf32>
    return %0 : tensor<1x16x32000xf32>
  }
  func.func private @__call__(%arg0: tensor<2048x256xbf16>, %arg1: tensor<2048x2048xbf16>, %arg2: tensor<2048x2048xbf16>, %arg3: tensor<2048x256xbf16>, %arg4: tensor<2048xbf16>, %arg5: tensor<5632x2048xbf16>, %arg6: tensor<2048x5632xbf16>, %arg7: tensor<2048x5632xbf16>, %arg8: tensor<2048xbf16>, %arg9: tensor<2048x256xbf16>, %arg10: tensor<2048x2048xbf16>, %arg11: tensor<2048x2048xbf16>, %arg12: tensor<2048x256xbf16>, %arg13: tensor<2048xbf16>, %arg14: tensor<5632x2048xbf16>, %arg15: tensor<2048x5632xbf16>, %arg16: tensor<2048x5632xbf16>, %arg17: tensor<2048xbf16>, %arg18: tensor<2048x256xbf16>, %arg19: tensor<2048x2048xbf16>, %arg20: tensor<2048x2048xbf16>, %arg21: tensor<2048x256xbf16>, %arg22: tensor<2048xbf16>, %arg23: tensor<5632x2048xbf16>, %arg24: tensor<2048x5632xbf16>, %arg25: tensor<2048x5632xbf16>, %arg26: tensor<2048xbf16>, %arg27: tensor<2048x256xbf16>, %arg28: tensor<2048x2048xbf16>, %arg29: tensor<2048x2048xbf16>, %arg30: tensor<2048x256xbf16>, %arg31: tensor<2048xbf16>, %arg32: tensor<5632x2048xbf16>, %arg33: tensor<2048x5632xbf16>, %arg34: tensor<2048x5632xbf16>, %arg35: tensor<2048xbf16>, %arg36: tensor<2048x256xbf16>, %arg37: tensor<2048x2048xbf16>, %arg38: tensor<2048x2048xbf16>, %arg39: tensor<2048x256xbf16>, %arg40: tensor<2048xbf16>, %arg41: tensor<5632x2048xbf16>, %arg42: tensor<2048x5632xbf16>, %arg43: tensor<2048x5632xbf16>, %arg44: tensor<2048xbf16>, %arg45: tensor<2048x256xbf16>, %arg46: tensor<2048x2048xbf16>, %arg47: tensor<2048x2048xbf16>, %arg48: tensor<2048x256xbf16>, %arg49: tensor<2048xbf16>, %arg50: tensor<5632x2048xbf16>, %arg51: tensor<2048x5632xbf16>, %arg52: tensor<2048x5632xbf16>, %arg53: tensor<2048xbf16>, %arg54: tensor<2048x256xbf16>, %arg55: tensor<2048x2048xbf16>, %arg56: tensor<2048x2048xbf16>, %arg57: tensor<2048x256xbf16>, %arg58: tensor<2048xbf16>, %arg59: tensor<5632x2048xbf16>, %arg60: tensor<2048x5632xbf16>, %arg61: tensor<2048x5632xbf16>, %arg62: tensor<2048xbf16>, %arg63: tensor<2048x256xbf16>, %arg64: tensor<2048x2048xbf16>, %arg65: tensor<2048x2048xbf16>, %arg66: tensor<2048x256xbf16>, %arg67: tensor<2048xbf16>, %arg68: tensor<5632x2048xbf16>, %arg69: tensor<2048x5632xbf16>, %arg70: tensor<2048x5632xbf16>, %arg71: tensor<2048xbf16>, %arg72: tensor<2048x256xbf16>, %arg73: tensor<2048x2048xbf16>, %arg74: tensor<2048x2048xbf16>, %arg75: tensor<2048x256xbf16>, %arg76: tensor<2048xbf16>, %arg77: tensor<5632x2048xbf16>, %arg78: tensor<2048x5632xbf16>, %arg79: tensor<2048x5632xbf16>, %arg80: tensor<2048xbf16>, %arg81: tensor<2048x256xbf16>, %arg82: tensor<2048x2048xbf16>, %arg83: tensor<2048x2048xbf16>, %arg84: tensor<2048x256xbf16>, %arg85: tensor<2048xbf16>, %arg86: tensor<5632x2048xbf16>, %arg87: tensor<2048x5632xbf16>, %arg88: tensor<2048x5632xbf16>, %arg89: tensor<2048xbf16>, %arg90: tensor<2048x256xbf16>, %arg91: tensor<2048x2048xbf16>, %arg92: tensor<2048x2048xbf16>, %arg93: tensor<2048x256xbf16>, %arg94: tensor<2048xbf16>, %arg95: tensor<5632x2048xbf16>, %arg96: tensor<2048x5632xbf16>, %arg97: tensor<2048x5632xbf16>, %arg98: tensor<2048xbf16>, %arg99: tensor<2048x256xbf16>, %arg100: tensor<2048x2048xbf16>, %arg101: tensor<2048x2048xbf16>, %arg102: tensor<2048x256xbf16>, %arg103: tensor<2048xbf16>, %arg104: tensor<5632x2048xbf16>, %arg105: tensor<2048x5632xbf16>, %arg106: tensor<2048x5632xbf16>, %arg107: tensor<2048xbf16>, %arg108: tensor<2048x256xbf16>, %arg109: tensor<2048x2048xbf16>, %arg110: tensor<2048x2048xbf16>, %arg111: tensor<2048x256xbf16>, %arg112: tensor<2048xbf16>, %arg113: tensor<5632x2048xbf16>, %arg114: tensor<2048x5632xbf16>, %arg115: tensor<2048x5632xbf16>, %arg116: tensor<2048xbf16>, %arg117: tensor<2048x256xbf16>, %arg118: tensor<2048x2048xbf16>, %arg119: tensor<2048x2048xbf16>, %arg120: tensor<2048x256xbf16>, %arg121: tensor<2048xbf16>, %arg122: tensor<5632x2048xbf16>, %arg123: tensor<2048x5632xbf16>, %arg124: tensor<2048x5632xbf16>, %arg125: tensor<2048xbf16>, %arg126: tensor<2048x256xbf16>, %arg127: tensor<2048x2048xbf16>, %arg128: tensor<2048x2048xbf16>, %arg129: tensor<2048x256xbf16>, %arg130: tensor<2048xbf16>, %arg131: tensor<5632x2048xbf16>, %arg132: tensor<2048x5632xbf16>, %arg133: tensor<2048x5632xbf16>, %arg134: tensor<2048xbf16>, %arg135: tensor<2048x256xbf16>, %arg136: tensor<2048x2048xbf16>, %arg137: tensor<2048x2048xbf16>, %arg138: tensor<2048x256xbf16>, %arg139: tensor<2048xbf16>, %arg140: tensor<5632x2048xbf16>, %arg141: tensor<2048x5632xbf16>, %arg142: tensor<2048x5632xbf16>, %arg143: tensor<2048xbf16>, %arg144: tensor<2048x256xbf16>, %arg145: tensor<2048x2048xbf16>, %arg146: tensor<2048x2048xbf16>, %arg147: tensor<2048x256xbf16>, %arg148: tensor<2048xbf16>, %arg149: tensor<5632x2048xbf16>, %arg150: tensor<2048x5632xbf16>, %arg151: tensor<2048x5632xbf16>, %arg152: tensor<2048xbf16>, %arg153: tensor<2048x256xbf16>, %arg154: tensor<2048x2048xbf16>, %arg155: tensor<2048x2048xbf16>, %arg156: tensor<2048x256xbf16>, %arg157: tensor<2048xbf16>, %arg158: tensor<5632x2048xbf16>, %arg159: tensor<2048x5632xbf16>, %arg160: tensor<2048x5632xbf16>, %arg161: tensor<2048xbf16>, %arg162: tensor<2048x256xbf16>, %arg163: tensor<2048x2048xbf16>, %arg164: tensor<2048x2048xbf16>, %arg165: tensor<2048x256xbf16>, %arg166: tensor<2048xbf16>, %arg167: tensor<5632x2048xbf16>, %arg168: tensor<2048x5632xbf16>, %arg169: tensor<2048x5632xbf16>, %arg170: tensor<2048xbf16>, %arg171: tensor<2048x256xbf16>, %arg172: tensor<2048x2048xbf16>, %arg173: tensor<2048x2048xbf16>, %arg174: tensor<2048x256xbf16>, %arg175: tensor<2048xbf16>, %arg176: tensor<5632x2048xbf16>, %arg177: tensor<2048x5632xbf16>, %arg178: tensor<2048x5632xbf16>, %arg179: tensor<2048xbf16>, %arg180: tensor<2048x256xbf16>, %arg181: tensor<2048x2048xbf16>, %arg182: tensor<2048x2048xbf16>, %arg183: tensor<2048x256xbf16>, %arg184: tensor<2048xbf16>, %arg185: tensor<5632x2048xbf16>, %arg186: tensor<2048x5632xbf16>, %arg187: tensor<2048x5632xbf16>, %arg188: tensor<2048xbf16>, %arg189: tensor<2048x256xbf16>, %arg190: tensor<2048x2048xbf16>, %arg191: tensor<2048x2048xbf16>, %arg192: tensor<2048x256xbf16>, %arg193: tensor<2048xbf16>, %arg194: tensor<5632x2048xbf16>, %arg195: tensor<2048x5632xbf16>, %arg196: tensor<2048x5632xbf16>, %arg197: tensor<2048xbf16>, %arg198: tensor<2048x32000xbf16>, %arg199: tensor<2048xbf16>, %arg200: tensor<32000x2048xbf16>, %arg201: tensor<1x16xi32>, %arg202: tensor<16x16xi1>) -> tensor<1x16x32000xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<16xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<16xi32>) -> tensor<1x16xi32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = call @_where(%arg202, %cst, %cst_0) : (tensor<16x16xi1>, tensor<f32>, tensor<f32>) -> tensor<16x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [2, 3] : (tensor<16x16xf32>) -> tensor<1x1x16x16xf32>
    %4 = stablehlo.convert %arg200 : (tensor<32000x2048xbf16>) -> tensor<32000x2048xf16>
    %5 = call @_take(%4, %arg201) : (tensor<32000x2048xf16>, tensor<1x16xi32>) -> tensor<1x16x2048xf16>
    %6 = call @__call___10(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %5, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf16>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %7 = call @__call___85(%arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %6, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %8 = call @__call___85(%arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %7, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %9 = call @__call___85(%arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %8, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %10 = call @__call___85(%arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %9, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %11 = call @__call___85(%arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %10, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %12 = call @__call___85(%arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %11, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %13 = call @__call___85(%arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %12, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %14 = call @__call___85(%arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %13, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %15 = call @__call___85(%arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %14, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %16 = call @__call___85(%arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %15, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %17 = call @__call___85(%arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %16, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %18 = call @__call___85(%arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %17, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %19 = call @__call___85(%arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %18, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %20 = call @__call___85(%arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %19, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %21 = call @__call___85(%arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %20, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %22 = call @__call___85(%arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %21, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %23 = call @__call___85(%arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %22, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %24 = call @__call___85(%arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %23, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %25 = call @__call___85(%arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %24, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %26 = call @__call___85(%arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %25, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %27 = call @__call___85(%arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %26, %1, %3) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<2048xbf16>, tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<2048xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %28 = call @__call___74(%arg199, %27) : (tensor<2048xbf16>, tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32>
    %29 = stablehlo.convert %arg198 : (tensor<2048x32000xbf16>) -> tensor<2048x32000xf32>
    %30 = stablehlo.dot_general %28, %29, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x32000xf32>) -> tensor<1x16x32000xf32>
    return %30 : tensor<1x16x32000xf32>
  }
  func.func private @_where(%arg0: tensor<16x16xi1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %2 = stablehlo.select %arg0, %0, %1 : tensor<16x16xi1>, tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }
  func.func private @_take(%arg0: tensor<32000x2048xf16>, %arg1: tensor<1x16xi32>) -> tensor<1x16x2048xf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0, SIGNED : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi1>
    %c_0 = stablehlo.constant dense<32000> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x16xi32>
    %4 = call @_where_2(%1, %3, %arg1) : (tensor<1x16xi1>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<1x16x1xi32>
    %c_1 = stablehlo.constant dense<31999> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x16x1xi32>
    %7 = stablehlo.compare GE, %5, %6, SIGNED : (tensor<1x16x1xi32>, tensor<1x16x1xi32>) -> tensor<1x16x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x16x1xi32>
    %10 = stablehlo.compare LE, %5, %9, SIGNED : (tensor<1x16x1xi32>, tensor<1x16x1xi32>) -> tensor<1x16x1xi1>
    %11 = stablehlo.and %7, %10 : tensor<1x16x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.reduce(%11 init: %c_3) applies stablehlo.and across dimensions = [2] : (tensor<1x16x1xi1>, tensor<i1>) -> tensor<1x16xi1>
    %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2048>}> : (tensor<32000x2048xf16>, tensor<1x16x1xi32>) -> tensor<1x16x2048xf16>
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x16xi1>) -> tensor<1x16x2048xi1>
    %cst = stablehlo.constant dense<0x7E00> : tensor<f16>
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<1x16x2048xf16>
    %16 = stablehlo.select %14, %13, %15 : tensor<1x16x2048xi1>, tensor<1x16x2048xf16>
    return %16 : tensor<1x16x2048xf16>
  }
  func.func private @_where_2(%arg0: tensor<1x16xi1>, %arg1: tensor<1x16xi32>, %arg2: tensor<1x16xi32>) -> tensor<1x16xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x16xi1>, tensor<1x16xi32>
    return %0 : tensor<1x16xi32>
  }
  func.func private @__call___10(%arg0: tensor<2048x256xbf16>, %arg1: tensor<2048x2048xbf16>, %arg2: tensor<2048x2048xbf16>, %arg3: tensor<2048x256xbf16>, %arg4: tensor<2048xbf16>, %arg5: tensor<5632x2048xbf16>, %arg6: tensor<2048x5632xbf16>, %arg7: tensor<2048x5632xbf16>, %arg8: tensor<2048xbf16>, %arg9: tensor<1x16x2048xf16>, %arg10: tensor<1x16xi32>, %arg11: tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32> {
    %0 = call @__call___11(%arg4, %arg9) : (tensor<2048xbf16>, tensor<1x16x2048xf16>) -> tensor<1x16x2048xf32>
    %1 = call @__call___19(%arg0, %arg1, %arg2, %arg3, %0, %arg10, %arg11) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %2 = stablehlo.convert %arg9 : (tensor<1x16x2048xf16>) -> tensor<1x16x2048xf32>
    %3 = stablehlo.add %2, %1 : tensor<1x16x2048xf32>
    %4 = call @__call___74(%arg8, %3) : (tensor<2048xbf16>, tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32>
    %5 = call @__call___75(%arg5, %arg6, %arg7, %4) : (tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x16x2048xf32>
    return %6 : tensor<1x16x2048xf32>
  }
  func.func private @__call___11(%arg0: tensor<2048xbf16>, %arg1: tensor<1x16x2048xf16>) -> tensor<1x16x2048xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<1x16x2048xf16>) -> tensor<1x16x2048xf32>
    %1 = chlo.square %0 : tensor<1x16x2048xf32> -> tensor<1x16x2048xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x2048xf32>, tensor<f32>) -> tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_0 = stablehlo.constant dense<2.048000e+03> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<1x16x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x16x1xf32>
    %8 = stablehlo.sqrt %7 : tensor<1x16x1xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %10 = stablehlo.divide %9, %8 : tensor<1x16x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x2048xf32>
    %12 = stablehlo.multiply %0, %11 : tensor<1x16x2048xf32>
    %13 = stablehlo.convert %12 : (tensor<1x16x2048xf32>) -> tensor<1x16x2048xf16>
    %14 = stablehlo.convert %arg0 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %15 = stablehlo.convert %13 : (tensor<1x16x2048xf16>) -> tensor<1x16x2048xf32>
    %16 = stablehlo.broadcast_in_dim %14, dims = [2] : (tensor<2048xf32>) -> tensor<1x1x2048xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x1x2048xf32>) -> tensor<1x16x2048xf32>
    %18 = stablehlo.multiply %17, %15 : tensor<1x16x2048xf32>
    return %18 : tensor<1x16x2048xf32>
  }
  func.func private @__call___19(%arg0: tensor<2048x256xbf16>, %arg1: tensor<2048x2048xbf16>, %arg2: tensor<2048x2048xbf16>, %arg3: tensor<2048x256xbf16>, %arg4: tensor<1x16x2048xf32>, %arg5: tensor<1x16xi32>, %arg6: tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<2048x2048xbf16>) -> tensor<2048x2048xf32>
    %1 = stablehlo.dot_general %arg4, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x2048xf32>) -> tensor<1x16x2048xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x16x2048xf32>) -> tensor<1x16x32x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1x16x32x64xf32>) -> tensor<1x32x16x64xf32>
    %4 = stablehlo.convert %arg0 : (tensor<2048x256xbf16>) -> tensor<2048x256xf32>
    %5 = stablehlo.dot_general %arg4, %4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x256xf32>) -> tensor<1x16x256xf32>
    %6 = stablehlo.reshape %5 : (tensor<1x16x256xf32>) -> tensor<1x16x4x64xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1, 3] : (tensor<1x16x4x64xf32>) -> tensor<1x4x16x64xf32>
    %8 = stablehlo.convert %arg3 : (tensor<2048x256xbf16>) -> tensor<2048x256xf32>
    %9 = stablehlo.dot_general %arg4, %8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x256xf32>) -> tensor<1x16x256xf32>
    %10 = stablehlo.reshape %9 : (tensor<1x16x256xf32>) -> tensor<1x16x4x64xf32>
    %11 = stablehlo.transpose %10, dims = [0, 2, 1, 3] : (tensor<1x16x4x64xf32>) -> tensor<1x4x16x64xf32>
    %12 = stablehlo.reshape %arg5 : (tensor<1x16xi32>) -> tensor<16xi32>
    %13:2 = call @__call___25(%12) : (tensor<16xi32>) -> (tensor<1x16x64xbf16>, tensor<1x16x64xbf16>)
    %14 = stablehlo.broadcast_in_dim %13#0, dims = [0, 2, 3] : (tensor<1x16x64xbf16>) -> tensor<1x1x16x64xbf16>
    %15 = stablehlo.broadcast_in_dim %13#1, dims = [0, 2, 3] : (tensor<1x16x64xbf16>) -> tensor<1x1x16x64xbf16>
    %16 = stablehlo.convert %14 : (tensor<1x1x16x64xbf16>) -> tensor<1x1x16x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x16x64xf32>) -> tensor<1x32x16x64xf32>
    %18 = stablehlo.multiply %3, %17 : tensor<1x32x16x64xf32>
    %19 = stablehlo.slice %3 [0:1, 0:32, 0:16, 0:32] : (tensor<1x32x16x64xf32>) -> tensor<1x32x16x32xf32>
    %20 = stablehlo.slice %3 [0:1, 0:32, 0:16, 32:64] : (tensor<1x32x16x64xf32>) -> tensor<1x32x16x32xf32>
    %21 = stablehlo.negate %20 : tensor<1x32x16x32xf32>
    %22 = stablehlo.concatenate %21, %19, dim = 3 : (tensor<1x32x16x32xf32>, tensor<1x32x16x32xf32>) -> tensor<1x32x16x64xf32>
    %23 = stablehlo.convert %15 : (tensor<1x1x16x64xbf16>) -> tensor<1x1x16x64xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x1x16x64xf32>) -> tensor<1x32x16x64xf32>
    %25 = stablehlo.multiply %22, %24 : tensor<1x32x16x64xf32>
    %26 = stablehlo.add %18, %25 : tensor<1x32x16x64xf32>
    %27 = stablehlo.convert %14 : (tensor<1x1x16x64xbf16>) -> tensor<1x1x16x64xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3] : (tensor<1x1x16x64xf32>) -> tensor<1x4x16x64xf32>
    %29 = stablehlo.multiply %7, %28 : tensor<1x4x16x64xf32>
    %30 = stablehlo.slice %7 [0:1, 0:4, 0:16, 0:32] : (tensor<1x4x16x64xf32>) -> tensor<1x4x16x32xf32>
    %31 = stablehlo.slice %7 [0:1, 0:4, 0:16, 32:64] : (tensor<1x4x16x64xf32>) -> tensor<1x4x16x32xf32>
    %32 = stablehlo.negate %31 : tensor<1x4x16x32xf32>
    %33 = stablehlo.concatenate %32, %30, dim = 3 : (tensor<1x4x16x32xf32>, tensor<1x4x16x32xf32>) -> tensor<1x4x16x64xf32>
    %34 = stablehlo.convert %15 : (tensor<1x1x16x64xbf16>) -> tensor<1x1x16x64xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x1x16x64xf32>) -> tensor<1x4x16x64xf32>
    %36 = stablehlo.multiply %33, %35 : tensor<1x4x16x64xf32>
    %37 = stablehlo.add %29, %36 : tensor<1x4x16x64xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 3, 4] : (tensor<1x4x16x64xf32>) -> tensor<1x4x1x16x64xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 4, 5] : (tensor<1x4x1x16x64xf32>) -> tensor<1x4x1x8x16x64xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x4x1x8x16x64xf32>) -> tensor<1x4x8x16x64xf32>
    %41 = stablehlo.reshape %40 : (tensor<1x4x8x16x64xf32>) -> tensor<1x32x16x64xf32>
    %42 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 3, 4] : (tensor<1x4x16x64xf32>) -> tensor<1x4x1x16x64xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 4, 5] : (tensor<1x4x1x16x64xf32>) -> tensor<1x4x1x8x16x64xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x4x1x8x16x64xf32>) -> tensor<1x4x8x16x64xf32>
    %45 = stablehlo.reshape %44 : (tensor<1x4x8x16x64xf32>) -> tensor<1x32x16x64xf32>
    %46 = stablehlo.transpose %41, dims = [0, 1, 3, 2] : (tensor<1x32x16x64xf32>) -> tensor<1x32x64x16xf32>
    %47 = stablehlo.reshape %26 : (tensor<1x32x16x64xf32>) -> tensor<32x16x64xf32>
    %48 = stablehlo.dot_general %47, %46, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x16x64xf32>, tensor<1x32x64x16xf32>) -> tensor<32x16x1x16xf32>
    %49 = stablehlo.transpose %48, dims = [2, 0, 1, 3] : (tensor<32x16x1x16xf32>) -> tensor<1x32x16x16xf32>
    %cst = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %50 = stablehlo.sqrt %cst : tensor<f32>
    %51 = stablehlo.convert %50 : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<f32>) -> tensor<1x32x16x16xf32>
    %53 = stablehlo.divide %49, %52 : tensor<1x32x16x16xf32>
    %54 = stablehlo.convert %53 : (tensor<1x32x16x16xf32>) -> tensor<1x32x16x16xbf16>
    %55 = stablehlo.convert %arg6 : (tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xbf16>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3] : (tensor<1x1x16x16xbf16>) -> tensor<1x32x16x16xbf16>
    %57 = stablehlo.add %54, %56 : tensor<1x32x16x16xbf16>
    %58 = stablehlo.convert %57 : (tensor<1x32x16x16xbf16>) -> tensor<1x32x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %59 = stablehlo.reduce(%58 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x16x16xf32>, tensor<f32>) -> tensor<1x32x16xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x16xf32>
    %61 = stablehlo.maximum %60, %59 : tensor<1x32x16xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2] : (tensor<1x32x16xf32>) -> tensor<1x32x16x1xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<1x32x16x1xf32>) -> tensor<1x32x16x16xf32>
    %64 = stablehlo.subtract %58, %63 : tensor<1x32x16x16xf32>
    %65 = stablehlo.exponential %64 : tensor<1x32x16x16xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %66 = stablehlo.reduce(%65 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x32x16x16xf32>, tensor<f32>) -> tensor<1x32x16xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<1x32x16xf32>) -> tensor<1x32x16x1xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<1x32x16x1xf32>) -> tensor<1x32x16x16xf32>
    %69 = stablehlo.divide %65, %68 : tensor<1x32x16x16xf32>
    %70 = stablehlo.convert %69 : (tensor<1x32x16x16xf32>) -> tensor<1x32x16x16xbf16>
    %71 = stablehlo.reshape %70 : (tensor<1x32x16x16xbf16>) -> tensor<32x16x16xbf16>
    %72 = stablehlo.convert %71 : (tensor<32x16x16xbf16>) -> tensor<32x16x16xf32>
    %73 = stablehlo.convert %45 : tensor<1x32x16x64xf32>
    %74 = stablehlo.dot_general %72, %73, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x16x16xf32>, tensor<1x32x16x64xf32>) -> tensor<32x16x1x64xf32>
    %75 = stablehlo.transpose %74, dims = [2, 0, 1, 3] : (tensor<32x16x1x64xf32>) -> tensor<1x32x16x64xf32>
    %76 = stablehlo.transpose %75, dims = [0, 2, 1, 3] : (tensor<1x32x16x64xf32>) -> tensor<1x16x32x64xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x16x32x64xf32>) -> tensor<1x16x2048xf32>
    %78 = stablehlo.convert %arg1 : (tensor<2048x2048xbf16>) -> tensor<2048x2048xf32>
    %79 = stablehlo.dot_general %77, %78, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x2048xf32>) -> tensor<1x16x2048xf32>
    return %79 : tensor<1x16x2048xf32>
  }
  func.func private @__call___25(%arg0: tensor<16xi32>) -> (tensor<1x16x64xbf16>, tensor<1x16x64xbf16>) {
    %0 = stablehlo.iota dim = 0 : tensor<32xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2 = stablehlo.multiply %1, %0 : tensor<32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %4 = stablehlo.add %3, %2 : tensor<32xf32>
    %cst_1 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %6 = stablehlo.divide %4, %5 : tensor<32xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %8 = stablehlo.power %7, %6 : tensor<32xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %10 = stablehlo.divide %9, %8 : tensor<32xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<16xi32>) -> tensor<1x16x1xi32>
    %13 = stablehlo.convert %12 : (tensor<1x16x1xi32>) -> tensor<1x16x1xf32>
    %14 = stablehlo.dot_general %11, %13, batching_dims = [0, 1] x [0, 2], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xf32>, tensor<1x16x1xf32>) -> tensor<1x1x32x16xf32>
    %15 = stablehlo.transpose %14, dims = [0, 3, 1, 2] : (tensor<1x1x32x16xf32>) -> tensor<1x16x1x32xf32>
    %16 = stablehlo.concatenate %15, %15, dim = 3 : (tensor<1x16x1x32xf32>, tensor<1x16x1x32xf32>) -> tensor<1x16x1x64xf32>
    %17 = stablehlo.cosine %16 : tensor<1x16x1x64xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x16x1x64xf32>) -> tensor<1x16x64xf32>
    %19 = stablehlo.convert %18 : (tensor<1x16x64xf32>) -> tensor<1x16x64xbf16>
    %20 = stablehlo.sine %16 : tensor<1x16x1x64xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x16x1x64xf32>) -> tensor<1x16x64xf32>
    %22 = stablehlo.convert %21 : (tensor<1x16x64xf32>) -> tensor<1x16x64xbf16>
    return %19, %22 : tensor<1x16x64xbf16>, tensor<1x16x64xbf16>
  }
  func.func private @__call___74(%arg0: tensor<2048xbf16>, %arg1: tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32> {
    %0 = chlo.square %arg1 : tensor<1x16x2048xf32> -> tensor<1x16x2048xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x2048xf32>, tensor<f32>) -> tensor<1x16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_0 = stablehlo.constant dense<2.048000e+03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %4 = stablehlo.divide %2, %3 : tensor<1x16x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %6 = stablehlo.add %4, %5 : tensor<1x16x1xf32>
    %7 = stablehlo.sqrt %6 : tensor<1x16x1xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %9 = stablehlo.divide %8, %7 : tensor<1x16x1xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x2048xf32>
    %11 = stablehlo.multiply %arg1, %10 : tensor<1x16x2048xf32>
    %12 = stablehlo.convert %arg0 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<2048xf32>) -> tensor<1x1x2048xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1x1x2048xf32>) -> tensor<1x16x2048xf32>
    %15 = stablehlo.multiply %14, %11 : tensor<1x16x2048xf32>
    return %15 : tensor<1x16x2048xf32>
  }
  func.func private @__call___75(%arg0: tensor<5632x2048xbf16>, %arg1: tensor<2048x5632xbf16>, %arg2: tensor<2048x5632xbf16>, %arg3: tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<2048x5632xbf16>) -> tensor<2048x5632xf32>
    %1 = stablehlo.dot_general %arg3, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x5632xf32>) -> tensor<1x16x5632xf32>
    %2 = call @silu(%1) : (tensor<1x16x5632xf32>) -> tensor<1x16x5632xf32>
    %3 = stablehlo.convert %arg2 : (tensor<2048x5632xbf16>) -> tensor<2048x5632xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x2048xf32>, tensor<2048x5632xf32>) -> tensor<1x16x5632xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1x16x5632xf32>
    %6 = stablehlo.convert %arg0 : (tensor<5632x2048xbf16>) -> tensor<5632x2048xf32>
    %7 = stablehlo.dot_general %5, %6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x5632xf32>, tensor<5632x2048xf32>) -> tensor<1x16x2048xf32>
    return %7 : tensor<1x16x2048xf32>
  }
  func.func private @silu(%arg0: tensor<1x16x5632xf32>) -> tensor<1x16x5632xf32> {
    %0 = stablehlo.negate %arg0 : tensor<1x16x5632xf32>
    %1 = stablehlo.exponential %0 : tensor<1x16x5632xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x5632xf32>
    %3 = stablehlo.add %2, %1 : tensor<1x16x5632xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x5632xf32>
    %5 = stablehlo.divide %4, %3 : tensor<1x16x5632xf32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<1x16x5632xf32>
    return %6 : tensor<1x16x5632xf32>
  }
  func.func private @__call___85(%arg0: tensor<2048x256xbf16>, %arg1: tensor<2048x2048xbf16>, %arg2: tensor<2048x2048xbf16>, %arg3: tensor<2048x256xbf16>, %arg4: tensor<2048xbf16>, %arg5: tensor<5632x2048xbf16>, %arg6: tensor<2048x5632xbf16>, %arg7: tensor<2048x5632xbf16>, %arg8: tensor<2048xbf16>, %arg9: tensor<1x16x2048xf32>, %arg10: tensor<1x16xi32>, %arg11: tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32> {
    %0 = call @__call___74(%arg4, %arg9) : (tensor<2048xbf16>, tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32>
    %1 = call @__call___19(%arg0, %arg1, %arg2, %arg3, %0, %arg10, %arg11) : (tensor<2048x256xbf16>, tensor<2048x2048xbf16>, tensor<2048x2048xbf16>, tensor<2048x256xbf16>, tensor<1x16x2048xf32>, tensor<1x16xi32>, tensor<1x1x16x16xf32>) -> tensor<1x16x2048xf32>
    %2 = stablehlo.add %arg9, %1 : tensor<1x16x2048xf32>
    %3 = call @__call___74(%arg8, %2) : (tensor<2048xbf16>, tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32>
    %4 = call @__call___75(%arg5, %arg6, %arg7, %3) : (tensor<5632x2048xbf16>, tensor<2048x5632xbf16>, tensor<2048x5632xbf16>, tensor<1x16x2048xf32>) -> tensor<1x16x2048xf32>
    %5 = stablehlo.add %2, %4 : tensor<1x16x2048xf32>
    return %5 : tensor<1x16x2048xf32>
  }
}
