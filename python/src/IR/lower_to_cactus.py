import numpy as np
from .stablehlo_ir import extract_constant
import re

def _extract_convert_target_dtype(raw):
    m = re.search(r"->\s*tensor<[^>]*x([a-z0-9]+)>", raw)
    if not m:
        return None
    return m.group(1).lower()


def _extract_tensor_element_dtype(raw):
    matches = re.findall(r"tensor<[^>]*x([a-z0-9]+)>", raw)
    if not matches:
        return None
    return matches[-1].lower()


def repeat_axis(g, x, axis, times):
    if times == 1:
        return x
    return g.cat([x for _ in range(times)], axis=axis)

    
def align_shapes(g, a, b, a_shape, b_shape):
    max_rank = max(len(a_shape), len(b_shape))

    a_shape = (1,) * (max_rank - len(a_shape)) + tuple(a_shape)
    b_shape = (1,) * (max_rank - len(b_shape)) + tuple(b_shape)

    if len(a.shape) != max_rank:
        a = g.reshape(a, a_shape)
    if len(b.shape) != max_rank:
        b = g.reshape(b, b_shape)

    target_shape = tuple(max(x, y) for x, y in zip(a_shape, b_shape))

    def expand(x, x_shape):
        for axis, (s, t) in enumerate(zip(x_shape, target_shape)):
            if s == t:
                continue
            if s == 1 and t > 1:
                x = repeat_axis(g, x, axis, t)
            else:
                raise Exception(f"Cannot broadcast {x_shape} to {target_shape}")
        return x

    a = expand(a, a_shape)
    b = expand(b, b_shape)

    return a, b, target_shape


def ensure_fp16(g, x):
    if getattr(x, "dtype", None) == g.FP16:
        return x
    return g.precision_cast(x, g.FP16)


def ensure_binary_fp16(g, a, b):
    if getattr(a, "dtype", None) != g.FP16:
        a = g.precision_cast(a, g.FP16)
    if getattr(b, "dtype", None) != g.FP16:
        b = g.precision_cast(b, g.FP16)
    return a, b


def ensure_binary_fp32(g, a, b):
    if getattr(a, "dtype", None) != g.FP32:
        a = g.precision_cast(a, g.FP32)
    if getattr(b, "dtype", None) != g.FP32:
        b = g.precision_cast(b, g.FP32)
    return a, b


def clamp_tensor_fp16(g, x, limit=60000.0):
    if getattr(x, "dtype", None) != g.FP16:
        return x
    hi = g.input((1,), g.FP16)
    lo = g.input((1,), g.FP16)
    g.set_input(hi, np.array([limit], dtype=np.float16))
    g.set_input(lo, np.array([-limit], dtype=np.float16))
    # min(x, hi)
    min_x = g.subtract(hi, g.relu(g.subtract(hi, x)))
    # max(min_x, lo)
    return g.add(lo, g.relu(g.subtract(min_x, lo)))


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n



def find_attention_blocks(nodes, verbose=False):
    candidates = []

    for t_idx, n in enumerate(nodes):
        if n.op != "transpose":
            continue

        # Skip transposes that are directly on graph input/weight args; these are
        # typically projection-weight transposes, not K^T in attention score.
        if n.inputs and n.inputs[0].startswith("%arg"):
            continue

        if verbose:
            print("\n--- transpose candidate ---")
            print("idx:", t_idx, "name:", n.name, "inputs:", n.inputs, "shape:", n.shape)

        # Print local window for debugging
        lo = max(0, t_idx - 6)
        hi = min(len(nodes), t_idx + 25)

        if verbose:
            for j in range(lo, hi):
                print(
                    f"{j:04d}",
                    nodes[j].name,
                    nodes[j].op,
                    "inputs=", nodes[j].inputs,
                    "shape=", nodes[j].shape,
                )

        # Find score matmul shortly after transpose:
        # score = dot_general(q, transpose(k))
        score_idx = None
        for j in range(t_idx + 1, min(len(nodes), t_idx + 8)):
            if nodes[j].op == "dot_general" and n.name in nodes[j].inputs:
                score_idx = j
                break

        if score_idx is None:
            if verbose:
                print("no score matmul after this transpose")
            continue

        # Find tril after score (present in prefill/full graphs; often absent for T=1 decode graphs)
        tril_idx = None
        for j in range(score_idx + 1, min(len(nodes), score_idx + 40)):
            if nodes[j].op == "tril":
                tril_idx = j
                break

        if tril_idx is None:
            if verbose:
                print("no tril after score matmul")
            # Be strict: require explicit causal tril to avoid false attention matches.
            continue
        softmax_start = tril_idx + 1

        # Find softmax divide after score/tril:
        # reduce_maximum -> exp -> reduce_add -> divide
        exp_idx = None
        red_add_idx = None
        softmax_div_idx = None

        for j in range(softmax_start, min(len(nodes), softmax_start + 120)):
            if nodes[j].op == "exponential":
                exp_idx = j
            elif exp_idx is not None and nodes[j].op == "reduce_add":
                red_add_idx = j
            elif red_add_idx is not None and nodes[j].op == "divide":
                softmax_div_idx = j
                break

        if softmax_div_idx is None:
            if verbose:
                print("no softmax divide after score/tril")
            continue

        # Find final attention matmul:
        # attn_out = dot_general(softmax_probs, v)
        out_idx = None
        for j in range(softmax_div_idx + 1, min(len(nodes), softmax_div_idx + 20)):
            if nodes[j].op == "dot_general" and nodes[softmax_div_idx].name in nodes[j].inputs:
                out_idx = j
                break

        if out_idx is None:
            if verbose:
                print("no output matmul after softmax divide")
            continue

        if verbose:
            print(f"🔥 Found attention block: transpose={t_idx}, score={score_idx}, tril={tril_idx}, softmax={softmax_div_idx}, out={out_idx}")
        candidates.append({
            "transpose": t_idx,
            "score": score_idx,
            "tril": tril_idx,
            "softmax": softmax_div_idx,
            "out": out_idx,
        })

    # Deduplicate: a single attention out node can be reachable from multiple
    # transpose candidates in the same local window. Keep the closest transpose
    # (highest index) to the score/out path.
    by_out = {}
    for c in candidates:
        prev = by_out.get(c["out"])
        if prev is None or c["transpose"] > prev["transpose"]:
            by_out[c["out"]] = c

    blocks = sorted(by_out.values(), key=lambda b: b["transpose"])
    return blocks


def _maybe_scalar_from_node(nodes_by_name, node_name):
    n = nodes_by_name.get(node_name)
    if n is None:
        return None
    if n.op == "constant":
        return extract_constant(n.raw)
    if n.op == "broadcast_in_dim" and n.inputs:
        src = nodes_by_name.get(n.inputs[0])
        if src is not None and src.op == "constant":
            return extract_constant(src.raw)
    return None


def _approx(a, b, tol=1e-4):
    return a is not None and abs(float(a) - float(b)) <= tol


def _subtree_has_input(nodes_by_name, root_name, target_name, depth=8):
    if depth < 0:
        return False
    if root_name == target_name:
        return True
    n = nodes_by_name.get(root_name)
    if n is None:
        return False
    for inp in n.inputs:
        if _subtree_has_input(nodes_by_name, inp, target_name, depth - 1):
            return True
    return False


def _subtree_has_scalar(nodes_by_name, root_name, target, depth=8):
    if depth < 0:
        return False
    v = _maybe_scalar_from_node(nodes_by_name, root_name)
    if _approx(v, target):
        return True
    n = nodes_by_name.get(root_name)
    if n is None:
        return False
    for inp in n.inputs:
        if _subtree_has_scalar(nodes_by_name, inp, target, depth - 1):
            return True
    return False


def find_layernorm_blocks(nodes, verbose=False):
    nodes_by_name = {n.name: n for n in nodes}
    idx_by_name = {n.name: i for i, n in enumerate(nodes)}
    blocks = []

    for out in nodes:
        if out.op != "add" or len(out.inputs) != 2:
            continue

        cand = None
        for mul_name, beta_b_name in ((out.inputs[0], out.inputs[1]), (out.inputs[1], out.inputs[0])):
            mul_n = nodes_by_name.get(mul_name)
            beta_b = nodes_by_name.get(beta_b_name)
            if mul_n is not None and mul_n.op == "multiply" and beta_b is not None and beta_b.op == "broadcast_in_dim":
                cand = (mul_n, beta_b)
                break
        if cand is None:
            continue

        mul_n, beta_b = cand
        if len(mul_n.inputs) != 2:
            continue

        div2 = nodes_by_name.get(mul_n.inputs[0])
        gamma_b = nodes_by_name.get(mul_n.inputs[1])
        if div2 is None or div2.op != "divide" or gamma_b is None or gamma_b.op != "broadcast_in_dim":
            continue

        sub2 = nodes_by_name.get(div2.inputs[0]) if len(div2.inputs) >= 2 else None
        invstd_b = nodes_by_name.get(div2.inputs[1]) if len(div2.inputs) >= 2 else None
        if sub2 is None or sub2.op != "subtract" or invstd_b is None or invstd_b.op != "broadcast_in_dim":
            continue

        x_name = sub2.inputs[0]
        sqrt_n = nodes_by_name.get(invstd_b.inputs[0]) if invstd_b.inputs else None
        if sqrt_n is None or sqrt_n.op != "sqrt" or not sqrt_n.inputs:
            continue

        add_eps = nodes_by_name.get(sqrt_n.inputs[0])
        if add_eps is None or add_eps.op != "add":
            continue

        # one branch should be variance divide
        var_div = None
        eps_node_name = None
        for a, b in ((add_eps.inputs[0], add_eps.inputs[1]), (add_eps.inputs[1], add_eps.inputs[0])):
            a_n = nodes_by_name.get(a)
            if a_n is not None and a_n.op == "divide":
                var_div = a_n
                eps_node_name = b
                break
        if var_div is None:
            continue

        var_num_b = nodes_by_name.get(var_div.inputs[0]) if len(var_div.inputs) >= 1 else None
        if var_num_b is None or var_num_b.op != "broadcast_in_dim" or not var_num_b.inputs:
            continue
        red2 = nodes_by_name.get(var_num_b.inputs[0])
        if red2 is None or red2.op != "reduce_add" or not red2.inputs:
            continue

        sq = nodes_by_name.get(red2.inputs[0])
        if sq is None or sq.op != "multiply" or len(sq.inputs) != 2 or sq.inputs[0] != sq.inputs[1]:
            continue

        sub1 = nodes_by_name.get(sq.inputs[0])
        if sub1 is None or sub1.op != "subtract" or len(sub1.inputs) != 2:
            continue
        if sub1.inputs[0] != x_name:
            continue

        mean1_b = nodes_by_name.get(sub1.inputs[1])
        if mean1_b is None or mean1_b.op != "broadcast_in_dim" or not mean1_b.inputs:
            continue
        mean1_div = nodes_by_name.get(mean1_b.inputs[0])
        if mean1_div is None or mean1_div.op != "divide" or not mean1_div.inputs:
            continue
        mean1_num_b = nodes_by_name.get(mean1_div.inputs[0])
        if mean1_num_b is None or mean1_num_b.op != "broadcast_in_dim" or not mean1_num_b.inputs:
            continue
        red1 = nodes_by_name.get(mean1_num_b.inputs[0])
        if red1 is None or red1.op != "reduce_add" or not red1.inputs or red1.inputs[0] != x_name:
            continue

        eps = _maybe_scalar_from_node(nodes_by_name, eps_node_name)
        if eps is None:
            eps = 1e-5

        gamma_name = gamma_b.inputs[0] if gamma_b.inputs else None
        beta_name = beta_b.inputs[0] if beta_b.inputs else None
        if gamma_name is None or beta_name is None:
            continue

        start = idx_by_name.get(red1.name, idx_by_name[out.name])
        end = idx_by_name[out.name]
        blocks.append({
            "start": start,
            "end": end,
            "x": x_name,
            "gamma": gamma_name,
            "beta": beta_name,
            "out": out.name,
            "eps": float(eps),
        })

    # Keep earliest non-overlapping blocks
    blocks.sort(key=lambda b: (b["start"], b["end"]))
    dedup = []
    last_end = -1
    for b in blocks:
        if b["start"] <= last_end:
            continue
        dedup.append(b)
        last_end = b["end"]
    if verbose:
        print("LayerNorm blocks:", len(dedup))
    return dedup


def find_rmsnorm_blocks(nodes, verbose=False):
    nodes_by_name = {n.name: n for n in nodes}
    idx_by_name = {n.name: i for i, n in enumerate(nodes)}
    blocks = []

    def _strip_convert_and_broadcast(name):
        cur = name
        while True:
            n = nodes_by_name.get(cur)
            if n is None or not n.inputs:
                return cur
            if n.op in ("convert", "broadcast_in_dim"):
                cur = n.inputs[0]
                continue
            return cur

    def _strip_convert(name):
        cur = name
        while True:
            n = nodes_by_name.get(cur)
            if n is None or n.op != "convert" or not n.inputs:
                return cur
            cur = n.inputs[0]

    def _parse_var_chain(var_name, x_name):
        # Expect var path equivalent to mean(x*x) with optional convert wrappers.
        cur = _strip_convert(var_name)
        v_n = nodes_by_name.get(cur)
        if v_n is None:
            return None

        start_candidates = [idx_by_name.get(cur, 10**9)]

        # Usually divide(sum(x*x), hidden_dim)
        if v_n.op == "divide" and len(v_n.inputs) == 2:
            num_name = _strip_convert(v_n.inputs[0])
        # Occasionally multiply(sum(x*x), reciprocal_hidden_dim)
        elif v_n.op == "multiply" and len(v_n.inputs) == 2:
            a, b = v_n.inputs
            if _maybe_scalar_from_node(nodes_by_name, a) is not None:
                num_name = _strip_convert(b)
            elif _maybe_scalar_from_node(nodes_by_name, b) is not None:
                num_name = _strip_convert(a)
            else:
                return None
        else:
            return None

        num_n = nodes_by_name.get(num_name)
        if num_n is None:
            return None
        start_candidates.append(idx_by_name.get(num_name, 10**9))

        if num_n.op == "broadcast_in_dim" and num_n.inputs:
            red_name = _strip_convert(num_n.inputs[0])
        elif num_n.op == "reduce_add":
            red_name = num_name
        else:
            return None

        red_n = nodes_by_name.get(red_name)
        if red_n is None or red_n.op != "reduce_add" or not red_n.inputs:
            return None
        start_candidates.append(idx_by_name.get(red_name, 10**9))

        sq_name = _strip_convert(red_n.inputs[0])
        sq_n = nodes_by_name.get(sq_name)
        if sq_n is None or sq_n.op != "multiply" or len(sq_n.inputs) != 2:
            return None
        if not (
            (sq_n.inputs[0] == x_name and sq_n.inputs[1] == x_name)
            or (sq_n.inputs[1] == x_name and sq_n.inputs[0] == x_name)
        ):
            return None
        start_candidates.append(idx_by_name.get(sq_name, 10**9))

        return min(start_candidates)

    by_out = {}
    for out in nodes:
        if out.op != "multiply" or len(out.inputs) != 2:
            continue

        for norm_name, weight_name in ((out.inputs[0], out.inputs[1]), (out.inputs[1], out.inputs[0])):
            norm_n = nodes_by_name.get(norm_name)
            if norm_n is None or norm_n.op != "multiply" or len(norm_n.inputs) != 2:
                continue

            x_name = None
            rsqrt_name = None
            for a, b in ((norm_n.inputs[0], norm_n.inputs[1]), (norm_n.inputs[1], norm_n.inputs[0])):
                b_n = nodes_by_name.get(b)
                if b_n is None or b_n.op != "broadcast_in_dim" or not b_n.inputs:
                    continue
                cand = _strip_convert(b_n.inputs[0])
                cand_n = nodes_by_name.get(cand)
                if cand_n is not None and cand_n.op == "rsqrt" and cand_n.inputs:
                    x_name = a
                    rsqrt_name = cand
                    break

            if x_name is None or rsqrt_name is None:
                continue

            rsqrt_n = nodes_by_name.get(rsqrt_name)
            add_name = _strip_convert(rsqrt_n.inputs[0]) if rsqrt_n and rsqrt_n.inputs else None
            add_n = nodes_by_name.get(add_name) if add_name else None
            if add_n is None or add_n.op != "add" or len(add_n.inputs) != 2:
                continue

            eps = None
            var_name = None
            for inp in add_n.inputs:
                s = _maybe_scalar_from_node(nodes_by_name, inp)
                if s is not None and eps is None:
                    eps = float(s)
                else:
                    var_name = inp
            if var_name is None:
                continue
            if eps is None:
                eps = 1e-6

            var_start = _parse_var_chain(var_name, x_name)
            if var_start is None:
                continue

            weight_src = _strip_convert_and_broadcast(weight_name)

            start = min(
                var_start,
                idx_by_name.get(norm_name, 10**9),
                idx_by_name.get(add_name, 10**9),
                idx_by_name.get(rsqrt_name, 10**9),
            )

            out_idx = idx_by_name.get(out.name)
            if out_idx is None:
                continue

            b = {
                "start": int(start),
                "end": int(out_idx),
                "x": x_name,
                "weight": weight_src,
                "out": out.name,
                "eps": float(eps),
            }
            prev = by_out.get(out.name)
            if prev is None or b["start"] < prev["start"]:
                by_out[out.name] = b
            break

    dedup = sorted(by_out.values(), key=lambda b: (b["start"], b["end"]))
    if verbose:
        print("RMSNorm blocks:", len(dedup))
    return dedup


def find_gelu_blocks(nodes, verbose=False):
    nodes_by_name = {n.name: n for n in nodes}
    idx_by_name = {n.name: i for i, n in enumerate(nodes)}
    blocks = []

    for out in nodes:
        if out.op != "multiply" or len(out.inputs) != 2:
            continue

        match = None
        for half_name, one_plus_name in ((out.inputs[0], out.inputs[1]), (out.inputs[1], out.inputs[0])):
            half_n = nodes_by_name.get(half_name)
            one_plus_n = nodes_by_name.get(one_plus_name)
            if half_n is None or half_n.op != "multiply" or one_plus_n is None or one_plus_n.op != "add":
                continue

            # one_plus_n should contain tanh branch + constant 1
            tanh_name = None
            for a, b in ((one_plus_n.inputs[0], one_plus_n.inputs[1]), (one_plus_n.inputs[1], one_plus_n.inputs[0])):
                if nodes_by_name.get(a) is not None and nodes_by_name[a].op == "tanh" and _approx(_maybe_scalar_from_node(nodes_by_name, b), 1.0):
                    tanh_name = a
                    break
            if tanh_name is None:
                continue

            # half_n should contain x * 0.5
            x_name = None
            for a, b in ((half_n.inputs[0], half_n.inputs[1]), (half_n.inputs[1], half_n.inputs[0])):
                if _approx(_maybe_scalar_from_node(nodes_by_name, b), 0.5):
                    x_name = a
                    break
            if x_name is None:
                continue

            tanh_in = nodes_by_name[tanh_name].inputs[0] if nodes_by_name[tanh_name].inputs else None
            if tanh_in is None:
                continue

            # Ensure this is really GELU tanh form and not a random tanh.
            if not _subtree_has_input(nodes_by_name, tanh_in, x_name, depth=10):
                continue

            match = {
                "x": x_name,
                "out": out.name,
                "start": min(idx_by_name.get(half_name, idx_by_name[out.name]), idx_by_name.get(one_plus_name, idx_by_name[out.name])),
                "end": idx_by_name[out.name],
            }
            break

        if match is not None:
            blocks.append(match)

    blocks.sort(key=lambda b: (b["start"], b["end"]))
    dedup = []
    last_end = -1
    for b in blocks:
        if b["start"] <= last_end:
            continue
        dedup.append(b)
        last_end = b["end"]
    if verbose:
        print("GELU blocks:", len(dedup))
    return dedup


def lower_to_cactus(
    nodes,
    g,
    input_map,
    input_shapes,
    raw_inputs=None,
    enable_attention_fusion=True,
    enable_rmsnorm_fusion=True,
    use_kv_cache=False,
    kv_cache_provider=None,
    position_offset=0,
    force_dense_kv_attention=False,
    debug_taps=None,
    fp32_layernorm=False,
    fp32_qkv_matmul=False,
    post_matmul_fp32=False,
    attention_scale=None,
    arg_specs=None,
    kv_cache_tensors=None,
    kv_cache_len=0,
    kv_num_heads=None,
    kv_head_dim=None,
    kv_cache_additive_mask=None,
    fused_attn_out_names=None,
):
    if arg_specs is None:
        arg_specs = {}

    attn_blocks = find_attention_blocks(nodes, verbose=False) if enable_attention_fusion else []
    rms_blocks = find_rmsnorm_blocks(nodes, verbose=False) if enable_rmsnorm_fusion else []
    ln_blocks = find_layernorm_blocks(nodes, verbose=False)
    gelu_blocks = find_gelu_blocks(nodes, verbose=False)
    nodes_by_name = {n.name: n for n in nodes}
    idx_by_name = {n.name: i for i, n in enumerate(nodes)}
    env = {}
    shapes = dict(input_shapes)
    attn_blocks = find_attention_blocks(nodes, verbose=False) if enable_attention_fusion else []
    attn_qkv_names = set()
    for b in attn_blocks:
        score_node = nodes[b["score"]]
        q_name = score_node.inputs[0]
        k_name = nodes[b["transpose"]].inputs[0]
        out_node = nodes[b["out"]]
        v_name = out_node.inputs[1]
        attn_qkv_names.add(q_name)
        attn_qkv_names.add(k_name)
        attn_qkv_names.add(v_name)
    rms_by_start = {b["start"]: b for b in rms_blocks}
    rms_input_names = {b["x"] for b in rms_blocks}
    ln_by_start = {b["start"]: b for b in ln_blocks}
    gelu_by_start = {b["start"]: b for b in gelu_blocks}
    if raw_inputs is None:
        raw_inputs = []
    constant_stats = {
        "total": 0,
        "scalars": 0,
        "fallback": 0,
    }

    # map inputs
    for k, v in input_map.items():
        env[k] = v

    if debug_taps is not None and rms_blocks:
        for bi, b in enumerate(rms_blocks[:12]):
            xn = nodes_by_name.get(b["x"])
            wn = nodes_by_name.get(b["weight"])
            print(
                f"[rms_block {bi}] start={b['start']} end={b['end']} "
                f"x={b['x']}({xn.op if xn else 'missing'}@{idx_by_name.get(b['x'], -1)}) "
                f"w={b['weight']}({wn.op if wn else 'missing'}@{idx_by_name.get(b['weight'], -1)}) "
                f"out={b['out']} eps={b['eps']}"
            )
        if len(rms_blocks) > 3:
            x_name = rms_blocks[3]["x"]
            x_idx = idx_by_name.get(x_name, -1)
            if x_idx >= 0:
                lo = max(0, x_idx - 25)
                hi = min(len(nodes), x_idx + 10)
                print(f"[rms_block 3 context] idx={x_idx} x={x_name}")
                for j in range(lo, hi):
                    n = nodes[j]
                    extra = ""
                    if n.op == "constant":
                        try:
                            extra = f" val={extract_constant(n.raw)} raw={n.raw}"
                        except Exception:
                            extra = f" raw={n.raw}"
                    print(f"  {j:04d} {n.name} {n.op} inputs={n.inputs} shape={n.shape}{extra}")

    skip_until_idx = -1
    attn_layer_idx = 0
    rms_block_idx = 0
    ln_block_idx = 0
    gelu_block_idx = 0

    for i, node in enumerate(nodes):

        if i <= skip_until_idx:
            continue

        rms_b = rms_by_start.get(i)
        if rms_b is not None and rms_b["x"] in env and rms_b["weight"] in env:
            x = env[rms_b["x"]]
            weight = env[rms_b["weight"]]

            orig_shape = tuple(shapes[rms_b["x"]])
            out_shape = tuple(nodes_by_name[rms_b["out"]].shape)

            x16 = ensure_fp16(g, x)
            w16 = ensure_fp16(g, weight)

            if debug_taps is not None:
                debug_taps[f"rms_in_{rms_block_idx}"] = x16
                debug_taps[f"rms_w_{rms_block_idx}"] = w16

            # Cactus rms_norm only supports 2D [rows, dim],
            # so flatten [..., dim] -> [prod(...), dim], then reshape back.
            if len(orig_shape) == 1:
                x2 = g.reshape(x16, (1, int(orig_shape[0])))
                y2 = g.rms_norm(x2, w16, eps=float(rms_b["eps"]))
                y = g.reshape(y2, orig_shape)

            elif len(orig_shape) == 2:
                y = g.rms_norm(x16, w16, eps=float(rms_b["eps"]))

            else:
                rows = 1
                for d in orig_shape[:-1]:
                    rows *= int(d)

                dim = int(orig_shape[-1])
                x2 = g.reshape(x16, (rows, dim))
                y2 = g.rms_norm(x2, w16, eps=float(rms_b["eps"]))
                y = g.reshape(y2, orig_shape)

            env[rms_b["out"]] = y
            shapes[rms_b["out"]] = out_shape

            if debug_taps is not None:
                debug_taps[f"rms_out_{rms_block_idx}"] = y

            rms_block_idx += 1
            skip_until_idx = rms_b["end"]
            continue

        ln_b = ln_by_start.get(i)
        if ln_b is not None:
            x = env[ln_b["x"]]
            gamma = env[ln_b["gamma"]]
            beta = env[ln_b["beta"]]
            if fp32_layernorm:
                x32 = g.precision_cast(x, g.FP32)
                g32 = g.precision_cast(gamma, g.FP32)
                b32 = g.precision_cast(beta, g.FP32)
                out32 = g.layernorm(x32, g32, b32, eps=float(ln_b["eps"]))
                out = g.precision_cast(out32, g.FP16)
            else:
                out = g.layernorm(x, gamma, beta, eps=float(ln_b["eps"]))
            env[ln_b["out"]] = out
            shapes[ln_b["out"]] = nodes_by_name[ln_b["out"]].shape
            if debug_taps is not None:
                debug_taps[f"ln_out_{ln_block_idx}"] = out
            ln_block_idx += 1
            skip_until_idx = ln_b["end"]
            continue

        gelu_b = gelu_by_start.get(i)
        if gelu_b is not None:
            x = env[gelu_b["x"]]
            out = g.gelu(x)
            env[gelu_b["out"]] = out
            shapes[gelu_b["out"]] = nodes_by_name[gelu_b["out"]].shape
            if debug_taps is not None:
                debug_taps[f"gelu_out_{gelu_block_idx}"] = out
            gelu_block_idx += 1
            skip_until_idx = gelu_b["end"]
            continue
        fused = False

        for b in attn_blocks:
            if i == b["out"]:
                score_node = nodes[b["score"]]
                q_name = score_node.inputs[0]

                k_name = nodes[b["transpose"]].inputs[0]

                out_node = nodes[b["out"]]
                softmax_name = nodes[b["softmax"]].name
                if len(out_node.inputs) < 2:
                    continue
                if out_node.inputs[0] == softmax_name:
                    v_name = out_node.inputs[1]
                elif out_node.inputs[1] == softmax_name:
                    v_name = out_node.inputs[0]
                else:
                    # Fallback to previous behavior if parser pattern shifts.
                    v_name = out_node.inputs[1]

                # Gemma decode graphs can contain dot/softmax/tril subgraphs that
                # resemble attention but do not expose GPT-2-style q/k/v names at
                # this exact point in topological order. If any input is missing,
                # skip fusion and let generic lowering handle this block.
                if q_name not in env or k_name not in env or v_name not in env:
                    if debug_taps is not None:
                        print(
                            "[attn_fuse_skip_missing]",
                            f"i={i}",
                            f"q={q_name in env}:{q_name}",
                            f"k={k_name in env}:{k_name}",
                            f"v={v_name in env}:{v_name}",
                            f"softmax={softmax_name}",
                            f"out_inputs={out_node.inputs}",
                        )
                    continue

                q = env[q_name]
                k = env[k_name]
                v = env[v_name]

                q_shape = shapes.get(q_name)
                k_shape = shapes.get(k_name)
                v_shape = shapes.get(v_name)
                if q_shape is None or k_shape is None or v_shape is None:
                    if debug_taps is not None:
                        print(
                            "[attn_fuse_skip_shape]",
                            f"i={i}",
                            f"q_shape={q_shape}",
                            f"k_shape={k_shape}",
                            f"v_shape={v_shape}",
                        )
                    continue

                layer_idx = attn_layer_idx
                attn_layer_idx += 1
                D_MODEL = None

                if use_kv_cache and kv_num_heads is not None and kv_head_dim is not None:
                    NUM_HEADS = int(kv_num_heads)
                    HEAD_DIM = int(kv_head_dim)
                elif use_kv_cache and kv_cache_provider is not None:
                    cache = kv_cache_provider(layer_idx, None, None)
                    NUM_HEADS = int(cache.get("num_kv_heads"))
                    HEAD_DIM = int(cache.get("head_dim"))
                else:
                    # fallback (prefill or no cache)
                    NUM_HEADS = 12
                    if len(q_shape) == 2:
                        D_MODEL = int(q_shape[1])
                        HEAD_DIM = D_MODEL // NUM_HEADS
                    else:
                        HEAD_DIM = int(q_shape[-1])
                        D_MODEL = NUM_HEADS * HEAD_DIM

                def _to_1thd(x_t, x_shape_t, tag):
                    if len(x_shape_t) == 2:
                        t_len, d = int(x_shape_t[0]), int(x_shape_t[1])
                        if d != NUM_HEADS * HEAD_DIM:
                            return None, None, None
                        x4_t = g.reshape(x_t, (1, t_len, NUM_HEADS, HEAD_DIM))
                        x2_t = x_t
                        return x4_t, t_len, x2_t
                    if len(x_shape_t) == 3:
                        a, bsz, c = int(x_shape_t[0]), int(x_shape_t[1]), int(x_shape_t[2])
                        if c != HEAD_DIM:
                            return None, None, None
                        if a == NUM_HEADS:
                            # (H, T, D) -> (T, H, D)
                            x_thd = g.permute(x_t, [1, 0, 2])
                            t_len = bsz
                        elif bsz == NUM_HEADS:
                            # (T, H, D)
                            x_thd = x_t
                            t_len = a
                        else:
                            return None, None, None
                        x4_t = g.reshape(x_thd, (1, t_len, NUM_HEADS, HEAD_DIM))
                        x2_t = g.reshape(x_thd, (t_len, NUM_HEADS * HEAD_DIM))
                        return x4_t, t_len, x2_t
                    return None, None, None

                q4, Tq, q2 = _to_1thd(q, q_shape, "q")
                k4, Tk, k2 = _to_1thd(k, k_shape, "k")
                v4, Tv, v2 = _to_1thd(v, v_shape, "v")
                if q4 is None or k4 is None or v4 is None or Tq != Tk or Tq != Tv:
                    if debug_taps is not None:
                        print(
                            "[attn_fuse_skip_shape]",
                            f"i={i}",
                            f"q_shape={q_shape}",
                            f"k_shape={k_shape}",
                            f"v_shape={v_shape}",
                            f"heads={NUM_HEADS}",
                            f"head_dim={HEAD_DIM}",
                        )
                    continue
                T = int(Tq)
                if D_MODEL is None:
                    D_MODEL = NUM_HEADS * HEAD_DIM

                q4 = ensure_fp16(g, q4)
                k4 = ensure_fp16(g, k4)
                v4 = ensure_fp16(g, v4)

                scale = np.float32(attention_scale if attention_scale is not None else (1.0 / np.sqrt(HEAD_DIM)))
                if use_kv_cache:
                    if T != 1:
                        raise Exception(f"KV-cache decode path expects T=1, got T={T}")
                    if kv_cache_tensors is not None:
                        if layer_idx >= len(kv_cache_tensors):
                            raise Exception(
                                f"Missing graph KV tensors for layer {layer_idx}; "
                                f"have {len(kv_cache_tensors)}"
                            )
                        k_cache4, v_cache4 = kv_cache_tensors[layer_idx]
                        cache_len = int(kv_cache_len)

                        if kv_cache_additive_mask is not None:
                            # Graph-reuse path: keep static key/value shapes and provide
                            # a runtime additive mask that selects valid cached positions
                            # plus the appended current token.
                            k_full4 = g.cat([k_cache4, k4], axis=1)
                            v_full4 = g.cat([v_cache4, v4], axis=1)
                            attn4 = g.attention(
                                q4,
                                k_full4,
                                v_full4,
                                scale,
                                is_causal=False,
                                position_offset=0,
                                mask=kv_cache_additive_mask,
                                additive_mask=True,
                            )
                        else:
                            if cache_len > 0:
                                k_old4 = g.slice(k_cache4, 1, 0, cache_len)
                                v_old4 = g.slice(v_cache4, 1, 0, cache_len)
                                k_full4 = g.cat([k_old4, k4], axis=1)
                                v_full4 = g.cat([v_old4, v4], axis=1)
                            else:
                                k_full4 = k4
                                v_full4 = v4

                            attn4 = g.attention(
                                q4,
                                k_full4,
                                v_full4,
                                scale,
                                is_causal=True,
                                position_offset=position_offset,
                            )
                    else:
                        if kv_cache_provider is None:
                            raise Exception(
                                "use_kv_cache=True requires kv_cache_provider or kv_cache_tensors"
                            )

                        cache = kv_cache_provider(layer_idx, NUM_HEADS, HEAD_DIM)
                        cache_len = int(cache.get("cache_len", 0))

                        if cache_len > 0 and not force_dense_kv_attention:
                            attn4 = g.attention_int8_hybrid(
                                q4, k4, v4, scale, position_offset,
                                cache["cached_keys"],
                                cache["cached_values"],
                                cache["k_scales"],
                                cache["v_scales"],
                                cache_len,
                                int(cache.get("num_kv_heads", NUM_HEADS)),
                                int(cache.get("head_dim", HEAD_DIM)),
                            )
                        else:
                            # Proper FP16 dense-cache fallback:
                            # old cache is numpy: (S, H, D)
                            k_old_np = cache["k_cache_float"]
                            v_old_np = cache["v_cache_float"]
                            S = int(k_old_np.shape[0])

                            if S > 0:
                                k_old4 = g.input((1, S, NUM_HEADS, HEAD_DIM), g.FP16)
                                v_old4 = g.input((1, S, NUM_HEADS, HEAD_DIM), g.FP16)

                                g.set_input(k_old4, np.ascontiguousarray(k_old_np[None].astype(np.float16)))
                                g.set_input(v_old4, np.ascontiguousarray(v_old_np[None].astype(np.float16)))

                                k_full4 = g.cat([k_old4, k4], axis=1)
                                v_full4 = g.cat([v_old4, v4], axis=1)
                            else:
                                k_full4 = k4
                                v_full4 = v4

                            attn4 = g.attention(
                                q4,
                                k_full4,
                                v_full4,
                                scale,
                                is_causal=True,
                                position_offset=position_offset,
                            )
                else:
                    attn4 = g.attention(q4, k4, v4, scale, is_causal=True)

                out_shape = tuple(nodes[b["out"]].shape) if nodes[b["out"]].shape is not None else None
                if out_shape is None:
                    continue
                if len(out_shape) == 2:
                    out = g.reshape(attn4, out_shape)
                elif len(out_shape) == 3:
                    out_thd = g.reshape(attn4, (T, NUM_HEADS, HEAD_DIM))
                    if out_shape == (T, NUM_HEADS, HEAD_DIM):
                        out = out_thd
                    elif out_shape == (NUM_HEADS, T, HEAD_DIM):
                        out = g.permute(out_thd, [1, 0, 2])
                    else:
                        if debug_taps is not None:
                            print(
                                "[attn_fuse_skip_outshape]",
                                f"i={i}",
                                f"out_shape={out_shape}",
                                f"T={T}",
                                f"heads={NUM_HEADS}",
                                f"head_dim={HEAD_DIM}",
                            )
                        continue
                else:
                    continue
                if debug_taps is not None:
                    debug_taps[f"attn_raw_{layer_idx}"] = out
                    print(
                        "[attn_fused]",
                        f"layer={layer_idx}",
                        f"q={q_name} shape={q_shape}",
                        f"k={k_name} shape={k_shape}",
                        f"v={v_name} shape={v_shape}",
                        f"T={T}",
                        f"heads={NUM_HEADS}",
                        f"head_dim={HEAD_DIM}",
                        f"out={out_node.name} shape={out_shape}",
                    )

                env[out_node.name] = out
                # Expose per-layer K/V for cache append from the pre-reshape path.
                # This avoids an extra cast/reshape round-trip before int8 quantization.
                env[out_node.name + "_k"] = k2
                env[out_node.name + "_v"] = v2
                if fused_attn_out_names is not None:
                    fused_attn_out_names.append(out_node.name)
                shapes[out_node.name] = nodes[b["out"]].shape

                skip_until_idx = b["out"]
                fused = True
                break   # 👈 IMPORTANT
        if fused:
            continue   # 👈 THIS IS THE KEY FIX
        op = node.op

        if op == "dot_general":
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]

            # Prefer runtime tensor shapes (authoritative) over tracked IR shapes.
            # The tracked map can drift in some Gemma graphs after aggressive rewrites.
            a_shape = tuple(getattr(a, "shape", ())) or tuple(shapes.get(node.inputs[0], ()))
            b_shape = tuple(getattr(b, "shape", ())) or tuple(shapes.get(node.inputs[1], ()))
            out_shape = node.shape
            b_name = node.inputs[1]
            pretransposed_rhs = False

            # Cactus-aware matmul lowering:
            # dot_general(lhs, transpose(weight_arg)) -> matmul(lhs, weight_arg, pretransposed_rhs=True)
            b_node = nodes_by_name.get(b_name)
            if b_node is not None and b_node.op == "transpose" and b_node.inputs:
                src_name = b_node.inputs[0]
                spec = arg_specs.get(src_name)
                src_t = env.get(src_name)
                src_shape = tuple(getattr(src_t, "shape", ())) if src_t is not None else None
                if not src_shape:
                    src_shape = tuple(shapes.get(src_name, ()))
                if (
                    spec is not None
                    and spec.get("matmul_weight", False)
                    and src_shape is not None
                    and len(src_shape) == 2
                    and len(b_shape) == 2
                ):
                    b = env[src_name]
                    b_shape = src_shape
                    pretransposed_rhs = bool(spec.get("pretransposed_rhs", True))
            else:
                spec = arg_specs.get(b_name)
                if spec is not None and spec.get("matmul_weight", False):
                    pretransposed_rhs = bool(spec.get("pretransposed_rhs", False))

            if len(a_shape) == 2 and len(b_shape) == 2:
                if fp32_qkv_matmul and node.name in attn_qkv_names:
                    a32 = g.precision_cast(a, g.FP32)
                    b32 = g.precision_cast(b, g.FP32)
                    env[node.name] = g.matmul(a32, b32)
                else:
                    env[node.name] = g.matmul(a, b, pretransposed_rhs=pretransposed_rhs)

            elif len(a_shape) == 3 and len(b_shape) == 2:
                B, T, K = a_shape
                Kb, N = b_shape

                if K != Kb:
                    raise Exception(f"Bad final matmul: {a_shape} @ {b_shape}")

                # pad N to multiple of 8 for backend matmul
                N_pad = ((N + 7) // 8) * 8

                if N_pad != N:
                    pad_cols = N_pad - N
                    zero_pad = g.input((K, pad_cols), g.FP16)
                    g.set_input(zero_pad, np.zeros((K, pad_cols), dtype=np.float16))
                    b = g.cat([b, zero_pad], axis=1)

                a2d = g.reshape(a, (B * T, K))
                y2d = g.matmul(a2d, b)

                y = g.reshape(y2d, (B, T, N_pad))

                # slice back to real vocab size
                if N_pad != N:
                    y = g.slice(y, 2, 0, N)

                env[node.name] = y

            elif len(a_shape) >= 3 and len(b_shape) >= 3:
                env[node.name] = lower_batched_matmul(
                    g, a, b, a_shape, b_shape, out_shape
                )

            else:
                a_name = node.inputs[0]
                b_name_dbg = node.inputs[1]
                a_node_dbg = nodes_by_name.get(a_name)
                b_node_dbg = nodes_by_name.get(b_name_dbg)
                raise Exception(
                    "Unsupported dot_general shapes: "
                    f"{a_shape} @ {b_shape}, "
                    f"a_name={a_name}, b_name={b_name_dbg}, "
                    f"a_tracked={shapes.get(a_name)}, b_tracked={shapes.get(b_name_dbg)}, "
                    f"a_runtime={tuple(getattr(a, 'shape', ()))}, b_runtime={tuple(getattr(b, 'shape', ()))}, "
                    f"a_node_op={getattr(a_node_dbg, 'op', None)}, b_node_op={getattr(b_node_dbg, 'op', None)}, "
                    f"a_node_shape={getattr(a_node_dbg, 'shape', None)}, b_node_shape={getattr(b_node_dbg, 'shape', None)}, "
                    f"raw={node.raw}"
                )

            if post_matmul_fp32 and getattr(env[node.name], "dtype", None) != g.FP32:
                env[node.name] = g.precision_cast(env[node.name], g.FP32)

            shapes[node.name] = out_shape

        elif op == "gather":
            src_name = node.inputs[0]
            idx_name = node.inputs[1]

            # Common GPT decode pattern: positional embedding gather from %arg3 using
            # computed/clamped %arg1 index. Use raw input index directly for stability.
            if (
                src_name.startswith("%arg")
                and raw_inputs
                and len(shapes.get(src_name, ())) == 2
                and node.shape is not None
                and len(node.shape) == 2
            ):
                src_arg_index = int(src_name.replace("%arg", ""))
                src_np = raw_inputs[src_arg_index].astype(np.float16)

                if src_arg_index == 3 and len(raw_inputs) > 1:
                    pos_raw = int(raw_inputs[1].reshape(-1)[0])
                    pos = pos_raw % src_np.shape[0]
                    gathered = src_np[pos:pos + 1]
                    t = g.input(gathered.shape, g.FP16)
                    g.set_input(t, np.ascontiguousarray(gathered))
                    env[node.name] = t
                    shapes[node.name] = node.shape
                    continue

            if src_name not in env or idx_name not in env:
                raise Exception(f"gather inputs missing: {src_name}, {idx_name}")

            env[node.name] = g.gather(env[src_name], env[idx_name])
            shapes[node.name] = node.shape
        
        elif op == "transpose":
            x = env[node.inputs[0]]
            input_shape = tuple(shapes[node.inputs[0]])
            rank = len(input_shape)

            # Scalar/vector transpose is identity in StableHLO practice.
            if rank <= 1:
                env[node.name] = x
                shapes[node.name] = input_shape
                continue

            try:
                perm = extract_permutation(node.raw)
            except Exception:
                perm = None

            def _default_last2_swap(r):
                p = list(range(r))
                if r >= 2:
                    p[-1], p[-2] = p[-2], p[-1]
                return p

            def _is_valid_perm(p, r):
                if p is None or len(p) != r:
                    return False
                try:
                    return sorted(int(v) for v in p) == list(range(r))
                except Exception:
                    return False

            if not _is_valid_perm(perm, rank):
                perm = _default_last2_swap(rank)

            # Transpose kernel currently supports FP16 only.
            x_t = ensure_fp16(g, x)

            try:
                if perm == list(range(rank)):
                    t = x_t
                elif rank == 2 and perm == [1, 0]:
                    t = g.transpose(x_t)
                else:
                    t = g.permute(x_t, perm)
            except Exception as e:
                raise RuntimeError(
                    f"transpose lowering failed for {node.name}: "
                    f"input_shape={input_shape}, perm={perm}, raw={node.raw}"
                ) from e

            env[node.name] = t
            shapes[node.name] = tuple(input_shape[j] for j in perm)
        
        elif op == "reshape":
            x = env[node.inputs[0]]
            env[node.name] = g.reshape(x, node.shape)
            shapes[node.name] = node.shape

        elif op == "concatenate":
            xs = [env[inp] for inp in node.inputs]
            m = re.search(r"dim\s*=\s*(\d+)", node.raw)
            if not m:
                raise Exception(f"Could not parse concatenate dim: {node.raw}")
            axis = int(m.group(1))
            env[node.name] = g.cat(xs, axis=axis)
            shapes[node.name] = node.shape
        
        elif op == "call":
            if "@relu" in node.raw:
                x = env[node.inputs[0]]

                env[node.name] = g.relu(x)
                shapes[node.name] = node.shape

            else:
                raise Exception(f"Unsupported call: {node.raw}")
        
        elif op == "relu" or op.startswith("relu_"):
            x = env[node.inputs[0]]
            env[node.name] = g.relu(x)
            shapes[node.name] = node.shape
        
        elif op == "reduce_maximum":
            x = env[node.inputs[0]]
            axis = extract_reduce_axis(node.raw)
            # Backend reduction kernels currently support FP16 only.
            x16 = ensure_fp16(g, x)
            env[node.name] = g.max(x16, axis)
            shapes[node.name] = node.shape

        elif op == "reduce_add":
            x = env[node.inputs[0]]
            axis = extract_reduce_axis(node.raw)
            # Backend reduction kernels currently support FP16 only.
            x16 = ensure_fp16(g, x)
            env[node.name] = g.sum(x16, axis)
            shapes[node.name] = node.shape
        
        # no negate in cactus graph???
        elif op == "negate":
            x = env[node.inputs[0]]
            env[node.name] = g.scalar_multiply(x, -1.0)
            shapes[node.name] = node.shape
        

        elif op == "rsqrt":
            x = env[node.inputs[0]]

            sqrt_x = g.scalar_sqrt(x)

            one = g.input((1,), g.FP16)
            g.set_input(one, np.array([1.0], dtype=np.float16))

            inv = g.divide(one, sqrt_x)

            env[node.name] = inv
            shapes[node.name] = node.shape

        
        elif op == "maximum":
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]

            a, b, out_shape = align_shapes(
                g, a, b,
                shapes[node.inputs[0]],
                shapes[node.inputs[1]]
            )
            if getattr(a, "dtype", None) == g.FP32 or getattr(b, "dtype", None) == g.FP32:
                a, b = ensure_binary_fp32(g, a, b)
            else:
                a, b = ensure_binary_fp16(g, a, b)

            env[node.name] = g.add(g.relu(g.subtract(a, b)), b)
            shapes[node.name] = out_shape
        
        elif op == "exponential":
            x = env[node.inputs[0]]
            x16 = ensure_fp16(g, x)
            env[node.name] = g.scalar_exp(x16)
            shapes[node.name] = node.shape   # 🔥 NO CAST BACK HERE


        elif op == "add":
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]

            a_shape0 = tuple(shapes[node.inputs[0]])
            b_shape0 = tuple(shapes[node.inputs[1]])

            a, b, out_shape = align_shapes(
                g, a, b,
                a_shape0,
                b_shape0
            )

            # Cactus binary ops are FP16-only in current backend.
            a, b = ensure_binary_fp16(g, a, b)

            # Residual adds are usually direct inputs into RMSNorm.
            # These can overflow in FP16, so use native clipped add ONLY there.
            is_same_shape_add = (
                tuple(a_shape0) == tuple(out_shape)
                and tuple(b_shape0) == tuple(out_shape)
            )
            is_rms_input_add = node.name in rms_input_names

            if is_same_shape_add and is_rms_input_add:
                env[node.name] = g.add_clipped(a, b)
            else:
                env[node.name] = g.add(a, b)

            shapes[node.name] = out_shape
        
        elif op == "subtract":
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]

            # 🔥 preserve FP32 if present
            if getattr(a, "dtype", None) == g.FP32 or getattr(b, "dtype", None) == g.FP32:
                a, b = ensure_binary_fp32(g, a, b)
                env[node.name] = g.subtract(a, b)
            else:
                a, b = ensure_binary_fp16(g, a, b)
                env[node.name] = g.subtract(a, b)

            shapes[node.name] = node.shape

        elif op == "divide":
                a = env[node.inputs[0]]
                b = env[node.inputs[1]]

                # 🔥 if either is FP32 → keep FP32
                if getattr(a, "dtype", None) == g.FP32 or getattr(b, "dtype", None) == g.FP32:
                    a, b = ensure_binary_fp32(g, a, b)
                    env[node.name] = g.divide(a, b)
                else:
                    a, b = ensure_binary_fp16(g, a, b)
                    env[node.name] = g.divide(a, b)

                shapes[node.name] = node.shape

        elif op == "multiply":
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]

            a, b, out_shape = align_shapes(
                g, a, b,
                shapes[node.inputs[0]],
                shapes[node.inputs[1]]
            )

            # 🔥 preserve FP32 if present
            if getattr(a, "dtype", None) == g.FP32 or getattr(b, "dtype", None) == g.FP32:
                a, b = ensure_binary_fp32(g, a, b)
                env[node.name] = g.multiply(a, b)
            else:
                a, b = ensure_binary_fp16(g, a, b)
                env[node.name] = g.multiply(a, b)

            shapes[node.name] = out_shape
        
        elif op == "sqrt":
            x = env[node.inputs[0]]
            env[node.name] = g.scalar_sqrt(x)
            shapes[node.name] = node.shape

        elif op == "select":
            cond_node = node.inputs[0]

            # 🔥 DETECT one-hot pattern
            cond_src = nodes_by_name.get(cond_node)

            if cond_src and cond_src.op == "compare":
                a_name = cond_src.inputs[0]  # tokens
                b_name = cond_src.inputs[1]  # range

                # Guard aggressively so we only rewrite true one-hot selects.
                is_rank2 = node.shape is not None and len(node.shape) == 2
                looks_vocab = is_rank2 and node.shape[-1] >= 128

                if a_name in env and b_name in shapes and is_rank2 and looks_vocab:
                    tokens_tensor = env[a_name]
                    T = shapes[a_name][0]
                    V = node.shape[-1]

                    # 🔥 build exact one-hot
                    tokens_np = tokens_tensor.numpy().astype(int)

                    if tokens_np.shape[0] == T and V == shapes[b_name][0]:
                        if np.any(tokens_np < 0) or np.any(tokens_np >= V):
                            raise Exception(f"one_hot token out of range: {tokens_np}, V={V}")

                        one_hot = np.zeros((T, V), dtype=np.float16)
                        one_hot[np.arange(T), tokens_np] = 1.0

                        t = g.input((T, V), Graph.FP16)
                        g.set_input(t, one_hot)

                        env[node.name] = t
                        shapes[node.name] = (T, V)
                        continue

            # Generic select: cond ? a : b
            cond = env[node.inputs[0]]
            a = env[node.inputs[1]]
            b = env[node.inputs[2]]

            a, b, out_shape = align_shapes(
                g, a, b,
                shapes[node.inputs[1]],
                shapes[node.inputs[2]],
            )
            cond, a_aligned, _ = align_shapes(
                g, cond, a,
                shapes[node.inputs[0]],
                out_shape,
            )
            cond = ensure_fp16(g, cond)
            a_aligned = ensure_fp16(g, a_aligned)
            b = ensure_fp16(g, b)

            one_minus_cond = g.scalar_add(g.scalar_multiply(cond, -1.0), 1.0)
            out = g.add(g.multiply(cond, a_aligned), g.multiply(one_minus_cond, b))
            env[node.name] = out
            shapes[node.name] = out_shape
            continue
        

        elif op == "tril":
            inp_name = node.inputs[0]

            if inp_name not in env:
                raise Exception(f"tril input missing: {inp_name}, env keys: {list(env.keys())}")

            inp = env[inp_name]
            shape = shapes[inp_name]

            if len(shape) != 2:
                raise Exception(f"Unsupported tril shape: {shape}")

            T0, T1 = shape

            # 🔥 build lower-triangular mask
            tril_np = np.tril(np.ones((T0, T1), dtype=np.float16))

            t = g.input((T0, T1), g.FP16)
            g.set_input(t, tril_np)

            env[node.name] = t
            shapes[node.name] = (T0, T1)

            continue

        if op == "_one_hot":
            tokens_name = node.inputs[0]

            if tokens_name not in env:
                raise Exception(f"_one_hot missing input {tokens_name}. Available: {list(env.keys())}")

            arg_index = int(tokens_name.replace("%arg", ""))
            tokens_np = raw_inputs[arg_index].astype(np.int64)

            T = tokens_np.shape[0]
            V = node.shape[-1]

            one_hot_np = np.zeros((T, V), dtype=np.float16)

            if np.any(tokens_np < 0) or np.any(tokens_np >= V):
                raise Exception(f"_one_hot token out of range: tokens={tokens_np}, V={V}")

            one_hot_np[np.arange(T), tokens_np] = np.float16(1.0)

            one_hot_t = g.input((T, V), g.FP16)
            g.set_input(one_hot_t, one_hot_np)

            env[node.name] = one_hot_t
            shapes[node.name] = (T, V)

            continue
        
        elif op == "tanh":
            x = env[node.inputs[0]]
            env[node.name] = g.tanh(x)
            shapes[node.name] = node.shape
        



        elif op == "compare":
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]
            a, b = ensure_binary_fp16(g, a, b)

            rel_m = re.search(r"compare\s+([A-Z]{2})", node.raw)
            rel = rel_m.group(1) if rel_m else "LT"

            # Build signed margin where positive means "true".
            if rel == "LT":
                margin = g.subtract(b, a)  # b - a > 0
            elif rel == "LE":
                margin = g.scalar_add(g.subtract(b, a), 0.5)  # include equality
            elif rel == "GT":
                margin = g.subtract(a, b)  # a - b > 0
            elif rel == "GE":
                margin = g.scalar_add(g.subtract(a, b), 0.5)  # include equality
            elif rel == "EQ":
                # Approximate equality with a sharp bell around zero.
                d = g.abs(g.subtract(a, b))
                margin = g.scalar_add(g.scalar_multiply(d, -8.0), 1.0)
            elif rel == "NE":
                d = g.abs(g.subtract(a, b))
                margin = g.scalar_add(g.scalar_multiply(d, 8.0), -1.0)
            else:
                margin = g.subtract(b, a)

            # Smooth step to near-binary mask in [0, 1].
            sharp = g.scalar_multiply(margin, 8.0)
            s = g.tanh(sharp)
            mask = g.scalar_multiply(g.scalar_add(s, 1.0), 0.5)

            env[node.name] = mask
            shapes[node.name] = node.shape

        # definetly better way to do this!! figure out how stablehlo does it behidn the scenes and copy
        elif op == "broadcast_in_dim":
            x = env[node.inputs[0]]
            input_shape = shapes[node.inputs[0]]
            target_shape = node.shape

            # Step 1: right-align via reshape
            if len(input_shape) != len(target_shape):
                new_shape = [1] * len(target_shape)
                offset = len(target_shape) - len(input_shape)

                for i, d in enumerate(input_shape):
                    new_shape[offset + i] = d

                x = g.reshape(x, tuple(new_shape))
                input_shape = tuple(new_shape)
            



            # 🔥 FIX: handle permuted broadcast (like (1,1,8) -> (1,8,1))
            if sorted(input_shape) == sorted(target_shape):
                # find permutation that maps input -> target
                perm = []
                used = [False] * len(input_shape)

                for t in target_shape:
                    for i, s in enumerate(input_shape):
                        if not used[i] and s == t:
                            perm.append(i)
                            used[i] = True
                            break

                if len(perm) == len(input_shape):
                    x = g.permute(x, perm)
                    input_shape = tuple(input_shape[i] for i in perm)
            
            # Step 2: ACTUAL broadcast via tile
            reps = []
            for s, t in zip(input_shape, target_shape):
                if s == t:
                    reps.append(1)
                elif s == 1:
                    reps.append(t)
                else:
                    raise Exception(f"Bad broadcast: {input_shape} → {target_shape}")

            for axis, r in enumerate(reps):
                if r != 1:
                    x = repeat_axis(g, x, axis, r)

            env[node.name] = x
            shapes[node.name] = target_shape
        

        elif op == "and":
            a = env[node.inputs[0]]
            b = env[node.inputs[1]]
            a, b, out_shape = align_shapes(
                g, a, b, shapes[node.inputs[0]], shapes[node.inputs[1]]
            )
            a, b = ensure_binary_fp16(g, a, b)
            env[node.name] = g.multiply(a, b)
            shapes[node.name] = out_shape

            
        elif op == "slice":
            x = env[node.inputs[0]]
            input_shape = shapes[node.inputs[0]]
            m = re.search(r"\[([0-9:\,\s]+)\]", node.raw)
            if not m:
                raise Exception(f"Could not parse slice: {node.raw}")

            ranges = []
            for part in m.group(1).split(","):
                start_s, end_s = part.strip().split(":")
                ranges.append((int(start_s), int(end_s)))

            out = x
            cur_shape = tuple(input_shape)

            for axis, (start, end) in enumerate(ranges):
                length = end - start

                # skip full-axis slices
                if start == 0 and length == cur_shape[axis]:
                    continue

                out = g.slice(out, axis, start, length)

                cur_shape = (
                    cur_shape[:axis]
                    + (length,)
                    + cur_shape[axis + 1:]
                )

            env[node.name] = out
            shapes[node.name] = node.shape




        elif op == "iota":
            shape = node.shape

            if shape is None or len(shape) != 1:
                raise Exception(f"Unsupported iota shape: {shape}")

            elem_dtype = _extract_tensor_element_dtype(node.raw)
            use_fp32 = elem_dtype == "f32"
            np_dtype = np.float32 if use_fp32 else np.float16
            g_dtype = g.FP32 if use_fp32 else g.FP16

            arr = np.arange(shape[0], dtype=np_dtype)

            t = g.input(shape, g_dtype)
            g.set_input(t, arr)

            env[node.name] = t
            shapes[node.name] = shape
        


        elif op.startswith("_take"):
            raise Exception(f"Unsupported op for clean baseline path: {op}")

        


        elif op == "convert":
            x = env[node.inputs[0]]
            target = _extract_convert_target_dtype(node.raw)
            if target in ("f16", "bf16"):
                env[node.name] = ensure_fp16(g, x)
            elif target == "f32":
                if getattr(x, "dtype", None) == g.FP32:
                    env[node.name] = x
                else:
                    env[node.name] = g.precision_cast(x, g.FP32)
            else:
                env[node.name] = x
            shapes[node.name] = node.shape

        elif op == "constant":
            constant_stats["total"] += 1
            shape = node.shape
            val = extract_constant(node.raw)
            elem_dtype = _extract_tensor_element_dtype(node.raw)
            use_fp32 = elem_dtype == "f32"
            np_dtype = np.float32 if use_fp32 else np.float16
            g_dtype = g.FP32 if use_fp32 else g.FP16

            if shape is None:
                raise Exception(f"Constant has no shape: {node.raw}")

            # scalar constants → keep same
            if shape == ():
                constant_stats["scalars"] += 1
                if val is None:
                    val = 0.0
                if not use_fp32:
                    val = max(min(float(val), 65504.0), -65504.0)

                arr = np.array([val], dtype=np_dtype)
                t = g.input((1,), g_dtype)
                g.set_input(t, arr)

                env[node.name] = t
                shapes[node.name] = ()
                continue

            constant_stats["fallback"] += 1
            if val is None:
                arr = np.zeros(shape, dtype=np_dtype)
            else:
                arr = np.full(shape, val, dtype=np_dtype)

            t = g.input(shape, g_dtype)
            g.set_input(t, arr)

            env[node.name] = t
            shapes[node.name] = shape

    return env






# helper fucntiosn for shpaes



def flatten_leading_for_matmul(g, x, lhs_shape):
    # (..., K) -> (prod(...), K)
    k = lhs_shape[-1]
    m = numel(lhs_shape[:-1])
    return g.reshape(x, (m, k)), (m, k)


def restore_leading_after_matmul(g, y, out_shape):
    return g.reshape(y, out_shape)


#generic version>?
def lower_batched_matmul(g, a, b, a_shape, b_shape, out_shape):
    # Supports:
    # a: batch_shape + (M, K)
    # b: batch_shape + (K, N)
    # out: batch_shape + (M, N)


    batch_shape = a_shape[:-2]
    b_batch_shape = b_shape[:-2]

    if batch_shape != b_batch_shape:
        raise Exception(f"Batch shapes differ: {a_shape} @ {b_shape}")


    M, K = a_shape[-2], a_shape[-1]
    Kb, N = b_shape[-2], b_shape[-1]

    if K != Kb:
        raise Exception(f"Bad matmul K dims: {a_shape} @ {b_shape}")

    B = numel(batch_shape)

    a_flat = g.reshape(a, (B, M, K))
    b_flat = g.reshape(b, (B, K, N))

    parts = []

    for i in range(B):
        a_i = g.slice(a_flat, 0, i, 1)
        b_i = g.slice(b_flat, 0, i, 1)

        a_i = g.reshape(a_i, (M, K))
        b_i = g.reshape(b_i, (K, N))

        y_i = g.matmul(a_i, b_i)
        y_i = g.reshape(y_i, (1, M, N))
        parts.append(y_i)

    y = parts[0] if len(parts) == 1 else g.cat(parts, axis=0)
    return g.reshape(y, out_shape)



# helper transpose finding extra valas

import re

def extract_permutation(raw):
    # Common StableHLO text forms:
    # dims = [0, 2, 1, 3]
    # permutation = [0, 2, 1, 3]
    # permutation = array<i64: 0, 2, 1, 3>

    m = re.search(r"dims\s*=\s*\[([0-9,\s]*)\]", raw)
    if m:
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]

    m = re.search(r"permutation\s*=\s*\[([0-9,\s]*)\]", raw)
    if m:
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]

    m = re.search(r"permutation\s*=\s*array<[^:>]+:\s*([^>]+)>", raw)
    if m:
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]

    raise Exception(f"Could not parse transpose permutation from: {raw}")



# helper for reduce _axis


def extract_reduce_axis(raw):
    import re
    m = re.search(r"dimensions\s*=\s*\[([0-9,\s]+)\]", raw)
    if not m:
        raise Exception(f"Could not parse reduce axis from: {raw}")
    axes = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    if len(axes) != 1:
        raise Exception(f"Only single-axis reduce supported for now: {raw}")
    return axes[0]
