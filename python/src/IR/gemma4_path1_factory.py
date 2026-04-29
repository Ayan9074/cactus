from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np


def build_for_export() -> Dict[str, Any]:
    """Build a tiny Gemma4-style JAX callable for Path-1 export.

    This intentionally uses Gemma4 modules from google-deepmind/gemma but with
    a very small config so local export/compile is practical.
    """
    # The user asked for Gemma4 code from DeepMind's repo.
    # We expect the repo to be present at /tmp/gemma.
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    repo_python = repo_root / "python"
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))
    if "/tmp/gemma" not in sys.path:
        sys.path.insert(0, "/tmp/gemma")
    # kauldron in current gemma deps expects np.float128 symbol, which macOS
    # arm64 NumPy may not expose. Alias to longdouble for local testing.
    if not hasattr(np, "float128"):
        np.float128 = np.longdouble  # type: ignore[attr-defined]

    from gemma.gm.nn.gemma4 import _config as g4_config  # pylint: disable=import-error
    from gemma.gm.nn.gemma4 import _modules as g4_modules  # pylint: disable=import-error
    from gemma.gm.nn.gemma4 import _transformer as g4_transformer  # pylint: disable=import-error

    # Tiny text-only Gemma4-style config for tractable local testing.
    cfg = g4_config.TransformerConfig(
        num_embed=2048,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        head_dim=32,
        num_kv_heads=2,
        final_logit_softcap=None,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attention_types=(
            g4_modules.AttentionType.LOCAL_SLIDING,
            g4_modules.AttentionType.GLOBAL,
        ),
        attn_logits_soft_cap=None,
        sliding_window_size=64,
        qk_norm_with_scale=True,
        global_rope_proportion=1.0,
        local_rope_proportion=1.0,
        per_layer_input_dim=0,
        vision_encoder=None,
        audio_encoder=None,
    )

    model = g4_transformer.Transformer(config=cfg, dtype=jnp.float32)

    # Small dummy token input.
    tokens = np.array([[1, 17, 42, 3, 9, 11, 27, 5]], dtype=np.int32)
    batch, seq = tokens.shape
    positions = np.tile(np.arange(seq, dtype=np.int32)[None, :], (batch, 1))
    # Causal full attention mask: [B, L, L]
    tri = np.tril(np.ones((seq, seq), dtype=np.bool_))
    attention_mask = np.tile(tri[None, :, :], (batch, 1, 1))

    key = jax.random.PRNGKey(0)
    params = model.init(
        key,
        tokens=jnp.asarray(tokens),
        positions=jnp.asarray(positions),
        attention_mask=jnp.asarray(attention_mask),
    )["params"]

    def fn(params_, tokens_, positions_, attention_mask_):
        out = model.apply(
            {"params": params_},
            tokens=tokens_,
            positions=positions_,
            attention_mask=attention_mask_,
        )
        return out.logits

    return {"fn": fn, "args": (params, tokens, positions, attention_mask)}
