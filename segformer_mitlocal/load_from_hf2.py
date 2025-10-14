# scripts/load_from_hf.py  (replace file)

import argparse, os
import torch
import torch.nn as nn
from transformers import SegformerModel
from mit import MixVisionTransformer, MIT_CONFIGS

def build_local_mit(variant: str):
    if variant not in MIT_CONFIGS:
        raise ValueError(f"Unknown MiT variant '{variant}'. Options: {list(MIT_CONFIGS.keys())}")
    return MixVisionTransformer(**MIT_CONFIGS[variant])

# --- utilities ---------------------------------------------------------------

def _get_any(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def _copy_ln_1d_to_ln2d(dst_ln2d, src_ln1d):
    # both are per-channel gammas/betas of size C
    dst_ln2d.ln.weight.data.copy_(src_ln1d.weight.data)
    dst_ln2d.ln.bias.data.copy_(src_ln1d.bias.data)

def _lin_to_conv1x1(dst_conv, src_linear):
    # dst_conv.weight: [C_out, C_in, 1, 1]
    # src_linear.weight: [C_out, C_in]
    w = src_linear.weight.data.view(dst_conv.weight.shape)
    dst_conv.weight.data.copy_(w)
    if dst_conv.bias is not None and src_linear.bias is not None:
        dst_conv.bias.data.copy_(src_linear.bias.data)

def _copy_patch_embed(local, hf):
    # HF: SegformerOverlapPatchEmbeddings has .proj (Conv2d) and .layer_norm (LayerNorm)
    proj = _get_any(hf, "proj", "projection")
    ln = _get_any(hf, "layer_norm", "norm")
    if proj is None or ln is None:
        raise RuntimeError("HF patch embed missing proj/layer_norm")
    local.proj.weight.data.copy_(proj.weight.data)
    local.proj.bias.data.copy_(proj.bias.data)
    _copy_ln_1d_to_ln2d(local.norm, ln)

def _copy_attention(local_attn, hf_attn):
    # HF structure: block.attention.self.{query,key,value}, block.attention.output.dense
    self_attn = _get_any(hf_attn, "self")
    out_proj = _get_any(hf_attn, "output")
    if self_attn is None or out_proj is None:
        # Some very old versions used a flatter structure; try direct names
        self_attn = hf_attn
        out_dense = _get_any(hf_attn, "dense", "proj", "projection")
    else:
        out_dense = _get_any(out_proj, "dense", "proj", "projection")

    # q, k, v are Linear in HF; convert to Conv1x1 in local
    _lin_to_conv1x1(local_attn.q, _get_any(self_attn, "query"))
    k = _get_any(self_attn, "key")
    v = _get_any(self_attn, "value")
    # pack k and v into kv
    kv_w = torch.cat([k.weight.data, v.weight.data], dim=0).view(local_attn.kv.weight.shape)
    kv_b = torch.cat([k.bias.data, v.bias.data], dim=0)
    local_attn.kv.weight.data.copy_(kv_w)
    local_attn.kv.bias.data.copy_(kv_b)

    # spatial reduction conv and its norm (if present)
    if hasattr(local_attn, "sr") and local_attn.sr is not None:
        sr_hf = _get_any(self_attn, "sr")
        if sr_hf is None:
            raise RuntimeError("HF attention missing sr for sr_ratio>1")
        local_attn.sr.weight.data.copy_(sr_hf.weight.data)
        local_attn.sr.bias.data.copy_(sr_hf.bias.data)
        sr_ln = _get_any(self_attn, "layer_norm", "norm", "sr_layer_norm")
        if sr_ln is None:
            raise RuntimeError("HF attention missing sr layer_norm")
        _copy_ln_1d_to_ln2d(local_attn.norm, sr_ln)

    # output projection: Linear -> Conv1x1
    if out_dense is None:
        raise RuntimeError("HF attention missing output dense/proj")
    _lin_to_conv1x1(local_attn.proj, out_dense)

def _copy_mlp(local_mlp, hf_mlp):
    # HF: dense1 (Linear), dwconv (module with .dwconv Conv2d), dense2 (Linear)
    _lin_to_conv1x1(local_mlp.fc1, hf_mlp.dense1)
    local_mlp.dwconv.weight.data.copy_(hf_mlp.dwconv.dwconv.weight.data)
    local_mlp.dwconv.bias.data.copy_(hf_mlp.dwconv.dwconv.bias.data)
    _lin_to_conv1x1(local_mlp.fc2, hf_mlp.dense2)

def _copy_block(local_blk, hf_blk):
    # norms: HF uses layer_norm_1 and layer_norm_2 (1D LN on channels)
    ln1 = _get_any(hf_blk, "layer_norm_1", "layer_norm1", "norm1", "ln1")
    ln2 = _get_any(hf_blk, "layer_norm_2", "layer_norm2", "norm2", "ln2")
    if ln1 is None or ln2 is None:
        raise RuntimeError("HF block missing layer norms")
    _copy_ln_1d_to_ln2d(local_blk.norm1, ln1)
    _copy_ln_1d_to_ln2d(local_blk.norm2, ln2)

    # attention
    _copy_attention(local_blk.attn, hf_blk.attention)
    # mlp
    _copy_mlp(local_blk.mlp, hf_blk.mlp)

def load_hf_encoder_to_local(variant: str, hf_id: str) -> dict:
    hf = SegformerModel.from_pretrained(hf_id)
    enc = _get_any(hf, "encoder", "segformer", "segformer.encoder")
    if hasattr(hf, "segformer") and enc is None:
        enc = hf.segformer.encoder

    local = build_local_mit(variant)

    # patch embeddings (4 stages)
    for i in range(4):
        _copy_patch_embed(local.patch_embeds[i], enc.patch_embeddings[i])

    # blocks per stage
    for i in range(4):
        local_stage = local.stages[i]
        hf_blocks = _get_any(enc, "block", "layer", "layers")[i]
        assert len(local_stage) == len(hf_blocks), f"Depth mismatch at stage {i}: local={len(local_stage)} hf={len(hf_blocks)}"
        for j in range(len(local_stage)):
            _copy_block(local_stage[j], hf_blocks[j])

    return local.state_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True, help="e.g. nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--variant", default="mit_b2", help="mit_b0..mit_b5")
    ap.add_argument("--out", default="checkpoints/mit_b2_from_hf.pth")
    args = ap.parse_args()

    sd = load_hf_encoder_to_local(args.variant, args.hf_id)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(sd, args.out)
    print(f"Saved local MiT weights converted from HF to {args.out}")

if __name__ == "__main__":
    main()

