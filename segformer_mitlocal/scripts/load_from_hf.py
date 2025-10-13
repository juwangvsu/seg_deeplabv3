import argparse
import torch
from transformers import SegformerModel
from mit import MixVisionTransformer, MIT_CONFIGS

def build_local_mit(variant: str):
    if variant not in MIT_CONFIGS:
        raise ValueError(f"Unknown MiT variant '{variant}'. Options: {list(MIT_CONFIGS.keys())}")
    return MixVisionTransformer(**MIT_CONFIGS[variant])

def _copy_patch_embed(local, hf):
    # local: OverlapPatchEmbed with .proj, .norm.ln
    # hf: SegformerOverlapPatchEmbeddings with .proj (or .projection) and .layer_norm
    if hasattr(hf, "proj"):
        local.proj.weight.data.copy_(hf.proj.weight.data)
        local.proj.bias.data.copy_(hf.proj.bias.data)
    elif hasattr(hf, "projection"):
        local.proj.weight.data.copy_(hf.projection.weight.data)
        local.proj.bias.data.copy_(hf.projection.bias.data)
    else:
        raise RuntimeError("HF patch embed missing proj/projection")
    # layer norm
    ln = getattr(hf, "layer_norm", None) or getattr(hf, "norm", None)
    if ln is None:
        raise RuntimeError("HF patch embed missing layer_norm/norm")
    local.norm.ln.weight.data.copy_(ln.weight.data)
    local.norm.ln.bias.data.copy_(ln.bias.data)

def _combine_kv_from_hf(attn_local, attn_hf, C):
    # attn_hf has query, key, value as Conv2d( C->C )
    # local has q (C->C) and kv (C->2C)
    attn_local.q.weight.data.copy_(attn_hf.query.weight.data)
    attn_local.q.bias.data.copy_(attn_hf.query.bias.data)
    # combine k and v along out_channels
    kv_w = torch.cat([attn_hf.key.weight.data, attn_hf.value.weight.data], dim=0)
    kv_b = torch.cat([attn_hf.key.bias.data, attn_hf.value.bias.data], dim=0)
    attn_local.kv.weight.data.copy_(kv_w)
    attn_local.kv.bias.data.copy_(kv_b)
    # sr conv + norm (if present)
    if hasattr(attn_local, "sr") and attn_local.sr is not None:
        assert hasattr(attn_hf, "sr"), "HF attention missing sr for sr_ratio>1"
        attn_local.sr.weight.data.copy_(attn_hf.sr.weight.data)
        attn_local.sr.bias.data.copy_(attn_hf.sr.bias.data)
        # norm
        ln = getattr(attn_hf, "layer_norm", None) or getattr(attn_hf, "norm", None)
        if ln is None:
            raise RuntimeError("HF attn sr missing norm")
        attn_local.norm.ln.weight.data.copy_(ln.weight.data)
        attn_local.norm.ln.bias.data.copy_(ln.bias.data)

    # output projection
    attn_local.proj.weight.data.copy_(attn_hf.proj.weight.data)
    attn_local.proj.bias.data.copy_(attn_hf.proj.bias.data)

def _copy_block(local_blk, hf_blk):
    # norms
    ln1 = getattr(hf_blk, "layer_norm1", None) or getattr(hf_blk, "norm1", None)
    ln2 = getattr(hf_blk, "layer_norm2", None) or getattr(hf_blk, "norm2", None)
    if ln1 is None or ln2 is None:
        raise RuntimeError("HF block missing layer norms")
    local_blk.norm1.ln.weight.data.copy_(ln1.weight.data)
    local_blk.norm1.ln.bias.data.copy_(ln1.bias.data)
    local_blk.norm2.ln.weight.data.copy_(ln2.weight.data)
    local_blk.norm2.ln.bias.data.copy_(ln2.bias.data)

    # attention
    _combine_kv_from_hf(local_blk.attn, hf_blk.attention, C=None)

    # mlp
    mlp_hf = hf_blk.mlp
    local_blk.mlp.fc1.weight.data.copy_(mlp_hf.dense1.weight.data)
    local_blk.mlp.fc1.bias.data.copy_(mlp_hf.dense1.bias.data)
    local_blk.mlp.dwconv.weight.data.copy_(mlp_hf.dwconv.weight.data)
    local_blk.mlp.dwconv.bias.data.copy_(mlp_hf.dwconv.bias.data)
    local_blk.mlp.fc2.weight.data.copy_(mlp_hf.dense2.weight.data)
    local_blk.mlp.fc2.bias.data.copy_(mlp_hf.dense2.bias.data)

def load_hf_encoder_to_local(variant: str, hf_id: str) -> dict:
    hf = SegformerModel.from_pretrained(hf_id)
    enc = hf.encoder
    local = build_local_mit(variant)

    # patch embeddings
    for i in range(4):
        _copy_patch_embed(local.patch_embeds[i], enc.patch_embeddings[i])

    # stages / blocks
    for i in range(4):
        local_stage = local.stages[i]
        hf_blocks = enc.block[i]
        assert len(local_stage) == len(hf_blocks), f"Depth mismatch at stage {i}"
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
    osdir = "/".join(args.out.split("/")[:-1])
    if osdir:
        import os
        os.makedirs(osdir, exist_ok=True)
    torch.save(sd, args.out)
    print(f"Saved local MiT weights converted from HF to {args.out}")

if __name__ == "__main__":
    main()
