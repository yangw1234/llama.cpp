"""
Parse llama.cpp GGML_VK_PERF_LOGGER output from perf.log and produce two PNGs
(prefill.png, decode.png). Each PNG has two subplots:
  (a) between-layer: avg time per full-attn block, per linear-attn block,
      plus embedding & lm_head bars
  (b) within-layer:  stacked breakdown inside a block (norms, QKV, attn core,
      O-proj, FFN gate+up, FFN down, DeltaNet compute, elementwise)

Architecture assumed for Qwen3.5-2B (hybrid):
  24 decoder blocks = 18 linear-attn (Gated DeltaNet) + 6 full-attn
  model_dim=2048, intermediate=6144, n_q_heads=16, n_kv_heads=2, head_dim=256,
  vocab=~248320
"""
import re
import os
import sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = r"C:\Users\yangwan7\sources\perf_bench.log"
OUT_DIR = r"C:\Users\yangwan7\sources"

# 0-indexed dump selection within the log. llama-bench order:
#   0 = pp warmup, 1 = pp real, 2 = tg warmup, 3..N = tg tokens
PREFILL_DUMP_IDX = 1
DECODE_DUMP_IDX = 3

N_FULL = 6
N_LIN = 18

def parse_dumps(path):
    dumps = []
    cur = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Vulkan Timings:"):
                cur = []
            elif line.startswith("Total time:"):
                if cur is not None:
                    dumps.append(cur)
                cur = None
            elif cur is not None and line.strip():
                cur.append(line)
    return dumps

def parse_dump(lines):
    """Return list of (op_key, count, total_us)."""
    entries = []
    # "NAME...: K x AVG us = TOTAL us" possibly followed by "(GFLOPS/s)"
    rx = re.compile(r"^(.*?):\s+(\d+)\s+x\s+([\d.]+)\s+us\s+=\s+([\d.]+)\s+us")
    for ln in lines:
        m = rx.match(ln)
        if not m:
            continue
        name = m.group(1).strip()
        cnt = int(m.group(2))
        tot = float(m.group(4))
        entries.append((name, cnt, tot))
    return entries

def classify(op_name, count):
    """Map an op entry to (block_type, component). block_type in
    {'full', 'lin', 'both', 'embed', 'lm_head', 'final', 'misc'}.
    component is the within-block sub-bucket."""
    n = op_name

    # LM head
    if "q6_K m=248320" in n:
        return ("lm_head", "lm_head")
    # embeddings
    if n.startswith("GET_ROWS"):
        return ("embed", "embed")
    # final rms norm on whole-hidden
    if "RMS_NORM(2048,1,1,1)" in n or "RMS_NORM(2048,16,1,1)" in n:
        return ("final", "final_norm")

    # Full-attention block pieces
    if n.startswith("FLASH_ATTN_EXT"):
        return ("full", "attn_core")
    # Non-flash attention path: MUL_MAT f16 m=256 k=256 batch=8 (QK and SV)
    if re.search(r"MUL_MAT(?:_VEC)? f16 m=256.*k=256.*batch=8", n):
        return ("full", "attn_core")
    if n.startswith("SOFT_MAX"):
        return ("full", "attn_core")
    # QKV shapes for full-attn: Q=4096, K=V=512 (head_dim=256, 2 KV heads)
    if re.search(r"q4_0 m=4096 .*k=2048", n):
        return ("full", "q_proj")
    if re.search(r"q4_0 m=512 .*k=2048", n):
        return ("full", "kv_proj")  # 12 entries -> K and V combined bucket
    if "RMS_NORM(256,8" in n:       # Q head norm (8 groups? or head dim norm)
        return ("full", "qk_norm")
    if "RMS_NORM(256,2" in n:
        return ("full", "qk_norm")
    if n.startswith("ROPE"):
        return ("full", "rope")

    # Linear-attention block pieces
    if n.startswith("GATED_DELTA_NET"):
        return ("lin", "deltanet_core")
    if n.startswith("SSM_CONV"):
        return ("lin", "ssm_conv")
    if n.startswith("SOFTPLUS"):
        return ("lin", "deltanet_core")
    if n.startswith("L2_NORM"):
        return ("lin", "qk_norm")
    if "RMS_NORM(128,16" in n:
        return ("lin", "deltanet_norm")
    # q5_K m=2048 k=2048 with 18 calls -> DeltaNet output projection
    if "q5_K m=2048" in n and "k=2048" in n:
        return ("lin", "o_proj")
    # q8_0 m=16 k=2048 with 36 calls -> small DeltaNet parameter projections (2 per block)
    if "q8_0 m=16" in n:
        return ("lin", "deltanet_params")

    # FFN (shared by both block types). We can't distinguish which block owns each.
    # gate+up: q4_0 m=6144 k=2048
    if re.search(r"q4_0 m=6144.*k=2048", n):
        return ("both", "ffn_gate_up")
    # down: q4_0 m=2048 k=6144
    if re.search(r"q4_0 m=2048.*k=6144", n):
        return ("both", "ffn_down")
    # q4_1 m=2048 k=6144 x3 -> unusual; bucket as ffn_down variant
    if re.search(r"q4_1 m=2048.*k=6144", n):
        return ("both", "ffn_down")
    # Fused MUL_MAT_ADD forms (decode uses _VEC variant)
    if re.search(r"MUL_MAT_ADD.*q4_0 m=2048.*k=6144", n) or \
       re.search(r"MUL_MAT_ADD.*q4_1 m=2048.*k=6144", n):
        return ("both", "ffn_down")
    if re.search(r"MUL_MAT_ADD.*q4_0 m=2048.*k=2048", n):
        return ("both", "attn_o_proj")
    if re.search(r"q4_0 m=2048.*k=2048", n):
        # 24 calls -> output projection per block (all 24 blocks)
        return ("both", "attn_o_proj")
    # Pre-attn / pre-FFN norms on hidden: RMS_NORM(2048,x,...) for x in {1,2,16,64}
    if re.match(r"RMS_NORM_MUL RMS_NORM\(2048,\d+", n) and "RMS_NORM(2048,1,1,1)" not in n:
        return ("both", "block_norm")

    # Elementwise / misc — bucket under 'both/elementwise'
    for pfx in ("ADD", "MUL", "CPY", "CONT", "CONCAT", "SCALE", "SET_ROWS",
                "SIGMOID", "SILU", "GLU"):
        if n.startswith(pfx):
            return ("both", "elementwise")
    return ("misc", "other")

def bucketize(entries):
    """
    Returns:
      per_block_us:  dict {block_type -> {component -> total_us_for_that_block_type}}
        - full/lin buckets divided by block count
        - 'both' components split proportionally to block count (full:lin = 6:18)
      standalone_us: dict {'embed': us, 'lm_head': us, 'final': us}
    """
    totals = defaultdict(lambda: defaultdict(float))  # totals[block_type][component]
    for name, cnt, tot in entries:
        bt, comp = classify(name, cnt)
        totals[bt][comp] += tot

    # Split 'both' proportionally across full + lin (weight = block count)
    both = totals.get("both", {})
    full = totals.setdefault("full", defaultdict(float))
    lin = totals.setdefault("lin", defaultdict(float))
    w_full = N_FULL / (N_FULL + N_LIN)
    w_lin = N_LIN / (N_FULL + N_LIN)
    for comp, v in both.items():
        full[comp] += v * w_full
        lin[comp] += v * w_lin
    totals["both"] = defaultdict(float)  # emptied

    # Per-block average (divide by number of blocks of that type)
    per_full = {c: v / N_FULL for c, v in full.items()}
    per_lin = {c: v / N_LIN for c, v in lin.items()}
    standalone = {
        "embed": totals.get("embed", {}).get("embed", 0.0),
        "lm_head": totals.get("lm_head", {}).get("lm_head", 0.0),
        "final_norm": totals.get("final", {}).get("final_norm", 0.0),
    }
    return per_full, per_lin, standalone

# Consistent component order + colors for stacked bar
COMPONENTS = [
    ("block_norm",        "#6c8ebf"),  # pre-attn & pre-ffn norms
    ("q_proj",            "#5b8f3a"),
    ("kv_proj",           "#7cb342"),
    ("qk_norm",           "#9ccc65"),
    ("rope",              "#c5e1a5"),
    ("attn_core",         "#d9534f"),
    ("deltanet_core",     "#e37933"),
    ("ssm_conv",          "#f0ad4e"),
    ("deltanet_norm",     "#f9d278"),
    ("deltanet_params",   "#fce29a"),
    ("o_proj",            "#8e44ad"),
    ("attn_o_proj",       "#8e44ad"),
    ("ffn_gate_up",       "#2c7fb8"),
    ("ffn_down",          "#41b6c4"),
    ("elementwise",       "#999999"),
    ("other",             "#bbbbbb"),
]
COMP_LABEL = {
    "block_norm": "RMSNorm (pre-attn+pre-ffn)",
    "q_proj": "Q proj",
    "kv_proj": "K+V proj",
    "qk_norm": "Q/K norm",
    "rope": "RoPE",
    "attn_core": "FlashAttention",
    "deltanet_core": "GatedDeltaNet core",
    "ssm_conv": "SSM conv (1D)",
    "deltanet_norm": "DeltaNet inner norm",
    "deltanet_params": "DeltaNet param projs",
    "o_proj": "Output proj (linear-attn)",
    "attn_o_proj": "Output proj",
    "ffn_gate_up": "FFN gate + up",
    "ffn_down": "FFN down",
    "elementwise": "Elementwise (add/mul/scale/cpy/…)",
    "other": "Other",
}

def plot_phase(per_full, per_lin, standalone, phase, out_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # --- left: between layers ---
    full_total = sum(per_full.values())
    lin_total = sum(per_lin.values())
    labels = [
        f"Linear-attn block\n(avg of {N_LIN})",
        f"Full-attn block\n(avg of {N_FULL})",
        "Embedding",
        "LM head",
        "Final norm",
    ]
    vals = [lin_total, full_total,
            standalone["embed"], standalone["lm_head"], standalone["final_norm"]]
    colors = ["#e37933", "#d9534f", "#6c8ebf", "#8e44ad", "#999999"]
    bars = ax1.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} us",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("GPU time (us)")
    ax1.set_title(f"{phase} — between components\n"
                  f"(per-block averages; standalone bars are whole-model totals)")
    ax1.grid(axis="y", alpha=0.3)

    # --- right: within a layer ---
    # stacked bar: one stack for linear-attn block, one for full-attn block
    stacks = [("Linear-attn block", per_lin), ("Full-attn block", per_full)]
    x = list(range(len(stacks)))
    bottom = [0.0] * len(stacks)
    legend_handles = []
    for comp, color in COMPONENTS:
        heights = []
        for _, per in stacks:
            heights.append(per.get(comp, 0.0))
        if sum(heights) <= 0:
            continue
        bar = ax2.bar([s[0] for s in stacks], heights, bottom=bottom, color=color,
                      label=COMP_LABEL[comp], edgecolor="white", linewidth=0.5)
        # annotate slice if big enough
        for xi, h in enumerate(heights):
            if h > 0 and h / max(sum(per_full.values()), sum(per_lin.values()), 1) > 0.05:
                ax2.text(xi, bottom[xi] + h / 2, f"{COMP_LABEL[comp]}\n{h:.0f}us",
                         ha="center", va="center", fontsize=8, color="black")
        bottom = [b + h for b, h in zip(bottom, heights)]
        legend_handles.append(bar)
    ax2.set_ylabel("GPU time per block (us)")
    ax2.set_title(f"{phase} — within a layer\n(avg time per block, decomposed)")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # sum annotations
    ax2.text(0, bottom[0], f"total {bottom[0]:.0f} us", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
    ax2.text(1, bottom[1], f"total {bottom[1]:.0f} us", ha="center", va="bottom",
             fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Qwen3.5-2B Q4_0 on Vulkan (Intel iGPU)  —  {phase}\n"
        f"24 decoder blocks: 18 Gated-DeltaNet (linear-attn) + 6 full-attn",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

def main():
    dumps = parse_dumps(LOG)
    if len(dumps) <= max(PREFILL_DUMP_IDX, DECODE_DUMP_IDX):
        print(f"only {len(dumps)} dumps found", file=sys.stderr)
        sys.exit(1)

    prefill_entries = parse_dump(dumps[PREFILL_DUMP_IDX])
    decode_entries = parse_dump(dumps[DECODE_DUMP_IDX])

    # summary print for sanity
    def brief(entries):
        return sum(t for _, _, t in entries)
    print(f"prefill total (us): {brief(prefill_entries):.0f}")
    print(f"decode  total (us): {brief(decode_entries):.0f}")

    pf_full, pf_lin, pf_std = bucketize(prefill_entries)
    dc_full, dc_lin, dc_std = bucketize(decode_entries)

    plot_phase(pf_full, pf_lin, pf_std, "Prefill (pp64, post-warmup)",
               os.path.join(OUT_DIR, "prefill_v2.png"))
    plot_phase(dc_full, dc_lin, dc_std, "Decode (tg, 1 token, post-warmup)",
               os.path.join(OUT_DIR, "decode_v2.png"))
    print("wrote prefill.png and decode.png in", OUT_DIR)

if __name__ == "__main__":
    main()
