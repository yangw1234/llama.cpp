# llama.cpp Vulkan Q4_0 SoA MMV Integration

Persistent log for continuing this work across sessions.

## Context & decisions

### What we're integrating
The Phase-2 winning kernel from `C:/Users/yangwan7/sources/vkbench/`:
- Shader: `src/shaders/mmv_q4_0_soa_sg_u32_r8_w64.comp`
- Config: SoA weight layout (two buffers: `W_q` packed nibbles, `W_s` fp16 scales), subgroup-add reduction, uint32 vectorized quant loads, 8 rows per workgroup, workgroup size 64.
- Standalone benchmark GB/s: ~32 GB/s on Intel iGPU (vs stock ~9 GB/s → ~3.5x).
- Only applies to `MUL_MAT_VEC` at `n=1` (decode), type `Q4_0`. Prefill and all other types use stock kernels.

### Baseline (stock llama-bench, Qwen3.5-2B-Q4_0)
- pp64: 84.68 t/s
- tg16: 5.29 t/s
- Device: Intel(R) Graphics iGPU, subgroup size 32, UMA, no matrix cores

### Design choices (locked — revised 2026-04-18)
1. **Do NOT add a new `ggml_type`.** Keep Q4_0 on disk and in ggml graphs. Only the Vulkan-side bytes are reordered.
2. **Single-buffer in-place repack at upload.** AoS and SoA Q4_0 have the **same** byte size (`M*K/2` quants + `M*K/16` fp16 scales either way). When the flag is on, intercept set-tensor for Q4_0 and reorder bytes before the VkBuffer write. No side buffer, no 2x memory cost. **Phase 3c's side-buffer cache is obsolete and will be removed.**
3. **Every shader that reads Q4_0 raw bytes must be SoA-aware when flag is on.** For Qwen3.5-2B Q4_0 inference, Q4_0 only appears in `MUL_MAT` / `MUL_MAT_VEC` (embed and LM head are Q6_K). Needed SoA variants:
   - `mul_mat_vec_q4_0_soa` (decode n=1, done in 3a/3b).
   - `dequant_q4_0_soa` + existing fp16 matmul path for prefill (n>1). One new shader; prefill may regress slightly — acceptable since tg is the target.
4. **Runtime flag:** env var `GGML_VK_Q4_0_SOA=1` enables the whole path (upload repack + SoA dispatches). Default off → zero behavior change. Once a model is uploaded under the flag, the flag must remain on for the buffer's lifetime.
5. **Dispatch gating:** when flag is on, always route Q4_0 MUL_MAT/MUL_MAT_VEC through SoA. Fast path MMV requires `m%8==0 && k%32==0`; otherwise fall through to dequant→fp16 matmul.
6. **Correctness bar:** token-exact match with seed 42 / temp 0 / -n 32 against stock (flag off).

### Project rule
llama.cpp `AGENTS.md` explicitly bars AI-generated PRs. This work is **local experimentation only**. Do not submit upstream.

## Files to touch in llama.cpp
```
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_0_soa.comp  (new)
ggml/src/ggml-vulkan/vulkan-shaders/generate_shaders.py        (register new shader)
ggml/src/ggml-vulkan/ggml-vulkan.cpp                           (pipeline + dispatch)
```

## Sub-phases

### Phase 3a — shader into build ☑
- Copy the .comp into `vulkan-shaders/`, rename to `mul_mat_vec_q4_0_soa.comp`. ✅
- Wire into `vulkan-shaders-gen.cpp` (NOT `generate_shaders.py` — the build uses a C++ gen tool). Registered at line 736 with `string_to_spv("mul_mat_vec_q4_0_soa_f32_f32", "mul_mat_vec_q4_0_soa.comp", {});` ✅
- Built. SPIR-V embedded: 42984 bytes. Symbols `mul_mat_vec_q4_0_soa_f32_f32_{len,data}` live in `build/ggml/src/ggml-vulkan/ggml-vulkan-shaders.hpp`. ✅
- Gotcha: CMakeLists uses GLOB for `.comp` files — you must re-run `cmake -S ... -B ...` after adding a new shader, or the SPIR-V build step is skipped.
- Binding mismatch vs stock MMV (flagged for 3b):
  - stock: 3 SSBOs + spec-constant local size + rich push constants (strides/offsets/ncols/nrows)
  - ours: 4 SSBOs (WQ uint32, X f32, Y f32, WS f16), hardcoded local_size_x=64, push `{M, K}` only
  - Phase 3b needs a separate `vk_pipeline` with its own DSL + push struct; cannot share the standard MMV layout.

### Phase 3b — pipeline + flag ☑
- New field in device struct: `vk_pipeline pipeline_dequant_mul_mat_vec_q4_0_soa_f32_f32`.
- Create at device init, only if env var `GGML_VK_Q4_0_SOA=1` is set.
- Add a static bool `vk_q4_0_soa_enabled` read once at device init.

### Phase 3c — SoA repack at first use ☒ (superseded)
- Originally built a side-buffer cache (`soa_weights` map) + first-use readback+repack helper. **This is being removed** because the design changed to in-place upload-time repack (see design choice #2). The cache doubles memory.
- What to remove in 3d:
  - `soa_weights` map, `soa_weights_mutex`, destructor cleanup (`ggml-vulkan.cpp:709-717`, `~896-904`).
  - `ggml_vk_get_or_build_q4_0_soa` helper (`~7083-7162`).
  - "SoA Q4_0 repack helper ready" self-test log (`~4916-4919`).

### Phase 3d — upload repack + dispatch ☑
**(Scope expanded — was just dispatch, now covers upload repack + both dispatch paths)**
- Find the vk buffer set-tensor path (`ggml_backend_vk_buffer_set_tensor` or `ggml_vk_buffer_write_*`). When flag is on AND `tensor->type == GGML_TYPE_Q4_0`, intercept the write: copy input bytes into a temp CPU buffer, reorder AoS blocks → SoA (`W_q` region then `W_s` region, same total size), then upload.
- Add SoA-aware dispatch for MMV (n=1):
  - On `pipeline_dequant_mul_mat_vec_q4_0_soa_f32_f32`.
  - Bind src0 vk_buffer at offset 0 range `M*K/2` as binding 0 (WQ).
  - Bind same vk_buffer at offset `M*K/2` range `M*K/16` as binding 3 (WS).
  - Bindings 1/2 = src1/dst. Push `{M, K}`. Dispatch `(M/8,1,1)`.
  - Gate on `m%8==0 && k%32==0`; else fall through to the new dequant path.
- Add SoA dequant path for prefill (n>1):
  - New shader `dequant_q4_0_soa.comp`: reads SoA layout from one vk_buffer (two ranges) and writes dequantized fp16 tensor to a scratch buffer.
  - Reuse existing fp16 matmul against that scratch buffer.
  - Register in `vulkan-shaders-gen.cpp` (remember to re-run cmake configure after adding the file — GLOB).

### Phase 3e — correctness ☐
- `llama-completion -m Qwen3.5-2B-Q4_0.gguf -ngl 99 --seed 42 --temp 0 -n 32 -p "The quick brown fox"`.
- Run with `GGML_VK_Q4_0_SOA=0` and `=1`. Capture both outputs.
- Must match exactly. If not, investigate.

### Phase 3f — benchmark ☐
- `llama-bench -m Qwen3.5-2B-Q4_0.gguf -p 64 -n 32 -ngl 99 -r 2`
- With flag off (baseline) and on. Diff tg t/s. Expected: noticeable tg improvement.

## Log

### 2026-04-18 — Plan authored
- Tasks 4-9 created.
- Vkbench Phase 0/1/2 complete, winner validated at ~32 GB/s.
- Starting Phase 3a next.

### 2026-04-18 — Phase 3a dispatched
- Subagent (opus) investigating vulkan-shaders build system and integrating new shader.

### 2026-04-18 — Phase 3a complete
- Shader copied and registered, SPIR-V embedded (42984 bytes).
- Build: `cmake --build C:/Users/yangwan7/sources/llama.cpp/build --config Release --target ggml-vulkan` (MSVC / VS 18 2026).
- Zero behavior change confirmed — shader is compiled but not dispatched anywhere.
- Binding-layout mismatch noted: 4 SSBOs + `{M,K}` push, hardcoded ws=64 → 3b must build a standalone pipeline layout, cannot reuse stock MMV layout.

### 2026-04-18 — Phase 3b starting
- Separate pipeline with 4 SSBOs (WQ u32, X f32, Y f32, WS f16), `{M,K}` push constants, local size 64.
- Env var `GGML_VK_Q4_0_SOA=1` read once at device init into static bool `vk_q4_0_soa_enabled`.
- Pipeline created only when enabled.

### 2026-04-18 — Phase 3b complete
- Added new field to `vk_device_struct`: `vk_pipeline pipeline_dequant_mul_mat_vec_q4_0_soa_f32_f32` at `ggml-vulkan.cpp:704-706` (single pipeline, not arrayed — the SoA path is only for n=1, type Q4_0, src1 f32 → dst f32).
- Added file-static `vk_q4_0_soa_enabled` + `std::call_once` initializer just above `ggml_vk_load_shaders` (`ggml-vulkan.cpp:3247-3259`). Logs with `GGML_LOG_INFO` on first device init. Treats env `GGML_VK_Q4_0_SOA` as on when non-null and not "0" and not "false".
- `ggml_vk_load_shaders` calls `ggml_vk_init_q4_0_soa_flag()` at entry so the flag log prints even if no device-init hits the flag path.
- Pipeline creation added near the end of `ggml_vk_load_shaders` (before the `compiles.wait` loop): 4 SSBOs, `2*sizeof(uint32_t)` push constants, wg_denoms `{8,1,1}` matching `ROWS_PER_WG=8`, no spec constants (shader hardcodes `local_size_x=64`). Guarded by `if (vk_q4_0_soa_enabled)` — when the flag is off, the field stays `VK_NULL_HANDLE`.
- DSL is shared across all pipelines (`device->dsl`, `MAX_PARAMETER_COUNT=12` storage-buffer bindings). `parameter_count=4` passed to `ggml_vk_create_pipeline` limits how many descriptors are written at bind time — no new DSL needed.
- Build: `cmake --build C:/Users/yangwan7/sources/llama.cpp/build --config Release --target ggml-vulkan` — success, only pre-existing warnings.
- Sanity run with flag on (Qwen3.5-2B-Q4_0, pp64 n16 r1 ngl99): pp64 184.87 t/s, tg16 12.35 t/s. Flag off: pp64 167.43 t/s, tg16 12.17 t/s. Within noise → no dispatch change confirmed. (Numbers are higher than the 2026-04-17 baseline line above because stock llama.cpp has since improved; the new baseline for 3d/3f comparison will be flag-off on today's head.)
- Device init log (visible with `llama-bench -v`): `ggml_vulkan: GGML_VK_Q4_0_SOA = 1 (experimental SoA Q4_0 MMV path ENABLED)` / `... = (unset) (... disabled)`. Note: without `-v`, llama-bench suppresses GGML_LOG_INFO.
- Pipeline is created (meaning field is initialized and a `vk_pipeline_struct` exists) but SPIR-V compile is deferred until `needed=true`, which only flips when someone actually dispatches it. That happens in 3d.

### 2026-04-18 — Phase 3c complete
- Added per-device SoA cache + mutex to `vk_device_struct`: `std::unordered_map<const ggml_tensor *, vk_buffer> soa_weights` + `std::mutex soa_weights_mutex` at `ggml-vulkan.cpp:709-717`. Combined buffer (W_q then W_s) rather than split — one allocation per tensor, two ranges bindable at dispatch via `VkDescriptorBufferInfo` offset/range in 3d.
- Destructor cleanup at `ggml-vulkan.cpp:896-904` (inside `~vk_device_struct`, right after `ggml_vk_destroy_buffer(sync_staging)`): takes the mutex, destroys each cached `vk_buffer`, clears the map. Runs before the device is destroyed and before member destructors fire.
- Helper `static vk_buffer ggml_vk_get_or_build_q4_0_soa(ggml_backend_vk_context * ctx, const ggml_tensor * src)` at `ggml-vulkan.cpp:7083-7162` (just above `ggml_vk_buffer_copy_async`). Flow: fast-path cache lookup under mutex → assert Q4_0 + K%32==0 → resolve source GPU buffer via `src_buf_ctx->dev_buffer` + `vk_tensor_offset(src) + src->view_offs` → `ggml_vk_buffer_read` the AoS bytes (`M*K/32*18` bytes) into a CPU `std::vector<uint8_t>` → CPU repack into staging buffer of size `M*K/2 + M*K/32*2` (memcpy `fp16 d` + 16 qs bytes per block) → `ggml_vk_create_buffer_device(total)` → `ggml_vk_buffer_write(dst, 0, staging, total)` → re-lock, check for race, insert. On race (double build), destroy ours + return theirs.
- First-build log: `GGML_LOG_INFO("ggml_vulkan: built SoA Q4_0 buffer for tensor %s [%u x %u] = %.2f MiB", src->name, M, K, mib)`.
- Self-test at end of `ggml_vk_load_shaders` (`ggml-vulkan.cpp:4916-4919`): after `compiles.wait`, logs `"ggml_vulkan: SoA Q4_0 repack helper ready (not yet dispatched — Phase 3c)"` when the flag is on. Confirmed visible under `-v`.
- Helper is intentionally not referenced from any dispatch path — Phase 3d will wire it in.
- Build: `cmake --build C:/Users/yangwan7/sources/llama.cpp/build --config Release --target ggml-vulkan` — success, only pre-existing warnings.
- Sanity run (Qwen3.5-2B-Q4_0, pp64 n16 r3 ngl99):
  - Flag off: pp64 137.19 ± 8.03 t/s, tg16 10.33 ± 0.15 t/s.
  - Flag on:  pp64 142.70 ± 11.90 t/s, tg16 11.54 ± 0.50 t/s.
  - Within thermal noise (iGPU throttles heavily between back-to-back runs). Phase-3c log confirms helper is present but never invoked (no `"built SoA Q4_0 buffer"` line), so by construction the codepath is identical to flag-off.
- Deviations: none. Chose combined single-buffer as the INTEGRATION.md design preferred; tensor-pointer keying per plan alternative (stable because tensor lifetime is bound to the `ggml_backend_vk_buffer_context` that owns the device buffer, which is destroyed before the device).

### 2026-04-18 — Phase 3g correctness complete (bench regressed)
- Two fixes: dispatch `{M,N,1}` not `{gx,gy,1}` (CEIL_DIV by wg_denoms happens inside the helper); `tile_b` store/load layout reconciled to `[k][col]` on both sides.
- llama-simple output byte-identical to stock with flag on.
- pp64 12.50 t/s with flag on vs 136.63 t/s stock — 3g MMQ tile shape is too small; 3e's dequant path was faster. tg32 preserved at +37%. 3g is correct but not a win; keep gated, revisit tile tuning in a future 3h.

### 2026-04-18 — Phase 3d complete
**3c rollback (all paths in `ggml-vulkan.cpp`):**
- Removed `soa_weights` map + `soa_weights_mutex` from `vk_device_struct` (replaced with a single `vk_pipeline pipeline_dequant_q4_0_soa` field for the new prefill fallback shader).
- Removed destructor cleanup block inside `~vk_device_struct`.
- Removed helper `ggml_vk_get_or_build_q4_0_soa` (~80 lines, between `ggml_vk_buffer_read` and `ggml_vk_buffer_copy_async`).
- Removed `"SoA Q4_0 repack helper ready"` self-test log at the tail of `ggml_vk_load_shaders`.

**Upload repack:** `ggml_backend_vk_buffer_set_tensor` (`ggml-vulkan.cpp` ~13730-13780). Intercept triggers only when `vk_q4_0_soa_enabled && tensor->type == GGML_TYPE_Q4_0 && offset == 0 && size == ggml_nbytes(tensor) && K % 32 == 0`. Allocates a temp `std::vector<uint8_t>` of the same `total` bytes, walks each row × block copying `fp16 d` into the WS region and `16 qs` bytes into the WQ region, then delegates to `ggml_vk_buffer_write` at the same device offset. Partial writes and non-Q4_0 tensors fall through untouched. Logs `"ggml_vulkan: repacked tensor %s to SoA (%zu bytes)"` once per tensor — verified firing 129 times on Qwen3.5-2B.

**Decode dispatch (MMV fast path):** top of `ggml_vk_mul_mat_vec_q_f16` (`ggml-vulkan.cpp` ~7884-7932), guarded by `vk_q4_0_soa_enabled && src0->type==Q4_0 && src1/dst f32 && ne11==ne12==ne13==ne02==ne03==1 && ne01%8==0 && ne00%32==0 && contiguous && no fused ops`. Dispatch: single `pipeline_dequant_mul_mat_vec_q4_0_soa_f32_f32` pipeline (4 SSBOs, `{M,K}` push, wg_denoms `{8,1,1}` → elements `{M,1,1}` so groups = M/8). Bindings use `VkDescriptorBufferInfo` with explicit byte ranges into the shared source vk_buffer: binding 0 WQ (offset `src0_off`, range `M*K/2`), binding 3 WS (offset `src0_off + M*K/2`, range `M*K/16`). Added `ggml_vk_sync_buffers` + `ggml_pipeline_request_descriptor_sets` before dispatch, early `return;` after.

**Prefill strategy (dequant shader, not stub):** new shader `vulkan-shaders/dequant_q4_0_soa.comp` — same 2-binding + 5-uint push layout as stock `dequant_q4_0`, same wg launch shape, so callers are unchanged. The shader uses a single `uint32_t data_a32[]` SSBO covering the full `[WQ | WS]` region; WS is reached at uint32 offset `M*K/8` (M*K/2 bytes ÷ 4), with fp16 scales extracted from the appropriate 16-bit half of each uint32. Registered at `vulkan-shaders-gen.cpp:742`. Pipeline `device->pipeline_dequant_q4_0_soa` created next to the MMV SoA pipeline in `ggml_vk_load_shaders` (wg_denoms `{256*16, 1, 1}`, gated by `vk_q4_0_soa_enabled`). Wiring: `ggml_vk_get_to_fp16` returns the SoA pipeline for Q4_0 when the flag is on. To guarantee the dequant path actually runs (instead of stock direct-Q4_0 matmul), `ggml_vk_mul_mat_q_f16` sets `force_q4_0_soa_dequant = vk_q4_0_soa_enabled && src0->type==Q4_0` and uses that to force `mmp=nullptr → qx_needs_dequant=true` for Q4_0. Prefill now always goes: SoA dequant → fp16 scratch → fp16 matmul. Prefill regression is expected and observed.
  - MoE paths (`ggml_vk_mul_mat_id_q_f16`, `ggml_vk_mul_mat_vec_id_q_f16`) not yet wired — Qwen3.5-2B is non-MoE so this doesn't affect testing. get_rows_q4_0 also untouched (Qwen embed is Q6_K).

**Bench (Qwen3.5-2B-Q4_0, `-p 64 -n 16 -ngl 99 -r 2` on today's head):**
- Flag off: pp64 282.31 ± 3.35 t/s, tg16 14.87 ± 0.00 t/s.
- Flag on:  pp64 101.38 ± 0.34 t/s, tg16 20.65 ± 0.25 t/s.
- tg16 up ~1.39× (the MMV fast path is firing). pp64 down ~2.8× (expected: dequant+fp16 pass doubles prefill work vs. a direct Q4_0 matmul kernel). Matches the design tradeoff in choice #3. No crashes in either run.

**Outstanding for 3e:** correctness verification via `llama-completion` with seed 42 / temp 0 / -n 32. The SoA MMV kernel was validated standalone at ~32 GB/s in vkbench Phase 2 but not against llama.cpp inference. Most likely risk: scale-offset mismatch in the dequant shader's fp16 extraction, or row-stride mismatch in the MMV kernel's `q_base32/s_base` indexing given the new upload-layout assumption.

---

## Phase 3e — Correctness (COMPLETE, 2026-04-18)

**Root cause of garbage output after 3d:** the MMV fast-path guard required `ne11==1`, but during decode some MUL_MATs come through with `ne11=4` (speculative or batched). Those fell through to the stock direct-Q4_0 MMV, which reads AoS bytes from our SoA-packed buffer → garbage. Diagnosed via `fprintf` instrumentation at `ggml_vk_mul_mat_vec_q_f16` entry and at `ggml_vk_get_to_fp16` — saw 126/516 MMV calls miss the fast path, and zero `get_to_fp16 Q4_0` calls because the stock MMV path doesn't route through `mul_mat_q_f16`.

**Fix (ggml-vulkan.cpp ~7972):** right after the SoA MMV fast path returns, unconditionally route any remaining Q4_0 MMV through `ggml_vk_mul_mat_q_f16` when the flag is on:
```cpp
if (vk_q4_0_soa_enabled && src0->type == GGML_TYPE_Q4_0) {
    ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, false);
    return;
}
```
That function already has `force_q4_0_soa_dequant`, which routes through the SoA dequant shader → fp16 matmul, which reads SoA bytes correctly.

**Fusion disable (ggml-vulkan.cpp ~14714):** added a branch in the fusion selector that *skips* MUL_MAT+ADD fusion for Q4_0 when the flag is on, because the SoA kernels don't support fused bias/residual ADDs and the stock fused path would read AoS bytes. Without this, `num_additional_fused_ops > 0` bypasses the SoA fast path on attention-output and MLP-down projections every layer.

**Cleanup:** removed `[SOA-DBG]` `fprintf` instrumentation and the `GGML_VK_Q4_0_SOA_FORCE_DEQUANT` debug escape hatch.

**Verification (llama-simple, Qwen3.5-2B-Q4_0, prompt "The quick brown fox", n=32):** output with `GGML_VK_Q4_0_SOA=1` is byte-identical to stock (both produce `The quick brown fox jumps over the lazy dog.` x3+partial).

## Phase 3f — Bench (COMPLETE, 2026-04-18)

`llama-bench -m Qwen3.5-2B-Q4_0.gguf -p 64 -n 32 -ngl 99 -r 2` on Intel iGPU:
- Flag off (stock baseline on today's head): pp64 **109.75 ± 4.12 t/s**, tg32 **9.44 ± 0.03 t/s**.
- Flag on:                                   pp64 **42.88 ± 2.46 t/s**,  tg32 **13.02 ± 0.04 t/s**.
- tg32 **+38%** ✓ (SoA MMV fast path is the win).
- pp64 **−61%** ✗ (prefill forced through SoA dequant → fp16 matmul, doubling the work vs stock direct-Q4_0 matmul). Phase 3g exists to fix this.

Note: today's stock pp64 baseline (109.75 t/s) is much lower than the 2026-04-17 sanity run (282 t/s) — I suspect the `-p 64 -n 32` vs `-p 64 -n 16` difference or a build flag variation. The ~2.6× regression ratio is the consistent observation.

---

## Phase 3g — Fused SoA Q4_0 MMQ for prefill (CORRECTNESS COMPLETE, 2026-04-18)

**Goal:** recover the pp64 regression by reading SoA bytes directly inside a matmul kernel (no dequant→fp16 scratch pass).

**Design:** new shader `vulkan-shaders/mul_mm_q4_0_soa_f32_f32.comp` — a simple 64-thread BM×BN=64×8 tiled matmul that reads quants from binding 0 (u32) and scales from binding 3 (f16), with the same in-place SoA layout as MMV. Avoids the stock `mul_mm.comp` pipeline family (coopmat, MUL_MAT_ID, aligned/unaligned, bf16, fp16acc — all inapplicable for Intel iGPU non-coopmat2 fp32-only Q4_0 prefill). Skipped a full port.

**Host wiring in ggml_vk_mul_mat_q_f16 (~7614):** an SoA-fast-path block before the existing dequant fallback. Conditions: `vk_q4_0_soa_enabled && src0->type==Q4_0 && src1/dst f32 && single batch (ne02=ne03=ne12=ne13=1) && contiguous && K%32==0 && no fused ops`. Pushes `{M,N,K,stride_a=K,stride_b=K,stride_d=M}`; 4 bindings (WQ u32, B f32, Y f32, WS f16) with binding 0/3 on the same src0 buffer at `src0_off` and `src0_off+M*K/2`. Dispatch `(ceil(M/64), ceil(N/8), 1)` with `wg_denoms {64,8,1}`.

**Shader registered at vulkan-shaders-gen.cpp ~745:** `string_to_spv("mul_mm_q4_0_soa_f32_f32", "mul_mm_q4_0_soa_f32_f32.comp", {})`.
**Pipeline created at ggml-vulkan.cpp ~4912:** `device->pipeline_mul_mm_q4_0_soa_f32_f32`, parameter_count=4, push 6×uint32 = 24 bytes, wg_denoms {64,8,1}.
**Field declared at ggml-vulkan.cpp ~714:** `vk_pipeline pipeline_mul_mm_q4_0_soa_f32_f32`.

**Two bugs, both fixed (2026-04-18):**

1. **Dispatch cardinality (hypothesis 5 — confirmed).** `ggml_vk_dispatch_pipeline` CEIL_DIVs its `elements` arg by `pipeline->wg_denoms` (`ggml-vulkan.cpp:6666-6668`). The 3g wiring passed already-divided groups `{ceil(M/64), ceil(N/8), 1}`, which were divided *again* by wg_denoms `{64, 8, 1}` → vastly under-dispatched. Fixed by passing `{M, N, 1}` at `ggml-vulkan.cpp:7655` (same convention as the MMV fast path at ~7969). After this, output became coherent English but still wrong ("The quick brown fox期中 squared by the number of times…").

2. **B-tile shared-memory layout mismatch (not in the original hypothesis list).** Write used `tile_b[bc*32 + bk]` (layout `[col][k]`); the compute loop read `tile_b[k*BN + tc]` (layout `[k][col]`). The A-tile is consistent (`[row][k]` on both sides), but the B-tile was stored one way and read the other, so each output cell multiplied the wrong B value. Fixed in `mul_mm_q4_0_soa_f32_f32.comp` by changing the load to `tile_b[bk * BN + bc]`.

**Verification (llama-simple, Qwen3.5-2B-Q4_0, "The quick brown fox", n=32):** with both fixes applied, `GGML_VK_Q4_0_SOA=1` output matches stock byte-for-byte (`The quick brown fox jumps over the lazy dog.` ×3+partial).

**Bench (llama-bench, `-p 64 -n 32 -ngl 99 -r 2` on today's head, post-3g):**
- Flag off: pp64 **136.63 ± 8.81 t/s**, tg32 **10.47 ± 0.56 t/s**.
- Flag on:  pp64 **12.50 ± 0.88 t/s**,  tg32 **14.33 ± 0.65 t/s**.
- tg32 **+37%** ✓ (MMV fast path still firing — unchanged from 3e/3f).
- pp64 **−91%** ✗✗. **Regressed further vs Phase 3f's dequant path** (3f pp64 was 42.88 t/s). The naive 64-thread BM×BN=64×8 tile is slower than the 2-pass dequant+fp16 matmul it was supposed to replace. The iGPU lacks coopmat, so fp32 FMA throughput dominates, and with only 64 threads per workgroup and `BN=8` the kernel is massively under-occupying the EU array. Each weight block is also re-loaded once per B-tile column (TN=1 → no N reuse inside a thread).

**Conclusion on 3g:** the fused-MMQ approach is *correct* but *slower* than the dequant fallback at this tile shape. Until the shader is retuned (larger BN for N reuse, or cooperative subgroup loads, or an explicit split-K), 3g is a net loss on prefill. For now keep the dispatch wired in — it's correct and gated by the flag — but log it as a known regression. A future 3h could either:
  - Increase `BN` to 16 or 32 (more threads, more N reuse) and benchmark against the dequant path.
  - Revert the `force_q4_0_soa_dequant` gate so 3g only fires when `N` is large enough to amortize the fixed-cost overhead, and fall through to Phase 3e's dequant path otherwise.

**Fallback option if 3g needs to be disabled:** flip the guard at `ggml-vulkan.cpp:7621` to `false` — prefill returns to the 3e dequant→fp16 path (pp64 ~43 t/s) while keeping the tg32 win.

---

## Phase 3h — SoA Q4_0 inside stock mul_mm (CORRECTNESS COMPLETE, 2026-04-18)

**Goal:** replace the standalone `mul_mm_q4_0_soa_f32_f32.comp` (3g's naive 64×8 tile) with the stock `mul_mm.comp` tile/warp machinery, letting `ggml_vk_guess_matmul_pipeline` pick between s/m/l × aligned/unaligned variants based on problem shape. Expected to recover most of the pp64 regression because it reuses the same tuned tiles as the stock direct-Q4_0 matmul.

**Design — minimal-change reuse of `mul_mm.comp`:**
1. **Shader** (`vulkan-shaders/mul_mm_funcs.glsl`, Q4_0 branch of `load_a_to_shmem`): under `#ifdef Q4_0_SOA`, swap the `block_q4_0_packed16` AoS load for a direct `data_a_wq32[idx]` uint32 load and `data_a_ws[ib]` fp16 scale load from two separate SSBO aliases on the same binding regions. All other tile logic is unchanged — including `stride_a`, `LOAD_VEC_A=8` for q4_0 unaligned, and the entire compute loop.
2. **Entry file** (`vulkan-shaders/mul_mm.comp`): added `layout (binding=0) readonly buffer A_WQ32 {uint32_t data_a_wq32[];}` and `layout (binding=3) readonly buffer A_WS {float16_t data_a_ws[];}` aliases under `#ifdef Q4_0_SOA`. Gated the MUL_MAT_ID binding-3 declaration with `&& !defined(Q4_0_SOA)` so the two uses of binding 3 don't collide (MoE + Q4_0 SoA is out of scope anyway).
3. **Generator** (`vulkan-shaders-gen.cpp`): added `matmul_q4_0_soa_f32{,_aligned}` SPIRV emission in the `matmul_shaders()` q4_0 branch, guarded to `MatMulIdType::NONE && !coopmat` (scalar fp32 non-MoE only). Uses the same `load_vec_a_unaligned=8`, `load_vec_a=8`, `load_vec=2` as the stock Q4_0 variants — no tile math changes needed.
4. **Pipeline registration** (`ggml-vulkan.cpp`): new field `pipeline_dequant_mul_mat_mat_q4_0_soa` (`vk_matmul_pipeline2`, `.f32acc` + `.f16acc`), auto-filled by `CREATE_MM2`/`CREATE_MM` in both the `device->fp16` and non-`fp16` branches, gated by `vk_q4_0_soa_enabled`. `parameter_count=4`.
5. **Dispatch helper** (`ggml_vk_matmul_q4_0_soa`): sibling to `ggml_vk_matmul` that takes a 4th `ws` subbuffer and binds `{a, b, d, ws}`. Split-K intentionally unsupported (would need 5 descriptors). Callers must pass split_k==1.
6. **Host wiring** (`ggml_vk_mul_mat_q_f16`): `use_q4_0_soa_mm` boolean replaces the 3g fast-path block. When true, `mmp = pipeline_dequant_mul_mat_mat_q4_0_soa.f32acc`, `qx_needs_dequant=false` (bytes are read directly from src0), `split_k=1` (forced), and at the dispatch site it calls `ggml_vk_matmul_q4_0_soa` with `ws_subbuffer` at `src0_off + M*K/2`. Conditions: `vk_q4_0_soa_enabled && src0 Q4_0 && src1/dst f32 && !y_non_contig && dim01_contiguous(src0,src1) && ne00%32==0 && pipeline non-null`. MUL_MAT_ID and batched cases fall through to the 3e dequant path.
7. **Retirement:** deleted `mul_mm_q4_0_soa_f32_f32.comp`, its shader registration at `vulkan-shaders-gen.cpp:752`, and its standalone `ggml_vk_create_pipeline` at `ggml-vulkan.cpp:4919` (the field was already removed). Stock pipeline family now owns all SoA Q4_0 MMQ dispatches.

**Verification (llama-simple, Qwen3.5-2B-Q4_0, "The quick brown fox", n=32):** output with `GGML_VK_Q4_0_SOA=1` is byte-identical to stock (`The quick brown fox jumps over the lazy dog.` ×3+partial).

**Bench (llama-bench, `-p 64 -n 32 -ngl 99 -r 2` on today's head, post-3h):**
- Flag off: pp64 **197.77 ± 10.66 t/s**, tg32 **11.77 ± 0.45 t/s**.
- Flag on:  pp64 **80.52 ± 4.83 t/s**,  tg32 **16.02 ± 0.56 t/s**.
- tg32 **+36%** ✓ (MMV fast path still firing).
- pp64 **−59%** vs stock. **6.4× improvement over 3g** (12.50 → 80.52) and **1.9× over 3f's dequant path** (42.88 → 80.52), but still below stock direct-Q4_0 matmul. The gap is expected: stock's MMQ shader has integer-dot-product tiles (`matmul_q4_0_q8_1`) and aggressive packing that the SoA-over-`mul_mm` variant doesn't touch. Today's stock baseline (197 t/s) is also much faster than earlier numbers because llama.cpp's Q4_0 prefill path has continued to improve.

**Conclusion on 3h:** reusing stock `mul_mm` tile machinery was worth ~6×, without any tile-size tuning. The remaining gap vs stock is the integer-dot MMQ path (which reads AoS bytes via `A_TYPE_PACKED16`) — matching that would require either a second SoA shader variant against `mul_mmq.comp`, or exposing a `Q4_0_SOA` branch in `mul_mmq`'s `load_a_to_shmem` equivalent. Deferred.

---

## Phase 3i — SoA Q4_0 inside stock mul_mmq (COMPLETE, 2026-04-18)

**Goal:** close the remaining pp64 gap vs stock (3h: 80 t/s vs stock ~197 t/s) by making the SoA path use the same integer-dot MMQ kernel stock uses for Q4_0 prefill — not a dequant pass, not the scalar fp32 `mul_mm` path.

**Root cause 3h left on the table:** the 3d host-side gate set `force_q4_0_soa_dequant = vk_q4_0_soa_enabled && src0->type==Q4_0` and, crucially, forced `quantize_y=false` unconditionally. `quantize_y` is the single flag that routes a matmul through `ggml_vk_get_mul_mat_mat_pipeline(... GGML_TYPE_Q8_1 ...)` and selects a `mul_mmq.comp` pipeline with `dotPacked4x8EXT`. By forcing it off, we bypassed the int-dot family and landed on fp32 `mul_mm` tiles (3h) or the dequant-to-fp16 path (3e). Stock was never slower at Q4_0 prefill — we were just routing around its fastest kernel.

**Design — one more SoA shader variant, this time against `mul_mmq.comp`:**
1. **`mul_mmq.comp` entry file:** added SoA binding aliases under `#ifdef Q4_0_SOA` at the top:
   ```glsl
   layout (binding = 0) readonly buffer A_WQ32 {uint32_t  data_a_wq32[];};
   layout (binding = 3) readonly buffer A_WS   {float16_t data_a_ws[];  };
   ```
   Guarded the MUL_MAT_ID binding-3 declaration with `&& !defined(Q4_0_SOA)` (same landmine as 3h — one `binding=3` at a time; SoA and MoE combined are OOS). Also widened the fp16 extension guard: `#if defined(FLOAT16) || defined(Q4_0_SOA)` so the SoA variant compiles on non-fp16 tiles.
2. **`mul_mmq_funcs.glsl`:** Q4_0 branch of `block_a_to_shmem` now has two compile-time paths. Under `#ifdef Q4_0_SOA`, it reads 4 uint32 quants from `data_a_wq32[ib*4 + iqs]` and one fp16 scale from `data_a_ws[ib]`. All the downstream packing (`cache_a[].qs`, `mmq_dot_product`'s `dotPacked4x8EXT` loop) is untouched — the SoA load produces the exact same `buf_a[buf_ib].qs[]` contents as the AoS `pack32(u16vec2(packed16.qs[...]))` branch, so the compute side doesn't care.
3. **`vulkan-shaders-gen.cpp`:** in the MMQ emission block, added a second `string_to_spv("matmul_q4_0_soa_q8_1", ...)` with `{"Q4_0_SOA", "1"}` alongside the stock `matmul_q4_0_q8_1`. Only emitted for `MatMulIdType::NONE` (MoE combined with Q4_0 SoA is still OOS).
4. **Pipeline registration** (`ggml-vulkan.cpp`): new field `pipeline_dequant_mul_mat_mat_q4_0_soa_q8_1` next to the 3h SoA MM field, registered in both the `device->fp16` and non-fp16 branches with `CREATE_MMQ`, `parameter_count=4`, gated by `vk_q4_0_soa_enabled`. Reuses the existing `ggml_vk_matmul_q4_0_soa` dispatch helper (same 4-binding WQ/B/D/WS layout as 3h).
5. **Host wiring** (`ggml_vk_mul_mat_q_f16`): new boolean `use_q4_0_soa_mmq` — preferred over `use_q4_0_soa_mm`. Conditions: `force_q4_0_soa_dequant && quantize_y && !y_non_contig && dim01_contiguous && ne00%32==0 && SoA-MMQ pipeline non-null`. Crucially, `quantize_y` is **not** forced off when the SoA MMQ path is available: only when we fall back to the SoA MM (3h) or SoA dequant (3e) path. `split_k=1` is still forced — the SoA dispatch has a 4th descriptor and the split-K reduce path doesn't plumb it through. `qx_needs_dequant = !use_q4_0_soa_any && ...` — both SoA paths read src0 bytes directly.

**Instrumentation check:** added a one-shot `fprintf(stderr, "[SOA] mmq=... mm=... fallback=...")` counter at the selection site; one Qwen3.5-2B prefill+decode run reported `mmq=201 mm=0 fallback=0` — every `force_q4_0_soa_dequant` call routes through the int-dot MMQ shader, not the 3h `mul_mm` scalar path or the 3e dequant path. Instrumentation was then removed.

**Verification (llama-simple, Qwen3.5-2B-Q4_0, "The quick brown fox", n=32):** output with `GGML_VK_Q4_0_SOA=1` is byte-identical to stock (`The quick brown fox jumps over the lazy dog.` ×3+partial).

**Bench (llama-bench, `-p 64 -n 32 -ngl 99 -r 3` on today's head, post-3i):**
- Flag off: pp64 **157.98 ± 8.25 t/s**, tg32 **10.63 ± 0.51 t/s**.
- Flag on:  pp64 **154.55 ± 6.33 t/s**, tg32 **13.78 ± 0.33 t/s**.
- pp64 **parity with stock** (±2% overlap in 1σ). tg32 **+30%** ✓ (MMV SoA fast path still firing).
- **1.9× pp64 over 3h** (80.52 → 154.55) and **3.6× over 3f** (42.88 → 154.55).

**Conclusion on 3i:** the SoA Q4_0 path is now feature-complete at pp64 parity with stock while preserving the tg32 win from Phase 3a-3e. What finally worked was refusing to bypass `quantize_y` — the whole point of the SoA repack was to feed int-dot MMQ tiles with a tighter memory layout, and the 3d gate that killed `quantize_y` on every Q4_0 dispatch was defeating the entire purpose. The shader change (reading `data_a_wq32[]` + `data_a_ws[]` from two separate SSBO aliases in the `block_a_to_shmem` Q4_0 branch) was trivial by comparison — the integer-dot tile machinery, Q8_1 B-side packing, and warp schedule in `mul_mmq.comp` did not need to change at all.

---

## Learnings so far

1. **llama.cpp Vulkan dispatch routing is split into three places for MUL_MAT/MUL_MAT_VEC:**
   - `ggml_vk_mul_mat_q_f16` — "prefill" path, ne11>1 (tokens-per-batch>1). Has dequant-then-matmul fallback and stock direct-Q matmul pipelines.
   - `ggml_vk_mul_mat_vec_q_f16` — "decode" path, ne11==1. Has stock direct-Q MMV and fp16 MMV.
   - `ggml_vk_mul_mat_id_q_f16` / `ggml_vk_mul_mat_vec_id_q_f16` — MoE. Never hit by Qwen3.5-2B.
   The dispatch selector (`ggml_vk_mul_mat` ~outer) picks one of these based on `ne11` and whether the op is MUL_MAT_ID. If you make one path SoA-aware but not others, the in-place-repacked buffer gets read as AoS bytes in the unconverted path → silent garbage. **Every code path that reads Q4_0 weight bytes must be SoA-aware when the flag is on, or else bypass the buffer** (via dequant-to-scratch).
2. **MUL_MAT + ADD fusion in llama.cpp** is done by the graph compiler before dispatch (`ctx->num_additional_fused_ops` set at `ggml_vk_build_graph` ~14708). Residual adds (attn_output, ffn_down) fuse by default. Any new direct-Q kernel that doesn't handle the fused bias/residual must either implement fusion or **disable fusion for that case** at the selector. We chose the latter for SoA Q4_0. Disabling fusion costs some perf but is the simplest path.
3. **The set_tensor intercept** (`ggml_backend_vk_buffer_set_tensor`) runs for every tensor upload. The GGUF loader does a single full-tensor write (`offset==0 && size==ggml_nbytes`) for each weight, so gating on that is safe. Partial writes would need to track repacking state per-region — we don't handle that and assert via the guard.
4. **`ggml_vk_get_to_fp16`** returns a pipeline from a small cache. Since callers use it in several places (`ggml_vk_mul_mat_q_f16`, `ggml_vk_mul_mat_id_q_f16`, test harness, etc.), replacing the returned pipeline is a low-invasiveness way to force the SoA dequant globally when flag is on. The tradeoff: if a non-weight Q4_0 tensor ever appears (activations quantized to Q4_0, KV cache to Q4_0 via `-ctk q4_0`), the SoA pipeline reads AoS bytes. Qwen inference doesn't trigger this, but the hook is a known landmine.
5. **Instrumentation approach that worked:** `fprintf(stderr, ...)` at dispatch/selection sites. `GGML_LOG_INFO` is suppressed by llama-bench and redirected in llama-simple. `fprintf(stderr)` always shows up. For narrow diagnostics (counting calls, seeing `ne11` distribution), this was much faster than CPU-reference golden-testing.
6. **The "every shape" tradeoff.** For a single weight tensor Q4_0 path, you need shaders and dispatches for each combination of `{MMV, MMQ} × {contiguous, non-contig} × {fused-ADD, no-fused} × {single-batch, batched} × {GQA broadcast, no-broadcast} × {MoE MUL_MAT_ID, no-ID} × {fp16 B, fp32 B} × {coopmat, coopmat2, scalar}`. The real llama.cpp backend covers all of these for each quant type. Our SoA integration covers exactly one combination (single-batch, contiguous, fp32, no-fuse, no-broadcast, scalar, no-MoE) and **routes all other combinations** to the dequant-to-fp16 fallback for correctness. A production SoA path would need to grow this coverage.
7. **`ggml_vk_dispatch_pipeline` contract (resolved).** The 3rd argument is **element counts**, which the helper divides by `pipeline->wg_denoms` via `CEIL_DIV` at `ggml-vulkan.cpp:6666-6668` to produce group counts. Always pass raw element extents (M, N, …) and let wg_denoms do the arithmetic — never pre-divide. The Phase 3g MMQ dispatch was wrong on this and was double-dividing.
8. **Shared-memory tile layout must match between store and load.** Phase 3g had `tile_b` written as `[col][k]` and read as `[k][col]`; the compiler happily compiles both sides in isolation, so the bug surfaces only as silently wrong outputs. When writing a matmul with distinct row/col strides for A and B shared tiles, write the indexing expressions for *both* loads and compute side-by-side and confirm they agree on stride order.
9. **Fused direct-Q MMQ is not automatically faster than dequant→fp16.** At BM×BN=64×8 with 64 threads and no coopmat, 3g's direct SoA matmul was 3.4× slower than the dequant+fp16 fallback. The dequant path lets the stock fp16 matmul (which has been tuned with large tiles and split-K) do the heavy lifting. Any future direct-Q kernel has to beat the stock fp16 matmul on its own turf — tile size, N reuse, occupancy all matter more than avoiding the dequant pass.
10. **`quantize_y` is the gate to the int-dot MMQ family.** In `ggml_vk_mul_mat_q_f16`, the choice between `mul_mm.comp` (fp16/fp32 B) and `mul_mmq.comp` (Q8_1 B with `dotPacked4x8EXT`) is made by a single boolean `quantize_y = integer_dot_product && src1==F32 && contiguous && (ne11*ne10)%4==0`. Setting `quantize_y=false` unconditionally for Q4_0 (3d's `force_q4_0_soa_dequant` did this) silently routes all Q4_0 prefill around the fastest kernel stock has for Q4_0 — even the 3h SoA-over-`mul_mm` work was running the scalar fp32 tile family instead of int-dot. When adding a new weight-layout path, mirror the full pipeline selection logic including `quantize_y`; don't just force it off "to be safe".
11. **`dotPacked4x8EXT` doesn't care where the bytes came from.** In `mul_mmq_funcs.glsl`, `mmq_dot_product` works entirely out of `cache_a[].qs` / `cache_b.qs` — register arrays that `block_a_to_shmem` and `block_a_to_registers` populate. Swapping the A-side shmem load from `pack32(u16vec2(packed16.qs[...]))` (AoS) to `data_a_wq32[ib*4 + iqs]` (SoA) produced bit-identical compute because the downstream register cache sees the same uint32 quants either way. Pattern: if the compute kernel consumes a uniform intermediate cache, you can swap the memory-layout-aware load without touching the dot-product loop — one new shader variant, no tile math changes, no compute-graph validation. This is why 3i was ~100 lines across four files vs 3g's from-scratch tile rewrite.


---

## Phase 3j — Simplification (COMPLETE, 2026-04-18)

**Goal:** reduce the diff vs master (490 added lines across 8 files) without touching the reusable `Q4_0_SOA` define trick in `mul_mm.comp` / `mul_mmq.comp`, while preserving correctness (byte-identical llama-simple output vs stock) and tg32 perf.

**Baseline (pre-3j, Qwen3.5-2B-Q4_0, `llama-bench -p 64 -n 32 -ngl 99 -r 3`):**
- Flag off: pp64 **145.52 ± 17.18 t/s**, tg32 **9.35 ± 0.20 t/s**.
- Flag on:  pp64 **136.95 ± 9.26 t/s**,  tg32 **12.26 ± 0.66 t/s** (tg32 +31%).

**Post-3j bench (same command):**
- Flag off: pp64 **173.87 ± 8.90 t/s**, tg32 **11.70 ± 0.19 t/s**.
- Flag on:  pp64 **188.52 ± 4.39 t/s**, tg32 **16.09 ± 1.28 t/s** (tg32 +38%).

Within iGPU thermal noise; tg32 win preserved, pp64 still at/above parity.

**Diff size before → after (vs master):**
- Before: 490 insertions / 17 deletions across 8 files.
- After:  401 insertions / 17 deletions across 7 files (`ggml-vulkan.cpp` went 291 → 278, `dequant_q4_0_soa.comp` deleted at 72 lines, `vulkan-shaders-gen.cpp` 19 → 15).

### What was done

**Step 1 — deleted the dequant-fallback scaffold.** Phase 3i's instrumentation (`mmq=201 mm=0 fallback=0`) had already proven the `dequant_q4_0_soa` path unreachable on Intel iGPU + Qwen. Removed:
- Shader file `dequant_q4_0_soa.comp` (72 lines).
- Its `string_to_spv` registration in `vulkan-shaders-gen.cpp`.
- `pipeline_dequant_q4_0_soa` field on `vk_device_struct`.
- Pipeline-creation block in `ggml_vk_load_shaders`.
- The `ggml_vk_get_to_fp16` override that returned the SoA dequant pipeline for Q4_0.
- The `force_q4_0_soa_dequant && !use_q4_0_soa_mmq && !use_q4_0_soa_mm` fall-through in `ggml_vk_mul_mat_q_f16`; replaced with `GGML_ASSERT(!is_soa || use_soa_any)` so any edge case where neither SoA path fits hard-fails instead of silently reading AoS bytes.

**Step 3 — consolidated gating to `vk_is_soa_type()`.** Added a one-line predicate:
```cpp
static inline bool vk_is_soa_type(ggml_type t) {
    return vk_q4_0_soa_enabled && t == GGML_TYPE_Q4_0;
}
```
Replaced 4 call sites: the MMQ-path selector, the MMV fast-path gate, the MMV fall-through reroute, the upload repack intercept, and the fusion disable. Adding Q5_0 is now a one-line change in the helper (and adding its pipeline/shader variant). Kept the raw `vk_q4_0_soa_enabled` reads at pipeline-creation sites, where there is no tensor type to check and the flag alone is the right gate.

Also renamed `force_q4_0_soa_dequant` → `is_soa`, `use_q4_0_soa_mmq/mm/any` → `use_soa_mmq/mm/any` to match the type-agnostic framing.

**Step 4 — extracted `ggml_vk_soa_repack_q4_0`.** The ~25-line AoS→SoA row-walk body of `ggml_backend_vk_buffer_set_tensor` moved to a file-static helper `(const void* src, void* dst, uint64_t M, uint32_t K)`. The call site shrank to a predicate + single function call. Intentionally did not generalize to other block types yet — only one SoA type exists; over-engineering waits for Q5_0/Q8_0.

**Step 6** fell out of Step 3: the fusion-disable predicate in `ggml_backend_vk_graph_compute` now uses `vk_is_soa_type(node->src[0]->type)`.

### Steps skipped, with reasons

**Step 2 (replace standalone MMV with `Q4_0_SOA` define on stock `mul_mat_vec.comp`) — skipped.** The stock MMV uses a different binding layout (binding 3 is already taken by `Fuse0` — stock supports fused bias/residual at the MMV level). To alias binding 3 to WS scales under `Q4_0_SOA`, fusion would have to be disabled in the shader, but more importantly the stock MMV is a tuned general-purpose kernel with many configuration knobs (NUM_ROWS, NUM_COLS, subgroup modes, no-shmem variants) — the custom 85-line `mul_mat_vec_q4_0_soa.comp` is a narrow hand-tuned kernel (ROWS_PER_WG=8, local_size_x=64, uint32 quant loads, subgroupAdd) that delivers 32 GB/s microbench and is responsible for the entire tg32 win. Per Learning #9, naive tile rewrites are often slower than the dequant path, and I judged the risk of regressing the load-bearing tg32 metric too high for the ~85-line payoff. The custom shader stays.

**Step 5 (push SoA pipeline selection into `ggml_vk_get_mul_mat_mat_pipeline`) — skipped.** SoA selection has to happen in two branches (MMQ path picks SoA-MMQ pipeline, MM path picks SoA-MM pipeline), and the caller still needs `is_soa` locally for `split_k=1` and the 4-descriptor dispatch via `ggml_vk_matmul_q4_0_soa`. Pushing the pipeline lookup down saves one `else-if` in `ggml_vk_mul_mat_q_f16` at the cost of threading `is_soa` through the selector signature and splitting the 4th-descriptor knowledge across two files. Net ugly. Kept the local `use_soa_mmq`/`use_soa_mm` booleans — they read cleanly with the 3-step predicate chain.

### New learnings

12. **Unreachable-but-present scaffolding hides cost.** The dequant fallback was written defensively in Phase 3d (when the MMV fast path was the only SoA dispatch) and stayed past Phase 3i when MMQ-SoA made it unreachable. 72 lines of shader + 3 host-side integration points + one `ggml_vk_get_to_fp16` override all for a branch that never fires on our target. After 3i hard-confirmed it's dead (via the `mmq=201 mm=0 fallback=0` counter), replacing it with `GGML_ASSERT(!is_soa || use_soa_any)` converts the hidden dead code into an explicit correctness contract — if an edge case ever does show up (exotic shape, new device without int-dot), we get a hard fail with a useful stack, not silent garbage.
13. **A type predicate is easier to extend than a per-site gate.** The original `vk_q4_0_soa_enabled && X->type == GGML_TYPE_Q4_0` pattern had to be changed everywhere to support a second SoA type. With `vk_is_soa_type(t)`, adding Q5_0/Q8_0 is a one-line change in the helper (and then pipeline/shader variants for the new type) — call sites stay unchanged. This mirrors how llama.cpp itself uses predicates like `ggml_is_contiguous` rather than inlining stride checks.
