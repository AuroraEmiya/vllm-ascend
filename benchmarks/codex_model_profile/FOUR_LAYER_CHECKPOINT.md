# Four-layer operator profile checkpoint

This tool creates a new checkpoint directory for operator-share diagnosis. The
source stays read-only. It retains decoder layers 0–3 with their original tensor
names and quantization, KDA `[1,2,3]`, DSA/full attention `[4]`, and the original
`attn_res_block_size=12`. All other configuration fields remain unchanged. It
rejects a source whose first four layer types differ.

```bash
python benchmarks/codex_model_profile/prepare_four_layer_checkpoint.py \
  --source /weights \
  --output /home/s30083169/perf-pr20878-pack-a3/weights-four-layer \
  --index quant_model_weights.safetensors.index.json
```

Run inside the task container with its installed torch/safetensors. Output must
not exist or be inside the source. The tool scans all shard headers, then loads
only selected tensors with `safe_open`. Each output shard targets 512 MiB; a
single larger tensor remains intact, so memory is bounded by selected shard/tensor
size plus serializer overhead. It rewrites real safetensors payloads, including
mixed source shards, and verifies every selected tensor's content checksum.

The original index basename is retained. An identical
`model.safetensors.index.json` is also written because vLLM's default HF loader
uses that canonical name to filter files. Both indexes reference only newly
written shards and contain an exact `metadata.total_size`. A completed
`four_layer_manifest.json` records source metadata hashes, all source shard
stat/header fingerprints, retained layers, tensor shapes/dtypes/checksums and
output checksums. Source payloads excluded by the layer filter are never loaded
or hashed. A failure leaves `PREPARING.json`; do not serve that output.

Non-decoder tensors, including embedding, final norm, output attention-residual
norm/projection, lm_head and vision/projector tensors, are retained. Small
top-level tokenizer/config/code resources are copied (64 MiB/file limit; no
pickle/weight/archive copying). Use the source tokenizer path explicitly if it
has larger or nested resources. Serve with the same `--language-model-only`
configuration as the original test. Quantization keys for excluded decoder
layers are removed; global metadata and all retained keys/values are preserved.

First verify the server's effective text layer count is 4, its DSA layer is 4,
and the target pack kernel actually appears in the trace. For loading validation,
fixed vLLM 4ba787 supports
`--model-loader-extra-config '{"enable_weights_track":true}'`. This checks retained
quantized parameters rather than silently relying on the default disabled check.
Keep the original batch/context/parallel settings for a directly interpretable
kernel trace; use a short collection window rather than a throughput matrix.
Four-layer operator shares do not establish the shares or throughput of the
93-layer model; residual block fill and layer mix also differ.

CPU fixture verification (NumPy backend is only for these synthetic fixtures;
real BF16 checkpoint preparation uses the default PyTorch backend):

```text
python -m unittest discover -s benchmarks/codex_model_profile -p test_prepare_four_layer_checkpoint.py
```
