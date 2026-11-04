# Up projection weight format trial

Additional optimization round7 retains round6 norm double buffering and all
other V stages. The same-binary control sets `_use_nz_up=False` and uses the
source ND transpose. The candidate selects an explicitly prepared format29
BF16 up-weight transpose for tokens<=64; larger token counts keep ND.
The packed down/gate projection is unchanged. Both public calls are matmul.

Round6 up tasks take12.5–13.7us at B2–B64. Historical single-projection NZ
micro Graphs take10.1us at B2/B32 and11.4us at B64. These are hypothesis
evidence; acceptance uses current full Graph AB/BA pairs, not old micro times.

Each object owns6,553,600 additional bytes of persistent BF16 weight data.
Initial preparation and `refresh_weights_` occur outside capture. Refresh
creates a temporary format29 weight, copies it to the captured address, and
checks address stability. Recreate the object and recapture after changing
source tensor identities. Enable `torch.npu.config.allow_internal_format`
before NPU construction. No input-dependent preparation occurs outside
forward; activations, matrix weights and outputs remain BF16.

Run the original15 cases including FP64 reference, eager, Graph, source
weight/input/gamma changes and restoration, unchanged inputs, and stable
prepared addresses. No epsilon or precision threshold changes. Full Graph
timing uses20 AB/BA pairs with10 complete forwards per Graph. Full msprof
must account for all execution tasks and actual matrix route/dtype/layout.
Reject the selected range on a correctness/update failure, activation cast,
or complete Graph regression. No native V rebuild is required.

```bash
python3 benchmarks/kernels/text_gated_residual/matrix_trial.py \
  --up-format-control --work-root /home/s30083169/dev-text-gated-residual \
  --source-commit PUBLISHED_SHA --out control-vs-candidate-all15.json
```
