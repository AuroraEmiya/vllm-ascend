# Decode merged-down public GMM trial

This is round 3 of the additional ten-round decode budget. Round 2's
bmm/contiguous route failed: B2/B16/B128 layout tasks took approximately
12.966/17.410/11.003 us, outweighing the lower bmm cost. This candidate restores
the round-1 TransposeBatchMatMul head route. Removing the failed round-2 path
is a rollback, not a new performance gain. The direct control is pinned
round 1 (`592d3963e3ef59fc147ef88ab0f114f35b731776`), not round 2.

The only new optimization hypothesis changes the merged-down projection for
the existing ND branch (B>=16 and B divisible by 16):

```python
torch_npu.npu_grouped_matmul(
    [x], [self._matrix_nd[0].t()],
    split_item=0, group_type=-1, output_dtype=torch.bfloat16,
)[0]
```

The existing prepared transposed-ND storage supplies the same logical RHS
through its existing transpose view. There is no new prepared weight, cast,
concatenation, grouping tensor or copy in forward. Preparation/refresh and
their stable-address contract remain unchanged. NZ small/tail batches, all up
projections, all head projections, Ascend C V, cache semantics, BF16 matrix
boundaries and the original precision thresholds stay as in round 1.
The matmul spelling retained from round 1 has no established speedup claim.
No native rebuild is required.

The round-1 full trace identified merged-down as the largest individual stage
at B128: 25.903 us (round 2: 25.821 us). This tests a distinct public grouped
matrix scheduling route, rather than another spelling of the same MatMulV2.
The [official public API](https://raw.githubusercontent.com/Ascend/op-plugin/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)
documents A3 BF16, independent matrix groups with `group_type=-1`, absent
group-list data and transposed weights. Installed-runtime schema, Graph
support and actual kernel routing still require the recorded device tests.
No MatMulV3 route or speedup is assumed.

The prediction is a lower first-projection cost and a repeatable complete-Graph
improvement for the affected ND cases. In the task-sum model, removing that
entire projection would give only about 1.44-1.45x at B16/B128; this is an upper
bound, not a complete-Graph performance result or a promise of 1.5x. Reject on
any ABI/precision/Graph/cache-update failure or if matrix/conversion overhead
does not improve the complete Graph against round 1. Retain all added tasks
and classify the real task types, including opaque MIX internals.

The round-1 control snapshot and frozen 15-case manifest remain hash-pinned.
No new V3 shape is considered implemented. Run from the native checkout with
the existing extension/custom OPP and a new output path for each attempt.
Publication and remote execution belong to the parent agent.

```bash
# CPU-only pinned plan.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan \
  --comparison gmm-down --mode micro --case-id decode-b16

# First affected runtime case; then all 15 fixed correctness/Graph/state cases.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison gmm-down \
  --mode verify --case-id decode-b16 --source-commit PUBLISHED_SHA \
  --out RESULTS/smoke-b16.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison gmm-down \
  --mode verify --source-commit PUBLISHED_SHA --out RESULTS/control-candidate-all15.json

# Same BF16 activation and same physical weight view, only the first projection.
# For NZ cases both methods keep the unchanged control route, explicitly marked.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison gmm-down \
  --mode micro --case-id decode-b16 --source-commit PUBLISHED_SHA \
  --out RESULTS/down-micro-b16.json

# Same-condition round-1 versus current complete Graph, 20 AB/BA groups,
# ten complete forwards per Graph. Select all required cases as appropriate.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison gmm-down \
  --mode paired --case-id decode-b16 --source-commit PUBLISHED_SHA \
  --out RESULTS/control-candidate-b16.json

# Unchanged original fused V1 comparison and full profiler entry points.
python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out RESULTS/fused-verify15.json
python3 benchmarks/kernels/mla_prolog/validate.py --case-id decode-b16 \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out RESULTS/fused-paired-b16.json
python3 benchmarks/kernels/mla_prolog/validate.py --profile serial --case-id decode-b16 \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out RESULTS/profile-b16.json
```

Use the generic project-local analyzer to preserve every execution task. Public
MatMul/GMM stages count as C; all other execution tasks count as V. The physical
split inside any MIX task remains unknown. Task C/V sums are diagnostic and
must not replace complete synchronized NPUGraph Event latency. Historical
complete projection/head trials require their original candidate wrapper;
their CPU plans and isolated microbenchmarks remain available explicitly.
