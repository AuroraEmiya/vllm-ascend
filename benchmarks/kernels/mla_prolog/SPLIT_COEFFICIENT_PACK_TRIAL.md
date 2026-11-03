# Decode split coefficient packing trial

This is round8 of the additional ten-round decode budget. Its exact control
is round7 published commit `7ada452c33c5748fe141a786250c2b30345b637f`, tree
`d4e46e97626b579482f71d0582b7ddffb9540330`. The snapshot
`baseline_serial_7ada452c.py` has SHA256
`1030da1293222b46d337840a496acc87d0bae152c532007121368da507096d56`.
Both classes use the same rebuilt V binary. Norm mode7 remains selected for
B>=32. Split DMA overlap remains limited to32<=B<=64 in both classes; B128
keeps the shared-input-buffer route. No rollback is counted as a new gain.

Round7 split/RoPE full-profile times were7.611/9.661/12.792us at B32/64/128.
The control broadcasts each of four32-element sin/cos halves into the same
`f0[0:64]` scratch window. Each reuse needs a barrier around its producer and
consumer. The round8 hypothesis replaces those repeated scratch overwrites
with four disjoint coefficient windows, allowing independent operations to
share a barrier. It retains eight Adds, four Mul, one Sub and one Add, with
the same per-element operation order and no FMA. Dynamic PIPE_V barriers per
tile decrease from12 to6. The measured split stage is the saving's upper
bound; no individual barrier duration is assumed.

New split mode8 is selected only for B>=32. After both GatherMask operations
and their existing PIPE_V, the original input in `f0` is dead. Mode8 packs:

| FP32 scratch region | Contents |
| --- | --- |
| `f0[0:64]` | Two copies of `cos[0:32]` |
| `f0[64:128]` | Two copies of `cos[32:64]` |
| `f0[128:192]` | Two copies of `sin[0:32]` |
| `f0[192:256]` | Two copies of `sin[32:64]` |

Each64-element coefficient block is reused across repeats with the existing
`src1RepStride=0`. N=1 uses mask32/repeat1; other legal tile sizes2/4/8/16/32
use mask64 and headCount/2 repeats. The public wrapper still accepts only
power-of-two head counts. The only legal odd tail is1; unsupported3/5/31
head counts are not added by this trial. CPU addressing and FP32 checks cover
all legal tile sizes and explicitly reject that unsupported odd domain.

| Phase | Producer / consumer dependence | Required ordering |
| --- | --- | --- |
| Input Cast to GatherMask | `f0` input writes -> both GatherMask reads | Existing PIPE_V |
| GatherMask to coefficient packing | Both reads of the original `f0` -> new `f0[0:256]` writes | Existing post-GatherMask PIPE_V; no early reuse |
| Pack to cos/sine multiplication | Eight independent Adds -> coefficient reads | One PIPE_V after all four disjoint windows |
| Cos products to in-place sine products | Both cos Mul read original `f3` halves -> sine Mul overwrite those halves | One PIPE_V after both cos Mul |
| Sine products to final sums | Both sine Mul write `f3` -> Sub/Add consume them | One PIPE_V after both sine Mul |
| Final sums to BF16 output Cast | Sub and Add write disjoint `f4` halves -> Cast reads them | One PIPE_V after both sums |

The two cos products write disjoint `f4` halves. The two sine products write
disjoint `f3` halves after both original-half readers finish. Sub and Add
write disjoint `f4` halves. No barrier is removed across a true RAW/WAR
dependency. The legacy mode1/6 branch retains its exact original sequence.

All input/output hard events remain. Mode8 uses the mode6 DMA route and
final MTE3_S drain at32<=B<=64; outside that interval it retains mode1's
shared-b0 wait and output completion behavior. Loop-entry V_MTE2 and
MTE3_MTE2 protect reused input/output storage. MTE2_V precedes input Cast,
and V_MTE3 precedes both output stores. Host inference and tiling recognize
mode8 as split; mode7 remains merged norm. No norm buffer or gamma logic changes.

UB remains59,424 bytes, with no new GM allocation. Public BF16 matrix routes,
weight layouts, preparation/refresh, output strides and continuous outputs,
cache mutation, fixed15 cases and precision thresholds are unchanged. B<32
stays on split mode1. Rebuild only the own `x_mla_vector` package; the Torch
adapter ABI is unchanged. No round8 A3 result is claimed by this handoff.

CPU evidence includes a FP32 reference with a separate rounding step for
every Mul/Sub/Add and a dependency model with each barrier independently
removed as a negative control. Sync-audit and flow-analyzer raw findings,
including their expanded-helper variants, are retained in project output.
Source/CPU reasoning supplements these candidates; it is not compiled-device
proof. SDK headers are not scanned on Windows.

Keep or reject concrete B>=32 ranges from20 AB/BA pairs of10 complete
forwards per Graph and full execution profiles, after all15 correctness and
cache/state checks pass. Graph latency is the primary metric. Matrix-stage
C/V sums remain separate and fused MIX internals remain unknown. The parent
owns GitHub publication, Docker Git synchronization, all A3 work and email.

```bash
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan \
  --comparison split-coefficient-pack --mode paired --case-id decode-b128

python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-coefficient-pack \
  --mode verify --case-id decode-b32 --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-coefficient-pack \
  --mode verify --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-coefficient-pack \
  --mode paired --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-coefficient-pack \
  --mode profile --implementation candidate --case-id CASE_ID \
  --source-commit PUBLISHED_SHA --out NEW.json
# Repeat profile with --implementation control for exact round7.

python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/validate.py --case-id CASE_ID \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
```
