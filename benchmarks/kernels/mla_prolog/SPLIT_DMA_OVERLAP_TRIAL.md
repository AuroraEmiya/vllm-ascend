# Decode split DMA overlap trial

This is round6 of the additional ten-round decode budget. Its fixed control is
the retained round5 wrapper at published commit
`5e35db3be11162a6edcb06e131d8dbaae7a2948e`, tree
`015fbe4527b58b3b0e84ee5e40df424a0280f39f`. The exact snapshot is
`baseline_serial_5e35db3b.py`, SHA256
`ff46e1fa24ed70794a191aaee14bdb68fa5512bffab0f09eedbe46ea9b254b42`.
Both classes load the same rebuilt V package. The control keeps split mode1;
the candidate selects mode6 only for B>=32. Both keep round5 norm mode5 for
B>=32 and mode4 below32. No earlier rejected path is used as this control.

Round5 full profiles measured split/RoPE7.491/9.802/12.846us at B32/64/128.
The existing split copies packed Q into `b0`, writes Q from `b0`, then waits
for MTE3 before loading RoPE into that same buffer. The single round6
hypothesis removes that storage dependency: load RoPE into the already
allocated `b3` while Q output still reads `b0`. This allows Q output to
overlap RoPE input and vector work. It adds no UB or GM allocation and no
preparation or refresh work. The total split stage is the upper bound on any
potential saving; the trace does not expose a measured substage wait cost.

| Producer / buffer | Consumer | Ordering in mode6 |
| --- | --- | --- |
| Q input DMA writes `b0` | Q output DMA reads `b0` | Existing `MTE2_MTE3` event |
| Q output reads `b0` | RoPE input DMA writes `b3` | Distinct buffers; the old shared-b0 wait is removed only here |
| RoPE/sin/cos input DMA writes `b3`/`b1`/`b2` | FP32 input casts read them | Existing `MTE2_V` event |
| Input cast reads `b3` into `f0` | Later BF16 output cast writes `b3` | Existing intervening `PIPE_V` barriers; the input is dead before output reuse |
| FP32 RoPE writes `f4`, then BF16 cast writes `b3` | Two output DMA reads of `b3` | Existing `V_MTE3` event |
| Previous unit's vector reads/writes | Next unit's input DMA | Existing `V_MTE2` event at loop entry |
| Previous unit's Q/RoPE output DMA reads `b0`/`b3` | Next unit's input DMA overwrites either buffer | Existing `MTE3_MTE2` event at loop entry |
| Last Q/RoPE output DMA | Kernel exits its split processing | New mode6-only `MTE3_S` drain |

The maximum Q tile uses4096 BF16 values in `b0`; RoPE uses2048 in `b3`.
The FP32 buffers and all numerical operations after the initial cast are
unchanged. The two RoPE stores retain the original even/odd-to-half layout.
All matrix GM boundaries remain BF16; the down/up matrix routes, head route,
continuous output/cache contracts, original15 cases and thresholds are fixed.
B31/B32/B33 protect the static threshold, including a core's repeated units.

The event reasoning follows the official description of
[queue ordering and SetFlag/WaitFlag](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html).
The sync-audit and flow-analyzer outputs are retained in the project output
directory, including every raw scanner candidate. A mechanical expansion of
the local `Sync<EVENT>` helper is additional inspection evidence, not a
compiled-device verification. SDK headers are not scanned on Windows.

Keep or reject B>=32 from the same-condition20 AB/BA pairs of10 complete
forwards per Graph, plus full execution profiles. A frozen correctness or
state-update failure rejects the path. B<32 retains mode1. No A3 gain is
claimed by this local handoff, and the formal V3 shape domain is not added.

Rebuild/install only the own `x_mla_vector` package for mode6 host inference
and dispatch. The Torch adapter ABI is unchanged. Record the loaded V binary
hashes and confirm real matrix dtypes and all execution task types with msprof.
The parent owns publication, Docker Git synchronization and all A3 work.

```bash
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan \
  --comparison split-dma-overlap --mode paired --case-id decode-b128

python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-dma-overlap \
  --mode verify --case-id decode-b32 --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-dma-overlap \
  --mode verify --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-dma-overlap \
  --mode paired --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison split-dma-overlap \
  --mode profile --implementation candidate --case-id CASE_ID \
  --source-commit PUBLISHED_SHA --out NEW.json
# Repeat the profile with --implementation control for the pinned round5 path.

python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/validate.py --case-id CASE_ID \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
```

Use the existing generic full-task analyzer. Matrix-stage C/V sums remain
separate from Graph Event latency; retain mixed-task internal C/V as unknown.
