# HC_Pre A2 expansion test tools

This directory prepares the original 24 HC_Pre cases for an independent A2
measurement epoch. It contains code and minimal execution configuration only.
It contains no measured timings, remote resource records or credentials.

The fused implementation is inherited unchanged from parent commit
`1b8b0c398e96dd3f8f3408aacbb61085ae43fcfe`. The serial implementation is the
original supplied AscendC source, with only the existing Graph tiling-cache
adaptation recorded in `source-config.json` and `patches/`. Three device kernel
files remain unchanged. `model.py` is a CPU reference and seeded input sampler.

`case-config.json` is a minimal execution projection of the original manifest:
24 identical cases, with 8 decode and 16 prefill labels. Its source-manifest
SHA256 and source CSV digests preserve lineage. Inputs are seeded synthetic
tensors matching CPU-template shapes; they are not captured model values.
The template's v2 label remains unresolved against the fixed source's v3
external-pre-mix, five-input and four-output function.

Run the build helper inside the task-owned Docker environment after obtaining
this exact commit from the authorized GitHub repository. Use a fresh task tree
and empty build directories. The helper takes the actual sync commit as an
argument; no source commit is inferred from branch names. Actual hardware
identity comes from the unmodified official `get_npu_arch.py` tool. A real
architecture record, CANN installation and matching Torch ABI are required
before compilation. Vendor installation and actual loaded-binary mappings
remain separate runtime checks.

Before every device action, refresh per-card health, processes, HBM and task
ownership. Resource collection never reads operator source or proxy contents.
The standalone build helper records only the new fixed-tree build inputs and
products; it neither connects by SSH nor examines old remote source trees.

Small-T eager and Graph output checks cover T4 and T16 with the explicit
scalars hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6 and norm_eps=1e-20.
Performance collection requires the current output evidence, same inputs and
10 paired Graph/msprof groups. Formal precision acceptance remains deferred;
finite outputs or source inspection do not establish equivalence. New A2
measurements must not replace historical A3 measurements.

Original notices are retained. `LICENSE.CANN` accompanies files that declare
CANN Open Software License Agreement Version 2.0. Other original files retain
their supplied provenance without a newly invented license attribution. The
parent repository's license continues to apply to its existing contents.

The five prepared runtime tools passed independent CPU review before
publication. Targeted Ruff checks use the parent configuration. The official
architecture tool and archived serial source retain their original bytes;
local lint exceptions preserve those identities. Full repository formatting
was attempted but its actionlint environment installation did not complete.
No full repository CI pass or NPU result is claimed by this code bundle.
