# HC_Pre A3 expansion test tools

This directory prepares the original 24 HC_Pre cases for an independent A3
measurement epoch on authorized host 80.48.33.118. It contains code and minimal
execution configuration, with no timing, resource observation or credentials.
The task root is `/home/s30083169/perf-serial-graph/hc-a3-20261102`.

The fused implementation is inherited unchanged from parent source commit
`1b8b0c398e96dd3f8f3408aacbb61085ae43fcfe`. The serial implementation is the
supplied AscendC source, with only the existing Graph tiling-cache adaptation
recorded in `source-config.json` and `patches/`. All three device kernel files,
the official architecture tool, the seeded CPU sampler and 24-case execution
configuration retain their previously published bytes. This A3 bundle adapts
execution/evidence handling; operator mathematics are unchanged.

`case-config.json` projects the original manifest: 8 decode and 16 prefill cases.
Its source-manifest and CSV digests preserve lineage. Seeded synthetic inputs
match CPU-template shapes; they are not runtime model values. The template v2
label remains unresolved against the source external-pre-mix v3 function,
with five inputs and four outputs.

Source synchronization uses the authorized GitHub repository and an actual
fixed commit. No source is transferred by SSH/SCP/SFTP. Build inside the owned
Docker environment using new empty build directories. The unmodified official
`get_npu_arch.py` must report the observed full `Ascend910_9362`, arch `2201`,
short `Ascend910_93`. Serial CMake takes the full SoC; the fused build takes
`--soc=ascend910_93`. The builder requires actual CANN >=9.1 and Torch C++11
ABI=1. Source identity, compilation, task-local vendor installation and actual
process mappings are separate evidence steps, not assumed successes.

The A3 resource guard uses the actual NPU/chip/logic/physical mapping, including
both chips per NPU. Every device action requires fresh health, HBM, process,
disk and host namespace ownership checks. Missing evidence fails closed.
The guard itself only observes resources; it does not scan operator sources or
read proxies. The outer executor verifies namespace and host PID/container
ownership, including the running compute process. A single observed PID alone
does not prove ownership. Stop specifically intercepted operations; never
automatically retry or change channels to bypass security.

Privileged task containers are explicitly authorized. They expose all host
devices despite an explicit device list. `ASCEND_RT_VISIBLE_DEVICES` selects
the runtime chip; the runner records actual framework device count/current
device below a fresh gate and requires one visible logical device 0 before
tensor allocation. This is runtime selection, not physical device isolation.

Small-T eager and Graph checks cover T4/T16 and all four real outputs, with
hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6 and norm_eps=1e-20. The independent
ACLNN adapter passes mult, iterations, hc_eps, norm_eps in the actual API order.
Collection requires current output evidence and identical seeded inputs, then
10 complete Graph/msprof groups per implementation. Warmup and preparation
remain separate from target tasks. Formal precision acceptance stays deferred;
finite values are not equivalence. New A3 measurements do not overwrite old A3
rows or historical A2 evidence.

Original notices are retained. `LICENSE.CANN` accompanies files that declare
CANN Open Software License Agreement Version 2.0. Other original files retain
their provenance without an invented blanket license. The parent repository
license continues to apply to its existing contents. Local Ruff exceptions
preserve official and archived source bytes. No full repository CI, compiled
operator, NPU output or performance pass is claimed by this code bundle.

For the completed build at `8db8c0af1efe96667367a5fb08d90ee8d74fc0e4`,
`finalize_hc_a3_fusion.py` recovers metadata only after the known CPack directory
hashing error. It preserves the original FAILED record and log, verifies the
original builder, source tree and matching package copies, then writes a separate
derived record. The original build return code is inferred from the fixed
builder control flow; it was not recorded as a raw field. Recovery performs no
compilation, installation or NPU work. Record its helper commit separately from
the original build commit. The corrected builder enumerates regular packages
under `build_out` and records future build return codes explicitly.
