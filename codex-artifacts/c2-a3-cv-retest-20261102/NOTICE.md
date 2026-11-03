This is a source/configuration-only review candidate for D19 C2 A3 full60
profiling. It is not a result package or a claim of numerical acceptance.

Operator source provenance: user-provided compressor_c2_a3_retest_v2.tar.gz,
SHA256 73a40269854a781ffca0b9e7a27ede0d91897984aae93b3ca988844e765fdec6.
The original package is immutable. Source file hashes and input-generator
source line ranges are recorded in source-manifest.json.

The source operator and official profiling assets retain their original
copyright notices. LICENSE.CANN is copied without modification from the
package's ops-transformer license, CANN Open Software License Agreement 2.0.

Official runner/collector files in official/ are unmodified references. The
stateful runner is a separate explicit extension: reset/poison precede each
warmup, setup may reuse target names, and every raw task is mapped exactly once.
It does not pretend the original name-only collector accepted this scenario.

Only the Python runtime model's required SoC parameter is adapted. Mathematical
C/V methods and all AscendC sources are unchanged. No historical DSO, result
JSON/CSV, latency, speedup, environment dump, profiling output or credential is
part of the publish allowlist. Resource guard is the explicitly authorized
unchanged A3 guard; its snapshot is not a lock or host-PID ownership proof.

The original plan's prose reference to numerical cases “55–59” is imprecise.
The frozen per-case configuration is authoritative: 54 zero_kv, 55 equal_scores,
56 random, 57 extreme_scores, 58 zero_norm, 59 small_kv. No fixture was renumbered.

Revision 2 corrects two independently reproduced preparation blockers: forwards
soc through the real prepare constructor call, and revalidates each pilot's
actual raw/topology/execution evidence against the current task build before
allowing remaining cases. Revision 1 source and its failed independent review
remain preserved locally and are excluded from publication. The correction
does not change AscendC mathematics, fixtures or historical performance.
