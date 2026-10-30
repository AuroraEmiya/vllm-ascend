# QLI old DSL baseline retest on CANN 9.2

The user requested adapting the supplied old QLI baseline package to CANN 9.2. The runner defaults to `--cann-mode native92`: it reads the actual toolkit metadata and uses the installed DSL's native ABI resolver. It does not relabel the toolkit as 9.3 or modify the frozen fused implementation. The original `--cann-mode compat93` setup remains available explicitly.

Frozen `sources/baseline_b_dsl.py` SHA256: `6222a4732d3b187dc9f6cd3a5b6f868a730772a2f34d7aee587b3f1449e1b308`.

Both implementations use the frozen 12-case manifest and full tensor input hashes. Prefill keeps q=128; decode/prefill and candidate off/on are separate groups. The old API uses `layout_kv='PA_BBND'`.

A measurement records ten adjacent 256MiB ArgMax/synchronize/full graph replay/synchronize groups. Each sample sums every device task in the complete graph; the case statistic is their median. Preserve all raw profiler CSVs, failures and unstable samples. CPU preflight is not an NPU measurement or a correctness acceptance.

Use a fresh directory under the task's `outputs/qli-old-retest-20261030` for every attempt. The user-supplied frozen ZIP is required separately. No credentials, host connection settings, or historical performance replacements are included here.
