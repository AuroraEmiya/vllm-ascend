# Source provenance and license

The operator source originates from the user-provided
compressor_c2_a3_retest_v2.tar.gz, SHA256
73a40269854a781ffca0b9e7a27ede0d91897984aae93b3ca988844e765fdec6.
The original package is retained unchanged outside this source tree.

The C/C++/AscendC and CMake files retain their original bytes and notices.
LICENSE.CANN is the unchanged CANN Open Software License Agreement Version 2.0
copied from the source package's ops-transformer license.

The actual model.py runtime adapter retains the previously reviewed explicit
keyword-only SoC argument, including forwarding soc to Prepared. Mathematical
cat/C/V calls are unchanged. Python is outside the AscendC LOC metric.

This source-only revision derives cv60n_routes.json with only the original
37 routes. Every route key, value and type is retained; unused comment/tuning
metadata is removed. It is a derived runtime configuration, not an assertion
that the whole JSON file is byte-identical to the original.

README.md and this notice are source-only documentation. No fixture, performance
shape list, original result, profiling tool, test script, environment dump,
credential, binary or build output forms part of this source-only directory.
