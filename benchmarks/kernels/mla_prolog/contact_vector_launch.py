# SPDX-License-Identifier: Apache-2.0
"""Run a frozen benchmark with compiler workers matching the task CPU quota.

Usage: python contact_vector_launch.py /path/to/contact_vector_bench.py [args]
This launcher changes preparation concurrency; it does not change the operator.
"""
import os
from pathlib import Path
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: contact_vector_launch.py BENCHMARK [arguments]")
    script = Path(sys.argv.pop(1)).resolve()
    os.cpu_count = lambda: 4
    sys.path.insert(0, str(script.parent))
    sys.argv[0] = str(script)
    print("CONTACT_STATIC_COMPILER_WORKER_CAP=4", flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
