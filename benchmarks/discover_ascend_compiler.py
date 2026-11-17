#!/usr/bin/env python3
"""Inspect the installed Triton-Ascend compiler source for dump/cache/IR hooks.
This avoids hard-coding one backend version's artifact names.
"""
import inspect
import re


def main():
    import triton.backends.ascend.compiler as compiler
    src = inspect.getsource(compiler)
    keys = [
        r"TRITON_[A-Z0-9_]+", r"ttir", r"ttadapter", r"\.mlir", r"dump",
        r"cache", r"tempfile", r"Path\(", r"write_text", r"triton-adapter",
    ]
    lines = src.splitlines()
    hits = set()
    for i, line in enumerate(lines, 1):
        if any(re.search(k, line, re.I) for k in keys):
            hits.add(i)
    for i in sorted(hits):
        lo, hi = max(1, i - 2), min(len(lines), i + 2)
        print(f"\n--- lines {lo}-{hi} ---")
        for n in range(lo, hi + 1):
            print(f"{n:5d}: {lines[n-1]}")

if __name__ == "__main__":
    main()
