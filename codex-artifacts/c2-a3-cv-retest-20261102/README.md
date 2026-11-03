# Compressor C2 A3 serial operator source

This directory contains the current BF16 C implementation, the reused AscendC V,
their dedicated host code, build files, and the actual Python runtime adapter.
It contains no performance cases, fixtures, test runner, warmup or collector.

`source/operators/compressor_c2_a3_bf16_c_only/model.py` implements the complete
runtime sequence: concatenate Wkv/Wgate on every call, run the new BF16 C, then
call the V-only entry `torch.ops.npu.cv60n_v_run`. It returns the latent tensor
and updates state through the V entry. It does not call the old C prepare/run.
The route file contains only the 37 original runtime route entries.

## Fresh build

Use the actual CANN9.2 toolkit and matching torch/torch_npu C++ ABI in the task
container. The following targets Ascend910_9362 explicitly; the original CMake
defaults are retained and must not be relied upon. OP_ROOT is this directory.
Both build directories must be absent before a fresh build.

```bash
cmake -S "$OP_ROOT/source/dependencies/original_a3/n_impl" \
  -B "$OP_ROOT/source/dependencies/original_a3/n_impl/build" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910_9362 \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$OP_ROOT/source/dependencies/original_a3/n_impl/build" --parallel 4
cmake -S "$OP_ROOT/source/operators/compressor_c2_a3_bf16_c_only" \
  -B "$OP_ROOT/source/operators/compressor_c2_a3_bf16_c_only/build" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910_9362 \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$OP_ROOT/source/operators/compressor_c2_a3_bf16_c_only/build" --parallel 4
```

Expected outputs are `libcv60n.so` and `libc2_a3_bf16_c_only.so` in their respective
build directories. The runtime adapter loads those two newly built libraries;
its `prepare(..., soc="Ascend910_9362")` passes the explicit target to the C++
prepare, which checks the actual SoC. A completed build is not runtime validation.

The unchanged V CMake also compiles `cv60n_gemm.cpp`; register.cpp references that
symbol. Those old C files are retained solely to preserve this build/link closure.
They are not the current S projection. No historical DSO or build output is supplied.

See LICENSE.CANN and NOTICE.md for provenance. Measurement and validation files
are created separately in the server task directory and are not in this source tree.
