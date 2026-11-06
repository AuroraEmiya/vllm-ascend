#!/usr/bin/env bash
# Build only our MLA operator; no vLLM installation or repository is required.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?source the selected CANN set_env.sh first}"
jobs=${MAX_JOBS:-4}
mkdir -p "$root/build" "$root/install" "$root/outputs"
python "$root/scripts/prepare_build.py"
project=$(cat "$root/build/opp-project-path.txt")
cmake -S "$project" -B "$project/build_out" --preset=default \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake"
cmake --build "$project/build_out" --target binary package -j"$jobs"
mapfile -t packages < <(find "$project/build_out" -maxdepth 2 -type f -name '*.run')
test "${#packages[@]}" -eq 1
bash "${packages[0]}" --quiet --install-path="$root/install"
source "$root/scripts/env.sh"
python -c 'import mla_superkernel; print("STANDALONE_MLA_BINDING_READY")'
find "$root/install" -type f \( -name '*.so' -o -name '*.o' \) -exec sha256sum '{}' + > "$root/outputs/build-binaries.sha256"
if test -n "${MLASK_EXISTING_BINDING:-}"; then
    sha256sum "$MLASK_EXISTING_BINDING" >> "$root/outputs/build-binaries.sha256"
else
    find "$root/build/binding" -type f \( -name '*.so' -o -name '*.o' \) -exec sha256sum '{}' + >> "$root/outputs/build-binaries.sha256"
fi
