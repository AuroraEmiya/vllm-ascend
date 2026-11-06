#!/usr/bin/env bash
# Build only our MLA operator; no vLLM installation or repository is required.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?source the selected CANN set_env.sh first}"
jobs=${MAX_JOBS:-4}
mkdir -p "$root/build" "$root/install" "$root/outputs"
args=(--pkg --ops=x_mla_vector,x_mla_norm_cache,x_mla_split_rope,x_mla_layout --soc=ascend910_93 --vendor_name=mlask --ccache false "-j$jobs")
if [[ -n ${MLASK_DEPENDENCY_DIR:-} ]]; then
    args+=("--cann_3rd_lib_path=$MLASK_DEPENDENCY_DIR")
fi
(
    cd "$root/csrc"
    bash build.sh "${args[@]}"
)
bash "$root/csrc/build_out/cann-ops-transformer-mlask_linux-aarch64.run" --quiet --install-path="$root/install"
source "$root/scripts/env.sh"
python -c 'import mla_superkernel; print("STANDALONE_MLA_BINDING_READY")'
find "$root/install" "$root/build/binding" -type f \( -name '*.so' -o -name '*.o' \) -exec sha256sum '{}' + > "$root/outputs/build-binaries.sha256"
