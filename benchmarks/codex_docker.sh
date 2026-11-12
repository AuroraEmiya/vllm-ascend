#!/usr/bin/env bash
# Host-specific defaults may be set after inspecting that server. Never pull implicitly.
set -euo pipefail

DEFAULT_IMAGE=""
DEFAULT_NPU_DEVICES="none"
DEFAULT_NETWORK_MODE="bridge"
WORKSPACE=/home/s30083169
PROJECT_NAME="${PROJECT_NAME:-dev-remote-env}"
PROJECT_DIR="$WORKSPACE/$PROJECT_NAME"
CONTAINER_NAME="${CONTAINER_NAME:-s30083169-$PROJECT_NAME}"
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
NPU_DEVICES="${NPU_DEVICES:-$DEFAULT_NPU_DEVICES}"
NETWORK_MODE="${NETWORK_MODE:-$DEFAULT_NETWORK_MODE}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-16g}"
SHM_SIZE="${SHM_SIZE:-1g}"
PIDS_LIMIT="${PIDS_LIMIT:-512}"

die() { printf '%s\n' "$*" >&2; exit 1; }
exists() { docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; }
owned() {
    local owner role project
    owner=$(docker inspect --format '{{index .Config.Labels "cannbot.owner"}}' "$CONTAINER_NAME")
    role=$(docker inspect --format '{{index .Config.Labels "cannbot.role"}}' "$CONTAINER_NAME")
    project=$(docker inspect --format '{{index .Config.Labels "cannbot.project"}}' "$CONTAINER_NAME")
    [[ "$owner" == s30083169 && "$role" == isolated-development-v4 && "$project" == "$PROJECT_NAME" ]] ||
        die "Refusing to operate on a container not created by this script."
}
running() {
    exists || die "Container absent. Run: bash $WORKSPACE/docker.sh start"
    owned
    [[ $(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME") == true ]] ||
        die "Container stopped. Start it before exec/shell."
}

[[ "$PROJECT_NAME" =~ ^(dev|fix|perf|test|refactor|docs|chore)-[a-z0-9]+(-[a-z0-9]+)*$ ]] ||
    die "PROJECT_NAME must use purpose-name, e.g. dev-add or fix-softmax."
[[ "$CONTAINER_NAME" == "s30083169-$PROJECT_NAME" || "$CONTAINER_NAME" == "s30083169-$PROJECT_NAME-"* ]] ||
    die "CONTAINER_NAME must match s30083169-PROJECT_NAME with an optional suffix."
[[ "$CONTAINER_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die "Invalid container name."
[[ "$NETWORK_MODE" == bridge || "$NETWORK_MODE" == host ]] || die "NETWORK_MODE must be bridge or host."
command -v docker >/dev/null || die "Docker is not installed; do not install it on the host automatically."

action="${1:-start}"
if (( $# )); then shift; fi
case "$action" in
    start)
        [[ -d "$WORKSPACE" && ! -L "$WORKSPACE" ]] || die "Workspace missing or symlinked."
        [[ $(readlink -f -- "$WORKSPACE") == "$WORKSPACE" ]] || die "Unexpected workspace path."
        [[ -n "$IMAGE" ]] || die "Set IMAGE to a locally available image ID."
        image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE") ||
            die "Image is not local. Inspect available images; this script will not pull."
        [[ ! -L "$PROJECT_DIR" ]] || die "Project directory must not be a symlink."
        mkdir -p -- "$PROJECT_DIR"
        [[ $(readlink -f -- "$PROJECT_DIR") == "$PROJECT_DIR" ]] || die "Unexpected project path."
        device_args=()
        group_args=()
        env_args=()
        declare -A device_groups=()
        if [[ "$NPU_DEVICES" != none ]]; then
            [[ "$NPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "NPU_DEVICES must be none or comma-separated physical IDs."
            IFS=, read -r -a ids <<< "$NPU_DEVICES"
            declare -A seen=()
            for id in "${ids[@]}"; do
                [[ -z "${seen[$id]:-}" ]] || die "Duplicate NPU ID: $id"
                seen[$id]=1
                [[ -c /dev/davinci$id ]] || die "Missing /dev/davinci$id"
                device_args+=(--device "/dev/davinci$id:/dev/davinci$id:rw")
                device_groups[$(stat -c %g "/dev/davinci$id")]=1
            done
            for device in davinci_manager devmm_svm hisi_hdc; do
                [[ -c /dev/$device ]] || die "Missing /dev/$device"
                device_args+=(--device "/dev/$device:/dev/$device:rw")
                device_groups[$(stat -c %g "/dev/$device")]=1
            done
            for group_id in "${!device_groups[@]}"; do
                group_args+=(--group-add "$group_id")
            done
            env_args+=(--env "ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICES" --env "ASCEND_VISIBLE_DEVICES=$NPU_DEVICES")
        fi
        mount_args=(--mount "type=bind,src=$PROJECT_DIR,dst=$PROJECT_DIR")
        if [[ -f "$WORKSPACE/proxy.sh" ]]; then
            mount_args+=(--mount "type=bind,src=$WORKSPACE/proxy.sh,dst=$WORKSPACE/proxy.sh,readonly")
        fi
        for directory in /usr/local/Ascend/driver /usr/local/dcmi; do
            if [[ -d "$directory" ]]; then
                mount_args+=(--mount "type=bind,src=$directory,dst=$directory,readonly")
            fi
        done
        if [[ -f /usr/local/bin/npu-smi ]]; then
            mount_args+=(--mount type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi,readonly)
        fi
        config_hash=$(printf '%s\0' isolated-development-v4 "$PROJECT_DIR" "$image_id" "$NPU_DEVICES" "$NETWORK_MODE" "$CPUS" "$MEMORY" "$SHM_SIZE" "$PIDS_LIMIT" "${group_args[@]}" | sha256sum)
        config_hash=${config_hash%% *}
        if exists; then
            owned
            existing_hash=$(docker inspect --format '{{index .Config.Labels "cannbot.config"}}' "$CONTAINER_NAME")
            [[ "$existing_hash" == "$config_hash" ]] ||
                die "Existing container has different settings. Choose another s30083169- name; do not recreate or remove it implicitly."
            if [[ $(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME") != true ]]; then
                docker start "$CONTAINER_NAME"
            else
                printf 'Already running: %s\n' "$CONTAINER_NAME"
            fi
        else
            docker run --detach --pull never --name "$CONTAINER_NAME" \
                --label cannbot.owner=s30083169 \
                --label cannbot.role=isolated-development-v4 \
                --label "cannbot.project=$PROJECT_NAME" \
                --label "cannbot.config=$config_hash" \
                --hostname s30083169-cannbot --init \
                --network "$NETWORK_MODE" --ipc private \
                --cap-drop ALL --security-opt no-new-privileges:true \
                --cpus "$CPUS" --memory "$MEMORY" --memory-swap "$MEMORY" \
                --pids-limit "$PIDS_LIMIT" --shm-size "$SHM_SIZE" \
                --tmpfs /tmp:rw,nosuid,nodev,size=1g \
                --log-driver json-file --log-opt max-size=10m --log-opt max-file=2 \
                --workdir "$PROJECT_DIR" \
                "${mount_args[@]}" "${device_args[@]}" "${group_args[@]}" "${env_args[@]}" \
                --entrypoint /bin/sleep "$image_id" infinity
        fi
        ;;
    exec)
        running
        (( $# )) || die "Usage: bash docker.sh exec COMMAND [ARG ...]"
        exec docker exec --workdir "$PROJECT_DIR" "$CONTAINER_NAME" "$@"
        ;;
    shell)
        running
        exec docker exec --interactive --tty --workdir "$PROJECT_DIR" "$CONTAINER_NAME" /bin/bash
        ;;
    status)
        exists || die "Container absent."
        owned
        docker inspect --format '{{.Name}} status={{.State.Status}} image={{.Image}} privileged={{.HostConfig.Privileged}} network={{.HostConfig.NetworkMode}}' "$CONTAINER_NAME"
        ;;
    stop)
        exists || die "Container absent."
        owned
        docker stop --time 15 "$CONTAINER_NAME"
        ;;
    *)
        die "Usage: bash docker.sh {start|shell|exec COMMAND...|status|stop}"
        ;;
esac
