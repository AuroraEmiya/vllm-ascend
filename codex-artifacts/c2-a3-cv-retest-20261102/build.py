"""Explicit fresh CMake build. This script never runs a device operation."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
from common import SOC, now, require, sha, task_root, verify_bundle, verify_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task-root', required=True)
    p.add_argument('--soc', required=True, choices=[SOC])
    p.add_argument('--cann-root', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    root = task_root(a.task_root)
    source_sha = verify_sources(root)
    bundle_sha = verify_bundle(root)
    require(1 <= a.jobs <= 8, 'Explicit bounded CPU parallelism required')
    sdk = Path(a.cann_root).resolve()
    require((sdk / 'compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake').is_file(), 'Actual toolkit CMake missing')
    outputs = root / 'build-evidence'
    outputs.mkdir(exist_ok=False)  # never overwrite an existing build identity
    # Pure CPU imports/version/ABI only. Actual SoC is separately verified by prepare().
    import torch
    import torch_npu
    env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
           'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
           'torch_CXX11_ABI': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
           'cann_root': str(sdk), 'soc_explicit_build_target': a.soc,
           'runtime_soc_verified': False, 'source_manifest_sha256': source_sha,
           'bundle_manifest_sha256': bundle_sha,
           'environment': {k: os.environ.get(k) for k in ('ASCEND_HOME_PATH', 'PATH', 'LD_LIBRARY_PATH')}}
    record = {'status': 'BUILDING', 'started_at': now(), 'environment': env, 'commands': [], 'artifacts': []}
    write_json(outputs / 'build-identity.json', record)
    try:
        for rel, name in [('dependencies/original_a3/n_impl', 'libcv60n.so'),
                          ('operators/compressor_c2_a3_bf16_c_only', 'libc2_a3_bf16_c_only.so')]:
            src = root / 'source' / rel
            build = src / 'build'
            require(not build.exists(), 'Build directory must be absent, not recycled: ' + str(build))
            commands = [['cmake', '-S', str(src), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                         '-DSOC_VERSION=' + a.soc, '-DASCEND_CANN_PACKAGE_PATH=' + str(sdk),
                         '-DPython3_EXECUTABLE=' + sys.executable],
                        ['cmake', '--build', str(build), '--parallel', str(a.jobs)]]
            for argv in commands:
                i = len(record['commands'])
                row = {'argv': argv, 'cwd': str(root), 'started_at': now(), 'returncode': None}
                record['commands'].append(row)
                write_json(outputs / 'build-identity.json', record)
                with (outputs / f'command-{i:02d}.log').open('wb') as log:
                    completed = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
                row.update(returncode=completed.returncode, finished_at=now(), log_sha256=sha(outputs / f'command-{i:02d}.log'))
                write_json(outputs / 'build-identity.json', record)
                require(completed.returncode == 0, 'Build failed; retain full build tree and logs')
            require((build / name).is_file(), 'Expected freshly built DSO missing')
            for path in sorted(build.rglob('*')):
                if path.is_file() and (path.suffix in ('.o', '.so', '.a', '.json') or path.name in ('CMakeCache.txt', 'compile_commands.json', 'link.txt', 'flags.make')):
                    record['artifacts'].append({'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
        record['status'] = 'PASS_CPU_BUILD_ONLY'
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        record['finished_at'] = now()
        write_json(outputs / 'build-identity.json', record)


if __name__ == '__main__':
    main()
