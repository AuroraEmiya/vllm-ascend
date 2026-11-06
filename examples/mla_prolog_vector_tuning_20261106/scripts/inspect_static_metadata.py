# SPDX-License-Identifier: Apache-2.0
"""Extract only kernel capability metadata from this run's own static packages."""
import hashlib
import json
from pathlib import Path
import subprocess


def inspect(out):
    out=Path(out).resolve();records=[];packages=[]
    for number,package in enumerate(sorted((out/'fresh-static-cache').rglob('*.run'))):
        target=out/'extracted-static-package'/f'package-{number:03d}'
        target.parent.mkdir(exist_ok=True)
        if target.exists():raise FileExistsError(target)
        with (out/f'static-extract-{number:03d}.log').open('w') as log:
            subprocess.run(['bash',str(package),'--noexec','--extract='+str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
        packages.append({'path':str(package.relative_to(out)),'sha256':hashlib.sha256(package.read_bytes()).hexdigest()})
        for path in sorted(target.rglob('*.json')):
            try:data=json.loads(path.read_text())
            except (ValueError,UnicodeError):continue
            if isinstance(data,dict) and any(k in data for k in ('kernelName','supportSuperKernel','sub_operator_kernel_name')):
                records.append({'path':str(path.relative_to(out)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                'kernelName':data.get('kernelName'),'supportSuperKernel':data.get('supportSuperKernel'),
                                'sub_operator_kernel_name':data.get('sub_operator_kernel_name'),'keys':list(data)})
    result={'packages':packages,'kernels':records}
    (out/'compiled-kernel-metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    return {'packages':len(packages),'kernel_records':len(records)}
