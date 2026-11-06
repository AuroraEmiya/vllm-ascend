# SPDX-License-Identifier: Apache-2.0
"""Use the installed CANN generator; ship only our operator sources."""
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parents[1]
build=root/'build'
build.mkdir(exist_ok=True)
target=build/'opp-project'
if target.exists():
    raise FileExistsError(f'{target}: use a fresh source/build directory')
ops=[('XMlaNormCache','x_mla_norm_cache'),('XMlaSplitRope','x_mla_split_rope'),
     ('XMlaLayout','x_mla_layout'),('XMlaVector','legacy/x_mla_vector')]
def tensor(name,dtype='bfloat16'):
    return dict(name=name,param_type='required',format=['ND'],type=[dtype])
spec=[]
for name,_ in ops:
    spec.append(dict(op=name,input_desc=[tensor(n) for n in ('x','aux','gamma_q','gamma_kv','sin','cos')]+[tensor('index','int64')],
                     output_desc=[tensor(n) for n in ('y','query_rope','kv_cache','kr_cache')],
                     attr=[dict(name=n,param_type='required',type='int') for n in ('mode','rows','heads','cache_slots')]+
                          [dict(name=n,param_type='required',type='float') for n in ('eps_q','eps_kv')]))
schema=build/'mla-operators.json'
schema.write_text(json.dumps(spec,indent=2)+'\n')
subprocess.run(['msopgen','gen','-i',str(schema),'-op','XMlaNormCache','-c','ai_core-ascend910_93','-f','aclnn','-lan','cpp','-out',str(target)],check=True)
projects=[p.parent for p in target.rglob('CMakePresets.json') if (p.parent/'op_host').is_dir()]
if len(projects)!=1: raise RuntimeError(f'Unexpected generated project roots: {projects}')
project=projects[0]
for sub in ('op_host','op_kernel'):
    folder=project/sub
    # Only replace generated C/C++ files inside this newly created project.
    for p in folder.iterdir():
        if p.is_file() and p.suffix in ('.cpp','.h'): p.unlink()
    for _,relative in ops:
        for p in (root/'csrc/attention/mla_prolog_serial'/relative/sub).iterdir():
            if p.is_file() and p.suffix in ('.cpp','.h'): shutil.copy2(p,folder/p.name)
preset=project/'CMakePresets.json'
data=json.loads(preset.read_text());variables=data['configurePresets'][0]['cacheVariables']
for key,value in [('vendor_name','mlask_transformer'),('ENABLE_TEST','False'),('ASCEND_COMPUTE_UNIT','ascend910_93')]:
    variables[key]['value']=value
preset.write_text(json.dumps(data,indent=2)+'\n')
(build/'opp-project-path.txt').write_text(str(project)+'\n')
print('PREPARED_OWN_MLA_ONLY',project)
