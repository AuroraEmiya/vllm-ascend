"""Extract only the frozen inputs_for function; never execute historical scripts.

The AST is compiled without rewriting its body. In particular the unused V
kv/scores draws and chunked FP32->BF16 X conversion retain RNG advancement.
"""
import ast
import hashlib
import importlib.util
from types import SimpleNamespace
from common import require

INPUT_NAMES = ('x', 'wkv', 'wgate', 'state_cache', 'metadata', 'norm_weight', 'epsilon', 'max_query_len')
IMMUTABLE = ('x', 'wkv', 'wgate', 'metadata', 'norm_weight')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frozen_function(root, torch):
    source = root / 'frozen/inputs_for.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'inputs_for']
    require(len(selected) == 1, 'Frozen inputs_for is ambiguous')
    scope = {'torch': torch, 'val': SimpleNamespace(
        ORIGINAL=root / 'source/dependencies/original_a3', load=load_module,
        h=SimpleNamespace(INPUT_NAMES=INPUT_NAMES))}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source) + ':inputs_for-only', 'exec'), scope)
    return scope['inputs_for']


def tensor_hash(tensor):
    """Exact contiguous raw bytes, streamed to CPU to bound transient memory."""
    require(tensor.is_contiguous(), 'Hash requires actual contiguous tensor')
    raw = tensor.detach().view(__import__('torch').uint8).reshape(-1)
    h = hashlib.sha256()
    for begin in range(0, raw.numel(), 16 * 1024**2):
        h.update(raw[begin:begin + 16 * 1024**2].cpu().numpy().tobytes())
    return h.hexdigest()


def describe(t):
    return {'shape': list(t.shape), 'dtype': str(t.dtype).removeprefix('torch.'),
            'stride': list(t.stride()), 'bytes': t.numel() * t.element_size(), 'device': str(t.device)}


def cpu_inputs(root, case, torch):
    inputs = frozen_function(root, torch)(case, 'cpu')
    for name, shape in case['shape'].items():
        require(list(inputs[name].shape) == shape, 'Input shape differs: ' + name)
        require(str(inputs[name].dtype) == 'torch.' + case['dtypes'][name], 'Input dtype differs: ' + name)
        require(inputs[name].is_contiguous(), 'Input layout differs: ' + name)
    require(inputs['metadata'].tolist() == case['metadata']['values'], 'Metadata content differs')
    require({k: inputs[k] for k in ('epsilon', 'max_query_len')} == case['scalars'], 'Scalar contract differs')
    hashes = {name: tensor_hash(inputs[name]) for name in IMMUTABLE}
    require(hashes == case['historical_immutable_input_hashes'], 'Historical five immutable input hashes differ; stop before device allocation')
    # S0 was not recorded historically. This is a NEW epoch digest, never a historical match.
    s0 = tensor_hash(inputs['state_cache'])
    return inputs, {'status': 'PASS', 'historical_immutable_input_hashes_match': True,
                    'hashes': hashes, 'new_S0_sha256': s0, 'historical_S0_sha256': None,
                    'tensors': {n: describe(inputs[n]) for n in case['shape']},
                    'RNG_strategy': 'unchanged inputs_for AST + unchanged frozen V generator'}
