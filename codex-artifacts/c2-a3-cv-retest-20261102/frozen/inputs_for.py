def inputs_for(case, device):
    frozen=val.load(val.ORIGINAL/'frozen_v/workload_inputs.py','full60_frozen_inputs')
    values=frozen.make_inputs({k:v for k,v in case['axes'].items() if k!='H'},'cpu')
    del values['kv'],values['scores']
    values={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in values.items()}
    rows=case['axes']['T']
    rng=torch.Generator(device='cpu').manual_seed(case['seed']+1000000007)
    x=torch.empty((rows,5120),dtype=torch.bfloat16,device=device)
    for begin in range(0,rows,16384):
        x[begin:begin+16384].copy_(torch.randn((min(16384,rows-begin),5120),generator=rng))
    wkv=torch.randn((512,5120),generator=rng)
    wgate=torch.randn((512,5120),generator=rng)
    mode=case.get('value_mode','random')
    if mode=='zero_kv': wkv.zero_()
    elif mode=='small_kv': wkv.mul_(1e-12)
    elif mode=='equal_scores': wgate.zero_()
    elif mode=='extreme_scores':
        x[0::2,0].fill_(10000.0)
        x[1::2,0].fill_(-10000.0)
        wgate.zero_(); wgate[:,0].fill_(1.0)
    elif mode not in ('random','zero_norm'): raise ValueError('Unknown original mode '+mode)
    values.update(x=x,wkv=wkv.to(device=device,dtype=torch.bfloat16),
                  wgate=wgate.to(device=device,dtype=torch.bfloat16))
    return {name:values[name] for name in val.h.INPUT_NAMES}
