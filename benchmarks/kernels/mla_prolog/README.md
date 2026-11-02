# MLA Prolog serial development

This directory validates an independently developed serial C + V implementation
against the public `aclnnMlaProlog` contract. The fused implementation is treated
as a black box. Only the downloaded public README and API documentation were read.

The initial scope is synthetic A3 BF16 decode (`S=1`), hidden size 7168,
query rank 1536, KV rank 512, 32 query heads, non-RoPE dimension 128,
RoPE dimension 64, and `PA_BSND` BF16 caches with block size 16.
These cases are not model traces and are not Qwen model shape claims.

The performance target is fused complete NPUGraph device-event latency divided
by serial complete NPUGraph latency at least 1.5 for each acceptance case, with
at most ten actual candidate iterations. Public Torch/Torch-NPU matrix APIs
implement C. Independently authored Ascend C kernels implement V, through the
repository's ACLNN, `EXEC_NPU_CMD`, PrivateUse1 and Meta registration path.

Public source: [MlaProlog 9.0.0-beta.1](https://gitcode.com/cann/ops-transformer/tree/9.0.0-beta.1/attention/mla_prolog).
The installed CANN/Torch-NPU binary's identity and black-box behavior are recorded
separately; it is not assumed to equal the source tag without evidence.
