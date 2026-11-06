# 十轮 Vector 优化结果

REPORT.md 是最终结论；B_S_RESULTS.md 是主80组 B/S 的完整可读表。
iteration-performance.csv 保存逐轮逐shape数据，pipeline-performance.csv 保存每个完整流程case的三方耗时。
source-size.json 按实际测试版本计数；tested-source.json 固定性能测量的代码身份。

当前文档提交与该实测代码的 csrc/、mla_superkernel/、scripts/、shape-plan.json 相同。
原始20组Event样本、CPU oracle、全输出融合比较、refresh和实际加载二进制hash在本地：
D:/Workflows/OP_DEV/CANNbot/outputs/dev-mla-vector-tuning-20261106/
每轮公开代码提交和邮件回执保留于该目录；所有构建失败、性能回退也保留。
公共仓库中的本目录是紧凑结果集，不含源码压缩包或服务器凭据。
