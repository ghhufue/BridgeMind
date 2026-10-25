# BridgeMind 文档索引

接手项目请先阅读 **[HANDOFF.md：当前进度与交接](HANDOFF.md)**，再根据任务查看代码和专题文档。

| 文档 | 用途 |
|---|---|
| [HANDOFF.md](HANDOFF.md) | 当前功能、用户目标、代码入口、本机产物、已知限制、未实现设计、接手步骤与维护规则 |
| [training.md](training.md) | 叫牌数据准备、模型、SL/PPO、续训、评估与推理的详细操作 |
| [training-validation.md](training-validation.md) | 叫牌训练实现轮次的测试、短实验和效果边界 |
| [cardplay.md](cardplay.md) | 完整出牌环境、Transformer、约束 DDS 搜索、数据、SL/蒸馏/PPO 和推理 |
| [cardplay-validation.md](cardplay-validation.md) | 出牌信息隔离、数据重放、续训与短实验验证 |
| [cardplay-expanded-experiment.md](cardplay-expanded-experiment.md) | 万副出牌训练、三随机种子、冻结选模与独立千副对战协议和结果 |
| [cardplay-long-sl-evaluation.md](cardplay-long-sl-evaluation.md) | 20 分钟 SL、全新 2,048 副对战、条件种子确认和独立搜索对照 |
| [dds-interface-design.md](dds-interface-design.md) | DDS 独立服务的接口与调度设计 |
| [dds-validation.md](dds-validation.md) | DDS 封装轮次的正确性、并发验证与小样本吞吐 |
| [DDS 构建说明](../artifacts/dds/windows-x64/README.md) | Windows 原生库构建和产物来源 |
| [训练机器可读汇总](../artifacts/training/pilot-report.json) | 数据来源、检查点配置、硬件与短训练结果 |
| [出牌机器可读汇总](../artifacts/cardplay/pilot-report.json) | 出牌数据逐条复核、检查点哈希与短训练结果 |

## 维护约定

- 影响功能进度、模型、数据或运行方式的改动，同步更新 `HANDOFF.md`；具体步骤见该文档第 10 节。
- 明确区分“已实现”“已验证”和“讨论中的方案”。出牌 Transformer、动态叫牌 token、隐藏手牌推断与采样搜索已实现并完成短实验，棋力改善、历史消融与联合训练仍待验证。
- 历史验证报告保留日期、样本量和覆盖范围；状态过期时增加当前说明，不把旧测试数字直接改成新数字。
- 每次实验注明模型、数据、参数和比较对象；本地 `data/`、`runs/` 被 Git 忽略，文档不能默认克隆即拥有这些文件。
- 代码与实际产物优先于状态描述，变更后检查命令和本地链接是否仍有效。
