# 20 分钟 SL 的独立对战与种子确认

日期：2026-10-25。初始基线为 `d834a3c`；本轮固定评估协议、冻结模型及原始记录位于 [artifacts/cardplay-long-sl-20261025](../artifacts/cardplay-long-sl-20261025/)。

## 20 分钟训练结果

`runs/play-sl-long-20261025/` 于北京时间 13:41:53 开始，14:01:54 正常结束，约 1,201 秒。学习率 3e-5，批量 64，种子 42，从扩大实验 `sl-42/best.pt` 初始化；训练 10,000 副，验证 1,024 副，没有使用测试牌。

总计 52,928 步、3,387,392 次决策抽样，约为 520,000 条训练决策的 6.51 倍有放回抽样量。每 1,000 步固定抽样 8,192 条验证决策，按 NLL 选 best：

| 检查点 | 步数 | 验证 NLL | 模仿准确率 |
|---|---:|---:|---:|
| 初始化 | 0 | 1.033856 | 56.43% |
| best | 50,000 | 0.734034 | 69.79% |
| last | 52,928 | 0.737866 | 70.09% |

训练日志与完成报告已归档为 `training-metrics.jsonl`、`training-report.json`。候选使用第 50,000 步 best，没有根据独立对战选择 last 或其他种子。

## 第一阶段：全新 2,048 副独立测试

在读取新测试牌的对战结果前，保存 [protocol.json](../artifacts/cardplay-long-sl-20261025/protocol.json) 和 [frozen-models.json](../artifacts/cardplay-long-sl-20261025/frozen-models.json)。

排除既有出牌数据各划分、叫牌监督训练/验证、叫牌牌表库，以及历史报告中所有显式 `board_ids`。共排除 241,024 个不同发牌 ID，原始监督测试集合还剩 8,976 个可用 ID；按原始轨迹顺序导入 2,048 副非四家不叫的完整牌局。每副牌导入时校验 52 张出牌合法性及 OpenSpiel 终局计分。

- 新集合：`data/cardplay-long-sl-holdout-v1/`，106,496 条决策。
- 旧数据/历史报告交集：0；证据在 [exclusion-audit.json](../artifacts/cardplay-long-sl-20261025/exclusion-audit.json) 与 [data-audit.json](../artifacts/cardplay-long-sl-20261025/data-audit.json)。完整排除快照留在本机 `data/cardplay-long-sl-excluded-ids.npy`，哈希在审计中固定。
- 相同完整发牌、原记录叫牌、局况与庄家，双方交换南北/东西实际打完两室；网络贪心，评估种子 20261026。
- 两项主要比较为训练起点 SL、旧 PPO；随机策略为辅助比较。预定扩展门槛为两项主要比较各自 97.5% 按牌局 bootstrap 区间下界都大于 0，采用两项比较的 Bonferroni 式校正；重采样 10,000 次。

| 对手 | 平均 IMP/副 | 95% 区间 | 主要比较的 97.5% 区间 | 胜 / 平 / 负 |
|---|---:|---|---|---|
| 本次训练起点 SL | **+5.0923** | [+4.8696, +5.3233] | [+4.8262, +5.3521] | 1613 / 266 / 169 |
| 旧 PPO pilot | **+5.3237** | [+5.0854, +5.5474] | [+5.0601, +5.5781] | 1661 / 240 / 147 |
| 随机策略 | +5.7134 | [+5.4878, +5.9400] | 辅助比较 | 1692 / 231 / 125 |

第一阶段通过预定门槛，支持这个训练种子的长训 SL 在该来源发牌上优于两个旧基线。它们是本项目此前训练的模型，不是 WBridge5 引擎；没有评估人类等级或证明超过强桥牌程序。详细结果在 [report.json](../artifacts/cardplay-long-sl-20261025/report.json)。

## 条件扩展：另外两个训练种子与新确认牌

通过门槛后，已启动种子 43、44 的训练，保存 [replication/protocol.json](../artifacts/cardplay-long-sl-20261025/replication/protocol.json)：

- 分别从此前对应种子的 SL best 初始化，匹配原训练的 52,928 次更新、批量 64、学习率 3e-5、每 1,000 步验证 8,192 条，按 NLL 选 best。
- 两任务并行使用同一 RTX 5070，匹配更新量而非墙钟时间；运行目录为 `runs/cardplay-long-sl-replicas-v1/sl-{43,44}/`。
- 自动流水线先等待两者达到目标、冻结三个种子的候选，再在另一组 2,048 副完全不同的确认牌上，对各自起点、同一旧 PPO、随机策略共做 9 组双室评估。
- 确认集合：`data/cardplay-long-sl-confirm-v1/`；排除原训练/验证/旧报告牌和第一阶段 2,048 副牌。与下面的搜索面板也无交集，审计见 [replication/data-audit.json](../artifacts/cardplay-long-sl-20261025/replication/data-audit.json)。
- 结果按训练种子与相同牌局列分别 bootstrap，不能把 3×2,048 个比较当作 6,144 个互不相关的新发牌。当前实际进度以 `runs/cardplay-long-sl-replicas-v1/status.json` 为准。

流水线结束会写入 `replication/frozen-models.json`、每组原始对战和 `replication/report.json`。本节不预告尚未完成的多种子结论，也没有启动新 PPO。

## 搜索对照：额外独立 32 副牌

另取 `data/cardplay-long-sl-search-v1/`，与上述两组 2,048 副牌及旧数据全部隔离。固定原种子 42 的第 50,000 步网络，不选择复现种子。所有搜索仅接收当时可见观察，每个非强制动作采样 8 个可能的隐藏局面，通过独立 DDS v3.1.0、4 工作线程求解。

| 当前策略，对手均为长训 SL | 平均 IMP/副 | 95% 区间 | 当前策略单步平均耗时 |
|---|---:|---|---:|
| 同一网络贪心自对局 | 0 | [0, 0] | 17.22 ms |
| 均匀约束采样 + DDS | **+7.2813** | [+5.5625, +8.9688] | 11.02 ms |
| 网络归属引导采样 + DDS | **+7.1250** | [+5.4055, +8.7188] | 32.94 ms |

每组 32 个双室比较，当前策略各做 1,664 个实际动作。逐动作耗时为 batch=1 的可见观察到结果的完整时间，包含推断、采样、RPC 和设备同步；**测量时另外两项训练正在占用 GPU**，不能据此判断纯网络生产部署比 DDS 更慢。搜索使用 CPU DDS，算力资源不同。未做样本量、并发或策略参数调优。

这组小面板支持搜索增强值得继续验证，尚不能证明网络引导采样优于均匀采样；两者没有直接相互对战。PIMC 按采样世界的平均原始得分选牌，仍存在策略融合偏差，不是不完全信息最优求解器。完整数据与时延见 [replication/search-report.json](../artifacts/cardplay-long-sl-20261025/replication/search-report.json)。

搜索流水线首次运行时，纯网络对照期间闲置连接超过 DDS 超时，导致查询服务统计失败；模型对局及两项复现训练不受影响。已改为统计查询使用新连接，并复用已校验的对局记录。重试后全部搜索对照完成，独立 DDS 进程已清理；没有重提未知状态的求解任务。

## 运行、恢复与交接

```powershell
# 第一阶段：已完成；现有文件只在模型、数据、牌数及种子匹配时复用。
.\.venv\Scripts\python.exe examples/cardplay_long_sl_eval.py --phase all

# 条件流水线：已有后台进程时不要重复启动；先查看 status.json。
.\.venv\Scripts\python.exe examples/cardplay_long_sl_confirm.py --phase all
```

后一个命令会检查门槛、准备互不重叠的面板、复用已完成搜索结果、等待两个训练进程、冻结模型、确认并汇总。失败时保留状态、stderr 和检查点，修复原因后可按阶段恢复。训练若提前中止，应先使用原参数 `--resume --steps 52928` 恢复对应种子；这些 CLI 学习参数见复制协议与运行目录的 `stdout.log` 起始记录。

新增历史发牌排除测试，核对嵌套报告正确收集 ID、不会误把模型哈希当作牌局、遇到格式错误时拒绝静默跳过。完整回归为 **88 项通过**。

本轮三个留出面板均应视作已使用或已预留，后续实验排除对应 `holdout-board-ids.npy`、`replication/*-board-ids.npy`，避免用已观察的结果调参后重复宣称独立评估。源数据、权重、完整排除快照和运行日志仍在被 Git 忽略的 `data/`、`runs/` 中。
