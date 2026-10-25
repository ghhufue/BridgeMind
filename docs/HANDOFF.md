# BridgeMind 项目交接与当前进度

最后核对：2026-10-25。接续基线：`9f79ef2`（出牌实现）。当前扩大出牌训练和独立评估的协议、产物及结论见 [扩大实验记录](cardplay-expanded-experiment.md)，最终提交以 `git log` 为准。

本页是接手项目的首读文档。**已实现的内容以代码、检查点和验证记录为准；出牌环境、模型、采样搜索和训练已跑通，尚无证据证明棋力提升。** 后续完成任务时按第 10 节维护本页。

## 1. 用户目标与当前阶段

- 最终希望做成可在安卓手机和电脑上体验的桥牌游戏，包含叫牌与出牌 AI。
- 当前优先开发后端算法，允许 Python、C++ 和推理模型协作，客户端与游戏业务服务尚未搭建。
- 近期目标是提高 AI 自身的对战能力；未来希望支持人与 AI 在叫牌上搭档，最终叫牌体系尚未确定。
- 用户希望 DDS 独立于环境和训练器，方便升级求解器、管理并发；当前独立服务已实现。
- 当前处于“DDS、叫牌和完整出牌训练流程已打通，完成短训练验证”的阶段，尚无证据证明模型棋力提升。
- 出牌模块已完成实现收尾；已按三个训练种子扩大 SL/PPO，并将选模验证牌与最终独立对战牌分开，协议与结果见扩大实验记录。

## 2. 已实现与未实现

| 模块 | 当前状态 | 入口 |
|---|---|---|
| DDS v3.1.0 | Windows x64 已编译，可求完整牌表和全知出牌局面 | `src/bridgemind/dds/native.py` |
| DDS 独立服务 | TCP、线程池、队列、缓存、在途合并、批量和异步任务 | `src/bridgemind/dds/server.py`、`service.py`、`client.py` |
| 叫牌环境 | OpenSpiel 规则；叫牌结束后外部 DDS/离线表结算 | `src/bridgemind/engine/bidding.py` |
| 叫牌模型 | 固定 480 维输入、四层 MLP、Actor–Critic、合法动作掩码 | `src/bridgemind/training/model.py`、`encoding.py` |
| 数据准备 | 下载和哈希校验、公开表导入、独立 DDS 生成新表、监督轨迹导入、集合交集审计 | `src/bridgemind/training/data.py`、`offline.py`、`__main__.py` |
| 训练和评估 | 监督学习、PPO、冻结历史对手池、双室 IMP、断点续训、推理接口 | `src/bridgemind/training/` |
| 完整出牌环境封装 | 已实现 52 张出牌、明手控制、私有观察、逐墩与终局计分 | `engine/play.py`、`BridgeBiddingEnv.start_play()` |
| 不完全信息出牌 AI | Transformer 策略/价值/归属头、约束采样 DDS / hybrid、SL/蒸馏/双室 PPO、恢复与评估已跑通 | `cardplay/`；[出牌说明](cardplay.md) |
| 人机搭档适配、客户端、游戏业务服务 | **未实现** | 体系与产品技术选型待后续确定 |

### 实际运行关系

```text
叫牌训练进程
  ├─ 直接调用 pyspiel/OpenSpiel 处理叫牌规则
  ├─ 策略仅获得本人手牌、公开历史与合法动作
  └─ 叫牌结束
       ├─ 离线训练：直接查当前牌的预计算 DDS 表
       └─ 在线求解：SolverClient → 独立 DDS 服务 → 原生 DLL

生成新牌库：训练数据工具 → SolverClient → 独立 DDS 服务
```

OpenSpiel 没有单独运行成服务。现有离线叫牌训练无需启动 DDS，仍在叫牌结束时用全知 DDS 表近似结算。完整游戏使用 `start_play()` 进入 `BridgePlayEnv` 逐张出牌；SL/PPO/纯网络出牌不需要 DDS，采样搜索与蒸馏通过独立服务调用 DDS。

## 3. 接手后先看什么、先运行什么

工作区当前为 `D:\code\BridgeMind`，命令默认在项目根目录、PowerShell 下执行。不要假设本机服务或训练进程仍在运行。

```powershell
git status --short
git log -3 --oneline
git submodule status
.\.venv\Scripts\python.exe -m bridgemind.training --help
```

阅读顺序：本页 → 本次用户请求涉及的源码 → [训练说明](training.md) / [DDS 接口设计](dds-interface-design.md) → 对应验证记录。

| 任务 | 优先阅读 |
|---|---|
| 调整叫牌输入与模型 | `training/encoding.py`、`model.py`、`policy.py` |
| 调整训练、奖励、对手 | `training/runner.py`、`rollout.py`、`optim.py`、`league.py` |
| 调整数据与评估 | `training/data.py`、`baseline.py`、`evaluation.py`、`__main__.py` |
| 出牌模型、训练与搜索 | `cardplay/encoding.py`、`model.py`、`sampling.py`、`search.py`、`runner.py`；[出牌说明](cardplay.md) |
| 扩展游戏环境 | `engine/bidding.py`、`scoring.py`，以及上游 bridge 源码 |
| 排查原生求解、任务或通信 | `dds/native.py`、`models.py`、`service.py`、`protocol.py`、`client.py`、`server.py` |

验证命令：

```powershell
# 完整测试，包含真实原生求解和独立服务进程
.\.venv\Scripts\python.exe -m pytest -q

# 需要本机 data/；核查六个数据集合、数量、指纹与交集
.\.venv\Scripts\python.exe -m bridgemind.training audit
```

最近完整回归 **82 项通过**（既有 58 项、出牌 24 项），新增缓存预算下采样及随机状态一致性验证。出牌实现轮次曾逐条重放核对原始 65,312 条编码和标签；扩大后的 626,496 条在导入时完成合法性及终局计分检查，三个集合的完整发牌 ID 互不重叠。历史记录见 [出牌验证](cardplay-validation.md) 和 [训练验证](training-validation.md)。

已有环境：Python **3.13.10**，OpenSpiel **2.0.2**，PyTorch **2.11.0+cu128**，NumPy **2.5.4**；GPU 为 RTX 5070、约 12 GB，CUDA 当前可用。完整依赖快照为 `requirements-dev.lock.txt`。新建环境的方法见 [training.md](training.md#环境)；已有 `.venv` 可以直接使用。

Git 当前使用 `main`，没有配置远端。出牌实现轮次接续了中断留下的代码，本次扩大实验从干净的 `9f79ef2` 开始。历史代码提交：

- `0e76295`：初始化 Git，纳入原有 DDS 服务和 OpenSpiel 叫牌适配器。
- `58f3153`：加入监督学习、双室 IMP PPO、数据、评估与推理。
- `9f79ef2`：完整出牌、受约束 DDS 搜索、Transformer SL/蒸馏/PPO 和信息隔离验证。

DDS 与 OpenSpiel 是固定提交的子模块。`git submodule update --init` 可取得根项目直接引用的源码；本地 OpenSpiel 原生源码不是当前运行的 wheel 构建。

## 4. DDS 接口与必须保留的语义

- 上游：`third_party/dds`，v3.1.0，提交 `fe6a6581a13535779eda5d9141aa704b3f6c8965`。
- 编译产物：`artifacts/dds/windows-x64/dds.dll`；构建说明、清单、许可证和日志在同目录。
- 默认监听 `127.0.0.1:18765`，面向可信本机调用；远程认证、持久化任务、运行中原生搜索的硬中断均未实现。
- `solve_table(deal)` 返回 5×4 表：花色按 `C,D,H,S,NT`，座位按 `N,E,S,W`。
- `solve_position(position)` 输入四家剩余手牌、将牌、当前墩首引者和已出牌；返回所有合法牌的评价、所有等价最优牌。它需要四家牌全知；实战策略由受限观察采样可能牌局，再调用此接口，不能直接提供真实隐藏手牌。
- `PositionResult` 按**开始调用时出牌方所在队伍**评价，剩余墩数包含尚未结束的当前墩；不要当成庄家视角或整局已得墩数。
- 公共牌字符串为 `SA`、`HT`、`C2` 等。OpenSpiel 牌 ID 和训练手牌位使用 `rank_index * 4 + suit_index`，点数从 2 到 A，花色为 `C,D,H,S`；不要与原生 DDS 编号混用。
- 每个工作线程持有独立的长期存活原生上下文；`NativeDdsBackend` 必须在创建它的线程内求解和关闭。
- 默认 8 个工作线程，单上下文 TT 默认/最大 64/128 MiB。封装要求 `32 <= tt_default_mb <= tt_max_mb <= 4096`；过小的 Large TT 配置曾出现原生访问错误。
- `DDS_TT_DEFAULT_MB`、`DDS_TT_LIMIT_MB` 会覆盖内存预算，封装要求取消这些环境变量后启动。
- 等待超时只停止等待，保留 `job_id`，可继续 `wait`；不会自动重提或停止原生计算。排队任务可取消，运行中的任务不保证中断。
- Windows `.venv` 启动器 PID 可能与实际服务 PID 不同。自动化测试启动服务后，使用 ready JSON / `get_info()` 中的 `process_id` 清理实际进程，再回收启动器；参考 `tests/test_transport.py`。

启动和演示：

```powershell
# 终端一；持续运行
.\.venv\Scripts\python.exe -m bridgemind.dds.server --workers 8 --port 18765
# 终端二
.\.venv\Scripts\python.exe examples/solver_demo.py --auction
```

历史小样本基准：32 副完整牌表，1 线程约 6.45 副/秒，4 线程约 19.89 副/秒。不能直接当作任意出牌局面或未来采样搜索的吞吐，条件见 [DDS 验证](dds-validation.md)。

## 5. 现有叫牌模型与训练规则

### 环境和输入

`BridgeBiddingEnv` 在叫牌完成时终止适配器回合。`result()` 返回原始桥牌分数，顺序 `N,E,S,W`，尚未转 IMP。四家不叫为零分。传入 `table=` 可离线结算，结果标记 `precomputed`，来源版本由数据 manifest 记录。

`observation()` 提供本人 `hand`、`player`、`dealer`、`vulnerable`、公开 `auction`、合法叫牌编号、`state_id` 以及原始 `tensor`；不提供四家手牌。`vulnerable` 顺序为南北、东西。

模型输入编码版本为 **`brl480_v1`**：

| 区间，右端不包含 | 维数 | 含义 |
|---|---:|---|
| `[0:4]` | 4 | 本方、对方局况，各两维 |
| `[4:8]` | 4 | 首次实叫前，各相对位置是否 Pass |
| `[8:428]` | 420 | 35 种实叫 × 实叫/加倍/再加倍 × 4 个相对位置 |
| `[428:480]` | 52 | 本人手牌，13 个位置为 1 |

相对位置依次为本人、下家、对家、上家。开叫后的 Pass 没有单独特征位。当前编码是固定长度压缩表示，`auction` 本身仍保存完整有序公开记录。

本机 wheel 默认配置的原始观察为 **571 维**：前 4 维阶段标识，接着是叫牌阶段的 480 维，叫牌时后 87 维为补零。该长度可能随 OpenSpiel 配置变化，不能作为所有版本的通用常数。

**本机 OpenSpiel 2.0.2 wheel 在非北家发牌时出现历史叫牌人位置缺少庄家偏移的问题。** 当前从显式历史、本人手牌和局况重建 480 维，不直接切原始 `tensor[4:484]`。北家发牌时两者可对照，其他庄家不能盲目照搬。当前检出的上游源码包含相应庄家偏移，不能把源码与已安装 wheel 当成同一构建。

### 网络、动作与训练

```text
480 → Linear(1024)+ReLU → Linear(1024)+ReLU
    → Linear(1024)+ReLU → Linear(1024)+ReLU
         ├─ Actor：Linear(38)
         └─ Critic：Linear(1)
```

默认模型有 **3,681,319** 个参数。合法动作掩码是另外的 38 维输入，不拼入 480 维主干。局部动作：`P=0,X=1,XX=2,1C=3,...,7NT=37`；OpenSpiel 叫牌动作在其基础上加 52。所有选择和训练都屏蔽非法动作。

- 监督学习使用合法动作上的交叉熵，更新主干和 Actor。
- 公开初始权重来自 brl 固定提交 `fdd958ff8adcb44844d70ea2e233bed18d5a7b70`。先核对固定 blob 哈希，再受限读取 NumPy 数组转 PyTorch；没有安装或运行 JAX。
- PPO 首次开始时将 Critic 置零；恢复训练时保留。只收集当前模型动作，冻结对手动作不进入 PPO buffer。
- 每副牌开两室：A 室当前模型打南北，B 室当前模型打东西；同发牌、庄家、局况和冻结对手。
- 收益为 `IMP(A室南北分 - B室南北分) / 24`。当前模型在两室所有动作共享该队伍收益，不能再根据行动座位翻转符号。完整叫牌回合、γ=1、Monte Carlo return，无 GAE。
- 对手槽位：50% 最近冻结快照、25% 历史快照、25% 初始锚点。每批先选一个历史模型，再逐副抽槽位；初期历史不足时会复用初始快照。
- 更新设备默认 `auto`；现有 MLP rollout 使用 CPU 冻结副本，GPU 用于参数更新。未来若改 Transformer，需要重新评估此推理调度，不能假设现有 CPU rollout 性能仍适用。
- `BiddingPolicy(checkpoint).predict(env.observation())` 返回动作、合法概率和 `state_id`，可供未来游戏服务使用。

## 6. 数据、检查点与复现

`data/`、`runs/`、`models/`、`.venv/`、`.build/` 被 Git 忽略；**本机有文件不代表其他机器克隆后也有**。已跟踪的机器可读实验记录在 `artifacts/training/`。

| 集合 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| DDS 牌表 | 100,000 副 | 10,000 副 | 20,000 副 |
| 监督牌局 | 100,000 副 | 10,000 副 | 10,000 副 |
| 监督叫牌决策 | 1,106,954 条 | 110,853 条 | 110,497 条 |

- `data/boards/{train,valid,test}.npz`：从公共 Pgx 250 万副表中固定种子抽样并划分，随机化庄家和局况。32 副用当前 DDS 重算全表一致，其余未全量重算。
- `data/supervised/{train,valid,test}/`：压缩 NPZ 分片、manifest 和 `board-ids.npy`。WBridge5 原始监督数据使用 SAYC、北家发牌、双方无局，导入保留原始条件。SAYC 只是当前数据的来源设定，不代表已确定最终人机搭档体系。
- 本次六个本地集合按完整发牌 SHA256 审计互不重叠；同发牌换庄家、换局况仍是同一 board ID。公共预训练模型的全部训练历史没有经本地审计。
- 原始轨迹在 `data/downloads/openspiel-{train,valid,test}.txt`，包含发牌、叫牌与出牌动作；出牌导入已处理其中 1,256 副，全部完成合法性、终局分数和编码前缀重放核对，其余未全量验证。
- `data/sources.json` 保存来源和哈希；公开模型在 `data/baselines/brl-sl.pt`，原始文件为同目录 `brl-sl.pkl`。
- `data/dds-smoke/` 是 16 副新牌的独立 DDS 生成验证产物，划分 8/4/4，不是主训练数据。

本机已存在的实验：

| 产物 | 状态 |
|---|---|
| `runs/sl-pilot/last.pt` | 监督学习第 300 步；初始化为公开 SL 模型 |
| `runs/ppo-pilot/last.pt` | PPO 第 10 次更新；从上述 SL 初始化，第 8 次后恢复过 |
| `runs/ppo-pilot/league/` | 初始锚点和第 0、2、4、6、8、10 次冻结快照 |
| `runs/*/metrics.jsonl` | 训练配置、过程、验证结果；完整副本也在 `artifacts/training/` |
| `data/cardplay/{train,valid,test}/` | 1,000 / 128 / 128 副出牌，沿用对应叫牌集合划分；共 65,312 条 |
| `runs/play-sl-pilot/{last,best}.pt` | 从随机初始化完成 100 步 SL；best 为第 75 步 |
| `runs/play-distill-pilot/last.pt` | 2 步后恢复至 3 步搜索蒸馏；每局面 2 样本，从第 40 张后采样 |
| `runs/play-ppo-pilot/last.pt` | 3 次完整出牌 PPO 更新、历史对手快照与锚点 |
| `artifacts/cardplay/` | 出牌配置/数据指纹、指标、原始评估及全局重放核对报告 |
| `data/cardplay-expanded-v1/{train,valid,test}/` | 扩大到 10,000 / 1,024 / 1,024 副；共 626,496 条，独立目录保留 pilot 数据 |
| `runs/cardplay-expanded-v1/{sl,ppo}-{42,43,44}/` | 三个训练随机种子；每种子 1,000 步 SL、30 次 PPO，按验证选 best |
| `artifacts/cardplay-expanded-v1/` | 固定协议、数据审计、初始化与冻结模型哈希、训练指标和独立对战记录 |

检查点包含优化器、随机状态、配置、数据指纹和软件/Git 元数据。`--steps`、`--updates` 是**累计目标**。恢复时使用 `--resume`，不要再给 `--init`，训练配置和数据需匹配。

例如，只有在用户要求继续该试验时，以下命令可把现有 PPO 从 10 次恢复到累计 20 次：

```powershell
.\.venv\Scripts\python.exe -m bridgemind.training train-ppo --run runs/ppo-pilot --resume --updates 20 --boards-per-update 128 --eval-boards 256 --eval-every 4 --snapshot-every 2 --device cuda
```

这组 pilot 的评估/快照参数不同于 CLI 默认值；漏传参数会被配置校验拒绝。CPU 上连续两次更新与恢复后训练的模型参数已验证逐位一致；跨硬件或 CUDA 不保证逐位一致。

新实验使用新的 `--run` 目录。数据缺失时按 [training.md](training.md) 的 `fetch → prepare-boards → prepare-sl → audit` 重建。已有牌表输出目录不会覆盖。SL 在评估间隔及最后保存，PPO 每轮保存；目前没有自动按棋力挑选 `best.pt`。活跃对手池默认最多 8 个，历史文件不会自动清理。

## 7. 已完成实验的结论

| 实验 | 结果 | 能支持的结论 |
|---|---|---|
| SL 300 步，固定抽样 8,192 条验证决策 | NLL 0.207479 → 0.185826；预测准确率 92.4316% → 93.1885% | 模仿示范叫牌的拟合改善 |
| SL 对公开初始模型，1,024 副双室验证牌 | +0.05957 IMP/副；95% 区间 [-0.18164, +0.30078] | 尚不能证明棋力提高 |
| PPO 10 次更新，1,280 个双室样本，13,474 条当前策略决策 | 实际参数更新、保存和恢复均跑通 | 训练流程可运行，尚未验证收敛 |
| PPO 对 SL，1,024 副双室验证牌 | -0.05273 IMP/副；95% 区间 [-0.21777, +0.10840] | 尚不能证明 PPO 改善棋力 |
| 初始模型贪心对自身，512 副双室牌 | 全部零 IMP | 支持座位交换和计分对称性检查 |

均为一个随机种子的短实验；测试集未用于效果评估。详情见 [训练验证](training-validation.md) 和 `artifacts/training/pilot-report.json`。报告记录训练时 Git 为 `0e76295` 且 dirty=true，是因为实验发生在训练实现提交前，不表示当前代码仍未提交。

## 8. 出牌当前实现与下一步

详细命令、输入和训练语义见 [出牌说明](cardplay.md)；测试、数据重放和短实验见 [出牌验证](cardplay-validation.md)。

### 已落地结构

- `engine/play.py`：完整出牌环境，`player` 为控制者、`to_play` 为实际出牌手；首攻后公开明手，庄家控制明手。只向策略提供该控制者可见信息，实际打完后结算。
- `cardplay/encoding.py`、`model.py`：`play_tokens_v1`，摘要 + 52 张牌 + 完整叫牌 + 出牌前缀，52 维元信息，4 层/256 维/8 头 Transformer，3,414,023 参数。庄家/防守策略头、价值头和归属头共享主干。
- 每次叫牌保留 Pass/X/XX、行动者和顺序；固定最终庄家坐标。序列动态补齐并屏蔽，最大叫牌数为 319，超长拒绝。体系 ID 0 表示未知，本机 WBridge5 SAYC 数据用 1。
- `cardplay/sampling.py`：对已知牌、公开缺门和精确张数约束的条件采样；默认均匀提议，也支持网络归属权重。对数 DP 求得逐牌独立权重在约束下的精确分布，不保证是真实桥牌后验。
- `cardplay/search.py`、`policy.py`：独立 DDS 批量求解采样局面，按当前队伍期望原始分数选牌；hybrid 用网络引导采样并打破同分平局。PIMC 仍存在策略融合偏差。
- `cardplay/data.py`、`runner.py`：完整轨迹监督学习、可见信息搜索蒸馏、实际 52 张出牌的双室 PPO、历史对手池、断点续训和推理。隐藏牌只用于监督标签，不进入策略输入。
- SL/蒸馏价值目标为实际本方分数/7600；PPO 初次清零价值输出层，再使用双室 IMP/24，全部当前策略动作共享同一配对收益。固定记录叫牌，尚未联合优化叫牌和出牌。

### 当前实验边界

- 已完成 [扩大实验](cardplay-expanded-experiment.md)：训练 10,000 副，验证/独立测试各 1,024 副，三个种子各 1,000 步 SL + 30 次 PPO。全部候选先冻结，再完成 15 组独立对战。
- 主结果：PPO 对各自 SL 平均 -0.008138 IMP/副，按种子与牌局 bootstrap 的 95% 区间 [-0.232104, +0.222664]。新 PPO 对旧 PPO 平均 -0.334635；没有证据支持本轮 PPO 改善或超过旧模型。完整原始报告在 `artifacts/cardplay-expanded-v1/`。本轮测试已使用，后续调参应另留独立牌。
- 分片缓存改为每 `PlayData` 实例默认 1 GiB 数组预算的 LRU，避免扩大训练后反复解压；不改变抽样或随机状态。完整回归为 82 项通过。

以下为保留的初始 pilot 记录，不能与扩大实验不同验证抽样的 NLL 直接比较：

- SL 100 步，固定 512 条验证样本的 best NLL 从 1.156484 降到 1.105088，准确率从 43.164% 到 51.563%；对随机策略 64 副双室为 -1.390625 IMP/副，区间跨零。
- PPO 完成 3 次更新，扩大到 64 副对初始锚点为 +0.015625 IMP/副，95% 区间 [-1.172266, 1.234375]，不能证明提升。
- DDS 两样本搜索、hybrid、蒸馏与完整叫牌到出牌演示已集成跑通；样本数过小，不构成棋力或吞吐结论。
- SL/蒸馏的 best 按模仿 NLL、PPO 的 best 按小验证面板 IMP 选取；现有 best 不是已证明的最强策略。精确 pilot 续训命令见出牌说明。

### 后续尚待完成

1. 扩大出牌监督数据和训练量，多随机种子与独立对手面板评估，确认相对随机/冻结网络的优势。
2. 对比纯网络、均匀 DDS 和学习权重 hybrid 的配对结果及逐动作耗时，增加搜索样本量并评估概率校准。
3. 完整叫牌历史与只给定约的消融，验证叫牌信息对策略和归属推断的收益。
4. 从固定版本叫牌模型生成新的自对弈定约和出牌数据，再评估联合训练；体系适配、人机搭档、客户端与业务服务仍未实现。

叫牌方面仍可扩大训练量、多随机种子和固定对手面板，按独立验证 IMP 选择模型。下一次接手根据用户最新请求推进；本文档不触发后台训练。

## 9. 接手时容易误判的事项

- 当前测试通过只验证已覆盖的行为；完整出牌后端已实现，不能推导为模型棋力已达标或客户端游戏已完成。
- DDS 牌表与 `solve_position` 是全知求解；不能把真实隐藏牌传给实战策略或采样器。
- 不要把出牌动态序列输入替换到现有叫牌 480 维检查点；二者是不同模型和编码。
- 不要用不断查看测试集的结果进行调参；本地预训练历史与外部公开模型训练历史的可审计范围不同。
- 修改公开输入、动作编号、奖励方向或检查点编码版本时，同步考虑已有数据和模型的兼容性。
- 监督原始数据的 SAYC、北家发牌和无局条件，以及当前 DDS 近似结算目标，都是实验边界。

## 10. 后续维护本页的方法

每次完成影响进度、接口、模型、数据或运行方式的任务时：

1. 更新页首核对日期与代码基线，并更新第 2 节状态表。
2. 新功能由“讨论/计划”移到“已实现”时，补上真实代码入口和验证证据；设计变更同步修订第 8 节。
3. 更新数据/检查点路径、步数、参数与复现命令，注明哪些只存在本机。
4. 实验结果写入对应验证文档和机器可读报告，再更新第 7 节；保留负结果与统计限制，区分本次新验证和历史记录。
5. 修改已有接口时同步维护专题文档；[文档索引](README.md) 指向当前入口，历史验证记录保留其日期与范围。

本页维护当前状态和接手入口，专题文档维护详细用法，验证文档维护实验事实；发现冲突时先核对源码与实际产物，再修正文档，避免只根据对话推断实现进度。
