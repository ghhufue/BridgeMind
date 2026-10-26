# 出牌牌力评分器 v1

日期：2026-10-26。实现入口：`src/bridgemind/rating/`；命令为 `python -m bridgemind.rating`。本体系评估固定叫牌之后的出牌能力，同一策略控制双方搭档，不评价叫牌、人机搭档或人类段位。

## 评分定义

- `random`：合法牌均匀随机，固定 **0 分**。
- `oracle`：可以看到真实四手牌的全知 DDS，固定 **100 分**。逐步最大化当前出牌阵营的剩余得墩；全部等价最优牌按引擎 card ID 升序选第一张，保证确定性。
- 其他规则、采样 DDS、网络和 hybrid 对手均通过实际比赛评分，不人为指定中间分数。
- 分数允许小于 0 或超过 100，不截断。100 是信息条件不同的参考锚点，不是胜率、正常信息条件下的理论上限或“掌握桥牌的百分比”。

v1 的固定标尺面板为 `random / lowest / rule-basic / oracle`，各占 25%。令 `M(A)` 为策略 A 对该固定面板的平均 IMP/副：

```text
rating(A) = 100 × [M(A) - M(random)] / [M(oracle) - M(random)]
```

对手池可以不断扩展，标尺面板、牌集、权重、随机种子和运行配置保持固定。新增对手不改变任何已有策略的定义分数。更换面板、扩大牌数、调整规则、升级模型编码或 DDS 时创建新版本，再重新校准；不同版本不能直接比较数值。

## 对手及信息边界

| kind | 行为 |
|---|---|
| `random` | 合法牌均匀随机 |
| `lowest` | 始终出 card ID 最小的合法牌，先按点数再按 C/D/H/S 排序 |
| `rule-basic` | 首攻/引牌从最长非将花色出小牌；搭档当前赢墩时出小牌；否则尝试用最小有效牌赢墩，不能赢则出小牌。确定性启发式，不保证强于随机或其他规则 |
| `oracle` | 全知 DDS；只能通过评测器专用的 `predict_position` 接口获得真实 Position |
| `sampling-dds` | 仅根据合法可见信息，采样固定数量的可能暗牌并调用 DDS；按采样世界的平均原始分选牌 |
| `neural` | 已冻结的出牌模型，贪心选牌 |
| `hybrid` | 固定网络提供归属分布与先验，固定采样量的 DDS 搜索 |

普通策略始终只收到 `BridgePlayEnv.observation()`，真实牌不进入观察或采样器。评测器只对 `kind=oracle` 构造特权 Position。全知 DDS 与采样 DDS 在登记和报告中使用不同名称。

一条登记记录绑定不可覆盖的 ID、策略版本、权重 SHA256、采样量和可声明的本地训练发牌 ID。相同 ID 的不同权重或配置会被拒绝。旧训练的 `PlayLeague` 继续用于训练快照抽样，评分器另维护长期评测对手库。

## 对战、统计和恢复

1. 每副完整发牌使用相同叫牌、庄家、局况开两室。A 在第一室打南北、第二室打东西，B 反向；真实打完每室 52 张牌。
2. 计算 `IMP(第一室南北分 - 第二室南北分)`。所有对手使用相同发牌列及重复随机种子。
3. 随机数由“重复种子、完整发牌 ID、策略 ID”派生；每副牌、每室重新建立策略。调换比赛执行顺序和恢复中断不改变随机流；相同策略自比较为精确零分。
4. 对所有对手、候选和两个锚点共同重采样相同发牌列及种子行，默认 2,000 次 bootstrap，输出归一化分数的 95% 区间。不会把 128 副 × 3 个重复种子当作 384 副独立发牌。
5. 锚点差必须为正；若 bootstrap 锚点差不能稳定保持为正，标为 `insufficient_anchor_separation`，不给可信分数区间。0/100 的端点区间是定义值，另行报告校准差及区间。
6. 每完成一个双室牌局原子保存缓存和记录校验和。正反向比赛共用缓存并翻转符号。恢复时验证协议、代码、运行时、DDS 身份、权重、发牌坐标和 IMP；不同配置不会混用旧结果。
7. 单写入者使用操作系统文件锁，进程退出后自动释放，防止两个评测任务同时修改同一评分库。中断牌局没有计入缓存，恢复时重做；求解等待超时不会向未知状态的 DDS 任务自动重提。

置信区间只覆盖固定策略在当前发牌来源和随机重复上的不确定性，不包含重新训练模型的随机性。逐对手报告保留平均 IMP、95% 区间和种子—发牌胜平负计数。单步时延是诊断记录，不是隔离负载下的正式性能承诺。

## 本机快速使用

先检查已有 DDS 服务；没有运行时在另一个终端启动，已有服务时无需重复启动：

```powershell
.\.venv\Scripts\python.exe -m bridgemind.dds.server --workers 4 --port 18765
```

本机已提供旧模型登记与新牌准备脚本，默认建立 `runs/cardplay-rating-v1/`，128 副、3 个重复随机种子、CUDA 网络推理：

```powershell
# 从已下载的原始记录中排除旧数据和已知历史评测牌，冻结新评分牌
.\.venv\Scripts\python.exe examples/cardplay_rating_setup.py --phase prepare

# 登记已有 SL、PPO、蒸馏、长训模型以及长训 hybrid；不训练新模型
.\.venv\Scripts\python.exe examples/cardplay_rating_setup.py --phase register

# 评估指定对手；会先校准并报告随机/全知 DDS，支持原命令恢复
.\.venv\Scripts\python.exe examples/cardplay_rating_setup.py --phase evaluate --id lowest rule-basic sampling-dds-8 sampling-dds-32 sl-pilot ppo-pilot-last sl-long-42 hybrid-long-42-8

# 多进程执行不同比赛，父进程独占评分库；各比赛文件互不重叠
.\.venv\Scripts\python.exe examples/cardplay_rating_setup.py --phase evaluate --workers 3 --id sl-long-42 sampling-dds-8

# 补测整个已登记对手池；复用已经完成且校验一致的比赛
.\.venv\Scripts\python.exe -m bridgemind.rating evaluate --all

# 输出本地榜单或导出可审查的协议、报告、比赛哈希
.\.venv\Scripts\python.exe -m bridgemind.rating leaderboard
.\.venv\Scripts\python.exe -m bridgemind.rating export --output artifacts/cardplay-rating-v1
```

新模型训练完成后的日常操作：

```powershell
# 一条命令登记、评估并显示榜单；同一模型同一ID可以原样恢复
.\scripts\rate-model.ps1 -Id sl-new-001 -Checkpoint runs/my-new-model/best.pt -TrainingData data/cardplay-expanded-v1/train

# 或分别使用Python命令
.\.venv\Scripts\python.exe -m bridgemind.rating register --id sl-new-001 --kind neural --checkpoint runs/my-new-model/best.pt --training-data data/cardplay-expanded-v1/train
.\.venv\Scripts\python.exe -m bridgemind.rating evaluate --id sl-new-001
```

训练数据可以重复传入 `--training-data`。与评分牌重叠会被拒绝；未声明或无法审计的预训练历史在登记中明确标为未知。仅改变显示 ID 来重新登记相同策略，会改变其派生随机种子；应保留原 ID，避免通过换名挑随机结果。

登记引用已有权重，不自动复制模型文件。每个版本应保留独立、停止写入的检查点；如果要继续训练同一个运行目录，先把需要保留的权重复制到独立版本路径再登记。已登记的权重被覆盖后会拒绝评测，不会静默更改历史模型。

需要验证两个相近分数模型是否直接互有优势时，可以对登记库中的任意两项加测。加测结果不改变固定面板的评分定义：

```python
import torch
from bridgemind.rating.system import RatingSystem, writer_lock

torch.set_num_threads(1)
with writer_lock("runs/cardplay-rating-v1"):
    evaluator = RatingSystem("runs/cardplay-rating-v1")
    values, match = evaluator.match("sl-long-42", "ppo-pilot-last")
    print("直接对战 IMP/副:", float(values.mean()))
```

固定面板评分不是对战收益的严格传递关系；某模型总分更高，不保证对每一个对手都占优。选择模型晋级时同时查看逐对手结果，必要时进行直接对战和独立确认。

自建另一个版本，使用任意已有的 `PlayData` 数据集：

```powershell
.\.venv\Scripts\python.exe -m bridgemind.rating --root runs/cardplay-rating-v2 init --name cardplay-rating-v2 --data data/my-rating-boards --count 512 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.rating --root runs/cardplay-rating-v2 evaluate --all
```

若作独立确认，设置 `--purpose confirmation`，并用多次 `--exclude-data` 声明需要排除的训练、验证和历史评估 `PlayData` 数据集；评分器验证交集，但无法自动获知未声明的外部训练历史。正式比较应冻结候选后只查看一次确认集。

## 输出及维护

2026-10-26追加SL v2：`sl-v2-20261026` 牌力 **62.54 [58.06,67.01]**，原长训SL为32.16；配对分数增量30.38 [24.41,36.80]。固定标尺未改变，登记22项、8项已评分；与基础规则的优势另在全新2,048副牌确认。训练课程和真实训练ID均已声明，完整统计和适用范围见 [SL v2验收](cardplay-sl-v2-evaluation.md)。

```text
runs/cardplay-rating-v1/
  protocol.json         # 固定面板、权重、来源、实现/运行时/DDS 身份
  boards.json           # 冻结发牌、叫牌、局况；仅供评测器
  opponents/*.json      # 可追加的不可覆盖对手登记
  matches/*.json        # 可恢复的逐副、逐种子双室结果
  reports/*.json        # 分数、置信区间、逐对手收益
  leaderboard.json      # 已完成评分和待评对手
  leaderboard.md        # 可读榜单
```

代码、编码、计分、采样器等实现哈希与 PyTorch/NumPy/OpenSpiel/Python 版本均冻结。升级后旧档案仍可读，但要在匹配环境下继续计算，或创建新标尺版本；不静默重算并沿用旧版本名称。完整模型、逐副缓存和数据只在本机 `runs/`、`data/`，导出的 `artifacts/` 保留协议、发牌 ID、模型指纹、评分报告和比赛文件哈希。

初版128副牌仍来源于 WBridge5 SAYC、北家发牌、双方无局的记录。首次取牌与已知旧数据隔离，日后反复用它比较版本后，它就是日常跟踪面板，不能再作为独立最终测试。应逐步另建覆盖局况、发牌人和不同叫牌来源的确认面板。

## 首批实测与验证（2026-10-26）

本机已冻结 `cardplay-rating-v1`，128副不同发牌、3个重复种子、2,000次联合bootstrap；排除246,192个已知旧发牌，选取牌与排除集合交集为0。登记21项策略，7项已完成评分，其他14项（含采样DDS、hybrid及其余历史模型）明确保留为未评分，不为未完成对战赋分。

| 策略 | 牌力 | 95%区间 |
|---|---:|---|
| 全知DDS `oracle` | 100.00 | 定义锚点 |
| 基础规则 `rule-basic` | 37.98 | [33.36, 42.69] |
| 长训SL种子42 `sl-long-42` | 32.16 | [26.55, 37.19] |
| 随机 `random` | 0.00 | 定义锚点 |
| 旧SL `sl-pilot` | -10.88 | [-16.57, -5.71] |
| 旧PPO末检查点 `ppo-pilot-last` | -11.20 | [-16.65, -6.14] |
| 最小合法牌 `lowest` | -20.21 | [-26.30, -14.75] |

两个锚点的固定面板平均IMP差为10.3555，95%区间[9.7187, 10.9812]，校准状态正常。旧SL/PPO的整体面板评分低于随机；这不等于对每个对手都更弱。基础规则与长训SL的总分区间重叠，不能仅据边际区间推断直接对战差异；实际同批128副的直接对战中，长训SL对基础规则为-1.8203 IMP/副，95%区间[-2.8049, -0.7813]，支持其在当前牌集上落后。进一步的六副重放和针对性案例见[诊断文档](cardplay-diagnostics.md)。该分数仍只覆盖当前固定来源的出牌任务。

机器可读协议、登记摘要、逐对手报告和榜单已导出至 [artifacts/cardplay-rating-v1](../artifacts/cardplay-rating-v1/leaderboard.md)。[verification.json](../artifacts/cardplay-rating-v1/verification.json) 记录全量pytest **169项通过，19.88秒**，全部权重和声明训练牌交集检查、逐副缓存/IMP校验，以及从逐副结果独立重算7项评分一致。22个完整配对缓存中4个为定义上的零分自比较，另3个采样策略比赛缓存只完成部分，未用于评分。一键PowerShell入口已用真实长训检查点及缓存完整运行，退出码0。

补测整个池可执行 `python -m bridgemind.rating evaluate --all`；要并行补测，使用上述setup脚本的 `--workers`。这些命令可能需要较长时间，已完成的校验一致比赛会复用。本次没有留下后台评分任务，原有DDS与分析服务保持运行。

测试见 `tests/test_rating.py`：端点与归一化、共享噪声联合重采样、弱锚点拒绝、规则合法性、特权通道、原生 DDS 小残局、双室对称/翻转、登记不可覆盖、恢复/缓存校验、代码/权重/发牌变更拒绝及新增对手后标尺稳定。
