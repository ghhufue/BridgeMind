# 完整出牌、采样搜索与训练

出牌模块独立于现有叫牌训练器，入口为 `python -m bridgemind.cardplay`，安装后也可使用 `bridgemind-play`。需要 `pip install -e '.[train,dev]'`；本机已有 `.venv`。下面命令在项目根目录的 PowerShell 运行。

## 环境与信息边界

`BridgePlayEnv` 使用 OpenSpiel 的合法动作和完整出牌规则，实际打完 52 张牌。适配层独立跟踪逐墩胜者、双方得墩、缺门和终局计分，终局分数与 OpenSpiel 交叉核对。四家不叫直接结束。

```python
from bridgemind.engine import BridgePlayEnv, Contract

env = BridgePlayEnv(deal, auction=["1NT", "P", "3NT", "P", "P", "P"])
# 或固定定约实验：BridgePlayEnv(deal, contract=Contract(3, "NT", "N"))
obs = env.observation()
env.apply_card(obs["legal_card_ids"][0])
```

实际叫牌完成后使用 `bidding.start_play(system_id=0)` 进入出牌。此步骤无需先调用叫牌环境的 DDS 结算。`contract=` 会构造合法的合成叫牌，标记 `auction_source=synthetic_contract`；它不提供真实叫牌证据。显式传入的完整叫牌标记为 `recorded`，也用于保留模型刚刚生成的叫牌。

`observation()` 的关键字段：

| 字段 | 含义 |
|---|---|
| `player` / `to_play` | 决策控制者 / 实际出牌手；庄家控制明手 |
| `visible_hands` | 控制者本人的剩余牌；首攻后再加入公开明手 |
| `auction` / `play_history` | 全部叫牌和截至当前的出牌前缀，含行动座位 |
| `contract` / `vulnerable` | 定约；南北、东西局况 |
| `current_trick` / `trick_leader` | 尚未完成的当前墩及首引者 |
| `tricks` / `remaining_counts` / `voids` | 双方已得墩数、四家剩余张数、公开缺门 |
| `legal_card_ids` | 强制跟牌后的合法牌 ID，`rank_index*4+suit_index` |
| `state_id` | 本局随机标识和当前出牌步数，用于关联推理与求解结果 |
| `system_id` | 0 为未知体系；当前公开监督数据用 1 表示 WBridge5 SAYC |

两个防守者各自只见本人私有牌。首攻前不公开明手。`training_owners()` 是训练专用的全知标签接口，不进入策略观察。`result()` 返回实际出牌所得的原始四家分数，标记 `played_out`。

## 采样与 DDS

`DealSampler(obs, probabilities=None)` 从观察采样完整剩余局面。默认均匀提议；也可传入形状 `[52,4]`、按绝对 `N,E,S,W` 排列的非负权重。对已知牌、已出牌、每家剩余张数和缺门施加硬约束，用对数空间动态规划精确采样“独立逐牌权重在这些约束下的条件分布”。这不是对真实桥牌后验准确性的保证。

```python
from bridgemind.dds import SolverClient
from bridgemind.cardplay.search import SamplingDdsPolicy

with SolverClient(port=18765) as solver:
    policy = SamplingDdsPolicy(solver, samples=32, seed=42)
    decision = policy.predict(obs)
    env.apply_card(decision["card_id"])
```

先另开终端启动 `python -m bridgemind.dds.server --workers 8 --port 18765`。每个采样局面批量提交给独立服务；结果按当前行动方、剩余墩数（包含当前墩）、定约和局况转成原始队伍得分，取样本均值。只有一张合法牌时直接出牌。

`HybridPlayPolicy(neural, search)` 用网络归属概率引导采样，并用策略概率打破同分平局。此版本没有混合网络价值与 DDS 分数，也没有搜索剪枝。搜索是 PIMC 近似，存在策略融合问题；默认目标为期望原始得分，未对每个样本转换 IMP。

## 数据

```powershell
.\.venv\Scripts\python.exe -m bridgemind.cardplay prepare --train-boards 1000 --valid-boards 128 --test-boards 128
.\.venv\Scripts\python.exe -m bridgemind.cardplay audit
```

原始轨迹来自已有 `data/downloads/openspiel-{train,valid,test}.txt`，入口检查固定来源 SHA256，沿用 `data/supervised/` 中的完整发牌划分。缺少原始文件或叫牌集合时先按 [叫牌训练说明](training.md) 准备。每条选中轨迹导入时都重放全部出牌、校验合法性，并交叉检查终局分数。

`data/cardplay/{split}/` 包含 manifest、`board-ids.npy`、重放用 `games.jsonl` 和压缩 NPZ 分片。每局 52 条决策，变长 token 通过 offsets 保存；真实归属仅作为标签。测试集只做来源、集合隔离和重放正确性检查，不用于棋力调参。`audit` 核查集合归属、隔离、重放身份及文件指纹，不会重新对每条分片执行重放对照。

本机已生成训练 1,000 副 / 52,000 条，验证和测试各 128 副 / 6,656 条。`data/` 和模型文件均被 Git 忽略，克隆项目后需要重建。

## 模型与监督训练

编码版本 `play_tokens_v1`。序列由摘要 token、52 个牌状态 token、完整叫牌事件和当前出牌前缀组成。座位固定以最终庄家为参照，事件包含类型、动作、座位、阶段内顺序和牌的可见状态；另有 52 维元信息和体系 embedding。叫牌最多 319 次，超长拒绝而不截断。批次按最长序列补齐，用 padding mask 屏蔽。

默认 4 层 Transformer、256 隐藏维、8 个头，3,414,023 个参数；共享主干后接庄家/防守两个策略头、价值头和逐牌四座位归属头。策略输出 52 个动作，非法牌始终屏蔽。归属损失只对当前未知牌计算。

```powershell
.\.venv\Scripts\python.exe -m bridgemind.cardplay train-sl --run runs/play-sl-main --steps 1000 --device cuda
```

SL 损失为策略交叉熵 + 0.2×归属交叉熵 + 0.1×价值 MSE；价值目标为实际终局本方原始分数 / 7600。`best.pt` 按固定验证决策的 NLL 选择，`last.pt` 包含优化器、随机状态、配置和数据指纹，可续训。NLL 改善不等于棋力提升。

## 搜索蒸馏与 PPO

```powershell
# 需要正在运行的 DDS 服务；优先从后半局开始控制求解开销。
.\.venv\Scripts\python.exe -m bridgemind.cardplay distill --run runs/play-distill-main --init runs/play-sl-main/best.pt --steps 100 --samples 8 --min-ply 28 --port 18765 --device cuda

# 不需要 DDS；实际完成两室出牌后计算 IMP。
.\.venv\Scripts\python.exe -m bridgemind.cardplay train-ppo --run runs/play-ppo-main --init runs/play-sl-main/best.pt --updates 100 --device cuda
```

蒸馏老师只接收当时的可见观察，使用均匀约束采样，不读取真实隐藏牌。将合法牌的平均原始队伍得分按温度 softmax 成软标签，替代模仿动作标签；归属和价值监督仍使用原始轨迹标签。此版本的 `best.pt` 仍按模仿 NLL 选取，未按搜索质量挑选。

PPO 固定每副牌的真实发牌和记录叫牌，A 室当前策略打南北，B 室打东西，对手在两室保持相同；两室都实际打完。收益为 `IMP(A室南北分 - B室南北分)/24`，当前模型两室的全部动作共享该收益。只收集当前策略的动作，Monte Carlo return，无 GAE；初次 PPO 将价值头输出层清零，恢复时保留。

历史对手池按 50% 最近快照、25% 历史快照、25% 初始锚点抽取。快照冻结，保留活跃历史容量，磁盘旧文件不自动删除。Transformer rollout 在模型所在设备批量推理。PPO `best.pt` 按固定验证牌上对锚点的平均 IMP 选择；短小验证集可能产生明显选择偏差。

`--steps` / `--updates` 是累计目标。恢复时使用相同配置及数据，加 `--resume` 并去掉 `--init`。本机 pilot 的精确续训参数为：

```powershell
.\.venv\Scripts\python.exe -m bridgemind.cardplay train-sl --run runs/play-sl-pilot --resume --steps 200 --eval-every 25 --eval-states 512 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.cardplay train-ppo --run runs/play-ppo-pilot --resume --updates 10 --boards-per-update 8 --epochs 2 --eval-boards 16 --eval-every 1 --snapshot-every 1 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.cardplay distill --run runs/play-distill-pilot --resume --steps 5 --batch-size 8 --eval-every 1 --eval-states 128 --samples 2 --min-ply 40 --port 18765 --device cuda
```

这些是复现示例，文档不会触发训练。`last.pt` 可续训，`best.pt` 仅用于模型推理或初始化。原有叫牌检查点编码不同，不能用来初始化出牌模型。

## 推理与真实对局评估

```python
from bridgemind.cardplay.policy import NeuralPlayPolicy

policy = NeuralPlayPolicy("runs/play-sl-pilot/best.pt", device="cuda")
decision = policy.predict(env.observation())
```

`predict_batch(observations, return_beliefs=True)` 还返回 52 维合法动作概率和绝对座位归属概率；已知牌位置强制固定。采样器进一步施加容量与缺门约束。`state_id` 随决策返回。

```powershell
.\.venv\Scripts\python.exe -m bridgemind.cardplay evaluate --model runs/play-ppo-pilot/last.pt --opponent neural --opponent-model runs/play-ppo-pilot/league/anchor.pt --count 64 --output runs/play-ppo-pilot/duplicate-vs-anchor.json --device cuda
.\.venv\Scripts\python.exe -m bridgemind.cardplay evaluate --policy dds --opponent random --count 8 --samples 32 --port 18765 --output runs/play-dds.json
.\.venv\Scripts\python.exe -m bridgemind.cardplay evaluate --model runs/play-sl-pilot/best.pt --supervised --states 1024 --seed 714
.\.venv\Scripts\python.exe -m bridgemind.cardplay demo --bidding-model runs/sl-pilot/last.pt --model runs/play-sl-pilot/best.pt --output runs/full-game-demo.json --device cuda
```

`evaluate` 支持 `neural/random/dds/hybrid` 双方策略，报告双室 IMP、bootstrap 区间、庄家得墩、数据/模型指纹和运行时间。配对评估固定记录叫牌；`demo` 才把现有叫牌模型接到实际出牌。当前没有联合优化叫牌与出牌、学习体系适配，也没有完成移除叫牌历史的消融实验。当前实验结果及边界见 [出牌验证记录](cardplay-validation.md)。
