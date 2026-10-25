# BridgeMind

当前已实现独立的 DDS v3.1.0 求解服务、Python 客户端、完整叫牌与出牌环境，以及 PyTorch 监督学习和 PPO 自对弈训练。出牌包含动态历史 Transformer、隐藏牌推断、受约束采样 DDS 搜索和搜索蒸馏。20 分钟 SL 首次在全新 2,048 副牌上明显优于旧模型，多训练种子的确认正在推进，详见 [长训 SL 评估](docs/cardplay-long-sl-evaluation.md)。

接手项目请先读 [项目交接与当前进度](docs/HANDOFF.md)。完整文档入口及维护约定见 [docs 文档索引](docs/README.md)。

## 叫牌训练

已有环境和数据可直接运行：

```powershell
.\.venv\Scripts\python.exe -m bridgemind.training train-sl --run runs/sl-main --init data/baselines/brl-sl.pt --steps 1000
.\.venv\Scripts\python.exe -m bridgemind.training train-ppo --run runs/ppo-main --init runs/sl-main/last.pt --updates 100
```

已准备 10 万训练牌表、1 万验证牌表、2 万测试牌表，以及约 110 万条监督叫牌决策。完整安装、数据重建、训练和评估命令见 [训练说明](docs/training.md)，已跑实验与效果限制见 [验证记录](docs/training-validation.md)。本轮短训练证明流程可运行，尚不能证明模型棋力提升。

本地 Git 使用 `main`，DDS 与 OpenSpiel 源码作为固定提交的子模块管理。克隆后使用 `git submodule update --init` 取得源码；数据、模型、训练输出和虚拟环境不提交到 Git。当前没有配置远端仓库。

## 出牌训练与完整对局

本机已准备出牌训练 1,000 副，验证和测试各 128 副。已有环境和数据可直接运行：

```powershell
.\.venv\Scripts\python.exe -m bridgemind.cardplay train-sl --run runs/play-sl-main --steps 1000 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.cardplay train-ppo --run runs/play-ppo-main --init runs/play-sl-main/best.pt --updates 100 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.cardplay demo --bidding-model runs/sl-pilot/last.pt --model runs/play-sl-pilot/best.pt --output runs/full-game-demo.json --device cuda
```

完整数据准备、采样搜索、蒸馏、评估和推理接口见 [出牌说明](docs/cardplay.md)，正确性与短实验结论见 [出牌验证](docs/cardplay-validation.md)。策略只接收当时可见信息；庄家控制明手，首攻后才公开明手，终局由实际出牌结算。

扩大训练使用独立目录中的 10,000 副训练牌、各 1,024 副验证/测试牌，以及三个 SL/PPO 训练随机种子；固定选模与独立对战协议、运行命令和结果见 [扩大实验记录](docs/cardplay-expanded-experiment.md)。

## Windows 快速运行

在项目根目录使用已有虚拟环境。在一个终端启动服务：

```powershell
.\.venv\Scripts\python.exe -m bridgemind.dds.server --workers 8 --port 18765
```

服务成功初始化全部线程后输出一行 JSON，包含监听地址、求解器版本和提交。保持该终端运行。在另一个终端运行示例：

```powershell
.\.venv\Scripts\python.exe examples/solver_demo.py --auction
```

示例会查询完整牌表、计算最后一墩的候选牌，并让 OpenSpiel 执行 `1NT-P-P-P`，由新版 DDS 结算。样例牌的南北得分应为 -50。

测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

DDS 与环境的验证范围见 `docs/dds-validation.md`；训练验证见 `docs/training-validation.md`。可重跑同一组随机牌的端到端基准：

```powershell
.\.venv\Scripts\python.exe examples/solver_benchmark.py --boards 32 --workers 1 4
```

如需新建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,bridge]'
```

DDS 适配层只依赖 Python 标准库；`dev` 安装 pytest，`bridge` 安装固定版本的 OpenSpiel 2.0.2，`train` 安装 PyTorch、NumPy 和 OpenSpiel。当前环境的依赖快照见 `requirements-dev.lock.txt`。OpenSpiel 通过 PyPI 的 Windows wheel 安装，未从本地稀疏检出编译；其包版本与本地源码提交分别记录，不将二者视为相同构建。

## 使用 Python 接口

```python
from bridgemind.dds import Options, SolverClient, parse_pbn

deal = parse_pbn(
    "N:QJ6.K652.J85.T98 873.J97.AT764.Q4 "
    "K5.T83.KQ9.A7652 AT942.AQ4.32.KJ3"
)
with SolverClient(port=18765) as solver:
    table = solver.solve_table(deal, Options(wait_timeout_ms=5000))
    print(table.strains)   # C, D, H, S, NT
    print(table.tricks)    # 每行按 N, E, S, W
```

`solve_position` 的输出始终按开始调用时的出牌方评价，包含未结束的当前墩。`best_cards` 包含全部等价最优牌，`moves` 包含全部合法牌。DDS 位掩码和内部花色编号不进入公共接口。

批量和异步：

- `solve_tables(deals, options)` / `solve_positions(positions, options)`：按输入顺序返回逐项结果，允许部分失败。
- `submit(tasks, request_id=..., scheduling=...)`：异步提交。
- `get_job(job_id)` / `wait(job_id, wait_timeout_ms)`：查询或等待。
- `cancel(job_id, item_ids=None)`：可取消排队任务；运行中的原生搜索不保证中断。
- `get_info()` / `get_stats()`：版本、并发配置、队列、缓存和最近时延样本统计。

同步便捷方法等待超时会抛 `SolverWaitTimeout`；异常的 `job_id` 可用于继续 `wait`，不会自动重提或取消计算。连接断开返回 `SERVICE_UNAVAILABLE`；不自动重试。需要跨连接重试提交时，复用同一个 `client_id` 与业务 `request_id`。服务实例 ID 变化意味着旧任务不能恢复。

## 原生层和调度

`NativeDdsBackend` 可独立使用，但必须在创建它的线程中求解和关闭，建议使用 `with`。默认 DLL 位于 `artifacts/dds/windows-x64/dds.dll`；服务可通过 `--library` 或 `BRIDGEMIND_DDS_LIBRARY` 指定别的已验证原生库。

每个工作线程独占一个长期存活的上下文。服务管理任务并行，不调用会额外扩展内部线程池的传统批处理接口。默认 8 个线程、每线程 TT 默认/最大 64/128 MiB，此外仍需预留上下文、队列、缓存和 Python 对象内存。

DDS 3.1.0 的 Large TT 在过小内存配置下会出现访问错误，因此适配层要求 `32 <= tt_default_mb <= tt_max_mb <= 4096`；这是本封装采用的保守下限，不声称上游支持任意小内存。`DDS_TT_DEFAULT_MB` 和 `DDS_TT_LIMIT_MB` 环境变量会覆盖上下文预算，服务要求先取消这些变量再启动。配置错误会在接收任务前报告。

缓存预算按规范化结果的 JSON 字节数和键长度计算，不是 Python 堆或进程 RSS 的硬限制。任务按优先级与等待时间调度，相同在途求解合并。已完成任务默认保留 300 秒，受最多 4096 个任务批次限制；容量满时可先淘汰已完成任务，因此提交去重不是永久保证。

服务仅面向可信本机调用，监听 127.0.0.1，没有远程认证或持久化任务。控制请求、连接、帧、批量和计算队列都有数量上限；单个 `wait` RPC 最多占用控制线程 1 秒，客户端会在总等待预算内继续查询。空闲连接约 30 秒后关闭，长时间停用后应新建客户端。

## OpenSpiel 叫牌环境

```python
from bridgemind.engine import BridgeBiddingEnv

env = BridgeBiddingEnv(solver, deal, dealer="N", vulnerable=(False, False))
observation = env.observation()
env.apply_call("1NT")   # 也可使用局部动作编号 0..37
```

环境的局部叫牌编号是 P=0、X=1、XX=2、1C=3 至 7NT=37。它内部转换到 OpenSpiel 编号，不直接向策略暴露完整状态。`observation()` 提供本人手牌、庄家与局况、原生观察向量、公开叫牌历史、合法叫牌编号及状态版本。原生观察向量不承诺与 brl/Pgx 编码兼容；训练模块单独重建 480 维输入，处理本机 wheel 的非北家历史位置问题。

`is_terminal()` 表示叫牌训练回合结束；`result()` 返回外部 DDS 结算后的原始分数，未转为 IMP。底层 OpenSpiel 此时可能刚进入出牌阶段；不能将本适配器当作完整出牌环境。四家不叫直接返回零分。可关闭 `prefetch`，只在确定定约后提交牌表。

离线训练可传入 `BridgeBiddingEnv(None, deal, table=table)`，直接用已校验的完整牌表结算；结果标记为 `precomputed`，原始来源由数据 manifest 记录。没有预计算表时继续使用外部求解器。

## 验证与目录

- `src/bridgemind/dds/`：模型与校验、原生绑定、调度、协议、客户端、服务入口。
- `src/bridgemind/engine/`：外部 DDS 叫牌环境、独立计分。
- `src/bridgemind/training/`：数据、编码、PyTorch 模型、SL/PPO、对手池、评估与推理。
- `src/bridgemind/cardplay/`：出牌历史 Transformer、归属推断、约束采样与 DDS 搜索、SL/蒸馏/PPO 和真实出牌评估。
- `tests/`：原生样例、独立小残局枚举、并发与协议、任务边界、OpenSpiel 交叉验证。
- `artifacts/dds/windows-x64/`：已编译的 DLL、导入库和上游示例。
- `third_party/`：保持上游原样的源码与许可证。
- `docs/dds-interface-design.md`：接口设计与本轮实现说明。

验证包括 DDS 上游 3 副牌的全部牌表、24 个随机小残局与独立 minimax 的逐动作对照、12 副随机完整牌的串并行一致性、真实独立进程通信，以及四种局况下 1680 个定约分数与 OpenSpiel 的交叉检查。
