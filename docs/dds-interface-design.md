# BridgeMind DDS 接口设计 v1

状态：v1 已实现，运行方式见项目根目录 README。基于已编译的主线 DDS v3.1.0（`fe6a6581a13535779eda5d9141aa704b3f6c8965`）。

实现补充：Large TT 配置采用经验证的至少 32 MiB 下限，并禁止环境变量覆盖托管预算；同步等待超时抛 `SolverWaitTimeout`，保留可继续查询的 `job_id`。传输信封为 `schema_version/request_id/client_id/method/params`，提交参数位于 `params`；传输 request ID 与 submit 的业务 request ID 分开。非法帧会断开连接。控制层的单次等待上限为 1 秒，客户端负责完整等待预算。当前已接入 OpenSpiel 2.0.2 wheel 的叫牌结算适配，尚未替换其编译时 DDS 依赖。

## 1. 边界和运行结构

OpenSpiel 管理规则与牌局；策略负责隐藏牌采样、动作选择；DDS 只计算给定完整牌局的全知最优结果。计分、IMP、成约概率与采样加权在 DDS 服务之外完成。

```text
训练环境 / 出牌搜索 / 离线数据生成
                ↓
       SolverClient（统一调用接口）
                ↓
   本机独立服务进程：DDS SolverService
   参数校验 → 规范化 → 缓存/合并 → 有界队列
                ↓
       固定工作线程，每线程独立上下文
                ↓
       NativeDdsBackend → DDS C ABI
```

分三个实现层：

| 层 | 职责 | 生命周期 |
| --- | --- | --- |
| `NativeDdsBackend` | 加载 DLL、绑定 ABI、转换编码、管理上下文、解释错误 | 服务进程内 |
| `SolverService` | 校验、队列、并发预算、缓存、任务状态与统计 | 独立进程 |
| `SolverClient` | 同步/异步调用、批量提交、响应关联、等待超时 | 每个调用方 |

第一版可用 Python `ctypes` 加载现有 DLL，由 Python 工作线程调度原生求解；计算在 C++ 中运行。使用 `ctypes.CDLL` 绑定无 `STDCALL` 的 `dds_c_*` 接口，精确声明参数和返回类型；不要用持有 GIL 的 `PyDLL`。传统导出函数另行按其调用约定绑定。未来可将服务替换为 C++，保持逻辑协议不变。

## 2. 公共数据约定

| 类型 | 约定 |
| --- | --- |
| `Seat` | `N`, `E`, `S`, `W`，固定顺时针顺序 |
| `Partnership` | `NS`, `EW` |
| `Strain` | `C`, `D`, `H`, `S`, `NT` |
| `Card` | 两字符，花色在前：`C2`, `DT`, `HQ`, `SA`；10 固定写 `T` |
| `Hand` | 不重复的 `Card` 数组，顺序没有语义 |
| `Hands` | 必须包含 `N/E/S/W` 四个键 |
| `schema_version` | 初始为 `1`；不兼容语义变化必须升级 |

外部不传 DDS 位掩码、数组索引、指针或上下文句柄。PBN 是导入导出辅助格式，通过 `parse_pbn`/`format_pbn` 转换为上述对象，避免同一请求存在多套牌局表示。

采用与 OpenSpiel 不绑定的编码。规范化后使用稳定的牌排序。DDS 自身的花色顺序为 S/H/D/C/NT，适配器必须显式映射；牌表响应固定输出 C/D/H/S/NT 行、N/E/S/W 列。

## 3. 两类求解操作

### 3.1 `solve_table`

输入 `Deal`：

- `hands`：四家初始手牌，各 13 张，合计完整 52 张。
- 不包含发牌人、局况、叫牌历史、阶数和加倍状态。

输出 `TableResult`：

- `strains`: `["C", "D", "H", "S", "NT"]`。
- `declarers`: `["N", "E", "S", "W"]`。
- `tricks`: 5×4 整数矩阵，每个值为 0～13。
- `exact`: `true`。
- `evaluation_model`: `double_dummy`。

`tricks[i][j]` 表示第 i 种将牌/无将、第 j 位庄家，在其左手方首攻、双方全知且最优攻防时，庄家一方可以取得的总墩数。这里的“精确”仅针对双明手模型。

### 3.2 `solve_position`

输入 `Position`：

| 字段 | 含义 |
| --- | --- |
| `remaining_hands` | 当前尚未打出的四家手牌，不含当前墩已经打出的牌 |
| `trump` | C/D/H/S/NT |
| `trick_leader` | 当前墩领出者；当前墩为空时，是下一张牌的出牌位置 |
| `current_trick` | 按出牌时间排列的 0～3 张牌；各出牌位置由领出者顺时针推导 |

不额外传 `to_play`，避免与当前墩矛盾。服务推导：

`to_play = (trick_leader + len(current_trick)) mod 4`。

`remaining_tricks = (剩余手牌总张数 + 当前墩牌数) / 4`。包含尚未结束的当前墩，不包含之前已完成的墩。v1 要求 1～13 个剩余墩；完整终局由游戏环境直接处理。

输出 `PositionResult`：

- `to_play`：实际出牌位置，不是控制该手牌的玩家身份。
- `evaluated_side`：`to_play` 所属的 NS 或 EW。
- `remaining_tricks`：上述总剩余墩数。
- `best_tricks`：当前出牌方在最优攻防下可获得的剩余墩数。
- `best_cards`：全部最优合法牌；不强制指定唯一一张最佳牌。
- `moves`：全部合法牌及对应 `tricks_for_evaluated_side`。
- `exact=true`、`evaluation_model="double_dummy"`。

所有候选分值的视角固定为调用开始时的 `evaluated_side`，不会随下一位出牌者变化。对方可获得墩数等于 `remaining_tricks - tricks_for_evaluated_side`。转换成庄家全局总墩数时，由业务层结合庄家身份和之前已获墩数处理。

内部固定使用完整候选精确求值（`target=-1, solutions=3, mode=1`）。DDS 的 `equals` 等价牌位掩码必须展开，保证一张合法牌对应一个候选。原始 `score`、`equals`、`mode` 不进入公共协议。不得将未求解标记或目标阈值当作精确墩数返回。

结果按分值降序、再按稳定牌序排序；`best_cards` 包含所有达到 `best_tricks` 的牌。排序不代表模型置信度。

示例请求（只剩最后一墩，N 已领出 SA）：

```json
{
  "remaining_hands": {"N": [], "E": ["SK"], "S": ["SQ"], "W": ["SJ"]},
  "trump": "NT",
  "trick_leader": "N",
  "current_trick": ["SA"]
}
```

该局面出牌位置为 E，评价方为 EW，共剩 1 墩；唯一动作 SK 的评价应为 EW 获 0 墩。此例作为后续视角和当前墩处理的验收用例。

## 4. 客户端逻辑方法

以下是契约说明，不是当前已经存在的实现。

| 方法 | 返回 | 行为 |
| --- | --- | --- |
| `solve_table(deal, options)` | `TableResult` | 同步便捷入口；单项错误抛结构化异常 |
| `solve_position(position, options)` | `PositionResult` | 同上 |
| `solve_tables(deals, options)` | `TaskOutcome[]` | 批量，结果与输入顺序一致 |
| `solve_positions(positions, options)` | `TaskOutcome[]` | 批量，允许部分失败 |
| `submit(tasks, scheduling)` | `Submission` | 异步提交，返回 job ID 和各项的接受/拒绝情况 |
| `get_job(job_id)` | `JobSnapshot` | 查询每项状态和已经完成的结果 |
| `wait(job_id, wait_timeout_ms)` | `JobSnapshot` | 等待全部已接受项目终止，或等待超时后返回快照 |
| `cancel(job_id, item_ids?)` | 逐项取消处理结果 | 取消排队订阅；运行项返回 already_running，仍可查询 |
| `get_info()` | `ServiceInfo` | 协议、DDS 版本、提交、平台、支持能力 |
| `get_stats()` | `ServiceStats` | 队列、在执行数量、缓存命中、错误、时延统计 |

`Options` 分为调用等待选项与任务调度选项：

- `wait_timeout_ms`：客户端最多等待多久，不限制底层搜索执行时间。
- `queue_timeout_ms`：服务接受项目后，最多等待多久开始执行；用服务单调时钟计时。
- `priority`：`interactive` / `training` / `offline`，默认 `training`。
- `state_id`：可选的应用状态版本，原样回显；不参与求解。调用方负责检查过期状态。
- 不允许每个请求指定线程数量、TT 配置或任意文件路径。

批量接口只是提交和收集接口，不保证使用 DDS 内部批处理线程池。服务可以拆分批次，使用统一线程池调度。

## 5. 请求、响应和状态

`Task` 包含：`item_id`（批次内唯一）、`operation`（`table`/`position`）、`payload`、可选 `state_id`。

提交信封包含 `schema_version`、`request_id`、`tasks`、调度选项。`request_id` 用于限定时间内的重试去重：同一客户端、同一 ID、同一内容返回原提交；同一 ID 不同内容返回冲突。该保证仅在服务实例生命周期及配置的保留窗口内有效。

`Submission` 包含服务生成的 `job_id`、`service_instance_id`、逐项接受结果。结构/帧错误整批拒绝；单个 payload 无效时只拒绝该项；队列容量不足时逐项报告拒绝，不默默丢弃。

`TaskOutcome` 包含：

- `item_id`、`state_id`、状态。
- 成功时 `result`；失败时 `error`。
- `solver_version`、`solver_commit`、`schema_version`。
- `cache_hit`、`coalesced`、可空 `queue_ms`、可空 `solve_ms`、可空 `search_nodes`。

`search_nodes` 仅在对应底层操作提供可信计数时返回；不能为完整牌表伪造计数。缓存命中的请求不得把历史计算耗时混作本次实际求解耗时。

状态转换：

```text
queued → running → succeeded / failed
queued → cancelled / expired
```

缓存命中可直接为 `succeeded`。不返回近似答案冒充成功结果。

运行中请求的取消为尽力而为：报告 `already_running`，不能保证中断底层 DDS。客户端可停止等待，底层完成后允许写入缓存。去重任务仍有其他订阅者时不得因为一个调用方取消就取消共享计算。v1 不承诺运行任务的硬超时。

## 6. 错误模型

统一错误字段：`code`、`message`、`field`（可空）、`retryable`、`native_code`（可空）。

| code | 含义 | 默认可重试 |
| --- | --- | --- |
| `INVALID_ARGUMENT` | 缺字段、非法枚举、牌数量错误、重复牌等 | 否 |
| `INVALID_POSITION` | 当前墩、剩余张数或出牌顺序不一致 | 否 |
| `UNSUPPORTED_SCHEMA` | 协议主版本不支持 | 否 |
| `IDEMPOTENCY_CONFLICT` | 同一请求 ID 对应不同内容 | 否 |
| `QUEUE_FULL` | 服务达到容量上限 | 是，退避 |
| `QUEUE_EXPIRED` | 排队超过允许时间 | 是，视应用预算 |
| `NATIVE_SOLVER_ERROR` | DDS 返回失败状态 | 根据 native_code 分类 |
| `SERVICE_UNAVAILABLE` | 无法连接、服务关闭或重启 | 是 |
| `JOB_NOT_FOUND` | job 不存在、已过保留期或来自旧实例 | 否，需明确重新提交 |

验证完整牌局的唯一性与 13 张约束；验证局面总张数可组成完整剩余墩、当前墩牌不在剩余手牌中，以及按领出顺序已出牌的家少一张的计数关系。当前墩中已能证明存在的非法不跟牌也应拒绝。未提供历史时，不声称验证了完整历史合法性。

## 7. 并发与缓存

选择服务管理并行：固定工作线程池，每线程独占一个长期存活的 `SolverContext`。初始化和配置在接收任务前完成，不在每个请求中反复改全局 DDS 设置。

绑定当前 DLL 已导出的 `dds_c_create_solvercontext`、`dds_c_calc_dd_table`、`dds_c_solve_board`、`dds_c_destroy_solvercontext`。本版本 context-aware 牌表路径顺序处理其内部子问题；不要从工作线程改调自动开启内部线程池的传统批量入口。

| 服务配置 | v1 建议起点 | 说明 |
| --- | --- | --- |
| `worker_count` | 8 | 固定计算并发数，后续按基准调整 |
| `max_queued_tasks` | 1024 | 按单项任务计数，不能仅按批次计数 |
| `max_batch_items` | 256 | 防止单次提交无限大 |
| `tt_default_mb` / `tt_max_mb` | 64 / 128 | 每个上下文的 TT 配置，需通过原生接口验证 |
| `cache_max_bytes` | 显式配置 | 与工作线程的内存预算分开 |

上述是初始配置，不是性能结论。TT 限制不是服务 RSS 硬上限；线程数 × TT 上限之外，还要预留临时数据、结果缓存、队列和进程开销。

缓存键：

- 牌表：规范化完整手牌 + 输出语义版本 + 求解器身份。
- 局面：规范化剩余手牌 + 将牌 + 当前墩领出位置 + 当前墩有序牌 + 输出语义版本 + 求解器身份。
- `request_id`、优先级、等待预算、应用 `state_id` 不进入缓存键。
- v1 不做花色/座位对称归一化，避免引入结果还原错误。

完全相同的在途求解合并；成功的精确结果才缓存。优先级采用带公平性的调度，允许低优先级随等待时间提升，避免离线任务永久饥饿。相关局面可以尽力分配到相同上下文，但仅是优化，不能影响正确性。

## 8. 进程通信

第一版建议本机 TCP 长连接，仅监听 `127.0.0.1`，请求与响应均为长度前缀的 UTF-8 JSON：4 字节无符号大端长度 + 消息体。单帧上限初始 8 MiB，超限明确拒绝。

消息由 `request_id` 关联，允许响应完成顺序不同；客户端串行化连接写入，避免帧交错。不要依赖 TCP 单次读取恰好拿到一条完整消息。

接口契约不依赖此传输。未来可以更换为命名管道或 gRPC。第一版不承诺远程认证、跨机器部署、流式推送、持久化任务或服务重启后的 job 恢复。

## 9. 与训练环境的连接

叫牌：环境生成完整牌局后可提前提交 `solve_table`；叫牌结束后用最终庄家/将牌查表，在环境适配层计分。固定训练集可直接预计算并持久化牌表，避免在线反复求解。

暗牌出牌：策略基于合法玩家观察生成隐藏牌样本，批量调用 `solve_positions`；按采样权重、计分方式汇总候选结果。服务不会生成隐藏牌，也不会从全知墩数自动得出真实成约概率。实际隐藏手牌不应流入正常策略的采样评价路径。

异步求解结果必须匹配调用方的 `state_id`，避免把旧牌局结果应用到更新后的局面。

## 10. 实现顺序和验收

1. 数据类型、合法性校验、牌编码转换；原生单线程 `solve_table`，与上游样例逐格比对。
2. `solve_position`；测试部分当前墩、分值视角、等价牌展开和唯一合法动作的精确分值。
3. 固定线程池与批量调用；串行/并行结果一致、上下文互不共享、结果保持输入对应关系。
4. 独立进程、客户端、缓存和背压；测试排队过期、重试去重、部分失败、取消共享请求和断连。
5. 接入 OpenSpiel 环境适配层，比较旧版 DDS 与新版结果，再测吞吐、p95/p99 等待时间和内存。

不在 v1 中加入叫牌策略、牌情模型、IMP 计算、任意搜索参数、求解中的硬取消或自动无限扩容。
