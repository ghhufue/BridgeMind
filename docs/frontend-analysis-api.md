# 前端局域网分析接口 v1

## 四家模型选择扩展

`GET /api/v1/models`返回`{protocol:'bridgemind.models.v1',play:[{id,name,available}],bidding:[{id,name,available}]}`。目录只列服务器允许列表中实际存在的固定权重，按需加载并缓存，不接受客户端权重路径。`default`使用启动时加载的策略；历史SL、PPO按独立ID选择，不改变人机对战的默认策略。`play`中的`dds`是独立全知求解选项，应调用原DDS接口，不作为AI模型ID发送。

`POST /api/v1/analysis/ai`与`POST /api/v1/bidding/ai`可额外发送可选`modelId`；响应会回显同一ID，客户端核对后才执行动作。未提供时保持原协议。选择模型不会改变信息边界，服务先校验原局面再按ID加载权重；非法ID、客户端路径、隐藏信息和DDS请求中的modelId均拒绝。明手按庄家模型选择，观战全知开关不进入任何模型请求。多个模型共享服务推理锁，默认与备选模型对象互不替换。

状态：**前端客户端、Python HTTP 网关、真实出牌模型与 DDS 接入均已实现**。网关代码在 `src/bridgemind/web/`，复用既有模型和 DDS TCP 服务，不改变训练流程。启动与连接见 [局域网使用说明](../frontend/LAN.md)。

## 传输

在界面设置中填写基地址，例如 `http://192.168.5.118:8000`。客户端发送：

| 方法与路径 | 语义 |
| --- | --- |
| `GET /api/v1/health` | 检查模型、DDS 就绪状态及版本 |
| `POST /api/v1/analysis/ai` | 根据当前控制者的可见信息，返回实战建议 |
| `POST /api/v1/analysis/dds` | 根据四家真实剩余牌，返回全知最优出牌 |
| `POST /api/v1/bidding/ai` | 根据当前叫牌者一手牌和公开叫牌，返回合法叫品 |

请求和响应均为 `Content-Type: application/json`。首次跨域请求会有 `OPTIONS` 预检；网关已允许当前电脑来源、localhost 开发来源和 Android debug 的 `http://localhost`，包括 `GET, POST, OPTIONS` 与 `Content-Type`。额外来源使用启动参数 `--origin`。当前客户端不携带认证 cookie，也没有跨用户登录功能；服务仅用于局域网调试。

开发时手机与电脑可同时使用普通局域网 HTTP，避免 HTTPS 页面调用 HTTP 服务被浏览器混合内容策略阻止。后续正式部署应让页面和接口通过同一 HTTPS 来源访问，或为两端都配置 HTTPS。

客户端超时为 15 秒。超时和取消只中止客户端等待，不承诺停止服务器上的模型或原生 DDS 计算。请求失败不自动重试；用户可重试。网关返回非 2xx 状态时，界面显示错误。

## 健康检查

设置中的“测试连接”会以 5 秒超时请求健康检查，响应示例：

```json
{
  "protocol": "bridgemind.analysis.v1",
  "service": "BridgeMind LAN",
  "status": "ready",
  "processId": 4812,
  "services": {
    "ai": {"ready": true, "model": "play-sl-long-20261025 · 纯网络"},
    "dds": {"ready": true, "version": "3.1.0"}
  }
}
```

`processId` 随启动变化；任一服务不可用时状态为 `degraded`。客户端会验证协议和服务标识，避免把普通 HTTP 页面当作就绪的分析服务。

## 公共类型与编号

- 座位：`N,E,S,W`，按顺时针排列。
- 牌：`SA,HT,C2`，花色 `C,D,H,S`，点数 `23456789TJQKA`；T 代表 10。
- 叫品：`P,X,XX,1C,...,7NT`。
- `vulnerable`：`[南北有局, 东西有局]`。
- `contract`：`{ level: 1..7, strain: C|D|H|S|NT, declarer: Seat, doubled: 0|1|2 }`。`doubled` 是前端字段，适配现有 Python `Contract` 时转换为 `double_status`。
- `toPlay`：实际出牌手；`controller`：作决策的人，明手行动时为庄家。两者不同是正常情况。
- `currentTrick`：当前墩已打出的 0–3 张牌，按出牌顺序排列。四张收墩期间前端禁止请求分析。
- `tricks`：已完成的墩，每项 `{ cards: [{seat,card},...], winner }`；`cards` 长度为 4。
- `stateId` 是不透明字符串，服务必须原样回传。它包含牌局 ID、修订号和公开行动历史指纹；撤销后不同的分支不会共用相同标识。

## AI 请求：只提供可见信息

```json
{
  "protocol": "bridgemind.analysis.v1",
  "stateId": "opaque-position-id",
  "board": 3,
  "dealer": "S",
  "vulnerable": [false, true],
  "contract": {"level": 3, "strain": "NT", "declarer": "S", "doubled": 0},
  "toPlay": "W",
  "controller": "W",
  "auction": [
    {"seat":"S","call":"1NT"}, {"seat":"W","call":"P"},
    {"seat":"N","call":"3NT"}, {"seat":"E","call":"P"},
    {"seat":"S","call":"P"}, {"seat":"W","call":"P"}
  ],
  "currentTrick": [],
  "tricks": [],
  "legalCards": ["SA","SQ","S9","S2","HK","H8","H3","DQ","D7","D4","CJ","C6","C2"],
  "information": "visible",
  "hand": {"seat":"W","cards":["SA","SQ","S9","S2","HK","H8","H3","DQ","D7","D4","CJ","C6","C2"]},
  "dummy": null,
  "remainingCounts": {"N":13,"E":13,"S":13,"W":13}
}
```

`hand` 只包含 `controller` 的剩余手牌。首攻前 `dummy=null`，首攻后才为 `{ seat, cards }`。明手行动时，`hand` 是庄家手牌，`dummy` 是明手手牌。其他隐藏牌、完整起手牌和 DDS 数据不会出现在 AI 请求中；界面“查看四手”不会改变该边界。

`tricks` 和 `currentTrick` 包含公开出牌，可按顺序拼成 `play_history`。模型需要的编码、公开缺门、体系 ID、OpenSpiel 动作 ID 等由网关适配，不要求浏览器构造张量。

## DDS 请求：全知

公共字段与 AI 相同，去掉 `hand,dummy,remainingCounts`，改为：

```typescript
{
  // ...公共字段
  information: 'full',
  hands: { N: Card[], E: Card[], S: Card[], W: Card[] }
}
```

`hands` 是四家**剩余牌**，不包含已出牌。无将按 `contract.strain === 'NT'` 处理。当前墩首引者是 `currentTrick[0].seat`；当前墩为空时，引牌者就是 `toPlay`。适配现有 `Position`/`SolverClient.solve_position` 时按实际公开字段转换，不直接复用原生 DDS 位掩码。

## 响应

AI 例子（只说明格式，数值不是模型表现）：

```json
{
  "protocol": "bridgemind.analysis.v1",
  "stateId": "opaque-position-id",
  "source": "ai",
  "bestCards": ["SA"],
  "candidates": [
    {"card":"SA","probability":0.65},
    {"card":"SQ","probability":0.22}
  ],
  "model": "checkpoint-name"
}
```

DDS 例子：

```json
{
  "protocol": "bridgemind.analysis.v1",
  "stateId": "opaque-position-id",
  "source": "dds",
  "bestCards": ["SA","SQ"],
  "candidates": [
    {"card":"SA","tricks":7},
    {"card":"SQ","tricks":7},
    {"card":"S9","tricks":6}
  ],
  "model": "DDS 3.1.0"
}
```

- `bestCards` 必须为非空、去重的合法牌数组，保留全部等价最优牌。
- `candidates` 可省略；存在时每项的 `card` 必须合法且不重复。客户端展示全部返回项。
- `probability` 可省略，存在时必须为 0–1 之间的有限数字，表示模型策略概率，不把它解释为胜率。候选可只返回一部分，不要求这些展示项之和为 1。
- `tricks` 可省略，存在时为 0 到 `13 - tricks.length` 之间的整数，表示**开始求解时实际出牌方队伍可获得的剩余墩数，包含尚未完成的当前墩**。不是庄家总墩数或绝对分数。
- `model` 是可选的展示名称。界面的耗时由客户端实测，包含网络时间。
- 客户端检查协议、来源、局面标识、合法牌和数值范围；不合规响应不会变成可执行建议。建议只展示和高亮，不自动替用户出牌。

## 已实现的网关适配

1. 校验请求、历史与手牌数量，重新验证合法动作，不能信任浏览器传入的 `legalCards`。
2. 将AI可见信息转换为既有出牌编码，与 `BridgePlayEnv` 逐字段对照。当前使用CPU纯网络SL v2，单次前向返回策略、归属和价值三个头，未接入hybrid搜索。叫牌端点使用独立的既有叫牌模型。
3. 将 DDS 全知请求适配到已有独立求解服务；不要把该请求体混入 AI 推理链。
4. 如需重建 OpenSpiel 环境，保留其输入要求；AI 请求并不提供四家完整起手牌，不能用隐藏真值来补齐模型观察。
5. 处理排队、超时、并发、结果来源与模型版本；回传原 `stateId`。

网关还校验叫牌结果、庄家、行动顺序、历史赢家、重复牌、手牌数量和明手公开时机；AI 请求不接受隐藏 `hands` 字段。DDS 每次通过独立客户端调用本地求解服务，等待超时为 11 秒、排队超时为 1 秒。无效请求返回 400，DDS 不可用返回 503，等待超时返回 504，繁忙返回 429。网关同时提供 `frontend/dist` 静态页面；启动脚本绑定本机 WLAN 地址的 8000 端口。

2026-10-26：60 项网关测试通过（Python 全量 156 项）；已实测电脑网页和 Android 手机通过 Wi-Fi 调用真实模型与 DDS。记录见 `artifacts/frontend/lan-validation-20261026.json`。

## 模型调试扩展（2026-10-26）

AI响应新增可选 `diagnostics`，保留v1请求和旧候选字段兼容性：

```typescript
{
  modelSha256: string | null,
  inference: 'greedy_neural',
  actionProbabilities: Array<{card: Card, legal: boolean, probability: number}>,
  ownership: Array<{
    card: Card, status: 'unknown'|'known'|'played',
    probabilities: {N: number, E: number, S: number, W: number}
  }>,
  ownershipMethod: string,
  value: {
    normalized: number, predictedScore: number, perspective: 'NS'|'EW',
    target: 'actual_terminal_team_score/7600'
  },
  winProbability: null,
  winProbabilityReason: string
}
```

`actionProbabilities`覆盖实际出牌手的全部剩余手牌，合法牌概率合计1，非法牌为0。`ownership`恰好52项，每项四座位概率合计1；未知牌使用归属头softmax原始概率，从庄家相对坐标转换为绝对座位，不作张数约束投影。可见手牌与已出牌标为已知事实并固定为one-hot，不把这些事实当成模型预测成绩。客户端仅对未知牌计算本局归属命中率，真值来自本地牌局，不进入请求。

价值视角是当前模型控制者的阵营，`predictedScore = normalized × 7600`。SL v2训练目标是实际终局本方原始得分，不是IMP、完成定约概率或赢牌概率；没有胜率头，所以 `winProbability`明确为空。权重SHA256在加载时记录。客户端验证全部动作、归属完整性、概率范围与归一化、价值缩放和阵营；局面变化立即清除并取消旧分析。

叫牌请求字段为 `protocol,stateId,board,dealer,vulnerable,turn,auction,hand,legalCalls`；其中 `hand={seat:turn,cards:13张牌}`，不得带四手牌。服务独立重放公开叫牌合法性、校验行动顺序、流局和结束条件及合法叫品集合，再只将私有当前手牌与公开字段交给 `BiddingPolicy`。响应为 `protocol,stateId,source:'bidding',call,probabilities,model`。客户端逐回合验证状态标识和合法叫品。

本轮新增28项Python测试，对照OpenSpiel的32局随机叫牌、全部庄家和多个出牌阶段的实际三个输出头；连同旧测试共88项网关测试通过，Python全量298项通过。前端108项测试覆盖主入口、南北控制、自动模型动作、取消/过期/错误处理及调试字段验证。浏览器已实测SL v2、真实叫牌和DDS，含844×390横屏；截图和服务状态在 `artifacts/frontend/model-workbench-20261026/`。Android新版APK构建通过，当前没有连接设备，本轮未安装或验收手机原生界面。
