import {
  controller,
  dummySeat,
  dummyVisible,
  legalCards,
  legalCalls,
  stateId,
} from "../bridge/engine";
import { SEATS, type Card, type Game, type Seat, type Call } from "../bridge/types";

export type AnalysisKind = "ai" | "dds";
export interface ServiceHealth {
  protocol: "bridgemind.analysis.v1";
  service: "BridgeMind LAN";
  status: "ready" | "degraded";
  services: {
    ai: { ready: boolean; model: string | null };
    dds: { ready: boolean; version: string | null };
    bidding?: { ready: boolean; model: string | null };
  };
}

export async function fetchHealth(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<ServiceHealth> {
  const response = await fetch(`${normalizeServerUrl(baseUrl)}/api/v1/health`, {
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(5000)])
      : AbortSignal.timeout(5000),
  });
  if (!response.ok) throw new Error(`服务返回 HTTP ${response.status}`);
  const data = await response.json();
  if (
    data?.protocol !== "bridgemind.analysis.v1" ||
    data?.service !== "BridgeMind LAN" ||
    !["ready", "degraded"].includes(data?.status) ||
    typeof data?.services?.ai?.ready !== "boolean" ||
    typeof data?.services?.dds?.ready !== "boolean"
  )
    throw new Error("这个地址不是 BridgeMind 分析服务");
  return data;
}
export interface AnalysisCandidate {
  card: Card;
  probability?: number;
  tricks?: number;
}
export interface AnalysisResponse {
  protocol: "bridgemind.analysis.v1";
  stateId: string;
  source: AnalysisKind;
  bestCards: Card[];
  candidates?: AnalysisCandidate[];
  model?: string;
  diagnostics?: ModelDiagnostics;
}

export interface ModelDiagnostics {
  modelSha256: string | null;
  inference: "greedy_neural";
  actionProbabilities: { card: Card; legal: boolean; probability: number }[];
  ownership: { card: Card; status: "known" | "unknown" | "played"; probabilities: Record<Seat, number> }[];
  ownershipMethod: string;
  value: { normalized: number; predictedScore: number; perspective: "NS" | "EW"; target: "actual_terminal_team_score/7600" };
  winProbability: null;
  winProbabilityReason: string;
}

export function biddingRequest(g: Game) {
  if (g.phase !== "auction") throw new Error("当前不在叫牌阶段");
  return { protocol: "bridgemind.analysis.v1", stateId: stateId(g), board: g.board,
    dealer: g.dealer, vulnerable: g.vulnerable, turn: g.turn, auction: g.auction,
    hand: { seat: g.turn, cards: g.hands[g.turn] }, legalCalls: legalCalls(g) };
}

export interface BiddingDecision { call: Call; probabilities: Partial<Record<Call, number>>; model?: string }
export async function fetchBiddingDecision(g: Game, baseUrl: string, signal: AbortSignal, modelId?: string): Promise<BiddingDecision> {
  const response = await fetch(`${normalizeServerUrl(baseUrl)}/api/v1/bidding/ai`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...biddingRequest(g), ...(modelId ? { modelId } : {}) }),
    signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]),
  });
  if (!response.ok) throw new Error(`叫牌服务返回 HTTP ${response.status}`);
  const data = await response.json();
  if (data?.protocol !== "bridgemind.analysis.v1" || data.source !== "bidding" ||
      data.stateId !== stateId(g) || (modelId && data.modelId !== modelId) || !legalCalls(g).includes(data.call))
    throw new Error("叫牌模型返回了过期或非法叫品");
  const probabilities = data.probabilities;
  if (!probabilities || typeof probabilities !== "object" || Array.isArray(probabilities) ||
      Object.entries(probabilities).some(([call, p]) => !legalCalls(g).includes(call as Call) || typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1) ||
      Math.abs(Object.values(probabilities).reduce<number>((sum, p) => sum + Number(p), 0) - 1) > 1e-4)
    throw new Error("叫牌概率格式错误");
  return { call: data.call, probabilities, model: data.model };
}

export async function fetchBidding(g: Game, baseUrl: string, signal: AbortSignal, modelId?: string): Promise<Call> {
  return (await fetchBiddingDecision(g, baseUrl, signal, modelId)).call;
}

export function analysisRequest(g: Game, kind: AnalysisKind) {
  if (g.phase !== "play" || !legalCards(g).length)
    throw new Error("请在出牌回合请求分析");
  const actor = controller(g);
  const dummy = dummySeat(g)!;
  const common = {
    protocol: "bridgemind.analysis.v1" as const,
    stateId: stateId(g),
    board: g.board,
    dealer: g.dealer,
    vulnerable: g.vulnerable,
    contract: g.contract,
    toPlay: g.turn,
    controller: actor,
    auction: g.auction,
    currentTrick: g.currentTrick,
    tricks: g.tricks,
    legalCards: legalCards(g),
  };
  if (kind === "dds")
    return { ...common, information: "full" as const, hands: g.hands };
  return {
    ...common,
    information: "visible" as const,
    hand: { seat: actor, cards: g.hands[actor] },
    dummy: dummyVisible(g) ? { seat: dummy, cards: g.hands[dummy] } : null,
    remainingCounts: Object.fromEntries(
      Object.entries(g.hands).map(([s, h]) => [s, h.length]),
    ),
  };
}

export function validateAnalysis(
  data: unknown,
  g: Game,
  kind: AnalysisKind,
): AnalysisResponse {
  if (!data || typeof data !== "object")
    throw new Error("分析服务返回格式错误");
  const r = data as Record<string, unknown>;
  if (
    r.protocol !== "bridgemind.analysis.v1" ||
    r.source !== kind ||
    r.stateId !== stateId(g)
  )
    throw new Error("服务返回的协议、来源或局面版本不匹配");
  const legal = legalCards(g);
  if (
    !Array.isArray(r.bestCards) ||
    r.bestCards.length === 0 ||
    r.bestCards.some((c) => !legal.includes(c)) ||
    new Set(r.bestCards).size !== r.bestCards.length
  )
    throw new Error("服务返回了非法或重复的最优牌");
  if (r.model !== undefined && typeof r.model !== "string")
    throw new Error("模型名称格式错误");
  if (r.candidates !== undefined) {
    if (
      !Array.isArray(r.candidates) ||
      r.candidates.some((c) => {
        if (!c || typeof c !== "object" || !legal.includes(c.card)) return true;
        if (
          c.probability !== undefined &&
          (typeof c.probability !== "number" ||
            !Number.isFinite(c.probability) ||
            c.probability < 0 ||
            c.probability > 1)
        )
          return true;
        return (
          c.tricks !== undefined &&
          (typeof c.tricks !== "number" ||
            !Number.isInteger(c.tricks) ||
            c.tricks < 0 ||
            c.tricks > 13 - g.tricks.length)
        );
      })
    )
      throw new Error("服务返回了无效的候选评价");
    if (new Set(r.candidates.map((c) => c.card)).size !== r.candidates.length)
      throw new Error("候选牌重复");
  }
  if (r.diagnostics !== undefined) {
    if (kind !== "ai") throw new Error("DDS 响应不能伪装为模型诊断");
    const d = r.diagnostics as ModelDiagnostics;
    const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
    const probability = (v: unknown): v is number => finite(v) && v >= 0 && v <= 1;
    const fullDeck = SEATS.flatMap((s) => g.initialHands[s]);
    const played = new Set([...g.tricks.flatMap((t) => t.cards), ...g.currentTrick].map((e) => e.card));
    const request = analysisRequest(g, "ai");
    const known = new Set("hand" in request ? [...request.hand.cards, ...(request.dummy?.cards || [])] : []);
    if (!d || d.inference !== "greedy_neural" ||
        !(d.modelSha256 === null || (typeof d.modelSha256 === "string" && /^[a-f0-9]{64}$/.test(d.modelSha256))) ||
        !d.value || !finite(d.value.normalized) || !finite(d.value.predictedScore) ||
        Math.abs(d.value.predictedScore-d.value.normalized*7600)>1e-3 ||
        d.value.target !== "actual_terminal_team_score/7600" ||
        d.value.perspective !== (["N", "S"].includes(controller(g)) ? "NS" : "EW") ||
        d.winProbability !== null || typeof d.winProbabilityReason !== "string" || typeof d.ownershipMethod !== "string")
      throw new Error("模型诊断或价值格式错误");
    if (!Array.isArray(d.actionProbabilities) || d.actionProbabilities.length !== g.hands[g.turn].length ||
        new Set(d.actionProbabilities.map((a) => a.card)).size !== d.actionProbabilities.length ||
        d.actionProbabilities.some((a) => !g.hands[g.turn].includes(a.card) || a.legal !== legal.includes(a.card) ||
          !probability(a.probability) || (!a.legal && a.probability !== 0)) ||
        Math.abs(d.actionProbabilities.reduce((sum, a) => sum+a.probability, 0)-1)>1e-4)
      throw new Error("动作概率不完整或未归一化");
    if (!Array.isArray(d.ownership) || d.ownership.length !== 52 || new Set(d.ownership.map((a) => a.card)).size !== 52 ||
        d.ownership.some((a) => !fullDeck.includes(a.card) ||
          a.status !== (played.has(a.card) ? "played" : known.has(a.card) ? "known" : "unknown") ||
          !a.probabilities || SEATS.some((s) => !probability(a.probabilities[s])) ||
          Math.abs(SEATS.reduce((sum, s) => sum+a.probabilities[s], 0)-1)>1e-4))
      throw new Error("归属概率不完整或未归一化");
  }
  return data as AnalysisResponse;
}

export function normalizeServerUrl(input: string) {
  const url = new URL(input.trim());
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new Error("请输入 HTTP(S) 服务地址，例如 http://192.168.1.10:8000");
  return url.toString().replace(/\/$/, "");
}

export async function fetchAnalysis(
  g: Game,
  kind: AnalysisKind,
  baseUrl: string,
  signal: AbortSignal,
  modelId?: string,
) {
  const url = normalizeServerUrl(baseUrl);
  const timeout = AbortSignal.timeout(15_000);
  const response = await fetch(`${url}/api/v1/analysis/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...analysisRequest(g, kind), ...(kind === "ai" && modelId ? { modelId } : {}) }),
    signal: AbortSignal.any([signal, timeout]),
  });
  if (!response.ok) throw new Error(`服务返回 HTTP ${response.status}`);
  const data = await response.json();
  if (kind === "ai" && modelId && data?.modelId !== modelId) throw new Error("服务返回的模型与选择不匹配");
  return validateAnalysis(data, g, kind);
}

export interface ModelOption { id: string; name: string; available: boolean }
export interface ModelCatalog { protocol: "bridgemind.models.v1"; play: ModelOption[]; bidding: ModelOption[] }
export async function fetchModels(baseUrl: string, signal: AbortSignal): Promise<ModelCatalog> {
  const response = await fetch(`${normalizeServerUrl(baseUrl)}/api/v1/models`, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(5000)]),
  });
  if (!response.ok) throw new Error(`模型列表返回 HTTP ${response.status}，请确认电脑服务已更新。`);
  const data = await response.json();
  if (data?.protocol !== "bridgemind.models.v1" || [data.play, data.bidding].some(rows =>
    !Array.isArray(rows) || !rows.length || rows.length > 32 || new Set(rows.map(row => row?.id)).size !== rows.length ||
    rows.some(row => !row || typeof row.id !== "string" || !/^[a-z0-9-]{1,64}$/.test(row.id) || typeof row.name !== "string" || typeof row.available !== "boolean")))
    throw new Error("模型列表格式错误");
  return data;
}
