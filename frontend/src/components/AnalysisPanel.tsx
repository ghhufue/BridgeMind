import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  BrainCircuit,
  Cable,
  ChevronRight,
  CircleHelp,
  LoaderCircle,
  ScanEye,
  Sparkles,
} from "lucide-react";
import {
  cardLabel,
  controller,
  legalCards,
  seatName,
  stateId,
} from "../bridge/engine";
import {
  fetchAnalysis,
  type AnalysisKind,
  type AnalysisResponse,
} from "../api/analysis";
import type { Card, Game } from "../bridge/types";
import { ModelDiagnostics } from "./ModelDiagnostics";

type Result = {
  stateId: string;
  serverUrl: string;
  status: "loading" | "success" | "error";
  data?: AnalysisResponse;
  error?: string;
  ms?: number;
};

export function AnalysisPanel({
  game,
  serverUrl,
  onSettings,
  onSuggest,
  onRequest,
  disabled,
  compact = false,
  modelId,
  onlyKind,
}: {
  game: Game;
  serverUrl: string;
  onSettings: () => void;
  onSuggest: (cards: Card[], positionId?: string) => void;
  onRequest: () => void;
  disabled: boolean;
  compact?: boolean;
  modelId?: string;
  onlyKind?: AnalysisKind;
}) {
  const [activeKind, setActiveKind] = useState<AnalysisKind>("ai");
  const [results, setResults] = useState<Partial<Record<AnalysisKind, Result>>>(
    {},
  );
  const aborts = useRef<Partial<Record<AnalysisKind, AbortController>>>({});
  const liveState = useRef(stateId(game));
  liveState.current = stateId(game);
  const currentId = stateId(game);
  useEffect(() => {
    Object.values(aborts.current).forEach((c) => c.abort());
    aborts.current = {};
    setResults({});
    onSuggest([]);
    return () => {
      Object.values(aborts.current).forEach((c) => c.abort());
    };
  }, [currentId, serverUrl, onSuggest, modelId]);

  async function request(kind: AnalysisKind) {
    onRequest();
    const requestId = stateId(game);
    const ac = new AbortController();
    aborts.current[kind]?.abort();
    aborts.current[kind] = ac;
    setResults((r) => ({
      ...r,
      [kind]: { stateId: requestId, serverUrl, status: "loading" },
    }));
    const started = performance.now();
    try {
      const data = await fetchAnalysis(game, kind, serverUrl, ac.signal, kind === "ai" ? modelId : undefined);
      if (ac.signal.aborted || liveState.current !== requestId) return;
      setResults((r) => ({
        ...r,
        [kind]: {
          stateId: requestId,
          serverUrl,
          status: "success",
          data,
          ms: Math.round(performance.now() - started),
        },
      }));
      if (kind === "ai") onSuggest(data.bestCards, requestId);
    } catch (error) {
      if (ac.signal.aborted || liveState.current !== requestId) return;
      const message =
        error instanceof Error && error.name === "TimeoutError"
          ? "请求超时，请检查电脑上的分析服务。"
          : error instanceof TypeError
            ? "无法连接服务，请检查地址、局域网和 CORS 配置。"
            : error instanceof Error
              ? error.message
              : "分析请求失败";
      setResults((r) => ({
        ...r,
        [kind]: {
          stateId: requestId,
          serverUrl,
          status: "error",
          error: message,
        },
      }));
    }
  }

  const available = game.phase === "play" && legalCards(game).length > 0;
  return (
    <div className={`analysis-content ${compact ? "compact-analysis" : ""}`}>
      <div className="section-heading">
        <div>
          <span className="eyebrow">A SECOND OPINION</span>
          <h2>{onlyKind === "dds" ? "DDS 全知求解" : "当前模型的真实输出"}</h2>
        </div>
        <Sparkles size={20} />
      </div>
      <p className="panel-intro">
        查看动作概率、牌张归属和价值预测；DDS 作为全知对照单独展示。
      </p>
      {compact && !onlyKind && (
        <div className="m-analysis-tabs" role="tablist" aria-label="分析来源">
          <button
            role="tab"
            aria-selected={activeKind === "ai"}
            onClick={() => setActiveKind("ai")}
          >
            <BrainCircuit size={14} />
            AI 实战
          </button>
          <button
            role="tab"
            aria-selected={activeKind === "dds"}
            onClick={() => setActiveKind("dds")}
          >
            <ScanEye size={14} />
            DDS 全知
          </button>
        </div>
      )}
      {(["ai", "dds"] as const)
        .filter((kind) => onlyKind ? kind === onlyKind : !compact || kind === activeKind)
        .map((kind) => {
          const stored = results[kind];
          const result =
            stored?.stateId === currentId && stored.serverUrl === serverUrl
              ? stored
              : undefined;
          return (
            <section className={`advice-card ${kind}`} key={kind}>
              <div className="advice-title">
                <span className="advice-icon">
                  {kind === "ai" ? (
                    <BrainCircuit size={20} />
                  ) : (
                    <ScanEye size={20} />
                  )}
                </span>
                <div>
                  <h3>{kind === "ai" ? "AI 实战建议" : "DDS 全知分析"}</h3>
                  <span>
                    {kind === "ai" ? "基于当前可见信息" : "基于四家完整手牌"}
                  </span>
                </div>
                <span className="small-tag">
                  {kind === "ai" ? "MODEL" : "SOLVER"}
                </span>
              </div>
              <button
                className="advice-action"
                disabled={
                  !!serverUrl &&
                  (!available || disabled || result?.status === "loading")
                }
                onClick={() => (serverUrl ? request(kind) : onSettings())}
              >
                {serverUrl
                  ? result?.status === "success"
                    ? "重新分析"
                    : "请求分析"
                  : "配置分析服务"}
                <ArrowUpRight size={15} />
              </button>
              {result?.status === "success" && result.data ? (
                <div className="advice-result">
                  <span className="muted">{seatName[game.turn]}家最佳出牌</span>
                  <div className="best-cards">
                    {result.data.bestCards.map((card) => (
                      <strong
                        key={card}
                        className={["H", "D"].includes(card[0]) ? "red" : ""}
                      >
                        {cardLabel(card)}
                      </strong>
                    ))}
                  </div>
                  {!(kind === "ai" && result.data.diagnostics) && result.data.candidates
                    ?.slice()
                    .sort(
                      (a, b) =>
                        (b.probability ?? b.tricks ?? 0) -
                        (a.probability ?? a.tricks ?? 0),
                    )
                    .map((candidate) => (
                      <div className="candidate" key={candidate.card}>
                        <span>{cardLabel(candidate.card)}</span>
                        <div className="candidate-bar">
                          <i
                            style={{
                              width: `${(candidate.probability ?? (candidate.tricks ?? 0) / 13) * 100}%`,
                            }}
                          />
                        </div>
                        <span>
                          {kind === "ai" && candidate.probability !== undefined
                            ? `${(candidate.probability * 100).toFixed(1)}%`
                            : candidate.tricks !== undefined
                              ? `${candidate.tricks} 墩`
                              : "—"}
                        </span>
                      </div>
                    ))}
                  <p className="result-meta">
                    {result.data.model || (kind === "ai" ? "远程模型" : "DDS")}{" "}
                    · {result.ms} ms
                  </p>
                  {kind === "ai" && <ModelDiagnostics key={currentId} data={result.data} game={game} />}
                </div>
              ) : (
                <div className="empty-advice">
                  <span className="empty-dash">
                    {result?.status === "loading" ? (
                      <LoaderCircle className="spin" size={25} />
                    ) : (
                      "—"
                    )}
                  </span>
                  <p role={result?.status === "error" ? "alert" : undefined}>
                    {result?.status === "error"
                      ? result.error
                      : result?.status === "loading"
                        ? "正在分析当前局面…"
                        : !available
                          ? "进入出牌阶段后可请求分析"
                          : !serverUrl
                            ? "连接电脑上的服务后获取建议"
                            : `准备分析${seatName[game.turn]}家当前出牌`}
                  </p>
                </div>
              )}

            </section>
          );
        })}
      <div className="analysis-note">
        <CircleHelp size={16} />
        <p>
          AI 建议与 DDS 最优可能不同。DDS
          知道所有隐藏牌，适合复盘对照。余墩评价以当前出牌方队伍为视角。
        </p>
      </div>
      {available && (
        <div className="controller-note">
          当前控制者：{seatName[controller(game)]}家 · 实际出牌：
          {seatName[game.turn]}家
        </div>
      )}
      <button className="connection-strip" onClick={onSettings}>
        <Cable size={17} />
        <span>
          {serverUrl ? "已配置服务 · 点击修改" : "分析服务未连接"}
          <small>{serverUrl || "支持局域网 HTTP / HTTPS 接口"}</small>
        </span>
        <ChevronRight size={17} />
      </button>
    </div>
  );
}
