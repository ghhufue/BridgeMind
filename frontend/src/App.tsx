import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  CircleHelp,
  Eye,
  EyeOff,
  Flag,
  History,
  Layers3,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Settings2,
  ShieldCheck,
  Spade,
  Trophy,
  Volume2,
  VolumeX,
  X,
  Cable,
  LoaderCircle,
} from "lucide-react";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { PlayingCard, TableCard } from "./components/PlayingCard";
import {
  applyCall,
  callLabel,
  cardLabel,
  collectTrick,
  controller,
  dummySeat,
  dummyVisible,
  gameResult,
  highCardPoints,
  legalCalls,
  legalCards,
  newGame,
  nextSeat,
  playCard,
  seatName,
  stateId,
  suitSymbol,
  trickCounts,
  trickWinner,
} from "./bridge/engine";
import {
  SEATS,
  STRAINS,
  type Card,
  type Game,
  type Seat,
  type Suit,
} from "./bridge/types";
import {
  fetchHealth,
  normalizeServerUrl,
  type ServiceHealth,
} from "./api/analysis";
import { playSound, setSound } from "./audio";
import { MobileTable, type TableModal } from "./components/MobileTable";
import { useTableLayout } from "./useTableLayout";
import { fetchAnalysis, fetchBidding } from "./api/analysis";
import { DebugWorkspace } from "./components/DebugWorkspace";
import { HomeScreen } from "./components/HomeScreen";
import { useAppUpdates, type AppUpdates } from "./updates";
import { UpdateNotice, UpdateSettings } from "./components/AppUpdates";

function loadPreferences() {
  try {
    const value = JSON.parse(
      localStorage.getItem("bridgemind.preferences.v1") || "{}",
    ) as {
      serverUrl?: string;
      sound?: boolean;
      manual?: boolean;
      reduced?: boolean;
      connectionVersion?: number;
    };
    return value && typeof value === "object" && !Array.isArray(value)
      ? value
      : {};
  } catch {
    return {};
  }
}
const preferences = loadPreferences();
const defaultServerUrl =
  window.location.port === "8000"
    ? window.location.origin
    : import.meta.env.VITE_ANALYSIS_SERVER_URL || "";
const systemReduced = window.matchMedia(
  "(prefers-reduced-motion: reduce)",
).matches;

function Modal({
  title,
  onClose,
  children,
  className = "",
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={`modal ${className}`}
      aria-label={title}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      <div className="modal-heading">
        <div>
          <span className="eyebrow">BRIDGEMIND</span>
          <h2>{title}</h2>
        </div>
        <button className="icon-button" aria-label="关闭" onClick={onClose}>
          <X size={21} />
        </button>
      </div>
      {children}
    </dialog>
  );
}

function SettingsModal({
  updates,
  onInstall,
  serverUrl,
  manual,
  reduced,
  onSave,
  onClose,
}: {
  updates: AppUpdates;
  onInstall: () => void;
  serverUrl: string;
  manual: boolean;
  reduced: boolean;
  onSave: (url: string, manual: boolean, reduced: boolean) => void;
  onClose: () => void;
}) {
  const [url, setUrl] = useState(serverUrl);
  const [mode, setMode] = useState(manual);
  const [motion, setMotion] = useState(reduced);
  const [error, setError] = useState("");
  const [health, setHealth] = useState<ServiceHealth | null>(null);
  const [testing, setTesting] = useState(false);
  const testAbort = useRef<AbortController | null>(null);
  useEffect(() => {
    setHealth(null);
    testAbort.current?.abort();
    setTesting(false);
    return () => testAbort.current?.abort();
  }, [url]);
  async function testConnection() {
    const ac = new AbortController();
    testAbort.current?.abort();
    testAbort.current = ac;
    setTesting(true);
    setError("");
    setHealth(null);
    try {
      const value = await fetchHealth(url, ac.signal);
      if (!ac.signal.aborted) setHealth(value);
    } catch (error) {
      if (!ac.signal.aborted)
        setError(
          error instanceof TypeError
            ? "连接失败，请确认电脑服务已启动，且手机和电脑在同一局域网。"
            : error instanceof Error && error.name === "TimeoutError"
              ? "连接超时，请检查电脑地址和防火墙。"
              : error instanceof Error
                ? error.message
                : "连接失败",
        );
    } finally {
      if (!ac.signal.aborted) setTesting(false);
    }
  }
  return (
    <Modal title="练习与服务设置" onClose={onClose}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          try {
            onSave(url.trim() ? normalizeServerUrl(url) : "", mode, motion);
            onClose();
          } catch (error) {
            setError((error as Error).message);
          }
        }}
      >
        <label className="field-label" htmlFor="server-url">
          电脑上的分析服务
        </label>
        <input
          id="server-url"
          type="text"
          inputMode="url"
          autoComplete="url"
          placeholder="http://192.168.1.10:8000"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
            setError("");
          }}
        />
        <p className="field-help">
          手机和电脑连接同一局域网后，填写电脑的 IP 和 HTTP
          服务端口。人机对战需要模型服务；四手手动练习可离线使用。
        </p>
        <div className="connection-test-actions">
          <button
            type="button"
            className="secondary-button"
            disabled={!url.trim() || testing}
            onClick={testConnection}
          >
            {testing ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Cable size={15} />
            )}
            {testing ? "正在连接…" : "测试连接"}
          </button>
          {defaultServerUrl && url !== defaultServerUrl && (
            <button
              type="button"
              className="text-button"
              onClick={() => setUrl(defaultServerUrl)}
            >
              使用电脑地址
            </button>
          )}
        </div>
        {health && (
          <div className="service-health" role="status">
            <strong>
              {health.status === "ready"
                ? "连接成功，分析服务已就绪"
                : "已连接电脑，部分服务未就绪"}
            </strong>
            <span className={health.services.ai.ready ? "service-ready" : ""}>
              AI {health.services.ai.ready ? "已就绪" : "未加载"}
              {health.services.ai.model ? ` · ${health.services.ai.model}` : ""}
            </span>
            <span className={health.services.dds.ready ? "service-ready" : ""}>
              DDS {health.services.dds.ready ? "已就绪" : "未连接"}
            </span>
            <span className={health.services.bidding?.ready ? "service-ready" : ""}>
              叫牌模型 {health.services.bidding?.ready ? `已就绪 · ${health.services.bidding.model}` : "未加载"}
            </span>
          </div>
        )}
        <div className="endpoint-preview">
          <code>POST /api/v1/analysis/ai</code>
          <code>POST /api/v1/analysis/dds</code>
          <code>POST /api/v1/bidding/ai</code>
          <span>请求包含局面版本；前端不运行模型或 DDS。</span>
        </div>
        <label className="toggle-row">
          <span>
            <strong>四手练习模式</strong>
            <small>
              手动控制所有座位；关闭后南家人工叫牌、南北人工出牌，其余由模型决策。
            </small>
          </span>
          <input
            type="checkbox"
            role="switch"
            checked={mode}
            onChange={(e) => setMode(e.target.checked)}
          />
        </label>
        <label className="toggle-row">
          <span>
            <strong>减少动态效果</strong>
            <small>关闭飞牌路径动画，保留出牌与声音反馈。</small>
          </span>
          <input
            type="checkbox"
            role="switch"
            checked={motion}
            onChange={(e) => setMotion(e.target.checked)}
          />
        </label>
        <UpdateSettings updates={updates} onInstall={onInstall} />
        {error && (
          <p className="error-text" role="alert">
            {error}
          </p>
        )}
        <button className="primary-button full-width" type="submit">
          <Check size={17} />
          保存设置
        </button>
        <p className="settings-footnote">
          服务需要允许当前网页来源的 CORS。现有 DDS TCP 端口不能直接作为 HTTP
          服务使用。
        </p>
      </form>
    </Modal>
  );
}

function AuctionTable({ game }: { game: Game }) {
  const offset = SEATS.indexOf(game.dealer);
  const slots = [
    ...Array.from({ length: offset }, () => null),
    ...game.auction,
  ];
  const count = Math.max(
    4,
    Math.ceil((slots.length + (game.phase === "auction" ? 1 : 0)) / 4) * 4,
  );
  return (
    <div className="auction-table">
      <div className="auction-table-head">
        {SEATS.map((s) => (
          <span key={s} className={s === "S" ? "your-seat" : ""}>
            {seatName[s]}
            {s === game.dealer && <i title="发牌人">D</i>}
          </span>
        ))}
      </div>
      <div className="auction-table-body">
        {Array.from({ length: count }, (_, i) => {
          const entry = slots[i];
          const active = game.phase === "auction" && i === slots.length;
          return (
            <span
              key={i}
              className={`${active ? "current-call" : ""} ${entry && /[HD]/.test(entry.call) ? "red" : ""} ${entry?.call === "P" ? "pass-call" : ""}`}
            >
              {entry ? callLabel(entry.call) : active ? "···" : "—"}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function Records({ game }: { game: Game }) {
  return (
    <div className="records-content">
      <div className="section-heading">
        <div>
          <span className="eyebrow">THE STORY SO FAR</span>
          <h2>对局记录</h2>
        </div>
        <History size={20} />
      </div>
      <h3 className="record-title">
        叫牌进程 <span>{game.auction.length} 次叫品</span>
      </h3>
      <AuctionTable game={game} />
      <h3 className="record-title">
        已完成的墩 <span>{game.tricks.length} / 13</span>
      </h3>
      {!game.tricks.length && (
        <div className="record-empty">
          <Layers3 size={27} />
          <p>每一墩，都会留在这里。</p>
        </div>
      )}
      {[...game.tricks].reverse().map((trick, i) => (
        <div className="trick-record" key={game.tricks.length - i}>
          <div>
            <b>第 {game.tricks.length - i} 墩</b>
            <span>
              {seatName[trick.winner]}家赢墩 <Flag size={12} />
            </span>
          </div>
          <div className="trick-record-cards">
            {trick.cards.map((c) => (
              <span
                key={c.seat}
                className={["H", "D"].includes(c.card[0]) ? "red" : ""}
              >
                <small>{seatName[c.seat]}</small>
                {cardLabel(c.card)}
                {c.seat === trick.winner && <i>✓</i>}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const mobile = useTableLayout();
  const [view, setView] = useState<"home" | "play" | "debug">("home");
  const [busy, setBusy] = useState(false);
  const [modelError, setModelError] = useState("");
  const [history, setHistory] = useState<Game[]>(() => [newGame()]);
  const game = history[history.length - 1];
  const [tab, setTab] = useState<"analysis" | "records">("analysis");
  const [modal, setModal] = useState<TableModal>(null);
  const [serverUrl, setServerUrl] = useState(
    preferences.connectionVersion === 2
      ? preferences.serverUrl || ""
      : preferences.serverUrl || defaultServerUrl,
  );
  const [manual, setManual] = useState(preferences.manual ?? false);
  const [reduced, setReduced] = useState(preferences.reduced ?? systemReduced);
  const [sound, setSoundEnabled] = useState(preferences.sound ?? false);
  const [reveal, setReveal] = useState(false);
  const [paused, setPaused] = useState(false);
  const updates = useAppUpdates(serverUrl);
  const installUpdate = () => { setPaused(true); void updates.install(); };
  const [level, setLevel] = useState(1);
  const [suggestion, setSuggestion] = useState<{
    stateId: string;
    cards: Card[];
  }>({ stateId: "", cards: [] });
  const suggested =
    suggestion.stateId === stateId(game) ? suggestion.cards : [];
  const [error, setError] = useState("");
  const stableSuggest = useCallback(
    (cards: Card[], positionId = "") =>
      setSuggestion({ stateId: positionId, cards }),
    [],
  );
  const myTurn = manual || (game.phase === "auction" ? game.turn === "S" : ["N", "S"].includes(game.turn));
  const currentPosition = useRef(stateId(game));
  currentPosition.current = stateId(game);
  const stepAbort = useRef<AbortController | null>(null);
  const inFlight = useRef(false);
  useEffect(() => () => stepAbort.current?.abort(), []);
  const calls = legalCalls(game);
  const cards = legalCards(game);
  const counts = trickCounts(game);
  const contract = game.contract;
  const dummy = dummySeat(game);
  const result = gameResult(game);

  useEffect(() => {
    if (
      !mobile &&
      (modal === "analysis" || modal === "records" || modal === "auction")
    )
      setModal(null);
  }, [mobile, modal]);

  useEffect(() => {
    setSound(sound);
    try {
      localStorage.setItem(
        "bridgemind.preferences.v1",
        JSON.stringify({
          serverUrl,
          sound,
          manual,
          reduced,
          connectionVersion: 2,
        }),
      );
    } catch {
      /* private browsing can disallow storage */
    }
  }, [serverUrl, sound, manual, reduced]);

  useEffect(() => {
    setSuggestion({ stateId: "", cards: [] });
  }, [game.id, game.revision, serverUrl]);

  const commit = useCallback((fn: (g: Game) => Game) => {
    setError("");
    setHistory((previous) => {
      const current = previous[previous.length - 1], next = fn(current);
      return next === current ? previous : [...previous, next];
    });
  }, []);

  useEffect(() => {
    if (
      modal ||
      view !== "play" ||
      paused ||
      game.phase !== "play" ||
      game.currentTrick.length !== 4
    )
      return;
    const timer = setTimeout(
      () => {
        commit(collectTrick);
        playSound("trick");
      },
      reduced ? 900 : 1500,
    );
    return () => clearTimeout(timer);
  }, [game, modal, paused, reduced, commit, view]);

  useEffect(() => {
    if (
      manual || myTurn || paused || view !== "play" ||
      modelError ||
      modal ||
      game.phase === "finished" ||
      game.currentTrick.length === 4
    )
      return;
    const ac = new AbortController();
    const timer = setTimeout(() => { if (!inFlight.current) void modelStep(ac); }, 650);
    return () => { clearTimeout(timer); ac.abort(); };
  }, [game, manual, myTurn, paused, modal, commit, view, modelError, serverUrl]);

  useEffect(() => {
    setModelError("");
    stepAbort.current?.abort();
    stepAbort.current = null; inFlight.current = false;
    setBusy(false);
  }, [game.id, serverUrl, view]);

  async function modelStep(ac: AbortController) {
    const position = stateId(game);
    stepAbort.current?.abort(); stepAbort.current = ac;
    inFlight.current = true;
    setBusy(true); setModelError("");
    try {
      if (!serverUrl) throw new Error("请先在设置中连接电脑模型服务。");
      if (game.phase === "auction") {
        const call = await fetchBidding(game, serverUrl, ac.signal);
        if (!ac.signal.aborted && currentPosition.current === position) {
          commit((g) => stateId(g) === position ? applyCall(g, call) : g); playSound("call");
        }
      } else if (game.phase === "play") {
        const response = await fetchAnalysis(game, "ai", serverUrl, ac.signal);
        if (!ac.signal.aborted && currentPosition.current === position) {
          commit((g) => stateId(g) === position ? playCard(g, response.bestCards[0]) : g); playSound("card");
        }
      }
    } catch (e) {
      if (!ac.signal.aborted && currentPosition.current === position) {
        setModelError(e instanceof TypeError ? "无法连接模型服务，请检查电脑服务和网络后重试。" : e instanceof Error ? e.message : "模型调用失败");
      }
    } finally {
      if (stepAbort.current === ac) { inFlight.current = false; setBusy(false); }
    }
  }

  useEffect(() => {
    if (!calls.some((c) => c.startsWith(String(level)))) {
      const next = calls.find((c) => /^[1-7]/.test(c));
      if (next) setLevel(Number(next[0]));
    }
  }, [game.revision, calls, level]);

  function makeCall(call: (typeof calls)[number]) {
    if (!myTurn) return;
    if (!calls.includes(call)) {
      setError("请选择合法叫品");
      return;
    }
    commit((g) => applyCall(g, call));
    playSound("call");
  }
  function makePlay(seat: Seat, card: Card) {
    if (!myTurn || game.turn !== seat) return;
    if (!cards.includes(card)) {
      setError("有首引花色时，必须跟出该花色。");
      return;
    }
    commit((g) => playCard(g, card));
    playSound("card");
  }
  function stepPractice() {
    if (busy) return;
    setModelError("");
    if (game.currentTrick.length === 4) {
      commit(collectTrick);
      playSound("trick");
    } else if (game.phase !== "finished") void modelStep(new AbortController());
  }
  function startNext() {
    setHistory([newGame(game.board + 1)]);
    setPaused(false);
    setLevel(1);
    setModal(null);
    setError("");
    setModelError("");
  }
  function exportGame() {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            format: "bridgemind.record.v1",
            ...game,
            result: game.phase === "finished" ? gameResult(game) : undefined,
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bridgemind-board-${game.board}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const status =
    game.phase === "auction"
      ? myTurn
        ? `轮到${seatName[game.turn]}家叫牌`
        : `${seatName[game.turn]}家正在叫牌`
      : game.phase === "finished"
        ? contract
          ? "本副对局已完成"
          : "四家不叫，本副流局"
        : game.currentTrick.length === 4
          ? `${seatName[trickWinner(game.currentTrick, contract!.strain)]}家赢得本墩`
          : myTurn
            ? `轮到${seatName[game.turn]}家出牌`
            : `${seatName[game.turn]}家正在出牌`;

  function seatHand(seat: Seat) {
    const visible =
      reveal ||
      manual ||
      seat === "S" ||
      (seat === dummy && dummyVisible(game)) ||
      game.phase === "finished";
    const active =
      game.phase !== "finished" &&
      game.turn === seat &&
      game.currentTrick.length !== 4;
    const hand = game.hands[seat];
    return (
      <div
        className={`seat-area seat-${seat} ${active ? "active-seat" : ""}`}
        key={seat}
      >
        <div className="seat-label">
          <span className="seat-avatar">{seat}</span>
          <span>
            <strong>
              {seatName[seat]}家{seat === "S" ? " · 你" : ""}
            </strong>
            <small>
              {seat === dummy
                ? "明手"
                : seat === contract?.declarer
                  ? "庄家"
                  : seat === "S"
                    ? manual
                      ? "手动练习"
                      : "你的座位"
                    : manual
                      ? "手动练习"
                      : "练习对手"}
              {seat === game.dealer ? " · 发牌人" : ""}
            </small>
          </span>
          {active && <i className="turn-dot" />}
        </div>
        {visible ? (
          <div className="compact-hand">
            {(["S", "H", "D", "C"] as Suit[]).map((s) => (
              <div
                className={`suit-row ${s === "H" || s === "D" ? "red" : ""}`}
                key={s}
              >
                <span>{suitSymbol[s]}</span>
                <div>
                  {hand
                    .filter((c) => c[0] === s)
                    .map((c) => (
                      <button
                        key={c}
                        title={cardLabel(c)}
                        disabled={
                          game.phase !== "play" ||
                          !active ||
                          !myTurn ||
                          !cards.includes(c)
                        }
                        onClick={() => makePlay(seat, c)}
                      >
                        {c[1] === "T" ? "10" : c[1]}
                      </button>
                    ))}
                  {!hand.some((c) => c[0] === s) && <i>—</i>}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div
            className="hidden-hand"
            aria-label={`${seatName[seat]}家隐藏手牌 ${hand.length} 张`}
          >
            {Array.from(
              { length: Math.min(hand.length, seat === "N" ? 10 : 5) },
              (_, i) => (
                <span className="card-back" key={i} />
              ),
            )}
            <b>{hand.length}</b>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`app-shell ${mobile ? "mobile-shell" : ""}`}>
      {view === "home" ? <HomeScreen board={game.board} onPlay={() => { setView("play"); setPaused(false); setManual(false); }}
        onDebug={() => { setView("debug"); setPaused(true); }} onSettings={() => setModal("settings")} /> : view === "debug" ?
        <DebugWorkspace game={game} serverUrl={serverUrl} reduced={reduced} sound={sound} blocked={!!modal} onCommit={commit}
          onHome={() => setView("home")} onPlay={() => { setView("play"); setPaused(false); setManual(false); }}
          onSettings={() => setModal("settings")} onNext={startNext} onModal={setModal} onSound={() => { setSound(!sound); setSoundEnabled(!sound); }}
          auction={<AuctionTable game={game} />} records={<Records game={game} />}
          onUndo={() => { stepAbort.current?.abort(); setBusy(false); setModelError(""); setHistory((h) => h.length > 1 ? h.slice(0, -1) : h); }}
          canUndo={history.length > 1} /> : mobile ? (
        <MobileTable
          game={game}
          manual={manual}
          partnership
          onHome={() => setView("home")}
          reveal={reveal}
          paused={paused}
          reduced={reduced}
          sound={sound}
          level={level}
          suggested={suggested}
          status={status}
          canUndo={history.length > 1}
          auction={<AuctionTable game={game} />}
          onModal={setModal}
          onReveal={() => setReveal(!reveal)}
          onPause={() => setPaused(!paused)}
          onSound={() => {
            setSound(!sound);
            setSoundEnabled(!sound);
            if (!sound) playSound("call");
          }}
          onLevel={setLevel}
          onCall={makeCall}
          onPlay={makePlay}
          onUndo={() => {
            setPaused(true);
            setHistory((h) => h.slice(0, -1));
            setError("");
          }}
          onStep={stepPractice}
          onNext={startNext}
        />
      ) : (
        <>
          <header className="app-header">
            <a className="brand" href="/" aria-label="BridgeMind 首页">
              <span className="brand-mark">
                <Spade fill="currentColor" size={23} />
              </span>
              <span>
                Bridge<span>Mind</span>
                <small>桥牌练习室</small>
              </span>
            </a>
            <nav aria-label="主导航">
              <span className="nav-active">
                <Layers3 size={17} />
                练习牌桌
              </span>
              <button onClick={() => setModal("help")}>
                <CircleHelp size={17} />
                玩法说明
              </button>
            </nav>
            <div className="header-actions">
              <span className="local-pill">
                <i />
                本地练习
              </span>
              <button
                className="icon-button"
                title={sound ? "关闭音效" : "开启音效"}
                aria-label={sound ? "关闭音效" : "开启音效"}
                onClick={() => {
                  setSound(!sound);
                  setSoundEnabled(!sound);
                  if (!sound) playSound("call");
                }}
              >
                {sound ? <Volume2 size={19} /> : <VolumeX size={19} />}
              </button>
              <button
                className="icon-button"
                aria-label="设置"
                onClick={() => setModal("settings")}
              >
                <Settings2 size={19} />
              </button>
            </div>
          </header>
          <main className="workspace">
            <div className="page-heading">
              <div>
                <span className="eyebrow">YOUR NEXT GREAT HAND</span>
                <h1>
                  把每一手，打得更好。<span>从容叫牌，认真出牌。</span>
                </h1>
              </div>
              <button
                className="primary-button"
                onClick={() => setModal("new")}
              >
                <Plus size={17} />
                新的一副
              </button>
            </div>
            <div className="workspace-grid">
              <section className="game-column" aria-label="桥牌对局">
                <div className="game-toolbar">
                  <div className="board-label">
                    <span>BOARD</span>
                    <strong>{String(game.board).padStart(2, "0")}</strong>
                  </div>
                  <div className="board-meta">
                    <span>
                      <b>{seatName[game.dealer]}</b> 发牌
                    </span>
                    <span
                      className={
                        game.vulnerable[0] || game.vulnerable[1]
                          ? "vulnerable"
                          : ""
                      }
                    >
                      {game.vulnerable.every(Boolean)
                        ? "双方有局"
                        : game.vulnerable[0]
                          ? "南北有局"
                          : game.vulnerable[1]
                            ? "东西有局"
                            : "双方无局"}
                    </span>
                  </div>
                  <div className="stage-track">
                    <span
                      className={
                        game.phase === "auction" ? "stage-active" : "stage-done"
                      }
                    >
                      <i>
                        {game.phase === "auction" ? "1" : <Check size={11} />}
                      </i>
                      叫牌
                    </span>
                    <ChevronRight size={13} />
                    <span
                      className={game.phase === "play" ? "stage-active" : ""}
                    >
                      <i>2</i>出牌
                    </span>
                    <ChevronRight size={13} />
                    <span
                      className={
                        game.phase === "finished" ? "stage-active" : ""
                      }
                    >
                      <i>3</i>结算
                    </span>
                  </div>
                </div>
                <div className={`table ${reduced ? "reduced-motion" : ""}`}>
                  <div className="table-topline">
                    <span>
                      <i />
                      {manual ? "四手练习" : "南家视角"}
                    </span>
                    <button
                      disabled={manual}
                      onClick={() => setReveal(!reveal)}
                      title={reveal ? "隐藏其他手牌" : "查看四家手牌"}
                    >
                      {reveal ? <EyeOff size={14} /> : <Eye size={14} />}
                      {manual ? "四手已公开" : reveal ? "隐藏手牌" : "查看四手"}
                    </button>
                  </div>
                  {seatHand("N")}
                  {seatHand("W")}
                  {seatHand("E")}
                  <div className="table-center">
                    {game.phase === "auction" ? (
                      <div className="auction-center">
                        <div className="table-watermark">
                          <Spade size={24} fill="currentColor" />
                        </div>
                        <span className="table-eyebrow">THE AUCTION</span>
                        <h2>好的一局，从叫牌开始</h2>
                        <div className="table-auction-scroll">
                          <AuctionTable game={game} />
                        </div>
                      </div>
                    ) : game.phase === "finished" ? (
                      <div className="result-center">
                        <span className="result-trophy">
                          <Trophy size={25} />
                        </span>
                        <span className="table-eyebrow">
                          {contract ? "BOARD COMPLETE" : "PASSED OUT"}
                        </span>
                        <h2>
                          {contract
                            ? `${contract.level}${suitSymbol[contract.strain]}${contract.doubled === 2 ? " XX" : contract.doubled ? " X" : ""} ${result.delta === 0 ? "完成定约" : result.delta > 0 ? `超 ${result.delta} 墩` : `宕 ${-result.delta} 墩`}`
                            : "四家不叫"}
                        </h2>
                        <p>
                          {contract
                            ? `${seatName[contract.declarer]}家庄家 · 获得 ${result.tricks} 墩`
                            : "流局计 0 分，准备好下一副了吗？"}
                        </p>
                        <div className="final-score">
                          <small>南北得分</small>
                          <strong>
                            {result.nsScore > 0 ? "+" : ""}
                            {result.nsScore}
                          </strong>
                        </div>
                        <button className="light-button" onClick={startNext}>
                          下一副
                          <ArrowRight size={16} />
                        </button>
                      </div>
                    ) : (
                      <div className="play-center">
                        <div className="compass">
                          <span>N</span>
                          <span>W</span>
                          <Spade size={27} fill="currentColor" />
                          <span>E</span>
                          <span>S</span>
                        </div>
                        {game.currentTrick.length === 0 && (
                          <div className="lead-hint">
                            {game.tricks.length === 0
                              ? "等待首攻"
                              : `${seatName[game.turn]}家引牌`}
                          </div>
                        )}
                        {game.currentTrick.map((c) => (
                          <TableCard
                            key={`${game.tricks.length}-${c.seat}-${c.card}`}
                            {...c}
                            reduced={reduced}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="table-bottomline">
                    <div className="south-badge">
                      <b>S</b>
                      <span>
                        南家 · 你
                        {game.turn === "S" && game.phase !== "finished" ? (
                          <i />
                        ) : null}
                      </span>
                    </div>
                    <button onClick={() => setTab("records")}>
                      <History size={13} />
                      {game.tricks.length
                        ? `上一墩 · ${seatName[game.tricks[game.tricks.length - 1].winner]}家赢墩`
                        : "尚无已完成的墩"}
                      <ChevronRight size={13} />
                    </button>
                  </div>
                </div>
                <div className="hand-section">
                  <div className="hand-heading">
                    <h3>
                      你的手牌 <span>南家</span>
                    </h3>
                    <div>
                      <span>{game.hands.S.length} 张</span>
                      <span className="hcp-pill">
                        {highCardPoints(game.hands.S)} HCP
                      </span>
                    </div>
                  </div>
                  <div className="your-hand" key={game.id}>
                    {game.hands.S.map((card, i) => (
                      <PlayingCard
                        key={card}
                        card={card}
                        selected={suggested.includes(card) && game.turn === "S"}
                        disabled={
                          game.phase !== "play" ||
                          game.turn !== "S" ||
                          !myTurn ||
                          !cards.includes(card)
                        }
                        onClick={() => makePlay("S", card)}
                        style={{ "--card-index": i } as React.CSSProperties}
                      />
                    ))}
                    {game.hands.S.length === 0 && (
                      <span className="hand-empty">所有牌都已打出。</span>
                    )}
                  </div>
                  <p className="hand-caption">
                    {game.phase === "auction"
                      ? "先观察牌力与牌型，再选择你的叫品。"
                      : game.phase === "finished"
                        ? "一副结束，新的思路才刚刚开始。"
                        : game.turn === dummy && controller(game) === "S"
                          ? "你是庄家，可点击明手或下方的可选牌出牌。"
                          : !myTurn
                            ? paused
                              ? "对局已暂停，可请求分析或单步代打。"
                              : "练习对手行动中…"
                            : game.currentTrick.length === 4
                              ? "本墩结束，正在收牌。"
                              : cards.length < game.hands[game.turn].length
                                ? `请跟出${suitSymbol[game.currentTrick[0].card[0] as Suit]}花色。亮起的牌均可打出。`
                                : "点击手牌出牌。"}
                  </p>
                </div>
                <div className="action-panel">
                  <div className="action-heading">
                    <div>
                      <span
                        className={`status-dot ${myTurn ? "is-yours" : ""}`}
                      />
                      <strong aria-live="polite">{status}</strong>
                      {paused && <span className="small-tag">已暂停</span>}
                    </div>
                    {game.phase !== "finished" && (
                      <button
                        className="text-button"
                        onClick={() => setPaused(!paused)}
                      >
                        {paused ? <Play size={14} /> : <Pause size={14} />}
                        {paused ? "继续自动对手" : "暂停自动对手"}
                      </button>
                    )}
                  </div>
                  {game.phase === "auction" ? (
                    <div className="bidding-controls">
                      <div className="bid-levels" aria-label="叫牌阶数">
                        {Array.from({ length: 7 }, (_, i) => i + 1).map((n) => (
                          <button
                            className={level === n ? "selected" : ""}
                            key={n}
                            aria-label={`${n}阶`}
                            aria-pressed={level === n}
                            onClick={() => setLevel(n)}
                          >
                            {n}
                          </button>
                        ))}
                      </div>
                      <div className="bid-actions">
                        <div className="bid-strains">
                          {STRAINS.map((s) => (
                            <button
                              key={s}
                              className={s === "D" || s === "H" ? "red" : ""}
                              disabled={
                                !myTurn || !calls.includes(`${level}${s}`)
                              }
                              onClick={() => makeCall(`${level}${s}`)}
                              aria-label={`叫${level}${suitSymbol[s]}`}
                            >
                              {level}
                              <span>{suitSymbol[s]}</span>
                            </button>
                          ))}
                        </div>
                        <div className="bid-specials">
                          <button
                            className="pass-button"
                            disabled={!myTurn}
                            onClick={() => makeCall("P")}
                          >
                            Pass
                          </button>
                          <button
                            disabled={!myTurn || !calls.includes("X")}
                            onClick={() => makeCall("X")}
                            title="加倍"
                          >
                            X
                          </button>
                          <button
                            disabled={!myTurn || !calls.includes("XX")}
                            onClick={() => makeCall("XX")}
                            title="再加倍"
                          >
                            XX
                          </button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="play-summary">
                      <div>
                        <span className="muted">当前定约</span>
                        <strong>
                          {contract
                            ? `${contract.level}${suitSymbol[contract.strain]}${contract.doubled === 2 ? " XX" : contract.doubled ? " X" : ""}`
                            : "流局"}
                          <small>
                            {contract
                              ? `${seatName[contract.declarer]}家庄家 · 目标 ${6 + contract.level} 墩`
                              : "本副计 0 分"}
                          </small>
                        </strong>
                      </div>
                      <div className="trick-score">
                        <span>
                          南北<strong>{counts[0]}</strong>
                        </span>
                        <i>:</i>
                        <span>
                          东西<strong>{counts[1]}</strong>
                        </span>
                      </div>
                      <span className="trick-progress">
                        {game.tricks.length} / 13 墩
                      </span>
                    </div>
                  )}
                  {game.phase === "play" &&
                    myTurn &&
                    game.turn !== "S" &&
                    cards.length > 0 && (
                      <div className="turn-card-options">
                        <span>
                          {seatName[game.turn]}家可选出牌
                          {game.turn === dummy ? " · 明手" : ""}
                        </span>
                        <div>
                          {cards.map((card) => (
                            <button
                              className={
                                ["H", "D"].includes(card[0]) ? "red" : ""
                              }
                              key={card}
                              onClick={() => makePlay(game.turn, card)}
                              aria-label={`替${seatName[game.turn]}家打出${cardLabel(card)}`}
                            >
                              {cardLabel(card)}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  {error && (
                    <p className="error-text" role="alert">
                      {error}
                    </p>
                  )}
                  <div className="action-footer">
                    <span>
                      <ShieldCheck size={13} />
                      {manual
                        ? "四手均由你控制"
                        : "东西由当前模型出牌，你控制南北两手"}
                    </span>
                    {game.phase !== "finished" && (
                      <button className="text-button" onClick={stepPractice}>
                        {game.currentTrick.length === 4
                          ? "收取本墩"
                          : "模型代打一步"}
                        <ArrowRight size={13} />
                      </button>
                    )}
                  </div>
                </div>
                <div className="below-table">
                  <button
                    disabled={history.length < 2}
                    onClick={() => {
                      setPaused(true);
                      setHistory((h) => h.slice(0, -1));
                      setError("");
                    }}
                  >
                    <RotateCcw size={14} />
                    撤销一步
                  </button>
                  <span>每一手，都是新的可能。</span>
                  <button
                    onClick={exportGame}
                    title="导出包含完整四手牌的 JSON 记录"
                  >
                    <ArrowDownToLine size={14} />
                    导出牌局
                  </button>
                </div>
              </section>
              <aside className="sidebar">
                <div
                  className="sidebar-tabs"
                  role="tablist"
                  aria-label="对局面板"
                >
                  <button
                    role="tab"
                    aria-selected={tab === "analysis"}
                    className={tab === "analysis" ? "active" : ""}
                    onClick={() => setTab("analysis")}
                  >
                    <Spade size={15} />
                    分析助手
                  </button>
                  <button
                    role="tab"
                    aria-selected={tab === "records"}
                    className={tab === "records" ? "active" : ""}
                    onClick={() => setTab("records")}
                  >
                    <History size={15} />
                    对局记录
                    {game.tricks.length > 0 && (
                      <span>{game.tricks.length}</span>
                    )}
                  </button>
                </div>
                <div role="tabpanel">
                  {tab === "analysis" ? (
                    <AnalysisPanel
                      game={game}
                      serverUrl={serverUrl}
                      onSettings={() => setModal("settings")}
                      onSuggest={stableSuggest}
                      onRequest={() => setPaused(true)}
                      disabled={!!modal}
                    />
                  ) : (
                    <Records game={game} />
                  )}
                </div>
              </aside>
            </div>
            <footer className="page-footer">
              <span>
                <Spade size={12} fill="currentColor" /> BRIDGEMIND
              </span>
              <span>专注一副牌，进步一点点。</span>
              <button onClick={() => setModal("help")}>
                练习指南
                <ArrowUpIcon />
              </button>
            </footer>
          </main>
        </>
      )}
      {view === "play" && modelError && <div className="model-error-toast" role="alert"><span>{modelError}</span><button onClick={stepPractice} disabled={busy}>重试模型</button><button onClick={() => setModal("settings")}>设置</button></div>}
      {mobile && modal === "analysis" && (
        <Modal
          title="对局信息"
          onClose={() => setModal(null)}
          className="m-sheet"
        >
          <AnalysisPanel
            game={game}
            serverUrl={serverUrl}
            onSettings={() => setModal("settings")}
            onSuggest={stableSuggest}
            onRequest={() => setPaused(true)}
            disabled={false}
            compact
          />
        </Modal>
      )}
      {mobile && modal === "records" && (
        <Modal
          title="对局记录"
          onClose={() => setModal(null)}
          className="m-sheet"
        >
          <Records game={game} />
        </Modal>
      )}
      {mobile && modal === "auction" && (
        <Modal
          title="完整叫牌"
          onClose={() => setModal(null)}
          className="m-sheet"
        >
          <Records game={game} />
        </Modal>
      )}
      {modal === "settings" && (
        <SettingsModal
          updates={updates}
          onInstall={installUpdate}
          serverUrl={serverUrl}
          manual={manual}
          reduced={reduced}
          onClose={() => setModal(null)}
          onSave={(url, mode, motion) => {
            setServerUrl(url);
            setManual(mode);
            setReduced(motion);
          }}
        />
      )}
      {modal === "new" && (
        <Modal title="开始新的一副" onClose={() => setModal(null)}>
          <p className="modal-copy">
            {game.phase === "finished"
              ? "继续挑战下一副，发牌人与局况将按复式桥牌规则轮换。"
              : mobile
                ? "开始新的一副后，当前练习进度将被清空。"
                : "当前进度将被清空。你可以先导出牌局，保留完整叫牌与出牌记录。"}
          </p>
          <div className="new-board-card">
            <Layers3 size={29} />
            <div>
              <span>NEXT BOARD</span>
              <strong>第 {game.board + 1} 副</strong>
              <small>{seatName[nextSeat(game.dealer)]}家发牌 · 随机洗牌</small>
            </div>
          </div>
          <div className="modal-buttons">
            <button className="secondary-button" onClick={() => setModal(null)}>
              <ArrowLeft size={15} />
              返回牌桌
            </button>
            <button className="primary-button" onClick={startNext}>
              开始发牌
              <ArrowRight size={15} />
            </button>
          </div>
        </Modal>
      )}
      {modal === "help" && (
        <Modal title="欢迎来到桥牌练习室" onClose={() => setModal(null)}>
          <div className="guide-list">
            <section>
              <b>01</b>
              <div>
                <h3>叫牌，确定目标</h3>
                <p>
                  你执南家叫牌，其余三家由电脑上的叫牌模型决策。选择阶数与花色叫牌；连续三次
                  Pass 后确定定约，四家开局全 Pass 则流局。
                </p>
              </div>
            </section>
            <section>
              <b>02</b>
              <div>
                <h3>跟牌，争取每一墩</h3>
                <p>
                  人机对战中你控制南北两手，东西由当前出牌模型决策。庄家左手方首攻，首攻后明手公开。必须跟出首引花色；缺门时可垫牌或将吃，赢墩方引出下一墩。13
                  墩结束后按复式桥牌规则计分。
                </p>
              </div>
            </section>
            <section>
              <b>03</b>
              <div>
                <h3>暂停，比较不同思路</h3>
                <p>
                  人机牌桌的“信息”打开当前牌局的AI/DDS分析；主界面的“模型调试”用于配置四家模型并观战。信息面板可查看动作概率、未知牌归属和终局得分预测。AI
                  使用合法可见信息，DDS 使用四家完整牌；真实归属仅在客户端对照。
                </p>
              </div>
            </section>
          </div>
          <button
            className="primary-button full-width"
            onClick={() => setModal(null)}
          >
            回到牌桌
            <ArrowRight size={16} />
          </button>
        </Modal>
      )}
      <UpdateNotice updates={updates} onInstall={installUpdate} />
    </div>
  );
}

function ArrowUpIcon() {
  return <ArrowRight size={12} style={{ transform: "rotate(-35deg)" }} />;
}
