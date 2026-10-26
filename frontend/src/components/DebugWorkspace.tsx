import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowLeft, Play, Settings2, Spade, X } from "lucide-react";
import { AnalysisPanel } from "./AnalysisPanel";
import { ModelDiagnostics } from "./ModelDiagnostics";
import { MobileTable, type TableModal } from "./MobileTable";
import { applyCall, callLabel, cardLabel, collectTrick, controller, playCard, seatName, stateId } from "../bridge/engine";
import { SEATS, type Game, type Seat } from "../bridge/types";
import { fetchAnalysis, fetchBiddingDecision, fetchModels, type AnalysisResponse, type BiddingDecision, type ModelCatalog } from "../api/analysis";
import { playSound } from "../audio";

type Selection = Record<Seat, { bidding: string; play: string }>;
const defaults = (): Selection => Object.fromEntries(SEATS.map(s => [s, { bidding: "default", play: "default" }])) as Selection;
type LastAction = { game: Game; actor: Seat; description: string; data?: AnalysisResponse; bidding?: BiddingDecision };

function InformationSheet({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current!; dialog.showModal(); return () => dialog.close(); }, []);
  return <dialog ref={ref} className="modal m-sheet observer-sheet" aria-label={title}
    onCancel={event => { event.preventDefault(); onClose(); }} onClick={event => { if (event.target === ref.current) onClose(); }}>
    <div className="modal-heading"><h2>{title}</h2><button className="icon-button" aria-label="关闭信息面板" onClick={onClose}><X size={20} /></button></div>
    {children}
  </dialog>;
}

export function DebugWorkspace({ game, serverUrl, reduced, sound, blocked, onHome, onPlay, onSettings,
  onCommit, onUndo, onNext, onModal, onSound, canUndo, auction, records }: {
  game: Game; serverUrl: string; reduced: boolean; sound: boolean; blocked: boolean;
  onHome: () => void; onPlay: () => void; onSettings: () => void;
  onCommit: (fn: (game: Game) => Game) => void; onUndo: () => void; onNext: () => void;
  onModal: (modal: TableModal) => void; onSound: () => void; canUndo: boolean; auction: ReactNode; records: ReactNode;
}) {
  const [setup, setSetup] = useState(true);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [reload, setReload] = useState(0);
  const [draft, setDraft] = useState(defaults);
  const [selection, setSelection] = useState(defaults);
  const [speed, setSpeed] = useState(1);
  const [omniscient, setOmniscient] = useState(true);
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sheet, setSheet] = useState<"info" | "records" | null>(null);
  const [infoTab, setInfoTab] = useState<"last" | "current">("last");
  const [last, setLast] = useState<LastAction | null>(null);
  const [stepRequested, setStepRequested] = useState(0);
  const request = useRef<AbortController | null>(null);
  const livePosition = useRef(stateId(game));
  livePosition.current = stateId(game);
  const generation = useRef(0);
  const stableSuggest = useCallback(() => {}, []);

  const stop = useCallback(() => {
    generation.current++;
    request.current?.abort(); request.current = null;
    setBusy(false); setRunning(false);
  }, []);
  useEffect(() => {
    stop(); setSetup(true); setLast(null); setError("");
  }, [game.id, serverUrl, stop]);
  useEffect(() => () => { request.current?.abort(); generation.current++; }, []);

  useEffect(() => {
    const ac = new AbortController();
    setCatalog(null); setCatalogError("");
    if (!serverUrl) { setCatalogError("请先连接电脑模型服务。"); return; }
    void fetchModels(serverUrl, ac.signal).then(data => {
      if (!ac.signal.aborted) setCatalog(data);
    }).catch(error => { if (!ac.signal.aborted) setCatalogError(error instanceof Error ? error.message : "无法获取模型列表"); });
    return () => ac.abort();
  }, [serverUrl, reload]);

  async function step() {
    if (request.current || blocked || sheet || setup || game.phase === "finished") return;
    setError("");
    if (game.currentTrick.length === 4) { onCommit(collectTrick); playSound("trick"); return; }
    const ac = new AbortController(), position = stateId(game), token = generation.current;
    request.current = ac; setBusy(true);
    const actor = controller(game), models = selection[actor];
    const current = () => !ac.signal.aborted && generation.current === token && livePosition.current === position;
    try {
      if (game.phase === "auction") {
        const decision = await fetchBiddingDecision(game, serverUrl, ac.signal, models.bidding);
        if (!current()) return;
        setLast({ game, actor, description: `${seatName[game.turn]}家叫 ${callLabel(decision.call)}`, bidding: decision });
        onCommit(g => stateId(g) === position ? applyCall(g, decision.call) : g); playSound("call");
      } else {
        const data = await fetchAnalysis(game, models.play === "dds" ? "dds" : "ai", serverUrl, ac.signal, models.play === "dds" ? undefined : models.play);
        if (!current()) return;
        setLast({ game, actor, description: `${seatName[game.turn]}家出 ${cardLabel(data.bestCards[0])}`, data });
        onCommit(g => stateId(g) === position ? playCard(g, data.bestCards[0]) : g); playSound("card");
      }
    } catch (cause) {
      if (current()) { setError(cause instanceof Error ? cause.message : "模型调用失败"); setRunning(false); }
    } finally {
      if (request.current === ac) { request.current = null; setBusy(false); }
    }
  }

  useEffect(() => {
    if (!running || setup || sheet || blocked || error || game.phase === "finished") return;
    const timer = window.setTimeout(() => { void step(); }, game.currentTrick.length === 4 ? Math.max(600, 1200 / speed) : 1000 / speed);
    return () => { window.clearTimeout(timer); request.current?.abort(); request.current = null; setBusy(false); generation.current++; };
  }, [game, running, speed, setup, sheet, blocked, selection, serverUrl]);
  useEffect(() => {
    if (stepRequested) void step();
  }, [stepRequested]);
  function singleStep() { stop(); setStepRequested(value => value + 1); }

  const modelName = (seat: Seat, phase = game.phase): string => {
    const kind = phase === "auction" ? "bidding" : "play";
    return catalog?.[kind].find(model => model.id === selection[seat][kind])?.name || "默认模型";
  };
  const valid = catalog && SEATS.every(seat => (["play", "bidding"] as const).every(kind =>
    catalog[kind].some(model => model.id === draft[seat][kind] && model.available)));
  function openInfo(kind: "info" | "records") { stop(); setSheet(kind); }
  function pauseOrPlay() {
    if (running) stop(); else { setError(""); setRunning(true); }
  }
  const actor = controller(game);
  const status = game.phase === "finished" ? "本副模型对局已结束" : busy ? `${seatName[actor]}家模型计算中…` :
    game.currentTrick.length === 4 ? "本墩已结束，等待收墩" : `${seatName[game.turn]}家${game.phase === "auction" ? "叫牌" : "出牌"} · ${running ? "播放中" : "已暂停"}`;

  if (setup) return <main className="debug-setup" aria-label="模型对局设置">
    <header><button onClick={onHome}><ArrowLeft size={18} />主界面</button><strong><Spade size={18} fill="currentColor" />模型调试</strong>
      <button onClick={onSettings}><Settings2 size={18} />服务设置</button></header>
    <div className="debug-setup-heading"><div><h1>配置四家模型</h1><p>第 {game.board} 副 · {game.phase === "auction" ? `公开叫牌记录 · ${game.auction.length} 次` : `已完成 ${game.tricks.length}/13 墩`} · 开始后继续当前牌局</p></div>
      <button className="text-button" onClick={onPlay}>人机对战</button></div>
    <div className="debug-model-grid">{SEATS.map(seat => <section key={seat} className="debug-model-seat">
      <h2><span>{seat}</span>{seatName[seat]}家</h2>
      {(["bidding", "play"] as const).map(kind => <label key={kind}>{kind === "bidding" ? "叫牌模型" : "出牌模型"}
        <select aria-label={`${seatName[seat]}家${kind === "bidding" ? "叫牌" : "出牌"}模型`} value={draft[seat][kind]} disabled={!catalog}
          onChange={event => setDraft(previous => ({ ...previous, [seat]: { ...previous[seat], [kind]: event.target.value } }))}>
          {!catalog && <option value="default">{catalogError ? "服务未连接" : "正在加载…"}</option>}
          {catalog?.[kind].map(model => <option key={model.id} value={model.id} disabled={!model.available}>{model.name}{!model.available ? "（不可用）" : ""}</option>)}
        </select>
      </label>)}
    </section>)}</div>
    <footer className="debug-setup-footer"><div className="debug-setup-options">
      <label><input type="checkbox" checked={omniscient} onChange={event => setOmniscient(event.target.checked)} />全知观战</label>
      <label>播放速度<select aria-label="初始播放速度" value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[0.5, 1, 2, 4].map(value => <option key={value} value={value}>{value}×</option>)}</select></label>
      <small>全知只影响观看；神经网络不接收隐藏手牌。明手由庄家模型控制。</small>
    </div><div className="debug-start-actions">{catalogError && <span role="alert">{catalogError}<button onClick={() => setReload(value => value + 1)}>重试</button></span>}
      <button className="primary-button" disabled={!valid} onClick={() => { stop(); setSelection(structuredClone(draft)); setSetup(false); setError(""); setLast(null); setRunning(true); }}><Play size={17} />开始观战</button>
    </div></footer>
  </main>;

  return <>
    <MobileTable game={game} manual={false} reveal={omniscient} paused={!running} reduced={reduced} sound={sound} level={1} suggested={[]}
      status={status} canUndo={canUndo && !busy} auction={auction} onHome={() => { stop(); onHome(); }}
      onModal={modal => { stop(); if (modal === "records" || modal === "auction") openInfo("records"); else onModal(modal); }}
      onReveal={() => setOmniscient(value => !value)} onPause={pauseOrPlay} onSound={onSound} onLevel={() => {}}
      onCall={() => {}} onPlay={() => {}} onUndo={() => { stop(); setLast(null); onUndo(); }}
      onStep={singleStep} onNext={() => { stop(); onNext(); }}
      observer={{ names: Object.fromEntries(SEATS.map(seat => [seat, modelName(seat)])) as Record<Seat, string>, speed, busy,
        onSpeed: setSpeed, onInfo: () => openInfo("info"), onConfigure: () => { stop(); setDraft(structuredClone(selection)); setSetup(true); } }} />
    {error && <div className="model-error-toast" role="alert"><span>{error}</span><button disabled={busy} onClick={() => { setError(""); singleStep(); }}>重试模型</button><button onClick={() => { stop(); onSettings(); }}>服务设置</button></div>}
    {sheet && <InformationSheet title={sheet === "info" ? "模型信息" : "对局记录"} onClose={() => setSheet(null)}>
      {sheet === "records" ? records : <>
        <p className="observer-info-summary">{status}<br />模型控制者：{seatName[actor]}家 · {modelName(actor)}<br />{omniscient ? "全知观战" : "南家视角"} · 观看设置不改变模型输入</p>
        <div className="m-analysis-tabs" role="tablist" aria-label="模型信息内容">
          <button role="tab" aria-selected={infoTab === "last"} onClick={() => setInfoTab("last")}>上一步输出</button>
          <button role="tab" aria-selected={infoTab === "current"} onClick={() => setInfoTab("current")}>当前局面分析</button>
        </div>
        {infoTab === "current" ? <AnalysisPanel game={game} serverUrl={serverUrl} onSettings={() => { setSheet(null); onSettings(); }}
          onlyKind={selection[actor].play === "dds" ? "dds" : undefined}
          modelId={selection[actor].play === "dds" ? undefined : selection[actor].play} onSuggest={stableSuggest} onRequest={stop} disabled={false} compact /> :
          last ? <section className="observer-last-action"><h3>{last.description}</h3><p>模型控制者：{seatName[last.actor]}家 · {last.data?.model || last.bidding?.model || modelName(last.actor, last.game.phase)}</p>
            {last.data?.diagnostics ? <ModelDiagnostics data={last.data} game={last.game} /> : last.bidding ? <div className="debug-actions">
              {Object.entries(last.bidding.probabilities).sort((a, b) => b[1]!-a[1]!).map(([call, probability]) => <p key={call}>{callLabel(call as BiddingDecision["call"])} <strong>{(probability! * 100).toFixed(2)}%</strong></p>)}
            </div> : <><p>DDS 全知求解，不提供神经网络概率或归属预测。</p><div className="debug-actions">{last.data?.candidates?.map(row => <p key={row.card}>{cardLabel(row.card)} <strong>{row.tricks} 墩</strong></p>)}</div></>}
          </section> : <p className="observer-empty">尚未产生模型动作。关闭面板后点击单步或播放。</p>}
      </>}
    </InformationSheet>}
  </>;
}
