import { useState, type CSSProperties, type ReactNode } from "react";
import {
  BrainCircuit,
  ChevronRight,
  Eye,
  EyeOff,
  History,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Settings2,
  Smartphone,
  Spade,
  Trophy,
  Volume2,
  VolumeX,
  WandSparkles,
} from "lucide-react";
import { PlayingCard, TableCard } from "./PlayingCard";
import {
  callLabel,
  cardLabel,
  controller,
  dummySeat,
  dummyVisible,
  gameResult,
  highCardPoints,
  legalCalls,
  legalCards,
  seatName,
  stateId,
  suitSymbol,
  trickCounts,
} from "../bridge/engine";
import {
  STRAINS,
  type Call,
  type Card,
  type Game,
  type Seat,
  type Suit,
} from "../bridge/types";

export type TableModal =
  | "settings"
  | "help"
  | "new"
  | "analysis"
  | "records"
  | "auction"
  | null;

interface Props {
  game: Game;
  manual: boolean;
  reveal: boolean;
  paused: boolean;
  reduced: boolean;
  sound: boolean;
  level: number;
  suggested: Card[];
  status: string;
  canUndo: boolean;
  auction: ReactNode;
  onModal: (modal: TableModal) => void;
  onReveal: () => void;
  onPause: () => void;
  onSound: () => void;
  onLevel: (level: number) => void;
  onCall: (call: Call) => void;
  onPlay: (seat: Seat, card: Card) => void;
  onUndo: () => void;
  onStep: () => void;
  onNext: () => void;
  partnership?: boolean;
  onHome?: () => void;
  observer?: {
    names: Record<Seat, string>;
    speed: number;
    busy: boolean;
    onSpeed: (speed: number) => void;
    onInfo: () => void;
    onConfigure: () => void;
  };
}

export function MobileTable(p: Props) {
  const { game, manual, reveal, paused, reduced, sound } = p;
  const [selection, setSelection] = useState<{
    position: string;
    card: Card;
  } | null>(null);
  const position = stateId(game);
  const chosen = selection?.position === position ? selection.card : null;
  const cards = legalCards(game);
  const calls = legalCalls(game);
  const mine = !p.observer && (manual || (game.phase === "auction" ? game.turn === "S" : p.partnership ? ["N", "S"].includes(game.turn) : controller(game) === "S"));
  const handSeat =
    game.phase === "play" && mine && cards.length ? game.turn : "S";
  const hand = game.hands[handSeat];
  const dummy = dummySeat(game);
  const counts = trickCounts(game);
  const contract = game.contract;
  const result = gameResult(game);

  function renderSeat(seat: Seat) {
    const active =
      game.phase !== "finished" &&
      game.turn === seat &&
      game.currentTrick.length < 4;
    const visible = manual || reveal || (p.partnership && seat === "N" && game.phase !== "auction") || (seat === dummy && dummyVisible(game));
    const latest = game.auction.findLast((c) => c.seat === seat);
    return (
      <section
        key={seat}
        className={`m-seat m-seat-${seat} ${active ? "m-active" : ""}`}
        aria-label={`${seatName[seat]}家座位`}
      >
        <div className="m-player">
          <span className="m-avatar">{seat}</span>
          <div>
            <strong>{seatName[seat]}家</strong>
            <small>
              {p.observer ? p.observer.names[seat] : seat === dummy
                ? "明手"
                : seat === contract?.declarer
                  ? "庄家"
                  : manual
                    ? "手动练习"
                    : seat === "N" && p.partnership ? "你控制的队友" : "模型对手"}
              {seat === game.dealer ? " · D" : ""}
            </small>
          </div>
          {active && <i />}
        </div>
        {game.phase === "auction" && latest && (
          <span
            className={`m-call-bubble ${/[HD]/.test(latest.call) ? "red" : ""}`}
          >
            {callLabel(latest.call)}
          </span>
        )}
        {visible ? (
          <div className="m-exposed-hand">
            {(["S", "H", "D", "C"] as Suit[]).map((suit) => (
              <span
                className={suit === "H" || suit === "D" ? "red" : ""}
                key={suit}
              >
                <b>{suitSymbol[suit]}</b>
                {game.hands[seat]
                  .filter((c) => c[0] === suit)
                  .map((c) => (c[1] === "T" ? "10" : c[1]))
                  .join(" ") || "—"}
              </span>
            ))}
          </div>
        ) : (
          <div
            className="m-backed-hand"
            aria-label={`${game.hands[seat].length} 张隐藏手牌`}
          >
            {Array.from(
              { length: Math.min(5, game.hands[seat].length) },
              (_, i) => (
                <i key={i} />
              ),
            )}
            <small>{game.hands[seat].length}</small>
          </div>
        )}
      </section>
    );
  }

  return (
    <div className={`mobile-table-app ${p.observer ? "observer-table" : ""}`}>
      <header className="m-hud">
        <button
          className="m-brand"
          aria-label={p.onHome ? "返回主界面" : "练习指南"}
          onClick={() => p.onHome ? p.onHome() : p.onModal("help")}
        >
          <Spade size={19} fill="currentColor" />
          <span>{p.observer ? "模型调试" : "BridgeMind"}</span>
        </button>
        <div className="m-board-info">
          <b>第 {String(game.board).padStart(2, "0")} 副</b>
          <span>
            {game.vulnerable.every(Boolean)
              ? "双方有局"
              : game.vulnerable[0]
                ? "南北有局"
                : game.vulnerable[1]
                  ? "东西有局"
                  : "双方无局"}
          </span>
        </div>
        <div className="m-scoreboard">
          <span className="m-contract">
            {contract
              ? `${contract.level}${suitSymbol[contract.strain]}${contract.doubled === 2 ? " XX" : contract.doubled ? " X" : ""}`
              : "叫牌中"}
            <small>
              {contract
                ? `${seatName[contract.declarer]}家庄家`
                : `${seatName[game.dealer]}家发牌`}
            </small>
          </span>
          <span className="m-score-pair">
            <small>南北</small>
            <b>{counts[0]}</b>
            <i>:</i>
            <b>{counts[1]}</b>
            <small>东西</small>
          </span>
        </div>
        <nav aria-label="牌桌工具">
          <button
            onClick={() => p.observer ? p.observer.onInfo() : p.onModal("analysis")}
            aria-label={p.observer ? "打开模型信息" : "打开对局信息"}
          >
            <BrainCircuit size={17} />
            <span>信息</span>
          </button>
          <button
            onClick={() => p.onModal("records")}
            aria-label="打开对局记录"
          >
            <History size={17} />
            <span>记录</span>
          </button>
          <button
            onClick={p.onSound}
            aria-label={sound ? "关闭音效" : "开启音效"}
          >
            {sound ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>
          <button onClick={() => p.observer ? p.observer.onConfigure() : p.onModal("settings")} aria-label={p.observer ? "配置四家模型" : "设置"}>
            <Settings2 size={18} />
          </button>
          <button onClick={() => p.onModal("new")} aria-label="新的一副">
            <Plus size={20} />
          </button>
        </nav>
      </header>
      <div className="m-table-mark" aria-hidden="true">
        <Spade fill="currentColor" />
        <span>BRIDGEMIND</span>
      </div>
      {renderSeat("N")}
      {renderSeat("W")}
      {renderSeat("E")}

      {game.phase === "auction" ? (
        <div className="m-auction-layout">
          <section className="m-auction-history">
            <div>
              <span>叫牌进程</span>
              <button
                onClick={() => p.onModal("auction")}
                aria-label="查看完整叫牌"
              >
                完整记录
                <ChevronRight size={12} />
              </button>
            </div>
            {p.auction}
          </section>
          <section className="m-bidding-box" aria-label={p.observer ? "模型叫牌" : "叫牌操作"}>
            {p.observer ? <div className="observer-auction-status">
              <span>{paused ? "已暂停" : p.observer.busy ? "模型思考中" : "自动叫牌"}</span>
              <h2>{seatName[game.turn]}家叫牌</h2><p>{p.observer.names[game.turn]}</p>
              <small>可暂停、单步推进，或打开信息面板。</small>
            </div> : <>
            <div className="m-bid-title">
              <i className={mine ? "is-yours" : ""} />
              {p.status}
            </div>
            <div className="m-bid-levels">
              {Array.from({ length: 7 }, (_, i) => i + 1).map((level) => (
                <button
                  key={level}
                  className={level === p.level ? "selected" : ""}
                  onClick={() => p.onLevel(level)}
                  aria-pressed={level === p.level}
                  aria-label={`${level}阶`}
                >
                  {level}
                </button>
              ))}
            </div>
            <div className="m-bid-strains">
              {STRAINS.map((s) => (
                <button
                  className={s === "H" || s === "D" ? "red" : ""}
                  key={s}
                  disabled={!mine || !calls.includes(`${p.level}${s}`)}
                  onClick={() => p.onCall(`${p.level}${s}`)}
                  aria-label={`叫${p.level}${suitSymbol[s]}`}
                >
                  {p.level}
                  <b>{suitSymbol[s]}</b>
                </button>
              ))}
            </div>
            <div className="m-bid-specials">
              <button
                disabled={!mine}
                className="m-pass"
                onClick={() => p.onCall("P")}
              >
                Pass
              </button>
              <button
                disabled={!mine || !calls.includes("X")}
                onClick={() => p.onCall("X")}
              >
                X
              </button>
              <button
                disabled={!mine || !calls.includes("XX")}
                onClick={() => p.onCall("XX")}
              >
                XX
              </button>
            </div>
            </>}
          </section>
        </div>
      ) : game.phase === "finished" ? (
        <section className="m-result">
          <Trophy size={26} />
          <div>
            <span>本副结束</span>
            <h1>
              {contract
                ? `${contract.level}${suitSymbol[contract.strain]} ${result.delta === 0 ? "完成定约" : result.delta > 0 ? `超 ${result.delta} 墩` : `宕 ${-result.delta} 墩`}`
                : "四家不叫 · 流局"}
            </h1>
          </div>
          <div className="m-final-score">
            <small>南北得分</small>
            <b>
              {result.nsScore > 0 ? "+" : ""}
              {result.nsScore}
            </b>
          </div>
          <button onClick={p.onNext}>
            下一副
            <ChevronRight size={15} />
          </button>
        </section>
      ) : (
        <div className="m-play-center">
          <span className="m-trick-number">
            第 {Math.min(13, game.tricks.length + 1)} 墩
          </span>
          <div className="m-compass" aria-hidden="true">
            <span>N</span>
            <span>W</span>
            <Spade size={25} fill="currentColor" />
            <span>E</span>
            <span>S</span>
          </div>
          {game.currentTrick.map((entry) => (
            <TableCard
              key={`${game.tricks.length}-${entry.seat}-${entry.card}`}
              {...entry}
              reduced={reduced}
            />
          ))}
          {!game.currentTrick.length && (
            <span className="m-lead-hint">
              {game.tricks.length ? `${seatName[game.turn]}家引牌` : "等待首攻"}
            </span>
          )}
        </div>
      )}

      <div className="m-south-identity">
        <span className="m-avatar">{handSeat}</span>
        <div>
          <strong>
            {seatName[handSeat]}家
            {p.observer ? ` · ${p.observer.names[handSeat]}` : handSeat === "S" ? " · 你" : handSeat === dummy ? " · 明手" : ""}
          </strong>
          <small>
            {highCardPoints(hand)} HCP · {hand.length} 张
          </small>
        </div>
      </div>
      <div
        className={`m-turn-message ${mine ? "m-your-turn" : ""}`}
        role="status"
      >
        {paused && <Pause size={11} />}
        {game.phase === "auction"
          ? mine ? "观察手牌，选择叫品" : p.status
          : game.phase === "finished"
            ? "每一手，都是新的可能"
            : chosen
              ? `已选 ${cardLabel(chosen)}`
              : mine && cards.length
                ? game.turn === dummy
                  ? "替明手选牌，再点击出牌"
                  : "点击选牌，再点击出牌"
                : p.status}
      </div>
      <div
        className="m-hand-deck"
        aria-label={`${seatName[handSeat]}家手牌`}
        key={game.id}
      >
        {hand.map((card, index) => (
          <div
            className={`m-hand-slot ${chosen === card ? "m-picked" : ""}`}
            key={card}
            style={
              {
                "--angle": `${(index - (hand.length - 1) / 2) * 0.65}deg`,
                "--drop": `${Math.abs(index - (hand.length - 1) / 2) * 0.8}px`,
              } as CSSProperties
            }
          >
            <PlayingCard
              card={card}
              disabled={
                game.phase !== "play" ||
                !mine ||
                game.turn !== handSeat ||
                !cards.includes(card)
              }
              selected={p.suggested.includes(card) && game.turn === handSeat}
              onClick={() =>
                setSelection(chosen === card ? null : { position, card })
              }
            />
          </div>
        ))}
      </div>
      <div className="m-left-actions">
        <button onClick={p.onUndo} disabled={!p.canUndo} aria-label="撤销一步">
          <RotateCcw size={16} />
          <span>撤销</span>
        </button>
        <button
          onClick={p.onReveal}
          disabled={manual}
          aria-label={p.observer ? reveal ? "关闭全知观战" : "开启全知观战" : reveal ? "隐藏其他手牌" : "查看四家手牌"}
        >
          {reveal ? <EyeOff size={16} /> : <Eye size={16} />}
          <span>{p.observer ? "全知观战" : "四手"}</span>
        </button>
      </div>
      <div className="m-right-actions">
        {p.observer && <label className="observer-speed">速度
          <select aria-label="播放速度" value={p.observer.speed} onChange={event => p.observer!.onSpeed(Number(event.target.value))}>
            {[0.5, 1, 2, 4].map(speed => <option key={speed} value={speed}>{speed}×</option>)}
          </select>
        </label>}
        {game.phase === "play" && !p.observer && (
          <button
            className="m-play-button"
            disabled={!mine || !chosen || !cards.includes(chosen)}
            onClick={() => chosen && p.onPlay(handSeat, chosen)}
          >
            {chosen ? cardLabel(chosen) : "选一张牌"}
            <b>出牌</b>
          </button>
        )}
        <div>
          <button
            onClick={p.onPause}
            disabled={!!p.observer && game.phase === "finished"}
            aria-label={p.observer ? paused ? "播放模型对局" : "暂停模型对局" : paused ? "继续自动对手" : "暂停自动对手"}
          >
            {paused ? <Play size={15} /> : <Pause size={15} />}
          </button>
          <button
            onClick={p.onStep}
            disabled={game.phase === "finished" || p.observer?.busy}
            aria-label={
              game.currentTrick.length === 4 ? "收取本墩" : "模型代打一步"
            }
          >
            <WandSparkles size={15} />
          </button>
        </div>
      </div>
      <span className="m-mode-label">
        {p.observer ? `模型观战 · ${reveal ? "全知视角" : "南家视角"}` : manual ? "四手手动练习" : p.partnership ? "人机对战 · 你控制南北出牌" : "模型对手 · 本地练习"}
      </span>
      <div className="m-rotate-hint">
        <Smartphone size={38} />
        <h2>横过来，开一桌</h2>
        <p>请将手机转为横屏，获得完整牌桌体验。</p>
      </div>
    </div>
  );
}
