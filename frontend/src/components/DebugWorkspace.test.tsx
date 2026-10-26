// @vitest-environment jsdom
import { useState } from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { applyCall, controller, legalCards, newGame, stateId } from "../bridge/engine";
import type { Call, Game } from "../bridge/types";
import { fetchAnalysis, fetchBiddingDecision, fetchModels } from "../api/analysis";
vi.mock("../api/analysis", async original => ({ ...await original<typeof import("../api/analysis")>(), fetchModels: vi.fn(), fetchBiddingDecision: vi.fn(), fetchAnalysis: vi.fn() }));
vi.mock("../audio", () => ({ playSound: vi.fn() }));
import { DebugWorkspace } from "./DebugWorkspace";

let game: Game;
function Harness({ initial = newGame(1, 42) }: { initial?: Game }) {
  const [history, setHistory] = useState([initial]);
  game = history[history.length - 1];
  return <DebugWorkspace game={game} serverUrl="http://computer:8000" reduced sound={false} blocked={false}
    onHome={() => {}} onPlay={() => {}} onSettings={() => {}} onNext={() => setHistory([newGame(2, 42)])} onModal={() => {}} onSound={() => {}}
    onCommit={fn => setHistory(previous => [...previous, fn(previous[previous.length - 1])])}
    onUndo={() => setHistory(previous => previous.slice(0, -1))} canUndo={history.length > 1} auction={<p>叫牌记录</p>} records={<p>对局记录内容</p>} />;
}
const advance = async (ms: number) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); };
async function start() { await act(async () => {}); fireEvent.click(screen.getByRole("button", { name: "开始观战" })); }

beforeEach(() => {
  vi.useFakeTimers();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; };
  vi.mocked(fetchModels).mockResolvedValue({ protocol: "bridgemind.models.v1", play: [
    { id: "default", name: "SL v2", available: true }, { id: "alternate", name: "长训 SL", available: true }, { id: "dds", name: "DDS 全知", available: true }],
    bidding: [{ id: "default", name: "SL 叫牌", available: true }, { id: "alternate", name: "PPO 叫牌", available: true }] });
  vi.mocked(fetchBiddingDecision).mockImplementation(async g => {
    const call = g.auction.length ? "P" : "1NT";
    return { call, probabilities: { [call]: 1 }, model: "fixture" };
  });
  vi.mocked(fetchAnalysis).mockImplementation(async (g, kind) => ({ protocol: "bridgemind.analysis.v1", stateId: stateId(g), source: kind, bestCards: [legalCards(g)[0]], model: "fixture" }));
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.resetAllMocks(); });

describe("模型观战", () => {
  it("选择四家的模型后才开始；按各家选择调用并自动走完13墩", async () => {
    render(<Harness />); await advance(5000);
    expect(fetchBiddingDecision).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("北家叫牌模型"), { target: { value: "alternate" } });
    fireEvent.change(screen.getByLabelText("东家出牌模型"), { target: { value: "alternate" } });
    fireEvent.change(screen.getByLabelText("初始播放速度"), { target: { value: "4" } });
    await start(); await advance(250);
    expect(vi.mocked(fetchBiddingDecision).mock.calls[0][3]).toBe("alternate");
    // Advance separate render ticks so each next action sees the committed position.
    for (let i = 0; i < 150 && game.phase !== "finished"; i++) await advance(600);
    expect(game.phase).toBe("finished"); expect(game.tricks).toHaveLength(13);
    expect(fetchAnalysis).toHaveBeenCalledTimes(52);
    expect(vi.mocked(fetchAnalysis).mock.calls.find(call => call[0].turn === "E")?.[4]).toBe("alternate");
  });
  it("暂停会取消在途调用，迟到结果不能推进；单步仍可执行", async () => {
    let resolve!: (value: { call: Call; probabilities: { P: number } }) => void;
    vi.mocked(fetchBiddingDecision).mockReturnValueOnce(new Promise(done => { resolve = done; }));
    render(<Harness />); await start(); await advance(1000);
    const signal = vi.mocked(fetchBiddingDecision).mock.calls[0][2];
    fireEvent.click(screen.getByRole("button", { name: "暂停模型对局" }));
    expect(signal.aborted).toBe(true);
    await act(async () => { resolve({ call: "P", probabilities: { P: 1 } }); });
    expect(game.auction).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "模型代打一步" }));
    await act(async () => {});
    expect(game.auction).toHaveLength(1);
    await advance(5000); expect(game.auction).toHaveLength(1);
  });
  it("信息面板按需打开并暂停，关闭后需要主动播放；速度可调整", async () => {
    render(<Harness />); await start(); await advance(1000);
    fireEvent.click(screen.getByRole("button", { name: "打开模型信息" }));
    expect(screen.getByRole("dialog", { name: "模型信息" })).toBeTruthy();
    const count = game.auction.length;
    await advance(5000); expect(game.auction).toHaveLength(count);
    fireEvent.click(screen.getByRole("button", { name: "关闭信息面板" }));
    await advance(5000); expect(game.auction).toHaveLength(count);
    fireEvent.change(screen.getByLabelText("播放速度"), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "播放模型对局" }));
    await advance(249); expect(game.auction).toHaveLength(count);
    await advance(1); expect(game.auction).toHaveLength(count + 1);
  });
  it("明手使用庄家模型，关闭全知不改变模型请求的座位", async () => {
    const initial = (["1NT", "P", "P", "P"] as Call[]).reduce((g, call) => applyCall(g, call), newGame(1, 42));
    render(<Harness initial={initial} />); await act(async () => {});
    fireEvent.change(screen.getByLabelText("南家出牌模型"), { target: { value: "alternate" } });
    fireEvent.change(screen.getByLabelText("初始播放速度"), { target: { value: "4" } });
    await start(); fireEvent.click(screen.getByRole("button", { name: "关闭全知观战" }));
    for (let i = 0; i < 5; i++) await advance(300);
    const dummyCall = vi.mocked(fetchAnalysis).mock.calls.find(call => call[0].turn === "S");
    expect(dummyCall).toBeTruthy(); expect(controller(dummyCall![0])).toBe("N"); expect(dummyCall![4]).toBe("default");
    expect(screen.getByRole("button", { name: "开启全知观战" })).toBeTruthy();
  });
  it("选择DDS明确使用全知接口，错误停止自动播放并允许重试", async () => {
    const initial = (["1NT", "P", "P", "P"] as Call[]).reduce((g, call) => applyCall(g, call), newGame(1, 42));
    render(<Harness initial={initial} />); await act(async () => {});
    fireEvent.change(screen.getByLabelText("东家出牌模型"), { target: { value: "dds" } });
    vi.mocked(fetchAnalysis).mockRejectedValueOnce(new Error("DDS 暂不可用"));
    await start(); await advance(1000);
    expect(vi.mocked(fetchAnalysis).mock.calls[0][1]).toBe("dds"); expect(vi.mocked(fetchAnalysis).mock.calls[0][4]).toBeUndefined();
    expect(screen.getByRole("alert").textContent).toContain("DDS 暂不可用");
    await advance(5000); expect(fetchAnalysis).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "重试模型" })); await act(async () => {});
    expect(fetchAnalysis).toHaveBeenCalledTimes(2); expect(game.currentTrick).toHaveLength(1);
  });
});
