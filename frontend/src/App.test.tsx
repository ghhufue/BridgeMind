// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchAnalysis, fetchBidding } from "./api/analysis";
import { legalCards, stateId } from "./bridge/engine";

vi.hoisted(() => {
  Object.defineProperty(window, "matchMedia", { writable: true, value: () => ({ matches: true, addEventListener() {}, removeEventListener() {} }) });
  localStorage.setItem("bridgemind.preferences.v1", JSON.stringify({ serverUrl: "http://localhost:8000", connectionVersion: 2, reduced: true }));
});
vi.mock("./api/analysis", async (original) => ({ ...await original<typeof import("./api/analysis")>(), fetchBidding: vi.fn(), fetchAnalysis: vi.fn() }));
vi.mock("./audio", () => ({ playSound: vi.fn(), setSound: vi.fn() }));
vi.mock("./useTableLayout", () => ({ useTableLayout: () => true }));
import App from "./App";

afterEach(() => { cleanup(); vi.useRealTimers(); vi.resetAllMocks(); });
const advance = async (ms: number) => {
  for (let remaining=ms; remaining>0; remaining-=650) {
    await act(async () => { await vi.advanceTimersByTimeAsync(Math.min(650, remaining)); });
  }
};

describe("主入口与模型对局生命周期", () => {
  it("停留主界面或调试台时不会自动推进牌局", async () => {
    vi.useFakeTimers(); render(<App />);
    await advance(5000);
    expect(fetchBidding).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /MODEL WORKBENCH/ }));
    await advance(5000);
    expect(fetchBidding).not.toHaveBeenCalled();
    expect(screen.getByRole("heading", { name: "配置四家模型" })).toBeTruthy();
  });
  it("只自动叫北东西的牌，等待南家叫牌；东西出牌后等待人打南北", async () => {
    vi.useFakeTimers();
    vi.mocked(fetchBidding).mockImplementation(async (g) => g.auction.length === 0 ? "1NT" : "P");
    vi.mocked(fetchAnalysis).mockImplementation(async (g) => ({ protocol: "bridgemind.analysis.v1", stateId: stateId(g), source: "ai", bestCards: [legalCards(g)[0]] }));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /PLAY WITH AI/ }));
    await advance(2000);
    expect(vi.mocked(fetchBidding).mock.calls.map((c) => c[0].turn)).toEqual(["N", "E"]);
    await advance(2000);
    expect(fetchBidding).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole("button", { name: "Pass" }));
    await advance(1500);
    expect(vi.mocked(fetchBidding).mock.calls.map((c) => c[0].turn)).toEqual(["N", "E", "W"]);
    expect(vi.mocked(fetchAnalysis).mock.calls.map((c) => c[0].turn)).toEqual(["E"]);
    expect(screen.getByLabelText("南家手牌").querySelector("button:not(:disabled)")).toBeTruthy();
    await advance(2000);
    expect(fetchAnalysis).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByLabelText("南家手牌").querySelector("button:not(:disabled)")!);
    fireEvent.click(screen.getByRole("button", { name: /出牌$/ }));
    await advance(1000);
    expect(vi.mocked(fetchAnalysis).mock.calls.map((c) => c[0].turn)).toEqual(["E", "W"]);
    expect(screen.getByLabelText("北家手牌").querySelector("button:not(:disabled)")).toBeTruthy();
    await advance(2000);
    expect(fetchAnalysis).toHaveBeenCalledTimes(2);
  });
  it("退出牌桌会取消请求，迟到的叫牌响应不能改动牌局", async () => {
    vi.useFakeTimers();
    let resolve!: (call: "1NT") => void;
    vi.mocked(fetchBidding).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /PLAY WITH AI/ }));
    await advance(700);
    const signal = vi.mocked(fetchBidding).mock.calls[0][2];
    fireEvent.click(screen.getByRole("button", { name: "返回主界面" }));
    expect(signal.aborted).toBe(true);
    await act(async () => { resolve("1NT"); });
    fireEvent.click(screen.getByRole("button", { name: /MODEL WORKBENCH/ }));
    expect(screen.getByText(/公开叫牌记录 · 0 次/)).toBeTruthy();
    expect(screen.queryByText("1NT", { exact: true })).toBeNull();
  });
  it("模型服务出错时停止推进，显示重试入口，不使用规则假装模型", async () => {
    vi.useFakeTimers();
    vi.mocked(fetchBidding).mockRejectedValue(new Error("模型服务不可用"));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /PLAY WITH AI/ }));
    await advance(2000);
    expect(screen.getByRole("alert").textContent).toContain("模型服务不可用");
    expect(screen.getByRole("button", { name: "重试模型" })).toBeTruthy();
    await advance(5000);
    expect(fetchBidding).toHaveBeenCalledOnce();
  });
});
