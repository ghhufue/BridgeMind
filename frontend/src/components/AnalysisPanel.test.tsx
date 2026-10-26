// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnalysisPanel } from "./AnalysisPanel";
import { fetchAnalysis, type AnalysisResponse } from "../api/analysis";
import {
  applyCall,
  legalCards,
  newGame,
  playCard,
  stateId,
} from "../bridge/engine";
import type { Call, Game } from "../bridge/types";

vi.mock("../api/analysis", () => ({ fetchAnalysis: vi.fn() }));
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

function pending<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
const game = () =>
  (["1NT", "P", "P", "P"] as Call[]).reduce(
    (g, c) => applyCall(g, c),
    newGame(1, 25),
  );
const response = (g: Game, source: "ai" | "dds"): AnalysisResponse => ({
  protocol: "bridgemind.analysis.v1",
  stateId: stateId(g),
  source,
  bestCards: [legalCards(g)[0]],
  model: `test-${source}`,
});
function propsFor(g: Game) {
  return {
    game: g,
    serverUrl: "http://localhost:8000",
    onSettings: vi.fn(),
    onSuggest: vi.fn(),
    onRequest: vi.fn(),
    disabled: false,
  };
}

describe("分析面板局面生命周期", () => {
  it("手机模式切换 AI / DDS 标签时保留同一局面的分析结果", async () => {
    const g = game();
    vi.mocked(fetchAnalysis).mockImplementation(async (_g, kind) =>
      response(g, kind),
    );
    render(<AnalysisPanel {...propsFor(g)} compact />);
    expect(screen.getAllByRole("button", { name: "请求分析" })).toHaveLength(1);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "请求分析" }));
    });
    expect(screen.getByText(/test-ai/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "DDS 全知" }));
    expect(screen.queryByText(/test-ai/)).toBeNull();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "请求分析" }));
    });
    expect(screen.getByText(/test-dds/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "AI 实战" }));
    expect(screen.getByText(/test-ai/)).toBeTruthy();
    expect(fetchAnalysis).toHaveBeenCalledTimes(2);
  });
  it("请求时暂停对手，真实响应才显示建议，两个来源分别展示", async () => {
    const g = game();
    const props = propsFor(g);
    vi.mocked(fetchAnalysis).mockImplementation(async (_game, kind) =>
      response(g, kind),
    );
    render(<AnalysisPanel {...props} />);
    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "请求分析" })[0]);
    });
    expect(props.onRequest).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/test-ai/)).toBeTruthy();
    expect(props.onSuggest).toHaveBeenCalledWith(
      [legalCards(g)[0]],
      stateId(g),
    );
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "请求分析" }));
    });
    expect(screen.getByText(/test-dds/)).toBeTruthy();
  });
  it("出牌后立即清除两个面板，迟到的请求不能恢复旧建议", async () => {
    const g = game();
    const props = propsFor(g);
    const delayed = pending<AnalysisResponse>();
    let signal!: AbortSignal;
    vi.mocked(fetchAnalysis).mockImplementation((_g, _kind, _url, s) => {
      signal = s;
      return delayed.promise;
    });
    const view = render(<AnalysisPanel {...props} />);
    fireEvent.click(screen.getAllByRole("button", { name: "请求分析" })[0]);
    view.rerender(
      <AnalysisPanel {...props} game={playCard(g, legalCards(g)[0])} />,
    );
    expect(signal.aborted).toBe(true);
    await act(async () => {
      delayed.resolve(response(g, "ai"));
    });
    expect(screen.queryByText(/test-ai/)).toBeNull();
    expect(props.onSuggest).not.toHaveBeenCalledWith(
      [legalCards(g)[0]],
      stateId(g),
    );
    expect(screen.getAllByText(/准备分析/)).toHaveLength(2);
  });
  it("修改服务地址会取消旧请求，网络错误保留重试入口", async () => {
    const g = game();
    const props = propsFor(g);
    const delayed = pending<AnalysisResponse>();
    let signal!: AbortSignal;
    vi.mocked(fetchAnalysis).mockImplementation((_g, _kind, _url, s) => {
      signal = s;
      return delayed.promise;
    });
    const view = render(<AnalysisPanel {...props} />);
    fireEvent.click(screen.getAllByRole("button", { name: "请求分析" })[0]);
    view.rerender(
      <AnalysisPanel {...props} serverUrl="http://localhost:9000" />,
    );
    expect(signal.aborted).toBe(true);
    await act(async () => {
      delayed.resolve(response(g, "ai"));
    });
    expect(screen.queryByText(/test-ai/)).toBeNull();
    vi.mocked(fetchAnalysis).mockRejectedValue(
      new TypeError("Failed to fetch"),
    );
    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "请求分析" })[0]);
    });
    expect(screen.getByRole("alert").textContent).toContain("无法连接服务");
    expect(screen.getAllByRole("button", { name: "请求分析" })).toHaveLength(2);
  });
});
