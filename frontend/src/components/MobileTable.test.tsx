// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MobileTable } from "./MobileTable";
import {
  applyCall,
  legalCards,
  newGame,
  playCard,
  cardLabel,
} from "../bridge/engine";
import type { Call, Game } from "../bridge/types";

afterEach(cleanup);
function playGame(board = 1) {
  return (["1NT", "P", "P", "P"] as Call[]).reduce(
    (g, c) => applyCall(g, c),
    newGame(board, 42),
  );
}
function propsFor(game: Game) {
  return {
    game,
    manual: true,
    reveal: false,
    paused: false,
    reduced: true,
    sound: false,
    level: 1,
    suggested: [],
    status: "轮到出牌",
    canUndo: false,
    auction: <div>叫牌记录</div>,
    onModal: vi.fn(),
    onReveal: vi.fn(),
    onPause: vi.fn(),
    onSound: vi.fn(),
    onLevel: vi.fn(),
    onCall: vi.fn(),
    onPlay: vi.fn(),
    onUndo: vi.fn(),
    onStep: vi.fn(),
    onNext: vi.fn(),
  };
}

describe("手机牌桌交互", () => {
  it("选中手牌不会立即出牌，必须再点出牌按钮确认", () => {
    const g = playGame();
    const props = propsFor(g);
    const card = legalCards(g)[0];
    render(<MobileTable {...props} />);
    fireEvent.click(
      screen.getByRole("button", {
        name: `打出${cardLabel(card)}`,
      }),
    );
    expect(props.onPlay).not.toHaveBeenCalled();
    expect(screen.getByRole("status").textContent).toContain(cardLabel(card));
    fireEvent.click(screen.getByRole("button", { name: /出牌$/ }));
    expect(props.onPlay).toHaveBeenCalledWith(g.turn, card);
  });
  it("局面变化后，旧牌的选中状态不能继续出牌", () => {
    const g = playGame();
    const props = propsFor(g);
    const card = legalCards(g)[0];
    const view = render(<MobileTable {...props} />);
    fireEvent.click(
      screen.getByRole("button", {
        name: `打出${cardLabel(card)}`,
      }),
    );
    view.rerender(<MobileTable {...props} game={playCard(g, card)} />);
    expect(
      (screen.getByRole("button", { name: /出牌$/ }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(props.onPlay).not.toHaveBeenCalled();
  });
  it("庄家操作明手时，底部直接显示明手的合法手牌", () => {
    let g = playGame(3);
    g = playCard(g, legalCards(g)[0]);
    const props = { ...propsFor(g), manual: false };
    render(<MobileTable {...props} />);
    expect(screen.getByLabelText("北家手牌")).toBeTruthy();
    const available = legalCards(g);
    const buttons = screen
      .getByLabelText("北家手牌")
      .querySelectorAll("button");
    expect([...buttons].filter((b) => !b.disabled)).toHaveLength(
      available.length,
    );
    expect(screen.getByRole("status").textContent).toContain("替明手选牌");
  });
  it("分析和记录由按钮打开；操作按钮不会触发出牌", () => {
    const props = propsFor(playGame());
    render(<MobileTable {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "打开对局信息" }));
    expect(props.onModal).toHaveBeenCalledWith("analysis");
    fireEvent.click(screen.getByRole("button", { name: "打开对局记录" }));
    expect(props.onModal).toHaveBeenCalledWith("records");
    expect(props.onPlay).not.toHaveBeenCalled();
  });
  it("有首引花色时，其他花色在手牌中不可选", () => {
    let g = playGame();
    g = playCard(g, legalCards(g)[0]);
    render(<MobileTable {...propsFor(g)} />);
    const hand = screen.getByLabelText(
      `${g.turn === "S" ? "南" : g.turn === "N" ? "北" : g.turn === "E" ? "东" : "西"}家手牌`,
    );
    const enabled = [...hand.querySelectorAll("button")]
      .filter((b) => !b.disabled)
      .map((b) => b.getAttribute("aria-label"));
    expect(new Set(enabled)).toEqual(
      new Set(legalCards(g).map((c) => `打出${cardLabel(c)}`)),
    );
  });
});
