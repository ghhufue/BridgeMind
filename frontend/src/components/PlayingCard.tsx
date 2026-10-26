import { useLayoutEffect, useRef, type CSSProperties } from "react";
import gsap from "gsap";
import { MotionPathPlugin } from "gsap/MotionPathPlugin";
import { cardLabel, suitSymbol } from "../bridge/engine";
import type { Card, Seat, Strain } from "../bridge/types";

gsap.registerPlugin(MotionPathPlugin);

export function PlayingCard({
  card,
  disabled,
  selected,
  onClick,
  style,
}: {
  card: Card;
  disabled?: boolean;
  selected?: boolean;
  onClick?: () => void;
  style?: CSSProperties;
}) {
  const suit = card[0] as Strain;
  const rank = card[1] === "T" ? "10" : card[1];
  return (
    <button
      type="button"
      style={style}
      className={`playing-card ${suit === "H" || suit === "D" ? "red" : ""} ${selected ? "recommended" : ""}`}
      disabled={disabled}
      onClick={onClick}
      aria-label={`打出${cardLabel(card)}`}
    >
      <span className="card-corner">
        <b>{rank}</b>
        <span>{suitSymbol[suit]}</span>
      </span>
      <span className="card-pip">{suitSymbol[suit]}</span>
      <span className="card-corner bottom">
        <b>{rank}</b>
        <span>{suitSymbol[suit]}</span>
      </span>
    </button>
  );
}

export function TableCard({
  card,
  seat,
  reduced,
}: {
  card: Card;
  seat: Seat;
  reduced: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (reduced) return;
    const origins: Record<Seat, { x: number; y: number }> = {
      S: { x: 0, y: 150 },
      N: { x: 0, y: -130 },
      E: { x: 180, y: 0 },
      W: { x: -180, y: 0 },
    };
    const start = origins[seat];
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ref.current,
        {
          x: start.x,
          y: start.y,
          rotation: seat === "E" ? 18 : seat === "W" ? -18 : -8,
          scale: 0.85,
          opacity: 0,
        },
        {
          motionPath: {
            path: [
              { x: start.x, y: start.y },
              { x: start.x * 0.45 + 12, y: start.y * 0.5 - 20 },
              { x: 0, y: 0 },
            ],
            curviness: 1.2,
          },
          rotation: seat === "E" ? 7 : seat === "W" ? -7 : 0,
          scale: 1,
          opacity: 1,
          duration: 0.46,
          ease: "power2.out",
        },
      );
    }, ref);
    return () => ctx.revert();
  }, [card, seat, reduced]);
  return (
    <div ref={ref} className={`table-card table-card-${seat}`}>
      <PlayingCard card={card} disabled />
    </div>
  );
}
