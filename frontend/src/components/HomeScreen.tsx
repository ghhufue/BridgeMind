import { BrainCircuit, ChevronRight, Settings2, Spade, Users } from "lucide-react";

export function HomeScreen({ onPlay, onDebug, onSettings, board }: {
  onPlay: () => void; onDebug: () => void; onSettings: () => void; board: number;
}) {
  return <main className="home-screen" aria-label="BridgeMind 主界面">
    <header className="home-header">
      <div className="home-brand"><Spade fill="currentColor" size={25} />BridgeMind</div>
      <button type="button" onClick={onSettings}><Settings2 size={18} />连接与设置</button>
    </header>
    <div className="home-intro">
      <div><h1>开始练习</h1><p>选择模式，进入当前牌局</p></div>
      <span className="home-board"><span>当前牌局</span><strong>第 {board} 副</strong></span>
    </div>
    <nav className="home-options" aria-label="练习模式">
      <button type="button" className="home-option play" onClick={onPlay}>
        <span className="home-option-top"><span className="home-option-icon"><Users size={26} /></span><small>PLAY WITH AI</small></span>
        <h2>人机对战</h2><p>你叫南家的牌，操作南北两手。<br />与模型完成一场完整对局。</p>
        <span className="home-option-link">进入牌桌<ChevronRight size={20} /></span>
      </button>
      <button type="button" className="home-option" onClick={onDebug}>
        <span className="home-option-top"><span className="home-option-icon"><BrainCircuit size={26} /></span><small>MODEL WORKBENCH</small></span>
        <h2>模型调试</h2><p>查看出牌概率、牌张归属与得分预测。<br />单步推进，与 DDS 分析对照。</p>
        <span className="home-option-link">打开调试台<ChevronRight size={20} /></span>
      </button>
    </nav>
    <footer><span className="home-session-dot" />两个模式共享当前牌局<span className="home-footer-detail">南北双手练习</span></footer>
  </main>;
}
