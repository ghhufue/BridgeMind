import { useState } from "react";
import type { AppUpdates } from "../updates";

function UpdateActions({ updates, onInstall }: { updates: AppUpdates; onInstall: () => void }) {
  return <div className="update-actions">
    {updates.stage === "ready"
      ? <button type="button" className="primary-button" onClick={onInstall}>安装更新</button>
      : updates.stage === "error"
        ? <button type="button" className="secondary-button" onClick={() => void updates.check()}>重试更新</button>
      : updates.release && !updates.busy
        ? <button type="button" className="primary-button" onClick={() => void updates.download()}>下载更新</button>
        : null}
    {updates.stage === "downloading" && <span>正在下载 {Math.round(updates.percent)}%</span>}
    {updates.stage === "installing" && <span>正在打开系统安装界面…</span>}
  </div>;
}

export function UpdateSettings({ updates, onInstall }: { updates: AppUpdates; onInstall: () => void }) {
  if (!updates.native) return <p className="field-help">网页版在重新打开或刷新页面后加载已发布的版本。</p>;
  return <section className="update-settings" aria-label="应用更新">
    <div className="section-heading"><strong>应用更新</strong><span>{updates.version ? `${updates.version.versionName} (${updates.version.versionCode})` : "Android"}</span></div>
    <p className="field-help">启动及回到前台时自动检查更新。安装由 Android 系统确认，安装后应用会重启，当前牌局不会保留。</p>
    <label className="toggle-row"><span><strong>自动下载新版本</strong><small>连接更新服务时自动下载，可能消耗移动流量。</small></span>
      <input type="checkbox" role="switch" checked={updates.autoDownload} onChange={event => updates.setAutoDownload(event.target.checked)} />
    </label>
    <button type="button" className="secondary-button" disabled={updates.busy} onClick={() => void updates.check()}>
      {updates.stage === "checking" ? "正在检查…" : "检查更新"}
    </button>
    {updates.release && <p>新版本 {updates.release.versionName} · {(updates.release.size / 1024 / 1024).toFixed(1)} MB</p>}
    {updates.release?.notes && <p className="update-notes">{updates.release.notes}</p>}
    <UpdateActions updates={updates} onInstall={onInstall} />
    {updates.message && <p role="status" className={updates.stage === "error" ? "error-text" : "field-help"}>{updates.message}</p>}
  </section>;
}

export function UpdateNotice({ updates, onInstall }: { updates: AppUpdates; onInstall: () => void }) {
  const [dismissed, setDismissed] = useState<number | null>(null);
  if (!updates.native || !updates.release || dismissed === updates.release.versionCode || updates.stage === "error") return null;
  return <aside className="update-notice" aria-label="发现应用更新">
    <div><strong>BridgeMind {updates.release.versionName} {updates.stage === "ready" ? "已下载" : "可更新"}</strong>
      <p>安装会重启应用，当前牌局不会保留。可以打完本局后再安装。</p></div>
    <UpdateActions updates={updates} onInstall={onInstall} />
    <button type="button" className="text-button" onClick={() => setDismissed(updates.release!.versionCode)}>稍后</button>
  </aside>;
}
