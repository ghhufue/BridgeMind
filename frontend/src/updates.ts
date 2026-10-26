import { Capacitor, registerPlugin, type PluginListenerHandle } from "@capacitor/core";
import { useCallback, useEffect, useRef, useState } from "react";

export type AppVersion = { versionCode: number; versionName: string };
export type Release = AppVersion & { notes: string; size: number };
export interface UpdaterPlugin {
  getInfo(): Promise<AppVersion>;
  check(options: { url: string }): Promise<{ available: boolean; release?: Release }>;
  download(): Promise<void>;
  install(): Promise<{ status: "installerOpened" }>;
  addListener(event: "resume", listener: () => void): Promise<PluginListenerHandle>;
  addListener(event: "downloadProgress", listener: (value: { percent: number }) => void): Promise<PluginListenerHandle>;
}
export const AppUpdater = registerPlugin<UpdaterPlugin>("AppUpdater");
export const isAndroidApp = () => Capacitor.getPlatform() === "android";
const AUTO_KEY = "bridgemind.updates.autoDownload.v1";
const INTERVAL = 30 * 60 * 1000;
type Stage = "idle" | "checking" | "downloading" | "ready" | "installing" | "error";

function storedAutoDownload() {
  try { return localStorage.getItem(AUTO_KEY) !== "false"; } catch { return true; }
}

export function useAppUpdates(serverUrl: string) {
  const native = isAndroidApp();
  const address = (import.meta.env.VITE_UPDATE_SERVER_URL || serverUrl).replace(/\/+$/, "");
  const manifestUrl = import.meta.env.VITE_UPDATE_MANIFEST_URL || (address ? `${address}/api/v1/updates/android` : "");
  const [version, setVersion] = useState<AppVersion | null>(null);
  const [release, setRelease] = useState<Release | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [percent, setPercent] = useState(0);
  const [message, setMessage] = useState("");
  const [autoDownload, setAuto] = useState(storedAutoDownload);
  const locked = useRef(false);
  const generation = useRef(0);
  const lastCheck = useRef(0);
  const automatic = useRef(autoDownload);
  automatic.current = autoDownload;

  const perform = useCallback(async (action: "check" | "download" | "install") => {
    if (!native || locked.current) return;
    const token = generation.current;
    const current = () => generation.current === token;
    locked.current = true;
    setMessage("");
    try {
      if (action === "install") {
        setStage("installing");
        await AppUpdater.install();
        if (current()) { setStage("ready"); setMessage("已打开系统安装界面；如取消安装，可再次点击安装。"); }
        return;
      }
      let shouldDownload = action === "download";
      if (action === "check") {
        setStage("checking");
        const info = await AppUpdater.getInfo();
        if (!current()) return;
        setVersion(info);
        if (!manifestUrl) throw new Error("请先在设置中配置电脑服务地址。");
        const result = await AppUpdater.check({ url: manifestUrl });
        if (!current()) return;
        lastCheck.current = Date.now();
        if (!result.available || !result.release) {
          setRelease(null); setStage("idle"); setMessage("当前没有可用的新版本。");
          return;
        }
        setRelease(result.release);
        shouldDownload = automatic.current;
        if (!shouldDownload) { setStage("idle"); setMessage("发现新版本，可以下载更新。"); }
      }
      if (shouldDownload) {
        if (!current()) return;
        setStage("downloading"); setPercent(0);
        await AppUpdater.download();
        if (current()) { setPercent(100); setStage("ready"); setMessage("更新已下载并校验，可以安装。"); }
      }
    } catch (error) {
      if (current()) { setStage("error"); setMessage(error instanceof Error ? error.message : "更新失败，请稍后重试。"); }
    } finally { locked.current = false; }
  }, [native, manifestUrl]);

  useEffect(() => {
    if (!native) return;
    let alive = true;
    const handles: PluginListenerHandle[] = [];
    const subscribe = async () => {
      const resume = await AppUpdater.addListener("resume", () => {
        if (alive && Date.now() - lastCheck.current > 60000) void perform("check");
      });
      if (alive) handles.push(resume); else void resume.remove();
      const progress = await AppUpdater.addListener("downloadProgress", ({ percent: value }) => {
        if (alive && Number.isFinite(value)) setPercent(Math.max(0, Math.min(100, value)));
      });
      if (alive) handles.push(progress); else void progress.remove();
    };
    void subscribe().catch(() => { /* Manual check still reports bridge errors. */ });
    // Defer initialization so StrictMode's effect probe cannot leave a stale operation in flight.
    const initialize = () => {
      if (locked.current) initial = window.setTimeout(initialize, 250);
      else void perform("check");
    };
    let initial = window.setTimeout(initialize, 0);
    const visible = () => {
      if (document.visibilityState === "visible" && Date.now() - lastCheck.current > 60000) void perform("check");
    };
    document.addEventListener("visibilitychange", visible);
    const timer = window.setInterval(() => { if (document.visibilityState === "visible") void perform("check"); }, INTERVAL);
    return () => {
      alive = false; generation.current++;
      window.clearTimeout(initial); window.clearInterval(timer);
      document.removeEventListener("visibilitychange", visible);
      for (const handle of handles) void handle.remove();
    };
  }, [native, perform]);

  // Changing services invalidates the native download selection and the visible old release.
  useEffect(() => { setRelease(null); setStage("idle"); setMessage(""); lastCheck.current = 0; }, [manifestUrl]);
  return {
    native, address, version, release, stage, percent, message, autoDownload,
    busy: ["checking", "downloading", "installing"].includes(stage),
    setAutoDownload(value: boolean) {
      setAuto(value);
      try { localStorage.setItem(AUTO_KEY, String(value)); } catch { /* Optional preference. */ }
    },
    check: () => perform("check"), download: () => perform("download"), install: () => perform("install"),
  };
}

export type AppUpdates = ReturnType<typeof useAppUpdates>;
