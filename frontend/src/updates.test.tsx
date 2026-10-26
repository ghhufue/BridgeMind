// @vitest-environment jsdom
import { StrictMode } from "react";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const native = vi.hoisted(() => ({
  platform: "android",
  getInfo: vi.fn(), check: vi.fn(), download: vi.fn(), install: vi.fn(), addListener: vi.fn(),
  listeners: new Map<string, (value?: unknown) => void>(),
}));
vi.mock("@capacitor/core", () => ({ Capacitor: { getPlatform: () => native.platform }, registerPlugin: () => native }));
import { useAppUpdates, type AppUpdates } from "./updates";
import { UpdateNotice, UpdateSettings } from "./components/AppUpdates";

let updates: AppUpdates;
function Harness({ url = "http://computer:8000" }: { url?: string }) {
  updates = useAppUpdates(url);
  return <><UpdateSettings updates={updates} onInstall={() => void updates.install()} /><UpdateNotice updates={updates} onInstall={() => void updates.install()} /></>;
}
const release = { versionCode: 3, versionName: "1.2.0", notes: "新版说明", size: 5000000 };
async function start() { await act(async () => { await vi.advanceTimersByTimeAsync(1); }); }

beforeEach(() => {
  vi.useFakeTimers(); localStorage.clear(); native.platform = "android";
  native.getInfo.mockResolvedValue({ versionCode: 2, versionName: "1.1.0" });
  native.check.mockResolvedValue({ available: true, release });
  native.download.mockResolvedValue(undefined);
  native.install.mockResolvedValue({ status: "installerOpened" });
  native.addListener.mockImplementation(async (name: string, listener: (value?: unknown) => void) => {
    native.listeners.set(name, listener);
    return { remove: () => { if (native.listeners.get(name) === listener) native.listeners.delete(name); } };
  });
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.resetAllMocks(); vi.unstubAllEnvs(); native.listeners.clear(); });

describe("Android 应用更新生命周期", () => {
  it("Release使用独立GitHub清单，不依赖局域网分析地址", async () => {
    vi.stubEnv('VITE_UPDATE_MANIFEST_URL', 'https://github.com/ghhufue/BridgeMind/releases/latest/download/latest.json');
    render(<Harness url="" />); await start();
    expect(native.check).toHaveBeenCalledExactlyOnceWith({ url: 'https://github.com/ghhufue/BridgeMind/releases/latest/download/latest.json' });
    expect(updates.stage).toBe('ready');
  });
  it("启动自动检查下载，但不会自动打开安装器", async () => {
    render(<StrictMode><Harness /></StrictMode>); await start();
    expect(native.check).toHaveBeenCalledExactlyOnceWith({ url: "http://computer:8000/api/v1/updates/android" });
    expect(native.download).toHaveBeenCalledOnce();
    expect(native.install).not.toHaveBeenCalled();
    expect(updates.stage).toBe("ready");
    expect(screen.getByLabelText("发现应用更新").textContent).toContain("当前牌局不会保留");
    await act(() => updates.install());
    expect(native.install).toHaveBeenCalledOnce();
    expect(updates.message).toContain("已打开系统安装界面");
    expect(updates.stage).toBe("ready");
  });
  it("关闭自动下载后只提示新版，可手动下载", async () => {
    localStorage.setItem("bridgemind.updates.autoDownload.v1", "false");
    render(<Harness />); await start();
    expect(native.download).not.toHaveBeenCalled();
    await act(() => updates.download());
    expect(native.download).toHaveBeenCalledOnce();
    expect(updates.stage).toBe("ready");
  });
  it("没有新版不下载，网页版不访问原生插件", async () => {
    native.check.mockResolvedValue({ available: false });
    const rendered = render(<Harness />); await start();
    expect(updates.release).toBeNull(); expect(native.download).not.toHaveBeenCalled();
    rendered.unmount(); native.platform = "web";
    render(<Harness />); await start();
    expect(native.getInfo).toHaveBeenCalledOnce();
    expect(native.check).toHaveBeenCalledOnce();
  });
  it("重复检查不会发起重叠请求，下载错误可重试", async () => {
    let reject!: (error: Error) => void;
    native.download.mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
    render(<Harness />); await start();
    await act(() => updates.check());
    expect(native.check).toHaveBeenCalledOnce();
    await act(async () => { reject(new Error("校验失败")); });
    expect(updates.stage).toBe("error"); expect(updates.message).toBe("校验失败");
    expect(native.install).not.toHaveBeenCalled();
    await act(() => updates.check());
    expect(updates.stage).toBe("ready");
  });
  it("切换服务时丢弃迟到响应，并检查新服务", async () => {
    let resolve!: (value: unknown) => void;
    native.check.mockReturnValueOnce(new Promise(done => { resolve = done; }));
    const rendered = render(<Harness />); await start();
    rendered.rerender(<Harness url="http://other:8000" />);
    await act(async () => { resolve({ available: true, release: { ...release, versionCode: 99 } }); });
    await act(async () => { await vi.advanceTimersByTimeAsync(300); });
    expect(native.check).toHaveBeenLastCalledWith({ url: "http://other:8000/api/v1/updates/android" });
    expect(updates.release?.versionCode).toBe(3);
    expect(native.download).toHaveBeenCalledOnce();
  });
  it("回到前台再次检查，移除组件后清理监听", async () => {
    const rendered = render(<Harness />); await start();
    await act(async () => { await vi.advanceTimersByTimeAsync(61000); native.listeners.get("resume")?.(); });
    expect(native.check).toHaveBeenCalledTimes(2);
    rendered.unmount();
    expect(native.listeners.size).toBe(0);
  });
  it("离线错误保持可重试，安装授权取消不会假报安装成功", async () => {
    native.check.mockRejectedValueOnce(new Error("连接失败"));
    render(<Harness />); await start();
    expect(updates.stage).toBe("error"); expect(updates.message).toBe("连接失败");
    await act(() => updates.check());
    native.install.mockRejectedValueOnce(new Error("未允许安装更新"));
    await act(() => updates.install());
    expect(updates.stage).toBe("error"); expect(updates.message).toBe("未允许安装更新");
  });
});
