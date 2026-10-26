# Android 调试版

正式发行与版本维护见[GitHub Releases说明](RELEASING.md)：`npm run android:release -- --compat-signing`构建不可调试的签名Release，`npm run release:publish`发布。下文保留调试版开发、真机验证和历史记录。

当前1.2.0（versionCode=5）已增加[四家模型观战牌桌](MODEL-DEBUG.md)，并通过真实Wi-Fi与App内更新安装完成。系统包信息已确认1.2.0，真机新版牌桌自动出牌截图在`artifacts/frontend/model-spectator-20261026/native-final.png`。以下1.1.x记录保留为历史。

最新1.1.2（versionCode=4）已采用[App单屏首页规范](DESIGN.md)，并完成实际Wi-Fi启动自动检查、下载校验及App内更新安装，系统包信息已确认1.1.2。本次未使用ADB覆盖安装，详见[更新记录](UPDATES.md)。下方1.1.1与更早记录为历史。

2026-10-26最新1.1.1（versionCode=3）已加入[应用内更新](UPDATES.md)，并通过ADB安装至已连接手机。支持启动/回到前台自动检查、自动下载、完整性与签名校验、点击安装及失败重试。115项前端测试、2项发布脚本测试及107项网关测试通过，Web/Android构建与Android lint通过。已真机验证更新下载与损坏包拦截，系统安装确认与锁屏期间的原生界面尚未验收。最新包在 `.build/frontend/BridgeMind-debug.apk`；下文旧安装记录保留作为历史。

React 页面通过 Capacitor 8.5.3 包装为 Android 应用，包名 `com.bridgemind.app`，桌面名称 BridgeMind。网页、字体、音效均内置于 APK；基础练习不依赖 Vite 服务器。

## 横屏牌桌

Android 应用固定采用传感器横屏，并隐藏系统状态栏与导航栏；从屏幕边缘滑动仍可临时呼出系统栏。手机上的主界面是一整张绿色牌桌：四家座位、当前墩和底部扇形手牌都在同一屏幕内，不使用网页的页头、介绍区和常驻侧栏。

- 点击手牌选中，再点击右下角“出牌”确认；有首引花色时只允许选合法跟牌。
- 庄家操作明手时，底部切换成明手的手牌；四手模式同样切换为当前行动方手牌。
- 顶部“分析”打开侧边抽屉，AI / DDS 使用标签切换；“记录”调出叫牌与逐墩历史。
- 打开分析、记录或设置时，自动对手和自动收墩暂时停住，关闭后按原暂停状态继续。
- 左下角可撤销与查看四手，右下角可暂停或单步代打。

电脑网页和手机共用这套牌桌交互，电脑会放大手牌和控件。手机浏览器竖屏时提示旋转。日常入口是启动局域网脚本输出的 8000 地址，开发入口仍为 `http://localhost:5173/`。

## 构建和安装

需要 Node.js 22+、JDK 21、Android SDK 36、ADB，以及已启用 USB 调试并授权电脑的手机。当前电脑已有 `D:\Java\jdk-21` 和 `D:\Android\SDK`；构建脚本从环境变量或 `adb` / `java` 路径发现位置，不把本机 SDK 路径提交到 Git。

```powershell
cd D:\code\BridgeMind\frontend
npm ci
npm run android:build
npm run android:install
```

`android:install` 会重新构建、同步资源、生成调试 APK，使用 `adb install -r` 安装并启动；保留已安装应用的数据。多台设备连接时指定设备：

```powershell
npm run android:install -- --serial=你的设备序列号
```

APK 路径：`D:\code\BridgeMind\.build\frontend\BridgeMind-debug.apk`，原始 Gradle 输出位于 `frontend/android/app/build/outputs/apk/debug/app-debug.apk`。它使用本机 Android 调试证书签名，不是商店发行版。

## 真机调试

- 修改页面后运行 `npm run android:install`，手机应用会更新为新的内置页面。
- 调试 APK 的 WebView 支持 Chrome 远程调试，可在电脑 Chrome 的 `chrome://inspect/#devices` 选择 BridgeMind WebView。不会开启整个手机的系统调试权限以外的额外权限。
- 仅查看应用日志，避免采集其他应用数据：

```powershell
adb shell pidof com.bridgemind.app
# 将下面的 PID 替换为上一条命令的结果
adb logcat --pid=PID
```

## 局域网分析接口

真实模型 / DDS 的 HTTP 网关已实现。电脑仓库根目录执行 `.\scripts\start-lan.ps1` 即可构建网页并启动两项服务，本次服务地址为 `http://192.168.5.118:8000`。新版 APK 已内置该地址，旧版的空地址设置会自动迁移；主动保存空地址则离线练习。在设置中点击“测试连接”可分别检查 AI 与 DDS。

调试版从 `http://localhost` 加载内置资源，只在 debug manifest 中允许明文 HTTP；网关已允许该来源的 CORS 预检。手机与电脑之间已通过 Wi-Fi 实测，无需 ADB reverse。IP 变化和重启后的恢复方式见 [局域网使用说明](LAN.md)。不要填写现有 DDS TCP 服务的 18765 端口。

## 配置边界

- `capacitor.config.ts` 不设置远程 `server.url`，所以安装后无需连接开发服务器。
- 构建脚本只在子进程中设置 `BRIDGEMIND_ANDROID_DEBUG=1`，并生成被忽略的 `android/local.properties`。
- 常规 `npx cap sync android` 不带上述环境变量时使用 HTTPS localhost；release manifest 没有明文流量放行。发布前需重新同步并配置正式签名与 HTTPS 服务。
- Capacitor 8.5 的 SystemBars 和页面的安全区 padding 配合，适配状态栏、刘海与底部手势区域。
- 浏览器版本的“导出牌局”使用 Blob 下载；Android 原生文件保存/分享尚未做专门适配。
- 当前对局保存在内存中；强制停止或重启应用会重新发牌。尚未提供原生后台对局恢复和联机多人。

## 本轮验证（2026-10-26）

初版已通过 ADB 安装、冷启动并检查桥牌页面，历史记录见 `artifacts/frontend/android-validation-20261026.json`。最新统一牌桌与局域网配置版已构建并覆盖安装；90 项前端测试和 156 项 Python 测试通过。已从实际手机经 Wi-Fi 调用真实模型和 DDS，记录见 `artifacts/frontend/lan-validation-20261026.json`。用户选择稍后自行体验，最新安装包没有继续做真机画面验收。

首轮 ADB 安装曾被手机以 `INSTALL_FAILED_USER_RESTRICTED` 拒绝；用户在手机允许 USB 安装后，第二轮安装成功。本轮没有关闭系统安全校验，也没有清除手机上的应用数据。
