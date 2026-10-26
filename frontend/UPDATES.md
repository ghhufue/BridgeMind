# Android 应用内更新

正式版本现在通过[GitHub Releases](RELEASING.md)维护。1.2.2 Release开始从GitHub独立检查后续版本，旧版首次迁移仍可使用原局域网清单。以下为此前局域网更新的设计与验证记录。

最新发布：1.2.1（versionCode=6），修复人机牌桌信息入口误跳模型调试。最近确认手机安装的是1.2.0（versionCode=5），该版本通过真实Wi-Fi与App内安装器完成更新；1.2.1发布时手机未连接ADB，等待应用后续自动检查与用户安装确认。此前1.1.x流程与验证记录见下文。

2026-10-26：已实现启动、回到前台及前台每30分钟检查；默认自动下载，设置中可关闭自动下载或手动检查。下载完成后提示安装，点击后才申请本应用的安装权限并打开 Android 安装界面。系统确认由用户完成，不能承诺所有设备都能静默安装。安装会重启应用，当前内存牌局不会保留，提示允许稍后安装。

当前版本由 `frontend/version.json` 统一管理，首次具备更新模块的是1.1.0（versionCode=2）。最新发布为1.1.2（versionCode=4），前序1.1.1通过ADB安装；1.1.2已在实际Wi-Fi通过手机启动自动检查下载并校验，用户完成系统安装后已核对安装版本为1.1.2。本次没有使用ADB覆盖安装。更早的应用需要先手动或通过ADB安装一次，才能发现后续更新。版本显示读取实际已安装APK，不依赖网页版本常量。

## 发布一个版本

1. 修改 `frontend/version.json`，递增整数 `versionCode`，并填写显示给用户的 `versionName`。改变APK内容时必须递增versionCode。
2. 在 `frontend/` 运行：

   ```powershell
   npm run android:publish
   # 可选：从UTF-8文件读取更新说明，路径相对frontend/
   npm run android:publish -- --notes-file=release-notes.txt
   ```

3. 保持电脑服务运行。首次启用新路由时，在仓库根目录执行：

   ```powershell
   .\scripts\start-lan.ps1 -RestartGateway
   ```

   `-RestartGateway`仅重启HTTP网关，保留运行中的DDS。后续发布只替换版本清单，已经运行的新网关无需重启。

发布命令构建Web、同步Capacitor资源、构建调试APK，使用Android SDK检查APK中的包名/版本并验证签名，再写入 `.build/releases/android/`。APK使用SHA256命名，先完整写入安装包，再原子替换 `latest.json`；同一个versionCode不能发布不同内容。保留旧APK，已经开始的下载不受新发布影响。该命令目前发布的是**本机签名的Android调试版**，用于现有私有客户端更新；没有发布到应用商店或公网。

安装包仍输出到 `.build/frontend/BridgeMind-debug.apk`。单纯 `npm run android:build` 不发布版本。网页构建独立生效，浏览器重新打开或刷新后获取新版。

## 更新服务

默认与用户已保存的分析服务共用地址，例如 `http://192.168.5.118:8000`：

- `GET /api/v1/updates/android`：当前清单；没有发布时404，配置损坏时503；禁止缓存。
- `GET /updates/android/<sha256>.apk`：流式下载命名安装包，内容固定。
- `--update-root`：Python网关可指定发布目录，默认为 `.build/releases/android`。不提供远程上传或发布接口。

清单字段为 `protocol='bridgemind.update.v1'`、`applicationId='com.bridgemind.app'`、`versionCode`、`versionName`、`size`、`sha256`、`apkPath` 与 `notes`。大小、哈希和路径由脚本生成，不要手填。

电脑需开机、服务需可访问；当前局域网地址只在能连到该网络时有效。应用完全关闭时不轮询，下一次启动或回到前台会检查。前台自动检查和下载不打断打牌，只有用户选择安装才暂停。

可在构建时设置 `VITE_UPDATE_SERVER_URL=https://稳定的更新域名`，把更新服务与模型服务独立部署。该地址是基地址，需提供上面两个相同路径；安装包与清单必须同源，不跟随重定向。正式包仅接受HTTPS，调试包支持现有HTTP局域网。当前没有配置公网托管、域名或商店更新。改动构建时地址也需要先把新地址随一个可获取的APK送到手机，故正式使用应保留稳定的更新域名。

## 完整性与安装

- 原生模块下载至应用私有目录，校验精确文件大小与SHA256，再读取APK包名、versionCode、versionName、最低Android版本和签名；仅允许与当前应用签名一致的更高版本。
- 最多128 MiB；连接10秒、单次读取15秒、下载总预算5分钟。拒绝重定向、跨域下载、非预期路径、损坏文件与降级；下载失败清理临时文件，重试从头下载。已有完整缓存也重新校验。
- 安装前再次校验，使用已有FileProvider仅分享更新文件。首次点击安装可能打开“允许来自此来源的应用”设置，授权返回后打开系统安装器；拒绝或取消可重试，不显示“安装成功”。
- 当前使用原来的调试签名证书，后续更新须保留同一证书。不要删除或替换本机调试keystore。正式发行需要单独配置稳定的正式签名和HTTPS分发；当前校验不支持签名轮换，换证书的包不能覆盖现有安装。

## 验证记录

2026-10-26：115项Vitest测试、2项发布脚本测试、107项Python网关测试（含19项更新测试）通过。Web与Android构建通过，Android `:app:lintDebug` 无错误（保留已有资源/模板警告）。

通过已连接Android真机，以临时旧版本与USB反向端口验证原生完整链路：损坏APK触发SHA256错误且不成为可安装包；真实1.1.1包自动下载，原生包信息和签名校验通过，再从应用私有目录取回逐字节核对SHA256与大小。结束后恢复正式地址配置并通过ADB覆盖安装已发布1.1.1，移除临时USB端口。报告在 `artifacts/frontend/app-updates-20261026/native-verification.json`。

手机当时锁屏，普通局域网请求连接失败，因此未验收当前Wi-Fi连通性、原生更新界面的视觉效果，以及用户授权/确认后的系统安装流程。最终安装通过ADB完成，不能视为应用内安装确认的端到端验收。

后续1.1.2首页改版发布：用户连接Wi-Fi后，通过命令重新打开**原有1.1.1 App**，真实触发启动自动检查，手机192.168.5.126向电脑网关读取清单并下载SHA256为`00caaadfd35dd628badb319f6e9d717e88d56b53b65e685aa7c67fd8552a570d`的新版，原生校验通过。截图显示“1.1.2已下载/安装更新”，见`artifacts/frontend/app-home-20261026/native-update-ready.png`。用户随后通过App内安装入口完成系统安装，已读取系统包信息确认versionCode=4、versionName=1.1.2，新进程再次检查返回available=false。本次未使用ADB覆盖安装，实际Wi-Fi检查、下载、安装链路已验收。
