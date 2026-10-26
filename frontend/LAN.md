# 手机与电脑局域网连接

电脑网页与Android应用共用主界面、人机对战和模型调试。模型和DDS在电脑运行；客户端发送当前合法可见局面并接收真实模型决策和调试输出。

## 启动

在 PowerShell 的仓库根目录执行：

```powershell
cd D:\code\BridgeMind
.\scripts\start-lan.ps1
```

脚本自动选择本机 WLAN IPv4，更新被 Git 忽略的 `frontend/.env.local`，构建网页，启动后台 DDS 和 HTTP 网关。已有健康服务会复用。脚本输出可用网址；本次为 **http://192.168.5.118:8000/**。网页与接口共用这个地址，使用时电脑需保持运行。

本机Node、JDK、Android SDK、Python环境和模型均已就绪。脚本默认出牌模型为 `runs/play-sl-v2-20261026-2h/best.pt`，叫牌模型为 `runs/sl-pilot/last.pt`；重新克隆需准备依赖和检查点。修改权重后使用 `-Restart`重载服务。

可选参数：

```powershell
# 已构建网页，仅恢复服务
.\scripts\start-lan.ps1 -SkipBuild
# 指定当前电脑地址
.\scripts\start-lan.ps1 -BindAddress 192.168.5.118
# 修改后重载自己启动的服务
.\scripts\start-lan.ps1 -Restart
```

日志保存在 `.build/lan/gateway.stderr.log` 与 `dds.stderr.log`，进程记录在 `.build/lan/state.json`。脚本没有配置开机自启；重启电脑后重新运行即可。服务绑定指定的局域网地址，`localhost:8000` 不是这个启动方式的入口。

## 手机配置

1. 电脑与手机连接同一局域网。本次电脑为 `192.168.5.118`，手机为 `192.168.5.126`。
2. 打开已更新的 BridgeMind，在设置中确认分析地址为 `http://192.168.5.118:8000`，点击“测试连接”。AI 与 DDS 都应显示就绪。
3. 主界面选择人机对战或模型调试。对局中点击“模型调试”暂停并查看当前局面，选择“AI实战”或“DDS全知”，点击“请求分析”。

新版 APK 已写入本机地址，旧版空地址设置会迁移；之后主动保存空地址可离线练习。地址改变时，手机设置里手动填写脚本输出的新地址即可。更新 APK 前运行启动脚本会同步构建默认地址，然后在 `frontend/` 运行 `npm run android:install`。

浏览器也可直接打开上述网页地址。开发模式仍可用 `frontend/` 中的 `npm run dev`，端口为 5173；日常体验推荐启动脚本输出的 8000 地址。

## 已验证与实现范围

2026-10-26 已从实际 Android 手机通过 Wi-Fi 请求健康检查、AI 和 DDS，均成功；未使用 USB 端口反向转发，未改动 Windows 防火墙。电脑网页中的连接测试、纯网络 AI 和 DDS 分析均已实际验证。新版 APK 已通过 ADB 覆盖安装，用户稍后自行体验，未继续操作锁屏手机。

- AI：SL v2在CPU推理，返回完整手牌动作概率、未知牌归属和价值预测；另加载独立叫牌模型，尚未接入hybrid搜索。胜率头尚未训练，明确显示不可用。
- DDS：通过电脑本地的 `127.0.0.1:18765` 独立服务计算，返回所有等价最优牌。该 TCP 端口不直接用于手机。
- 人机对战使用真实叫牌模型和SL v2出牌模型：南家人工叫牌，南北人工出牌；其余动作由模型完成。调试分析本身只展示结果，模型代打需要另外点击或返回对局。
- 前端仅在 AI 请求中发送控制者手牌、已公开明手与公开历史；网关拒绝混入四家隐藏牌的 AI 请求。DDS 使用独立的全知请求。
- HTTP 网关默认限制允许的来源，支持 Android debug 的 `http://localhost`、电脑网页同源和本机 Vite 开发来源。新增来源可用网关的 `--origin` 参数。当前是局域网调试服务，未配置公网访问或账户体系。

协议与健康检查见 [接口说明](../docs/frontend-analysis-api.md)，验证记录见 `artifacts/frontend/lan-validation-20261026.json`。

本轮新主界面和调试台已在电脑及844×390浏览器验证，服务已切换SL v2；最新APK构建通过但未安装，因为当前无ADB设备。上面的手机Wi-Fi和安装记录属于此前版本。本轮记录见 `artifacts/frontend/model-workbench-20261026/verification.json`。
