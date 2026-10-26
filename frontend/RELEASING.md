# GitHub Releases 与 Android Release 包

公开仓库：[ghhufue/BridgeMind](https://github.com/ghhufue/BridgeMind)。App 的版本以 `frontend/version.json` 和对应的 GitHub Release 标签 `v<versionName>` 为准；APK只作为Release附件分发，不提交到Git。当前首个正式构建为1.2.2（versionCode=7）。

每个Release包含：

- `BridgeMind-v<版本>.apk`：已签名、不可调试的Android Release构建。
- `latest.json`：应用内更新协议清单，含包名、版本、精确大小、SHA256、固定版本APK地址及源码提交。
- `SHA256SUMS`：安装包校验和。

## 发布步骤

1. 修改 `frontend/version.json`，递增 `versionCode`，更新 `versionName`；编辑 `frontend/release-notes.md`。
2. 在 `frontend/` 运行 `npm ci`（依赖有变化时）、`npm test`，按改动范围完成后端检查。
3. 提交所有源码并推送到 `origin/main`。发布脚本要求工作区干净、当前提交已推送，防止二进制与源码版本不一致。
4. 在 `frontend/` 构建签名Release：

   ```powershell
   npm run android:release -- --compat-signing
   ```

5. 使用已登录的GitHub CLI发布：

   ```powershell
   npm run release:publish
   # 可选更新说明，路径相对frontend/
   npm run release:publish -- --notes-file=release-notes.md
   ```

发布脚本先创建草稿，上传全部附件，再下载并校验已上传的APK与清单，最后发布并设为Latest。已有标签拒绝覆盖；后续变更必须递增版本。发布中途失败会保留草稿，不能把未完整验证的草稿当作已发布版本。

产物位于被忽略的 `.build/frontend/BridgeMind-release.apk`，构建元数据包含源码提交、工作区状态及APK哈希。发布时再次核对这三项、版本号和远端仓库。构建阶段使用Android SDK核对APK包名/版本、验证签名，并拒绝可调试的Release APK。

## 签名保管

`--compat-signing` 使用本机已有的 `~/.android/debug.keystore` 为**Release变体**签名，目的是让先前已安装的本机调试包可以原地升级。它仍然是不可调试的Release构建，但签名证书沿用历史证书，不能擅自换证书。keystore仅保留在本机，已被Git忽略；不要删除它，不上传到公开仓库或Release。

如果另行提供签名，使用环境变量 `BRIDGEMIND_KEYSTORE_PATH`、`BRIDGEMIND_KEYSTORE_PASSWORD`、`BRIDGEMIND_KEY_ALIAS`、`BRIDGEMIND_KEY_PASSWORD`，省略兼容签名参数。现有客户端只接受相同签名，不支持证书轮换。脚本不把密码放到命令行或输出日志，也不向GitHub上传签名密钥。

## 应用内更新与旧版迁移

Release构建自动读取 `release.config.json`，将以下稳定地址编入APK：

```text
https://github.com/ghhufue/BridgeMind/releases/latest/download/latest.json
```

更新只需要能访问GitHub，不依赖电脑开机。原生更新器只接受同一仓库、相同目标版本的APK地址；允许GitHub正常的HTTPS附件重定向到指定分发域名，拒绝降级HTTP、跨仓库和任意第三方重定向。下载后仍校验文件、包名、版本与相同签名，最后由系统确认安装。

旧版1.2.1及更早APK尚不知道GitHub更新源。首次迁移可手动下载Release APK覆盖安装，或通过现有局域网更新。公开Release发布成功后，脚本同时更新本机 `.build/releases/android/` 的兼容清单和安装包，旧版在现有电脑网关启动后即可检测到1.2.2；安装后转用GitHub更新。无需卸载，原先保存的设置保留。

模型推理仍由用户配置的电脑网关提供，不随公开仓库上传权重或训练数据。Release保持现有HTTP localhost内置资源和局域网HTTP分析连接；**更新清单及Release安装包始终使用HTTPS**。新安装不内置发布者的局域网IP，用户自行设置电脑地址，已有安装保留先前地址。

## 本地开发

`npm run android:build` / `android:install`仍构建调试版；`android:publish`保留为局域网调试发布。正式版本维护使用上面的Release流程。不要以`android:publish`代替GitHub发布，也不要向Git提交APK、Gradle输出、node_modules、模型、数据或任何签名文件。
