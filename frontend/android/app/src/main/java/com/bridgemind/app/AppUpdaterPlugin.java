package com.bridgemind.app;

import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.net.Uri;
import android.os.Build;
import android.provider.Settings;
import androidx.activity.result.ActivityResult;
import androidx.core.content.FileProvider;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.File;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** Sideloaded APK updates. The OS owns installation and user confirmation. */
@CapacitorPlugin(name = "AppUpdater")
public class AppUpdaterPlugin extends Plugin {
    private static final long MAX_APK_BYTES = 128L * 1024 * 1024;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Release latest;
    private File prepared;

    private static class Release {
        long code, size;
        String name, notes, sha256;
        URL apk;
        JSObject json() {
            JSObject value = new JSObject();
            value.put("versionCode", code);
            value.put("versionName", name);
            value.put("notes", notes);
            value.put("size", size);
            return value;
        }
    }

    @PluginMethod
    public void getInfo(PluginCall call) {
        try {
            PackageInfo info = installed();
            JSObject value = new JSObject();
            value.put("versionCode", code(info));
            value.put("versionName", info.versionName);
            call.resolve(value);
        } catch (Exception error) { fail(call, error); }
    }

    @PluginMethod
    public void check(PluginCall call) {
        worker.execute(() -> {
            try {
                // Invalid or missing manifests must not leave an old installable release.
                latest = null;
                prepared = null;
                String address = call.getString("url");
                if (address == null) throw new Exception("未配置更新服务地址");
                URL url = checkedUrl(address);
                HttpURLConnection connection = connect(url);
                try {
                    int status = connection.getResponseCode();
                    if (status == 404) { call.resolve(new JSObject().put("available", false)); return; }
                    if (status != 200) throw new Exception("更新服务返回 HTTP " + status);
                    byte[] bytes;
                    try (InputStream stream = connection.getInputStream(); ByteArrayOutputStream body = new ByteArrayOutputStream()) {
                        byte[] buffer = new byte[1024]; int count;
                        while ((count = stream.read(buffer)) != -1) {
                            body.write(buffer, 0, count);
                            if (body.size() > 16384) throw new Exception("版本信息过大");
                        }
                        bytes = body.toByteArray();
                    }
                    if (bytes.length > 16384) throw new Exception("版本信息过大");
                    JSONObject data = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
                    if (!"bridgemind.update.v1".equals(data.getString("protocol")) ||
                        !getContext().getPackageName().equals(data.getString("applicationId")))
                        throw new Exception("更新服务与应用不匹配");
                    Release release = new Release();
                    release.code = data.getLong("versionCode");
                    release.size = data.getLong("size");
                    release.name = data.getString("versionName");
                    release.notes = data.optString("notes", "");
                    release.sha256 = data.getString("sha256");
                    if (release.code < 1 || release.code > 2100000000 || release.size < 1 || release.size > MAX_APK_BYTES ||
                        !release.sha256.matches("[a-f0-9]{64}") || release.name.trim().isEmpty() || release.name.length() > 80 ||
                        release.notes.length() > 4000) throw new Exception("版本信息无效");
                    release.apk = checkedUrl(new URL(url, data.getString("apkPath")).toString());
                    if (!UpdateSource.allowsApk(url, release.apk, release.sha256, release.name)) throw new Exception("安装包地址无效");
                    JSObject result = new JSObject();
                    result.put("available", release.code > code(installed()));
                    if (release.code > code(installed())) {
                        latest = release;
                        result.put("release", release.json());
                    }
                    call.resolve(result);
                } finally { connection.disconnect(); }
            } catch (Exception error) { fail(call, error); }
        });
    }

    @PluginMethod
    public void download(PluginCall call) {
        worker.execute(() -> {
            File partial = null;
            try {
                Release release = latest;
                if (release == null) throw new Exception("请先检查新版本");
                File directory = new File(getContext().getFilesDir(), "updates");
                if (!directory.isDirectory() && !directory.mkdirs()) throw new Exception("无法创建更新目录");
                File apk = new File(directory, release.sha256 + ".apk");
                prepared = null;
                if (apk.isFile()) {
                    try { verify(apk, release); prepared = apk; call.resolve(); return; }
                    catch (Exception invalid) { if (!apk.delete()) throw new Exception("无法清理旧安装包"); }
                }
                // Only private updater files are removed; keep at most one completed package.
                File[] obsolete = directory.listFiles();
                if (obsolete != null) for (File file : obsolete) file.delete();
                partial = new File(directory, release.sha256 + ".part");
                HttpURLConnection connection = connect(release.apk);
                try {
                    if (connection.getResponseCode() != 200) throw new Exception("下载安装包失败");
                    long declared = connection.getContentLengthLong();
                    if (declared != -1 && declared != release.size) throw new Exception("安装包大小不匹配");
                    try (InputStream in = connection.getInputStream(); FileOutputStream out = new FileOutputStream(partial)) {
                        byte[] buffer = new byte[32768];
                        long total = 0, lastEvent = 0;
                        long deadline = System.nanoTime() + 300_000_000_000L;
                        int count;
                        while ((count = in.read(buffer)) != -1) {
                            total += count;
                            if (total > release.size || System.nanoTime() > deadline) throw new Exception("下载大小超限或超时");
                            out.write(buffer, 0, count);
                            if (total - lastEvent >= 131072 || total == release.size) {
                                notifyListeners("downloadProgress", new JSObject().put("percent", total * 100 / release.size));
                                lastEvent = total;
                            }
                        }
                        out.getFD().sync();
                    }
                } finally { connection.disconnect(); }
                verify(partial, release);
                if (!partial.renameTo(apk)) throw new Exception("无法保存安装包");
                partial = null;
                prepared = apk;
                call.resolve();
            } catch (Exception error) { fail(call, error); }
            finally { if (partial != null) partial.delete(); }
        });
    }

    @PluginMethod
    public void install(PluginCall call) {
        worker.execute(() -> {
            try {
                if (prepared == null || latest == null) throw new Exception("请先下载更新");
                verify(prepared, latest);
                getActivity().runOnUiThread(() -> {
                    try {
                        if (Build.VERSION.SDK_INT >= 26 && !getContext().getPackageManager().canRequestPackageInstalls()) {
                            Intent settings = new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                                Uri.parse("package:" + getContext().getPackageName()));
                            startActivityForResult(call, settings, "installPermissionResult");
                        } else openInstaller(call);
                    } catch (Exception error) { fail(call, error); }
                });
            } catch (Exception error) { fail(call, error); }
        });
    }

    @ActivityCallback
    private void installPermissionResult(PluginCall call, ActivityResult result) {
        if (call == null) return;
        if (Build.VERSION.SDK_INT >= 26 && !getContext().getPackageManager().canRequestPackageInstalls()) {
            call.reject("未允许安装更新，可稍后点击安装重试");
            return;
        }
        try { openInstaller(call); } catch (Exception error) { fail(call, error); }
    }

    private void openInstaller(PluginCall call) throws Exception {
        if (prepared == null || !prepared.isFile()) throw new Exception("安装包已失效，请重新检查更新");
        Uri uri = FileProvider.getUriForFile(getContext(), getContext().getPackageName() + ".fileprovider", prepared);
        Intent intent = new Intent(Intent.ACTION_VIEW).setDataAndType(uri, "application/vnd.android.package-archive")
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        getActivity().startActivity(intent);
        call.resolve(new JSObject().put("status", "installerOpened"));
    }

    private void verify(File file, Release release) throws Exception {
        if (file.length() != release.size) throw new Exception("安装包大小校验失败");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new FileInputStream(file)) {
            byte[] buffer = new byte[32768]; int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        if (!hex(digest.digest()).equals(release.sha256)) throw new Exception("安装包 SHA256 校验失败");
        PackageManager manager = getContext().getPackageManager();
        PackageInfo archive = manager.getPackageArchiveInfo(file.getAbsolutePath(), packageFlags());
        PackageInfo current = installed();
        if (archive == null || !current.packageName.equals(archive.packageName) || code(archive) != release.code ||
            !release.name.equals(archive.versionName) || code(archive) <= code(current))
            throw new Exception("安装包应用或版本不匹配，不能降级");
        if (archive.applicationInfo == null || archive.applicationInfo.minSdkVersion > Build.VERSION.SDK_INT)
            throw new Exception("此版本不支持当前 Android 系统");
        Set<String> signatures = signers(current);
        if (signatures.isEmpty() || !signatures.equals(signers(archive))) throw new Exception("安装包签名与当前应用不一致");
    }

    private PackageInfo installed() throws Exception {
        return getContext().getPackageManager().getPackageInfo(getContext().getPackageName(), packageFlags());
    }
    @SuppressWarnings("deprecation")
    private int packageFlags() { return Build.VERSION.SDK_INT >= 28 ? PackageManager.GET_SIGNING_CERTIFICATES : PackageManager.GET_SIGNATURES; }
    @SuppressWarnings("deprecation")
    private long code(PackageInfo info) { return Build.VERSION.SDK_INT >= 28 ? info.getLongVersionCode() : info.versionCode; }
    @SuppressWarnings("deprecation")
    private Set<String> signers(PackageInfo info) throws Exception {
        Signature[] values = Build.VERSION.SDK_INT >= 28
            ? (info.signingInfo == null ? null : info.signingInfo.getApkContentsSigners()) : info.signatures;
        Set<String> result = new HashSet<>();
        if (values != null) for (Signature value : values)
            result.add(hex(MessageDigest.getInstance("SHA-256").digest(value.toByteArray())));
        return result;
    }
    private static String hex(byte[] bytes) {
        StringBuilder value = new StringBuilder();
        for (byte b : bytes) value.append(String.format("%02x", b & 255));
        return value.toString();
    }
    private URL checkedUrl(String address) throws Exception {
        URL url = new URL(address);
        boolean debug = (getContext().getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        if ((!"https".equals(url.getProtocol()) && !(debug && "http".equals(url.getProtocol()))) ||
            url.getHost().isEmpty() || url.getUserInfo() != null) throw new Exception("更新地址必须使用 HTTPS（调试包支持局域网 HTTP）");
        return url;
    }
    private static HttpURLConnection connect(URL url) throws Exception {
        URL current = url;
        for (int redirects = 0; redirects <= 5; redirects++) {
            HttpURLConnection connection = (HttpURLConnection) current.openConnection();
            connection.setConnectTimeout(10000);
            connection.setReadTimeout(15000);
            connection.setInstanceFollowRedirects(false);
            connection.setRequestProperty("Accept-Encoding", "identity");
            connection.setRequestProperty("Cache-Control", "no-cache");
            connection.setRequestProperty("User-Agent", "BridgeMind-Updater");
            try {
                int status = connection.getResponseCode();
                if (status != 301 && status != 302 && status != 303 && status != 307 && status != 308) return connection;
                String location = connection.getHeaderField("Location");
                if (location == null) throw new Exception("更新重定向缺少地址");
                URL next = new URL(current, location);
                if (redirects == 5 || !UpdateSource.allowsRedirect(url, next)) throw new Exception("不允许的更新重定向");
                connection.disconnect(); current = next;
            } catch (Exception error) { connection.disconnect(); throw error; }
        }
        throw new Exception("更新重定向次数超限");
    }
    private static void fail(PluginCall call, Exception error) {
        call.reject(error.getMessage() == null ? "更新失败，请稍后重试" : error.getMessage());
    }
    @Override
    protected void handleOnResume() { notifyListeners("resume", new JSObject()); }
    @Override
    protected void handleOnDestroy() { worker.shutdownNow(); }
}
