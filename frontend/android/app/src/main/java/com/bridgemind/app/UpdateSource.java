package com.bridgemind.app;

import java.net.URL;

/** URL policy is independent of Android so release transport rules can be tested. */
final class UpdateSource {
    private static int port(URL url) { return url.getPort() < 0 ? url.getDefaultPort() : url.getPort(); }
    private static boolean clean(URL url) { return url.getUserInfo() == null && url.getRef() == null; }
    private static boolean https(URL url) { return "https".equals(url.getProtocol()) && port(url) == 443 && clean(url); }
    private static String repository(URL url) {
        if (!https(url) || !"github.com".equalsIgnoreCase(url.getHost()) || url.getQuery() != null) return null;
        String[] parts = url.getPath().split("/");
        if (parts.length < 6 || !parts[1].matches("[A-Za-z0-9_-]+") || !parts[2].matches("[A-Za-z0-9_.-]+") ||
            !"releases".equals(parts[3]) || !("download".equals(parts[4]) || "latest".equals(parts[4]))) return null;
        return "/" + parts[1] + "/" + parts[2] + "/releases/";
    }
    static boolean allowsApk(URL manifest, URL apk, String sha256, String version) {
        if (!clean(apk) || apk.getQuery() != null) return false;
        String repo = repository(manifest);
        if (repo != null) {
            if (!version.matches("[A-Za-z0-9][A-Za-z0-9._-]{0,79}")) return false;
            return https(apk) && "github.com".equalsIgnoreCase(apk.getHost()) &&
                apk.getPath().equals(repo + "download/v" + version + "/BridgeMind-v" + version + ".apk");
        }
        return manifest.getProtocol().equals(apk.getProtocol()) && manifest.getHost().equalsIgnoreCase(apk.getHost()) &&
            port(manifest) == port(apk) && apk.getPath().equals("/updates/android/" + sha256 + ".apk");
    }
    static boolean allowsRedirect(URL original, URL next) {
        String repo = repository(original);
        if (repo == null || !https(next)) return false;
        String host = next.getHost().toLowerCase(java.util.Locale.ROOT);
        if (host.equals("github.com")) return next.getQuery() == null && next.getPath().startsWith(repo);
        return host.equals("release-assets.githubusercontent.com") || host.equals("objects.githubusercontent.com");
    }
}
