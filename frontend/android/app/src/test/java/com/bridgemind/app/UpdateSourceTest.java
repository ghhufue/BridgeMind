package com.bridgemind.app;

import java.net.URL;
import org.junit.Test;
import static org.junit.Assert.*;

public class UpdateSourceTest {
    private final String sha = "a".repeat(64);
    @Test public void githubPackageMustStayInSelectedRepositoryAndVersion() throws Exception {
        URL feed = new URL("https://github.com/ghhufue/BridgeMind/releases/latest/download/latest.json");
        assertTrue(UpdateSource.allowsApk(feed, new URL("https://github.com/ghhufue/BridgeMind/releases/download/v1.2.2/BridgeMind-v1.2.2.apk"), sha, "1.2.2"));
        assertFalse(UpdateSource.allowsApk(feed, new URL("https://github.com/other/BridgeMind/releases/download/v1.2.2/BridgeMind-v1.2.2.apk"), sha, "1.2.2"));
        assertFalse(UpdateSource.allowsApk(feed, new URL("https://github.com/ghhufue/BridgeMind/releases/download/v1.2.1/BridgeMind-v1.2.1.apk"), sha, "1.2.2"));
        assertFalse(UpdateSource.allowsApk(feed, new URL("http://github.com/ghhufue/BridgeMind/releases/download/v1.2.2/BridgeMind-v1.2.2.apk"), sha, "1.2.2"));
    }
    @Test public void redirectsOnlyUseGithubHttpsDeliveryHosts() throws Exception {
        URL feed = new URL("https://github.com/ghhufue/BridgeMind/releases/latest/download/latest.json");
        assertTrue(UpdateSource.allowsRedirect(feed, new URL("https://github.com/ghhufue/BridgeMind/releases/download/v1.2.2/latest.json")));
        assertTrue(UpdateSource.allowsRedirect(feed, new URL("https://release-assets.githubusercontent.com/github-production-release-asset/1/file?sig=example")));
        for (String target : new String[]{"https://attacker.example/file", "http://release-assets.githubusercontent.com/file", "https://github.com/other/repo/releases/download/v1/file", "https://release-assets.githubusercontent.com:444/file", "https://user@release-assets.githubusercontent.com/file"})
            assertFalse(UpdateSource.allowsRedirect(feed, new URL(target)));
    }
    @Test public void lanPackagesKeepOriginalHashPathAndNoRedirects() throws Exception {
        URL feed = new URL("http://192.168.1.2:8000/api/v1/updates/android");
        assertTrue(UpdateSource.allowsApk(feed, new URL("http://192.168.1.2:8000/updates/android/" + sha + ".apk"), sha, "1.2.2"));
        assertFalse(UpdateSource.allowsApk(feed, new URL("http://192.168.1.3:8000/updates/android/" + sha + ".apk"), sha, "1.2.2"));
        assertFalse(UpdateSource.allowsApk(feed, new URL("http://192.168.1.2:8000/private.apk"), sha, "1.2.2"));
        assertFalse(UpdateSource.allowsRedirect(feed, new URL("https://github.com/ghhufue/BridgeMind/releases/download/v1/file")));
    }
}
