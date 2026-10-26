import hashlib
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from bridgemind.web.server import LanServer
from bridgemind.web.updates import latest_release


def publish(root, *, code=2, content=b"test APK bytes"):
    root.mkdir(exist_ok=True)
    sha = hashlib.sha256(content).hexdigest()
    (root / (sha + ".apk")).write_bytes(content)
    manifest = {"protocol": "bridgemind.update.v1", "applicationId": "com.bridgemind.app",
                "versionCode": code, "versionName": "1.1.0", "size": len(content), "sha256": sha,
                "apkPath": f"/updates/android/{sha}.apk", "notes": "更新说明"}
    (root / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def gateway(tmp_path):
    root = tmp_path / "releases"
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<h1>BridgeMind</h1>")
    server = LanServer(("127.0.0.1", 0), object(), web, update_root=root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", root
    server.shutdown()
    server.server_close()
    thread.join()


def test_no_release_returns_404_and_keeps_web_working(gateway):
    base, _ = gateway
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/v1/updates/android")
    assert error.value.code == 404
    assert urlopen(base + "/").read() == b"<h1>BridgeMind</h1>"


def test_manifest_and_apk_available_without_restart(gateway):
    base, root = gateway
    first = publish(root)
    request = Request(base + "/api/v1/updates/android", headers={"Origin": "http://localhost"})
    with urlopen(request) as response:
        assert json.load(response) == first
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost"
    with urlopen(base + first["apkPath"]) as response:
        content = response.read()
        assert response.headers["Content-Type"] == "application/vnd.android.package-archive"
        assert len(content) == first["size"]
        assert hashlib.sha256(content).hexdigest() == first["sha256"]
    second = publish(root, code=3, content=b"new test APK")
    assert json.load(urlopen(base + "/api/v1/updates/android")) == second
    # Existing downloads remain valid after publishing the next version.
    assert urlopen(base + first["apkPath"]).read() == b"test APK bytes"


@pytest.mark.parametrize("path", [
    "/updates/android/latest.json", "/updates/android/../latest.json",
    "/updates/android/%2e%2e/latest.json", "/updates/android/x.apk",
    "/updates/other/" + "a" * 64 + ".apk", "/updates/android/" + "a" * 64 + ".apk",
])
def test_update_route_only_serves_named_apks(gateway, path):
    base, root = gateway
    publish(root)
    with pytest.raises(HTTPError) as error:
        urlopen(base + path)
    assert error.value.code == 404


@pytest.mark.parametrize("field,value", [
    ("applicationId", "another.app"), ("protocol", "another.protocol"),
    ("versionCode", True), ("versionCode", 0), ("versionCode", 2.5),
    ("size", 99999), ("sha256", "../private"), ("apkPath", "https://other.example/app.apk"),
    ("versionName", ""), ("notes", "x" * 4001),
])
def test_invalid_manifests_are_rejected(tmp_path, field, value):
    manifest = publish(tmp_path)
    manifest[field] = value
    (tmp_path / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        latest_release(tmp_path)


def test_bad_manifest_is_503_not_partial_release(gateway):
    base, root = gateway
    root.mkdir()
    (root / "latest.json").write_text("{not json")
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/v1/updates/android")
    assert error.value.code == 503
