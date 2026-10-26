"""Read locally published Android releases; never accept uploads over HTTP."""

import json
from pathlib import Path
import re


MAX_APK_BYTES = 128 * 1024 * 1024


def apk_file(root: Path, filename: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}\.apk", filename):
        raise FileNotFoundError("Unknown update package")
    target = (root / filename).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise FileNotFoundError("Update package not found")
    return target


def latest_release(root: Path) -> dict:
    manifest = root / "latest.json"
    if not manifest.resolve().is_relative_to(root.resolve()) or manifest.stat().st_size > 16384:
        raise ValueError("Invalid update manifest")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid update manifest")
    if value.get("protocol") != "bridgemind.update.v1" or value.get("applicationId") != "com.bridgemind.app":
        raise ValueError("Invalid update application")
    for key, limit in (("versionCode", 2100000000), ("size", MAX_APK_BYTES)):
        if type(value.get(key)) is not int or not 1 <= value[key] <= limit:
            raise ValueError(f"Invalid {key}")
    for key, limit in (("versionName", 80), ("notes", 4000)):
        if not isinstance(value.get(key), str) or len(value[key]) > limit:
            raise ValueError(f"Invalid {key}")
    if not value["versionName"].strip() or not re.fullmatch(r"[a-f0-9]{64}", value.get("sha256", "")):
        raise ValueError("Invalid version or checksum")
    filename = value["sha256"] + ".apk"
    if value.get("apkPath") != "/updates/android/" + filename or apk_file(root, filename).stat().st_size != value["size"]:
        raise ValueError("Invalid update package")
    # Return only public fields. Ignore accidental local build metadata.
    return {key: value[key] for key in ("protocol", "applicationId", "versionCode", "versionName", "size", "sha256", "apkPath", "notes")}
