"""Verify a Git backup and restore its exact source, data and training run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def restore(backup, destination):
    backup, destination = Path(backup).resolve(), Path(destination).resolve()
    manifest = json.loads((backup/"manifest.json").read_text())
    for name, info in manifest["files"].items():
        path = (backup/name).resolve()
        if not path.is_relative_to(backup) or sha(path) != info["sha256"]:
            raise ValueError(f"Invalid backup file: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        for assembly in sorted((backup/"assets").glob("*.assembly.json")):
            info = json.loads(assembly.read_text())
            archive = Path(temporary)/Path(info["archive"]).name
            with archive.open("wb") as output:
                for part in info["parts"]:
                    path = backup/"assets"/part["name"]
                    if sha(path) != part["sha256"]:
                        raise ValueError(f"Invalid archive part: {path}")
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, output)
            if sha(archive) != info["sha256"]:
                raise ValueError(f"Invalid assembled archive: {archive}")
            with tarfile.open(archive) as tar:
                tar.extractall(destination, filter="data")
    run = destination/manifest["run"]
    if not run.resolve().is_relative_to(destination):
        raise ValueError("Invalid restore run path")
    if (backup/"run").exists():
        shutil.copytree(backup/"run", run, dirs_exist_ok=True)
    shutil.copytree(backup/"logs", destination/"restored-job-logs", dirs_exist_ok=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--backup", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    print(restore(args.backup, args.destination))
