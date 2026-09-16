# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage an allowlisted build context from the pinned vision11 release wheels."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import pybase64 as base64

ASSETS = Path(__file__).resolve().parent
SOURCE = ASSETS.parents[1]
SOURCE_REVISION = "678d8f531e5e498e4060b21df254d27467e0cef9"
RELEASE_URL = (
    "https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/"
    "v0.28.1rc1-vision9-sm89-sm120-cu130/"
)
VLLM_VERSION = "0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130"
FLASHINFER_VERSION = "0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213"
WHEELS = {
    "vllm": (
        f"vllm-{VLLM_VERSION}-cp312-cp312-linux_x86_64.whl",
        "e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed",
        247193774,
        4931,
    ),
    "flashinfer": (
        f"flashinfer_python-{FLASHINFER_VERSION}-py3-none-any.whl",
        "078aa7682d699c67dce619fc8acf29ec41934ab5ba916268d1fe6f8850459de8",
        23983606,
        6246,
    ),
}
ASSET_NAMES = (
    "manifest.json",
    "constraints.txt",
    "overrides.txt",
    ".dockerignore",
    "verify_image.py",
    "cold_jit_smoke.py",
)
RUNTIME_PATHS = (
    "vllm",
    "csrc",
    "requirements",
    "cmake",
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
)


class StageError(Exception):
    """An input failed a required staging gate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(SOURCE), *args], capture_output=True, check=False
    )
    require(result.returncode == 0, "Git source verification failed")
    return result.stdout


def verify_source(manifest: dict) -> int:
    git("cat-file", "-e", f"{SOURCE_REVISION}^{{commit}}")
    require(
        not git("diff", "--name-only", SOURCE_REVISION, "HEAD", "--", *RUNTIME_PATHS),
        "Runtime paths differ from the pinned source revision",
    )
    require(
        not git("status", "--porcelain", "-z", "--", *RUNTIME_PATHS),
        "Runtime paths have local changes",
    )
    tracked = {
        name.decode()
        for name in git(
            "ls-tree", "-r", "--name-only", "-z", SOURCE_REVISION, "--", "vllm"
        ).split(b"\0")
        if name
    }
    payload = manifest["payloads"]["vllm"]
    require(len(tracked) == 2950, "Pinned source file count differs from the audit")
    require(tracked <= payload.keys(), "Tracked source is missing from the payload")
    for name in sorted(tracked):
        path = SOURCE / name
        require(
            path.is_file() and not path.is_symlink(),
            f"Source is not a regular file: {name}",
        )
        require(digest(path) == payload[name], f"Source hash mismatch: {name}")
    return len(tracked)


def read_manifest() -> dict:
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    expected = {
        "source_revision": SOURCE_REVISION,
        "wheel_source_revision": "86d35745c8797c9d2524877fb235084a6a5f5a7d",
        "vllm_version": VLLM_VERSION,
        "flashinfer_version": FLASHINFER_VERSION,
        "source_files_checked": 2950,
        "source_runtime_matches_wheel": True,
        "source_documentation_changes": ["README.md", "README_EN.md"],
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")
    require(set(manifest["payloads"]) == set(WHEELS), "Manifest payload set mismatch")
    for prefix, (_, sha, _, count) in WHEELS.items():
        require(
            manifest.get(f"{prefix}_wheel_sha256") == sha,
            f"Manifest wheel SHA mismatch: {prefix}",
        )
        require(
            len(manifest["payloads"][prefix]) == count,
            f"Manifest payload count mismatch: {prefix}",
        )
    return manifest


def audit_wheel(path: Path, prefix: str, expected_payload: dict[str, str]) -> None:
    _, expected_sha, expected_size, _ = WHEELS[prefix]
    require(path.is_file() and not path.is_symlink(), f"Missing wheel: {path.name}")
    require(path.stat().st_size == expected_size, f"Wheel size mismatch: {path.name}")
    require(digest(path) == expected_sha, f"Wheel SHA mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        for member in members:
            name = member.filename
            parts = name.rstrip("/").split("/")
            require(
                not PurePosixPath(name).is_absolute()
                and "\\" not in name
                and not any(part in ("", ".", "..") for part in parts)
                and not stat.S_ISLNK(member.external_attr >> 16),
                "Unsafe ZIP member",
            )
        files = {member.filename for member in members if not member.is_dir()}
        record_names = [name for name in files if name.endswith(".dist-info/RECORD")]
        require(len(record_names) == 1, "Expected exactly one wheel RECORD")
        record_name = record_names[0]
        records = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
        require(all(len(row) == 3 for row in records), "Malformed wheel RECORD")
        record_paths = [row[0] for row in records]
        require(len(record_paths) == len(set(record_paths)), "Duplicate RECORD rows")
        require(set(record_paths) == files, "RECORD membership differs from ZIP")
        payload = {}
        for name, encoded, size in records:
            if name == record_name:
                require(not encoded and not size, "RECORD must not hash itself")
                continue
            algorithm, separator, value = encoded.partition("=")
            require(
                algorithm == "sha256" and bool(separator), "Unsupported RECORD hash"
            )
            hasher = hashlib.sha256()
            actual_size = 0
            with archive.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
                    actual_size += len(chunk)
            actual = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode()
            require(actual == value, f"RECORD hash mismatch: {name}")
            require(str(actual_size) == size, f"RECORD size mismatch: {name}")
            if name.startswith(prefix + "/"):
                payload[name] = hasher.hexdigest()
        require(payload == expected_payload, f"Payload mapping mismatch: {prefix}")


def download_wheel(directory: Path, prefix: str) -> Path:
    filename, _, expected_size, _ = WHEELS[prefix]
    path = directory / filename
    url = RELEASE_URL + urllib.parse.quote(filename, safe="")
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            path.open("xb") as out,
        ):
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                require(
                    size <= expected_size, f"Download exceeds wheel size: {filename}"
                )
                out.write(chunk)
    except urllib.error.HTTPError as error:
        raise StageError(f"Download HTTP {error.code}: {filename}") from None
    except urllib.error.URLError:
        raise StageError(f"Download failed (URLError): {filename}") from None
    return path


def main() -> None:
    require(__debug__, "Optimized Python is not permitted for verification")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wheel-dir", type=Path, help="Use local wheels; never download"
    )
    args = parser.parse_args()
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "Output already exists")
    inputs = {
        "Dockerfile": SOURCE / "docker/Dockerfile.sm89-sm120-runtime",
        "LICENSE": SOURCE / "LICENSE",
        **{name: ASSETS / name for name in ASSET_NAMES},
    }
    for name, path in inputs.items():
        require(
            path.is_file() and not path.is_symlink(), f"Missing regular asset: {name}"
        )
    manifest = read_manifest()
    source_count = verify_source(manifest)
    if args.wheel_dir is not None:
        require(args.wheel_dir.is_dir(), "Local wheel directory does not exist")
    with tempfile.TemporaryDirectory(prefix="vision11-wheels-") as temporary:
        wheels = {}
        for prefix, (filename, _, _, _) in WHEELS.items():
            path = (
                args.wheel_dir / filename
                if args.wheel_dir is not None
                else download_wheel(Path(temporary), prefix)
            )
            audit_wheel(path, prefix, manifest["payloads"][prefix])
            wheels[filename] = path
        output.mkdir(parents=True, exist_ok=False)
        (output / "wheels").mkdir()
        for name, path in inputs.items():
            shutil.copyfile(path, output / name)
        for name, path in wheels.items():
            shutil.copyfile(path, output / "wheels" / name)
        for filename, sha, size, _ in WHEELS.values():
            staged = output / "wheels" / filename
            require(
                staged.stat().st_size == size and digest(staged) == sha,
                f"Staged wheel integrity mismatch: {filename}",
            )
        actual = {
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
        }
        expected = set(inputs) | {f"wheels/{name}" for name in wheels}
        require(actual == expected, "Staged context allowlist mismatch")
    print(
        json.dumps(
            {
                "context": str(output),
                "source_revision": SOURCE_REVISION,
                "source_files_checked": source_count,
                "payload_counts": {key: value[3] for key, value in WHEELS.items()},
                "context_files": len(expected),
                "wheel_sha256": {key: value[1] for key, value in WHEELS.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except StageError as error:
        print(f"Staging failed: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # Do not expose redirect URLs or credentials from network exception text.
        print(f"Staging failed ({type(error).__name__})", file=sys.stderr)
        sys.exit(1)
