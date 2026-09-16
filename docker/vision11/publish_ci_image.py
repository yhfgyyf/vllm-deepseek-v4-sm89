# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish this verified tag only when absent, then verify remote identity."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pybase64 as base64

REPOSITORY = "yhfgyyf/vllm-deepseek-v4-sm89"
IMAGE = f"ghcr.io/{REPOSITORY}:0.28.1rc1-vision11-sm89-sm120-cu130"
SCRIPTS = Path(__file__).resolve().parent
TAG = IMAGE.rsplit(":", 1)[1]
ACCEPT = ",".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        require(urllib.parse.urlsplit(new_url).scheme == "https", "Unsafe redirect")
        redirected = super().redirect_request(
            request, response, code, message, headers, new_url
        )
        if (
            urllib.parse.urlsplit(request.full_url).netloc
            != urllib.parse.urlsplit(new_url).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def registry_token(opener):
    require(os.environ["GITHUB_ACTOR"] == "yhfgyyf", "Unexpected publishing actor")
    credentials = base64.b64encode(
        ("yhfgyyf:" + os.environ["GHCR_TOKEN"]).encode()
    ).decode()
    query = urllib.parse.urlencode(
        {
            "service": "ghcr.io",
            "scope": f"repository:{REPOSITORY}:pull,push",
        }
    )
    request = urllib.request.Request(
        "https://ghcr.io/token?" + query,
        headers={"Authorization": "Basic " + credentials},
    )
    with opener.open(request, timeout=60) as response:
        return json.load(response)["token"]


def fetch(opener, token, path):
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{REPOSITORY}/{path}",
        headers={"Authorization": "Bearer " + token, "Accept": ACCEPT},
    )
    with opener.open(request, timeout=120) as response:
        data = response.read(8 * 1024 * 1024 + 1)
        require(len(data) <= 8 * 1024 * 1024, "Oversized registry metadata")
        return data, response.headers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence
    receipt = json.loads((evidence / "ci-validation.json").read_text())
    require(receipt["passed"] and receipt["tag"] == IMAGE, "Validation required")
    require(receipt.get("media_decode_passed") is True, "Media validation required")
    require(
        receipt.get("cold_sm89_sampling_compile_passed") is True,
        "Cold SM89 compilation validation required",
    )
    local = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE]))[
        0
    ]
    require(local["Id"] == receipt["image_id"], "Image changed after validation")
    opener = urllib.request.build_opener(SafeRedirect())
    token = registry_token(opener)
    try:
        fetch(opener, token, f"manifests/{TAG}")
    except urllib.error.HTTPError as error:
        errors = json.loads(error.read(8192)).get("errors", [])
        require(
            error.code == 404
            and errors
            and all(
                item.get("code") in ("MANIFEST_UNKNOWN", "NAME_UNKNOWN")
                for item in errors
            ),
            "Target absence was not established",
        )
    else:
        raise RuntimeError("Refusing to overwrite an existing image tag")
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPTS / "run_logged.py"),
            str(evidence / "push.log"),
            "--",
            "docker",
            "push",
            IMAGE,
        ],
        check=True,
    )
    token = registry_token(opener)
    raw, headers = fetch(opener, token, f"manifests/{TAG}")
    manifest_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    require(headers["Docker-Content-Digest"] == manifest_digest, "Manifest mismatch")
    manifest = json.loads(raw)
    require(manifest["config"]["digest"] == local["Id"], "Remote config mismatch")
    config_raw, _ = fetch(opener, token, f"blobs/{local['Id']}")
    require(
        "sha256:" + hashlib.sha256(config_raw).hexdigest() == local["Id"],
        "Remote config bytes mismatch",
    )
    config = json.loads(config_raw)
    require(config["rootfs"]["diff_ids"] == local["RootFS"]["Layers"], "Layer drift")
    require(len(manifest["layers"]) == len(local["RootFS"]["Layers"]), "Layer count")
    require(
        all(layer["size"] < 10_000_000_000 for layer in manifest["layers"]),
        "Oversized GHCR layer",
    )
    (evidence / "remote-manifest.json").write_bytes(raw)
    (evidence / "remote-config.json").write_bytes(config_raw)
    result = {
        "passed": True,
        "tag": IMAGE,
        "manifest_digest": manifest_digest,
        "config_digest": local["Id"],
        "source_revision": receipt["source_revision"],
        "public_visibility_verified": False,
        "anonymous_access_verified": False,
    }
    (evidence / "remote-publication.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        require(__debug__, "Optimized Python is not permitted for publication")
        main()
    except RuntimeError as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        sys.exit(1)
    except urllib.error.HTTPError as error:
        print(json.dumps({"passed": False, "http_status": error.code}))
        sys.exit(1)
    except Exception as error:
        # Registry redirects and authentication values must never enter logs.
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        sys.exit(1)
