# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a task command while retaining redacted output and its real exit code."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import regex as re


def redact(text: str) -> str:
    def clean_url(match: re.Match[str]) -> str:
        value = match.group()
        try:
            url = urlsplit(value)
            host = url.netloc.rsplit("@", 1)[-1]
            query = "REDACTED" if url.query else ""
            return urlunsplit((url.scheme, host, url.path, query, url.fragment))
        except ValueError:
            return "[URL redacted]"

    text = re.sub(r"https?://[^\s\"<>]+", clean_url, text)
    return re.sub(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)\b", "[REDACTED]", text
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("A command is required")
    with args.log.open("x") as log:
        heading = "$ " + redact(shlex.join(command)) + "\n"
        log.write(heading)
        log.flush()
        print(heading, end="", flush=True)
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                safe = redact(line)
                log.write(safe)
                log.flush()
                print(safe, end="", flush=True)
            code = process.wait()
        footer = f"\nExit code: {code}\n"
        log.write(footer)
        print(footer, end="", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
