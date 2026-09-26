"""Copy the current candidate into a clean, isolated Git checkout for verification."""

import shutil
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / ".artifacts" / "release-checkout"
    if destination.exists():
        raise SystemExit(f"An isolated checkout already exists: {destination}")
    destination.mkdir(parents=True)
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=root
        )
        .decode("utf-8")
        .split("\0")
    )
    count = 0
    for name in sorted(set(tracked)):
        if not name:
            continue
        source = (root / name).resolve()
        target = (destination / name).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(destination):
            raise SystemExit(f"Refusing a path outside the checkout: {name}")
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            count += 1
    subprocess.run(["git", "init", "--quiet"], cwd=destination, check=True)
    subprocess.run(["git", "add", "--all"], cwd=destination, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PSPSO verification",
            "-c",
            "user.email=verification@localhost",
            "commit",
            "--quiet",
            "--message",
            "Isolated PSPSO 1.0 verification snapshot",
        ],
        cwd=destination,
        check=True,
    )
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=destination)
    print(f"Clean candidate checkout: {destination} ({count} files)")


if __name__ == "__main__":
    main()
