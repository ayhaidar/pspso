"""Enforce the release's overall and lifecycle coverage gates."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = []
    overall = report["totals"]["percent_covered"]
    print(f"Overall coverage: {overall:.2f}% (required: 80%)")
    if overall < 80:
        failures.append("overall")
    for module in ("manager", "tracking", "worker"):
        filename = f"pspso/dashboard/{module}.py"
        match = next(
            (
                value
                for name, value in report["files"].items()
                if name.replace("\\", "/").endswith(filename)
            ),
            None,
        )
        coverage = match["summary"]["percent_covered"] if match else 0
        print(f"{module} coverage: {coverage:.2f}% (required: 90%)")
        if coverage < 90:
            failures.append(module)
    if failures:
        raise SystemExit("Coverage gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
