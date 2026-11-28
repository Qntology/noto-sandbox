#!/usr/bin/env python3
"""Verify Tier 2-C / Track C-2 dependency pins and absence of deprecated/SAM crates in Qntology/noto-sandbox."""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

CANDLE_PIN_REV = "f73415d4d149d7949e8a6a94911af670e4e10f7f"
PD_LIB_PIN_REV_2 = "b024dbb5e12bdb4fc1fcf83d0d62e5a645baccb5"
LV_LIB_PIN_REV_2 = "5ea66a0f8727d8560bd16991cccff7f1e6a00132"

FORBIDDEN_CRATE_DIRS = [
    "crates/noto-cpu-kernels",
    "crates/noto-ml",
    "crates/noto-sds",
    "crates/noto-ts",
    "crates/noto-vision",
]

FORBIDDEN_PATTERN = re.compile(
    r"mobile_sam|NOTO_SAM|sammie-sam3|sammie-accel|SamModel|SamTracker"
)


def parse_lock_packages(text: str) -> list[dict[str, str]]:
    pkgs: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line == "[[package]]":
            if cur:
                pkgs.append(cur)
            cur = {}
            continue
        if cur is not None and "=" in line and not line.startswith("#"):
            k, v = [p.strip() for p in line.split("=", 1)]
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            cur[k] = v
    if cur:
        pkgs.append(cur)
    return pkgs


def main() -> int:
    errors: list[str] = []

    for rel in FORBIDDEN_CRATE_DIRS:
        if (ROOT / rel).exists():
            errors.append(f"forbidden deprecated crate directory still exists: {rel}")

    cargo_toml = ROOT / "Cargo.toml"
    toml_text = cargo_toml.read_text(encoding="utf-8")
    for line_no, line in enumerate(toml_text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or "git =" not in s:
            continue
        if "candle-rocm-vulkan" in s and CANDLE_PIN_REV not in s:
            errors.append(f"Cargo.toml:{line_no}: expected {CANDLE_PIN_REV} in `{s}`")
        if "Qntology/premium-discount" in s and PD_LIB_PIN_REV_2 not in s:
            errors.append(f"Cargo.toml:{line_no}: expected {PD_LIB_PIN_REV_2} in `{s}`")
        if "Qntology/louis-vision" in s and LV_LIB_PIN_REV_2 not in s:
            errors.append(f"Cargo.toml:{line_no}: expected {LV_LIB_PIN_REV_2} in `{s}`")

    cargo_lock = ROOT / "Cargo.lock"
    if not cargo_lock.is_file():
        errors.append("Cargo.lock not found")
    else:
        pkgs = parse_lock_packages(cargo_lock.read_text(encoding="utf-8"))
        for pkg in pkgs:
            name = pkg.get("name", "")
            src = pkg.get("source", "")
            if name in {"candle-core", "candle-nn", "candle-rocm", "candle-flash-attn"}:
                if CANDLE_PIN_REV not in src:
                    errors.append(f"Cargo.lock: {name} source `{src}` != {CANDLE_PIN_REV}")
            if name.startswith("premium-discount-"):
                if PD_LIB_PIN_REV_2 not in src:
                    errors.append(f"Cargo.lock: {name} source `{src}` != {PD_LIB_PIN_REV_2}")
            if name.startswith("lv-"):
                if LV_LIB_PIN_REV_2 not in src:
                    errors.append(f"Cargo.lock: {name} source `{src}` != {LV_LIB_PIN_REV_2}")

    scan_roots = [cargo_toml, ROOT / "crates", ROOT / "src-tauri"]
    for item in scan_roots:
        files = [item] if item.is_file() else [
            p for p in item.rglob("*")
            if p.is_file()
            and "target" not in p.parts
            and ".git" not in p.parts
            and p.suffix in {".rs", ".toml", ".py", ".json"}
        ]
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(text.splitlines(), 1):
                if FORBIDDEN_PATTERN.search(line):
                    rel = path.relative_to(ROOT).as_posix()
                    errors.append(f"{rel}:{idx}: forbidden match `{line.strip()}`")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(
        f"verify-pins OK: candle={CANDLE_PIN_REV[:8]} pd={PD_LIB_PIN_REV_2[:8]} lv={LV_LIB_PIN_REV_2[:8]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
