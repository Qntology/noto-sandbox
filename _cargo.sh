#!/usr/bin/env bash
# =====================================================================
#  macOS / Linux twin of _cargo.bat: one entry point behind the *_metal.sh scripts and the CI jobs.
#    _cargo.sh <metal|cuda|rocm|vulkan|cpu> <dev|build|check|test|serve> [extra cargo arguments]
#  Sets the cargo feature flags and NOTO_DEVICE (which device the app and the tests open;
#  override it before calling). The compilers and SDKs come from the environment.
# =====================================================================
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

usage() {
    echo "usage: _cargo.sh <metal|cuda|rocm|vulkan|cpu> <dev|build|check|test|serve> [cargo arguments]" >&2
    exit 2
}
[ $# -ge 2 ] || usage
BACKEND="$1"
ACTION="$2"
shift 2

# Feature flags: the app forwards them to noto-runtime, noto-vision and noto-ml.
FEAT=""
WSFEAT=""
PKGFEAT=""
case "$BACKEND" in
    cpu) NOTO_DEVICE=cpu ;;
    metal | cuda | rocm | vulkan)
        FEAT="--features $BACKEND"
        WSFEAT="--features noto-ml/$BACKEND,noto-vision/$BACKEND,noto-runtime/$BACKEND,noto-sim-app/$BACKEND"
        # For one package (-p): every member with models depends on noto-ml and noto-vision.
        PKGFEAT="--features noto-ml/$BACKEND,noto-vision/$BACKEND"
        ;;
    *) usage ;;
esac
if [ "$BACKEND" = metal ] && [ "$(uname -s)" != Darwin ]; then
    echo "[Metal] Metal builds need macOS." >&2
    exit 1
fi
export BACKEND WSFEAT PKGFEAT
export NOTO_DEVICE="${NOTO_DEVICE:-$BACKEND}"
export RUST_BACKTRACE=1

# LanceDB's build needs protoc. Use the one on PATH, else a copy under tools/ (same version as _env_msvc.bat).
if [ -z "${PROTOC:-}" ] && ! command -v protoc > /dev/null 2>&1; then
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) PACKAGE=osx-aarch_64 ;;
        Darwin-x86_64) PACKAGE=osx-x86_64 ;;
        Linux-aarch64) PACKAGE=linux-aarch_64 ;;
        Linux-x86_64) PACKAGE=linux-x86_64 ;;
        *) PACKAGE=win64 ;;
    esac
    PROTOC="$ROOT/tools/protoc/bin/protoc"
    if [ "$PACKAGE" = win64 ]; then PROTOC="$PROTOC.exe"; fi
    if [ ! -f "$PROTOC" ]; then
        echo "[SETUP] protoc not found - downloading protoc 29.3 for the LanceDB build..."
        mkdir -p tools
        curl -fsSL --retry 3 -o tools/protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v29.3/protoc-29.3-$PACKAGE.zip"
        unzip -q -o tools/protoc.zip -d tools/protoc
    fi
    # Git Bash on Windows: the build scripts are native programs and need a native path.
    if command -v cygpath > /dev/null 2>&1; then PROTOC="$(cygpath -w "$PROTOC")"; fi
    export PROTOC
fi

# AI edits run the SolidCraft engine out of sight; its web package is not in git. Without it the runtime exports designs itself.
case "$ACTION" in
    dev | build | serve)
        if [ ! -f ui/vendor/solidcraft/index.html ]; then bash ./fetch_solidcraft.sh; fi
        ;;
esac

TAURI_CLI="cargo tauri"
if [ -x node_modules/.bin/tauri ]; then TAURI_CLI="$ROOT/node_modules/.bin/tauri"; fi

case "$ACTION" in
    dev)
        cd src-tauri
        echo "[DEV] Tauri app, backend $BACKEND (NOTO_DEVICE=$NOTO_DEVICE)"
        exec $TAURI_CLI dev -- $FEAT "$@"
        ;;
    build)
        cd src-tauri
        echo "[BUILD] Tauri app bundle, backend $BACKEND"
        exec $TAURI_CLI build -- $FEAT "$@"
        ;;
    check)
        echo "[CHECK] cargo check --workspace --all-targets $WSFEAT"
        exec cargo check --workspace --all-targets $WSFEAT "$@"
        ;;
    serve)
        echo "[SERVE] development server http://127.0.0.1:8787/ , backend $BACKEND (NOTO_DEVICE=$NOTO_DEVICE)"
        exec cargo run --release -p noto-runtime --example serve $FEAT "$@"
        ;;
    test)
        if [ $# -gt 0 ]; then
            echo "[TEST] cargo test $PKGFEAT $*"
            exec cargo test $PKGFEAT "$@"
        fi
        exec bash ./test_runner.sh
        ;;
    *) usage ;;
esac
