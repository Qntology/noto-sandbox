#!/usr/bin/env bash
# ============================================================
#  Sequential test run for macOS, Linux and the CI jobs: the bash twin of test_runner.bat, with
#  the same step numbers, commands and logs. A step added to one belongs in the other
#  (the cross-verify workflow compares the two lists).
#    test_metal.sh             Apple Metal
#    _cargo.sh <backend> test  any backend
#  _cargo.sh sets BACKEND, WSFEAT, PKGFEAT (cargo feature flags) and NOTO_DEVICE, then calls
#  this file. Do not run it directly.
#
#  The game logic, the physics and the stand-in models run on the CPU for every backend, so
#  steps 02-06 must give the same results everywhere. The backend changes the models only
#  (SigLIP2, TIPS, SAM, TTM); step 07 compares them with the CPU. No step measures speed.
#
#  Every step writes its own log:  test_logs/<timestamp>_<backend>/NN_name.log
#  Failures do not stop the run, except a failed step 01: nothing can run without the build.
#  At the end:
#    summary.log     exit code and duration of every step, test results, check counts,
#                    the cross-area deviation report ([UX] [UX-SUM])
#    highlights.log  every check line ([GPU] [NN] [PRESET] [DYN] and the benchmarks'
#                    [AX] [FX] [CAM] [VX] [CX] [TX] [KX] [MX] [UX] with -SUM / -HELD / -WHY / -CELL)
#
#  Options (set before running):
#    ONLY=03,07         run only these step numbers (step 01 build always runs)
#    CARGO_BUILD_JOBS=2 parallel rustc/link jobs
#    REST=1             then one more step for every integration test this list does not name
#                       (CI: the cpu jobs; test_runner.bat has no such steps)
# ============================================================
set -u
cd "$(dirname "$0")"
# The logs hold Korean lines: the text tools below work on bytes, so a cut line never stops them.
export LC_ALL=C
BACKEND="${BACKEND:-cpu}"
WSFEAT="${WSFEAT:-}"
PKGFEAT="${PKGFEAT:-}"
NOTO_DEVICE="${NOTO_DEVICE:-}"
export RUST_BACKTRACE=1
# Two parallel rustc/link jobs by default, as in test_runner.bat: several links map the 160 MB lancedb rlib.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
CT="cargo test $WSFEAT"
# One package with models (noto-vision, noto-runtime): only its own dependencies take features.
PT="cargo test $PKGFEAT"
# Physics and data crates do not depend on the models: no GPU features.
CPT="cargo test"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$PWD/test_logs/${TS}_$BACKEND"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.log"
FAILED=0
{
    echo "========================================"
    echo " noto-sandbox test run $TS  backend=$BACKEND  features=$WSFEAT  NOTO_DEVICE=$NOTO_DEVICE"
    echo " options: ONLY=${ONLY:-} CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS REST=${REST:-}"
    echo "========================================"
} > "$SUMMARY"
echo "Logs: $LOGDIR"

# CI only: the runner logs cannot be read from outside the job, annotations can.
annot() {
    if [ -n "${GITHUB_ACTIONS:-}" ]; then bash .github/scripts/annot.sh "$@"; fi
}

# run <NN> <name> <command...>
run() {
    local id="$1" step="$1_$2"
    shift 2
    local log="$LOGDIR/$step.log"
    if [ -n "${ONLY:-}" ] && [ "$id" != 01 ]; then
        case ",$ONLY," in
            *",$id,"*) ;;
            *)
                echo "[$step] skipped - not in ONLY=$ONLY"
                echo "$step: skipped - ONLY" >> "$SUMMARY"
                return 0
                ;;
        esac
    fi
    echo "[$step] $*"
    {
        echo "[$(date +%T)] start $step  backend=$BACKEND"
        echo "\$ $*"
        echo
    } > "$log"
    local t0=$SECONDS rc
    "$@" >> "$log" 2>&1
    rc=$?
    local dur=$((SECONDS - t0))
    {
        echo
        echo "[$(date +%T)] end $step exit=$rc ($dur s)"
    } >> "$log"
    {
        echo "$step: exit=$rc  $dur s"
        grep -a '^test result:' "$log"
        grep -a -E 'FAILED|panicked at' "$log"
    } >> "$SUMMARY"
    echo "  -> exit=$rc  $dur s  ($log)"
    if [ "$rc" != 0 ]; then
        FAILED=$((FAILED + 1))
        annot error "$step" "$log"
    fi
    return "$rc"
}

# bench <NN> <test>: runs crates/<crate>/tests/<test>.rs of the crate that has it (noto-core and world-sim
# without model features, noto-runtime with them); a test that is not in the tree is reported as skipped.
bench() {
    local id="$1" t="$2"
    if [ -f "crates/noto-core/tests/$t.rs" ]; then
        run "$id" "$t" $CPT -p noto-core --test "$t" --no-fail-fast -- --nocapture --test-threads=1
    elif [ -f "crates/world-sim/tests/$t.rs" ]; then
        run "$id" "$t" $CPT -p world-sim --test "$t" --no-fail-fast -- --nocapture --test-threads=1
    elif [ -f "crates/noto-runtime/tests/$t.rs" ]; then
        run "$id" "$t" $PT -p noto-runtime --test "$t" --no-fail-fast -- --nocapture --test-threads=1
    else
        echo "[${id}_$t] skipped - no test file"
        echo "${id}_$t: skipped - not in the tree" >> "$SUMMARY"
    fi
}

# Collects the check lines into highlights.log and appends the check summary to summary.log (test_summary.ps1).
summarize() {
    local tag='^\[(GPU|PRESET|DYN|SRC|(NN|AX|FX|CAM|VX|CX|TX|KX|MX|UX)(-SUM|-HELD|-WHY|-CELL)?)\]'
    local logs=("$LOGDIR"/[0-9][0-9]_*.log) f lines
    : > "$LOGDIR/highlights.log"
    for f in "${logs[@]}"; do
        # With --nocapture the first line a test prints follows cargo's 'test <name> ... '.
        lines="$(sed -E 's/^test [^ ]+ \.\.\. //' "$f" | grep -a -E "$tag")"
        if [ -n "$lines" ]; then
            {
                echo "===== $(basename "$f") ($(printf '%s\n' "$lines" | wc -l | tr -d ' ') lines) ====="
                printf '%s\n\n' "$lines"
            } >> "$LOGDIR/highlights.log"
        fi
    done
    {
        echo
        echo "======== check summary (full lines in highlights.log) ========"
        echo "backend: $BACKEND (physics and game logic run on the CPU for every backend; steps 02-06 must match across backends)"
        lines="$(grep -a -E '^\[(NN-SUM|NN-HELD)\]' "$LOGDIR/highlights.log")"
        if [ -n "$lines" ]; then
            echo "N:N composite scenarios:"
            printf '%s\n' "$lines" | sed 's/^/  /'
        fi
        lines="$(grep -a -o -E '^\[(AX|FX|CAM|VX|CX|TX|KX|MX)-SUM\]' "$LOGDIR/highlights.log" | sort | uniq -c | awk '{gsub(/[][]|-SUM/, "", $2); print "  " $2 ": " $1}')"
        if [ -n "$lines" ]; then
            echo "reference benchmarks (scenario lines per area; cells in highlights.log):"
            printf '%s\n' "$lines"
        fi
        lines="$(grep -a -E '^\[UX(-SUM)?\]' "$LOGDIR/highlights.log")"
        if [ -n "$lines" ]; then
            echo "cross-area deviation (median |z| per area; the farthest area is the next work item):"
            printf '%s\n' "$lines" | sed 's/^/  /'
        fi
        lines="$(grep -a -E '^\[GPU\]' "$LOGDIR/highlights.log")"
        if [ -n "$lines" ]; then
            echo "GPU parity: $(printf '%s\n' "$lines" | wc -l | tr -d ' ') lines"
            printf '%s\n' "$lines" | head -30 | sed 's/^/  /'
        fi
        echo "compiler warnings: $(cat "${logs[@]}" | grep -a '^warning: ' | grep -a -c -v 'generated [0-9]* warning')"
        lines="$(grep -a -H -E '^test .+ \.\.\. FAILED' "${logs[@]}" | sed -E 's|^.*/([0-9][0-9]_[a-z_]+)\.log:test (.+) \.\.\. FAILED.*$|  \1: \2|')"
        if [ -n "$lines" ]; then
            echo "failed tests:"
            printf '%s\n' "$lines"
        fi
        lines="$(grep -a -h -E 'panicked at ' "${logs[@]}" | sed -E 's/^.*panicked at /  /' | head -20)"
        if [ -n "$lines" ]; then
            echo "panics:"
            printf '%s\n' "$lines"
        fi
    } >> "$SUMMARY"
}

finish() {
    {
        echo "========================================"
        if [ "$FAILED" = 0 ]; then echo "ALL STEPS PASSED"; else echo "STEPS FAILED: $FAILED"; fi
    } >> "$SUMMARY"
    summarize
    echo
    cat "$SUMMARY"
    echo
    echo "Logs saved in: $LOGDIR"
    annot lines "steps $BACKEND" "$SUMMARY" '^([0-9][0-9]_|ALL STEPS|STEPS FAILED)'
    annot lines "GPU parity $BACKEND" "$LOGDIR/highlights.log" '^\[GPU\]'
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            echo '```'
            cat "$SUMMARY"
            echo '```'
        } >> "$GITHUB_STEP_SUMMARY"
    fi
    exit "$FAILED"
}

# ---- 00: environment snapshot ----
{
    echo "--- date ---"
    date
    echo "--- git ---"
    git rev-parse HEAD
    git status --short
    echo "--- rustc / cargo ---"
    rustc -Vv
    cargo -V
    echo "--- os ---"
    uname -a
    echo "--- backend $BACKEND ---"
    case "$BACKEND" in
        cuda)
            nvcc --version
            nvidia-smi
            ;;
        rocm)
            if command -v hipconfig > /dev/null; then hipconfig --version; fi
            ;;
        vulkan)
            if command -v vulkaninfo > /dev/null; then vulkaninfo --summary; else echo "vulkaninfo not installed"; fi
            ;;
        metal)
            if command -v system_profiler > /dev/null; then system_profiler SPDisplaysDataType; fi
            ;;
    esac
    echo "--- env ---"
    echo "BACKEND=$BACKEND"
    echo "WSFEAT=$WSFEAT"
    echo "NOTO_DEVICE=$NOTO_DEVICE"
    echo "CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-}"
    echo "ROCM_PATH=${ROCM_PATH:-}"
    echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-}"
    echo "CANDLE_VULKAN_ALLOW_CPU=${CANDLE_VULKAN_ALLOW_CPU:-}"
} > "$LOGDIR/00_env.log" 2>&1

# ---- 01: build every test binary first (separates compile errors from test failures) ----
run 01 build_tests $CT --workspace --no-run || finish
# ---- 02: unit tests of every crate (physics, presets, balance, SDS, models with random weights) ----
run 02 unit $CT --workspace --lib --no-fail-fast -- --nocapture --test-threads=1
# ---- 03: joint dynamics and presets: every preset change must reach the physics ----
run 03 presets_physics $CPT -p noto-core --test presets_physics --no-fail-fast -- --nocapture --test-threads=1
# ---- 04: N:N composite scenarios against the research references (no tuning to them) ----
run 04 nn_scenarios $CPT -p noto-core --test nn_scenarios --no-fail-fast -- --nocapture --test-threads=1
# ---- 05: world scenarios (quests, game logic) ----
run 05 scenarios $PT -p noto-runtime --test scenarios --no-fail-fast -- --nocapture --test-threads=1
# ---- 06: integrated test (game pass, ML perception pass, ML decision pass) ----
run 06 integrated $PT -p noto-runtime --test integrated --no-fail-fast -- --nocapture --test-threads=1
# ---- 07: the models on the GPU backend compared with the CPU (skipped on the cpu backend) ----
if [ "$BACKEND" = cpu ]; then
    echo "07_gpu_parity: skipped - cpu backend" >> "$SUMMARY"
else
    run 07 gpu_parity $PT -p noto-vision --test gpu_backend --no-fail-fast -- --nocapture --test-threads=1
fi
# ---- 08: Lance tables ----
run 08 lance $CPT -p noto-data --no-fail-fast -- --test-threads=1
# ---- 09-16: reference benchmarks, one area each (docs/research/*.md; no tuning to them) ----
bench 09 action_bench
bench 10 effects_bench
bench 11 camera_bench
bench 12 vision_bench
bench 13 collision_bench
bench 14 tactile_bench
bench 15 coop_bench
bench 16 morph_bench
# ---- 17: no engine file names a benchmark source; decisions never read a reference ----
bench 17 source_split
# ---- 18-19: cross-area deviation ("uncanniness") report: core areas, then with camera and vision ----
bench 18 uncanny_report
bench 19 uncanny_runtime
# ---- 20-21: bodies stop at what they can hit; a sentence changes a unit's body and a lost load is fetched again ----
bench 20 collisions
bench 21 rig_mcp

# ---- REST=1: the integration tests the steps above do not name (noto-data ran whole in step 08) ----
if [ "${REST:-}" = 1 ]; then
    id=22
    for f in crates/*/tests/*.rs; do
        t="$(basename "$f" .rs)"
        crate="$(basename "$(dirname "$(dirname "$f")")")"
        if [ "$crate" = noto-data ] || grep -q -E "^ *(bench [0-9]+|run [0-9]+ [a-z_]+ .*--test) $t( |\$)" "$0"; then continue; fi
        run "$id" "$t" $CPT -p "$crate" --test "$t" --no-fail-fast -- --nocapture --test-threads=1
        id=$((id + 1))
    done
fi

finish
