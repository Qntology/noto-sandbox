#!/usr/bin/env bash
# env-report.sh <label> — record the runner environment as annotations (and in the step summary).
# Runner logs cannot be read from outside a job, annotations can, so whatever explains a build failure
# (compilers, SDKs, PATH, build inputs) is written here, up front.
# GitHub shows at most 10 notices per step, so the facts are packed into four annotations.
# Never fails the job. Works in Git Bash (Windows) and bash (Linux, macOS).
set -u
label="${1:-env}"
os="$(uname -s 2>/dev/null || echo unknown)"
case "$os" in MINGW* | MSYS* | CYGWIN*) win=1 ;; *) win=0 ;; esac

have() { command -v "$1" > /dev/null 2>&1; }
# line NAME CMD... : first 3 lines of the command's output (stdout+stderr), CR stripped
line() {
    local n="$1"
    shift
    local v
    v="$("$@" 2>&1 | head -3 | tr -d '\r' | tr '\n' '|')" || true
    printf '%s: %s\n' "$n" "${v:-<none>}"
}
present() {
    local n="$1" v="${2:-}"
    if [ -n "$v" ]; then printf '%s: %s\n' "$n" "${v:0:160}"; else printf '%s: <unset>\n' "$n"; fi
}

emit() {
    local title="$1" body="$2" msg
    msg="${body//%/%25}"
    msg="${msg//$'\r'/}"
    msg="${msg//$'\n'/%0A}"
    echo "::notice title=env ${label} / ${title}::${msg}"
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            echo "#### env ${label} / ${title}"
            echo '```'
            printf '%s\n' "$body"
            echo '```'
        } >> "$GITHUB_STEP_SUMMARY"
    fi
}

# ── 1. runner ──
g1="$(
    echo "uname: $(uname -a 2>/dev/null | cut -c1-200)"
    if [ "$win" = 1 ]; then
        line "windows" powershell -NoProfile -Command '(Get-CimInstance Win32_OperatingSystem | % { $_.Caption + " " + $_.Version + " " + $_.OSArchitecture })'
        line "memory-GB" powershell -NoProfile -Command '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)'
        echo "cores: ${NUMBER_OF_PROCESSORS:-?}"
        line "disk-free" powershell -NoProfile -Command '(Get-PSDrive -Name (Get-Location).Drive.Name | % { [math]::Round($_.Free/1GB,1).ToString() + " GB free" })'
    elif [ "$os" = Darwin ]; then
        line "os" sw_vers
        line "cpu" sysctl -n machdep.cpu.brand_string
        echo "cores: $(sysctl -n hw.ncpu 2>/dev/null)"
        echo "memory-GB: $(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824))"
        line "disk-free" bash -c "df -h . | tail -1"
        line "xcode" xcodebuild -version
        line "metal-gpu" bash -c "system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Metal' | head -2"
    else
        line "os" bash -c '. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME"'
        echo "cores: $(nproc 2>/dev/null)"
        line "memory-MB" bash -c "free -m | awk '/Mem:/{print \$2\" total, \"\$7\" available\"}'"
        line "disk-free" bash -c "df -h . | tail -1"
    fi
    echo "image: ${ImageOS:-?} ${ImageVersion:-?}"
    echo "cwd: $(pwd)"
)"
emit "runner" "$g1"

# ── 2. toolchain ──
g2="$(
    line "rustc" rustc -vV
    line "cargo" cargo -V
    line "git" git --version
    line "protoc" protoc --version
    present "PROTOC" "${PROTOC:-}"
    if [ "$win" = 1 ]; then
        line "cl (PATH)" where.exe cl
        line "link (PATH)" where.exe link
        line "autocrlf" git config --get core.autocrlf
    else
        line "cc" cc --version
        line "pkg-config" pkg-config --version
        line "webkit2gtk-4.1" pkg-config --modversion webkit2gtk-4.1
        line "gtk+-3.0" pkg-config --modversion gtk+-3.0
    fi
)"
emit "toolchain" "$g2"

# ── 3. GPU SDKs ──
g3="$(
    if have nvcc; then line "nvcc" bash -c 'nvcc --version | tail -2'; else echo "nvcc: <not in PATH>"; fi
    present "CUDA_PATH" "${CUDA_PATH:-}"
    present "NVCC_CCBIN" "${NVCC_CCBIN:-}"
    present "NVCC_PREPEND_FLAGS" "${NVCC_PREPEND_FLAGS:-}"
    present "CUDA_COMPUTE_CAP" "${CUDA_COMPUTE_CAP:-}"
    present "CANDLE_ROCM_SKIP_KERNEL_BUILD" "${CANDLE_ROCM_SKIP_KERNEL_BUILD:-}"
    present "VK_ICD_FILENAMES" "${VK_ICD_FILENAMES:-}"
    present "CANDLE_VULKAN_ALLOW_CPU" "${CANDLE_VULKAN_ALLOW_CPU:-}"
    if have vulkaninfo; then
        line "vulkaninfo" bash -c 'vulkaninfo --summary 2>&1 | grep -E "deviceName|apiVersion|driverName" | head -4'
    else
        echo "vulkaninfo: <not installed>"
    fi
)"
emit "gpu-sdk" "$g3"

# ── 4. build inputs + PATH names ──
g4="$(
    root="${GITHUB_WORKSPACE:-.}"
    if [ -f "$root/ui/index.html" ]; then echo "ui/index.html: present"; else echo "ui/index.html: MISSING"; fi
    echo "candle rev: $(grep -m1 -oE 'rev = "[0-9a-f]+"' "$root/Cargo.toml")"
    present "RUSTFLAGS" "${RUSTFLAGS:-}"
    present "CARGO_BUILD_JOBS" "${CARGO_BUILD_JOBS:-}"
    echo "PATH entries (compiler / SDK related): $(echo "$PATH" | tr ':;' '\n' | grep -iE 'cuda|msvc|visual studio|vc/tools|windows kits|rocm|hip|vulkan|cargo|rustup' | head -12 | tr '\n' '|')"
)"
emit "build-inputs" "$g4"
exit 0
