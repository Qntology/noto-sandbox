#!/usr/bin/env bash
# Annotations for the CI jobs: the runner logs cannot be read from outside a job, annotations can.
#   annot.sh error <title> <log>           the first compiler / build-script / test errors + the last log lines
#   annot.sh lines <title> <log> <regex>   the matching lines as one notice (proof of what really ran)
# An annotation is cut around 4 KB; line breaks and % are escaped. Never fails the step.
set -u
kind="$1"
title="$2"
log="$3"

escape() { tr -d '\r' | sed 's/%/%25/g' | awk '{printf "%s%%0A", $0}' | cut -c1-3800; }

case "$kind" in
    error)
        msg="$(
            {
                grep -a -oE '\((exit (code|status)|signal)[^)]*\)' "$log" | head -3 | sed 's/^/ended with /'
                grep -a -A3 -E '^(error|warning: build failed)|panicked at|failed to run custom build|--- stderr|--- stdout|nvcc|fatal|^failures:|FAILED|exit code|STATUS_|not found' "$log" | cut -c1-300 | head -30
                echo '--- last lines ---'
                tail -n 20 "$log" | cut -c1-300
            } | escape
        )"
        echo "::error title=$title::$msg"
        ;;
    lines)
        msg="$(grep -a -E "$4" "$log" | cut -c1-400 | head -32 | escape)"
        if [ -n "$msg" ]; then echo "::notice title=$title::$msg"; fi
        ;;
esac
exit 0
