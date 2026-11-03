#!/usr/bin/env bash
# dev for the metal backend. Extra arguments go to cargo. See _cargo.sh.
exec bash "$(dirname "$0")/_cargo.sh" metal dev "$@"
