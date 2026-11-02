#!/usr/bin/env bash
# test for the metal backend. Extra arguments go to cargo. See _cargo.sh.
exec bash "$(dirname "$0")/_cargo.sh" metal test "$@"
