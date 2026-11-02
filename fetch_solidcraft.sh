#!/usr/bin/env bash
# =====================================================================
#  macOS / Linux twin of fetch_solidcraft.bat: the official release of the SolidCraft
#  web package AI edits run out of sight, pinned by version and SHA-256, unpacked to
#  ui/vendor/solidcraft. SolidCraft is MIT OR Apache-2.0; its licences and NOTICE come with it.
# =====================================================================
set -eu
VERSION=0.27.1
SHA256=3bd142af6fbc5b209738f7ddc69d58d9baf649085dc5bd718b61900f5c572a1f
FOLDER="solidcraft-web-$VERSION"
URL="https://github.com/tangency-labs/solidcraft/releases/download/v$VERSION/$FOLDER.zip"
DEST="$(cd "$(dirname "$0")" && pwd)/ui/vendor/solidcraft"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Downloading $URL"
curl -fsSL --retry 3 -o "$TMP/$FOLDER.zip" "$URL"
if command -v sha256sum > /dev/null 2>&1; then
    GOT="$(sha256sum "$TMP/$FOLDER.zip" | cut -d' ' -f1)"
else
    GOT="$(shasum -a 256 "$TMP/$FOLDER.zip" | cut -d' ' -f1)"
fi
if [ "$GOT" != "$SHA256" ]; then
    echo "SHA-256 mismatch: expected $SHA256, got $GOT" >&2
    exit 1
fi
unzip -q "$TMP/$FOLDER.zip" -d "$TMP"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$TMP/$FOLDER" "$DEST"
echo "SolidCraft web package is in $DEST"
