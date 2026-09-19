#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for name in viewer policy; do
  NODE_PATH="$PWD/.webdeps/node_modules" .webdeps/node_modules/.bin/esbuild "web/$name.js" \
    --bundle --minify --format=iife --outfile="src/dex_manipulation/static/$name.bundle.js"
done
cp .webdeps/node_modules/three/LICENSE src/dex_manipulation/static/THREE_LICENSE.txt
