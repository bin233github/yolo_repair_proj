#!/usr/bin/env bash
set -euo pipefail
PREFIX=${1:-/usr/local/lib}
mkdir -p "$PREFIX"
gcc -shared -fPIC scripts/libgl_stub.c -o "$PREFIX/libGL.so.1"
if [ -d /usr/lib ]; then
    ln -sf "$PREFIX/libGL.so.1" /usr/lib/libGL.so.1
fi
echo "libGL stub built at $PREFIX/libGL.so.1"
