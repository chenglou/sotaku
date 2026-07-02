#!/bin/sh
# Viridian installer:
#   curl -fsSL https://console.viridianresear.ch/install.sh | sh                   -> vd (the CLI)
#   curl -fsSL https://console.viridianresear.ch/install.sh | sh -s -- workbench   -> vd-workbench too
API="${VD_API:-https://console.viridianresear.ch}"
os=$(uname -s | tr 'A-Z' 'a-z')
arch=$(uname -m)
case "$arch" in
  x86_64|amd64)  arch=x86_64 ;;
  arm64|aarch64) arch=aarch64 ;;
  *) echo "vd: unsupported architecture '$arch'" >&2; exit 1 ;;
esac
target="${os}-${arch}"

bins="vd"
[ "$1" = "workbench" ] && bins="vd vd-workbench"

bindir="${VD_BIN:-/usr/local/bin}"
if ! ( mkdir -p "$bindir" 2>/dev/null && [ -w "$bindir" ] ); then
  bindir="${HOME}/.local/bin"
  mkdir -p "$bindir" || { echo "vd: cannot create $bindir" >&2; exit 1; }
fi

for bin in $bins; do
  tmp=$(mktemp)
  echo "$bin: downloading ${target}..."
  if command -v curl >/dev/null 2>&1; then
    curl -fSL "${API}/dl/${bin}-${target}" -o "$tmp"
  else
    wget -O "$tmp" "${API}/dl/${bin}-${target}"
  fi
  if [ $? -ne 0 ] || [ ! -s "$tmp" ]; then
    rm -f "$tmp"
    echo "$bin: no prebuilt binary for ${target} yet — build from source: cargo build --release" >&2
    exit 1
  fi
  chmod +x "$tmp" && mv "$tmp" "${bindir}/${bin}"
  echo "$bin: installed to ${bindir}/${bin}"
done
case ":${PATH}:" in
  *":${bindir}:"*) ;;
  *) echo "vd: add ${bindir} to your PATH" ;;
esac
printf '\n  next:  vd auth      log in via your browser\n         vd --help    list commands\n'
[ "$1" = "workbench" ] && printf '         vd-workbench start a research session\n'
exit 0
