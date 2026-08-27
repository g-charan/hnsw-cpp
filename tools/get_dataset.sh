#!/usr/bin/env bash
# Fetch an ANN benchmark dataset into data/.
#
#   ./tools/get_dataset.sh siftsmall   10k x 128, ~5 MB    (tests)
#   ./tools/get_dataset.sh sift        1M  x 128, ~168 MB  (README numbers)
#
# Both ship ground truth computed by exhaustive search, which is what makes
# recall measurable rather than self-reported.
set -euo pipefail

name="${1:-siftsmall}"
case "$name" in
  siftsmall|sift|gist) ;;
  *) echo "unknown dataset: $name (want siftsmall, sift or gist)" >&2; exit 1 ;;
esac

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/data"
mkdir -p "$dest"

if [ -d "$dest/$name" ]; then
  echo "$dest/$name already present"
  exit 0
fi

url="ftp://ftp.irisa.fr/local/texmex/corpus/${name}.tar.gz"
echo "fetching $url"
curl -SL --fail -o "$dest/${name}.tar.gz" "$url"
tar xzf "$dest/${name}.tar.gz" -C "$dest"
rm -f "$dest/${name}.tar.gz"

# The learn split is only used for training quantisers, which this project does
# not do. It is by far the largest file, so it goes.
rm -f "$dest/$name/${name}_learn.fvecs"
echo "ready: $dest/$name"
ls -la "$dest/$name"
