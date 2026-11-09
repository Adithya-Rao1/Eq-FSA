#!/usr/bin/env bash
set -u
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
DEST="$SCRATCH_ROOT/data"
TAR="$DEST/rmd17.tar.bz2"
SIZE=1066301513
SHA=cddeea2ec2c4b22a9de16a9a65a3eeedac5b29da9c99c20bbff0153912bf922b
URLS=(
  "https://figshare.com/ndownloader/files/23950376"
  "https://archive.materialscloud.org/records/466/files/rmd17.tar.bz2?download=1"
)
mkdir -p "$DEST"

size_ok() { [[ -f "$TAR" && "$(stat -c %s "$TAR" 2>/dev/null)" == "$SIZE" ]]; }

if ! size_ok; then
  for url in "${URLS[@]}"; do
    echo "[fetch_rmd17] trying $url"
    wget -q --tries=3 --timeout=60 -O "$TAR.part" "$url" && mv -f "$TAR.part" "$TAR"
    size_ok && break
    echo "[fetch_rmd17]   not usable: $(stat -c %s "$TAR" 2>/dev/null || echo 'no file') bytes (expected $SIZE)"
    rm -f "$TAR.part"
  done
fi
if ! size_ok; then
  echo "[fetch_rmd17] could not download automatically." >&2
  echo "  Get rmd17.tar.bz2 in a browser from https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038" >&2
  echo "  and place it at $TAR, then run this script again (it will skip the download)." >&2
  echo "  For large uploads through a web file manager, split into pieces and verify each; whole-file uploads were truncated before." >&2
  exit 1
fi

echo "[fetch_rmd17] checking sha256 ..."
echo "$SHA  $TAR" | sha256sum -c - || { echo "[fetch_rmd17] sha256 mismatch; the file is corrupt or a different release" >&2; exit 1; }
if [[ ! -d "$DEST/rmd17/npz_data" ]]; then
  echo "[fetch_rmd17] extracting ..."
  mkdir -p "$DEST/rmd17"
  tar -xjf "$TAR" -C "$DEST/rmd17" || { echo "[fetch_rmd17] extraction failed" >&2; exit 1; }
  # the archive's top-level folder is rmd17/ (holding npz_data/, splits/, readme.txt); the loaders want npz_data/ and splits/ directly under data/rmd17/
  if [[ -d "$DEST/rmd17/rmd17/npz_data" ]]; then
    mv "$DEST/rmd17/rmd17/npz_data" "$DEST/rmd17/rmd17/splits" "$DEST/rmd17/"
  fi
fi
n=$(ls "$DEST/rmd17/npz_data"/rmd17_*.npz 2>/dev/null | wc -l)
s=$(ls "$DEST/rmd17/splits"/index_*.csv 2>/dev/null | wc -l)
echo "[fetch_rmd17] molecules: $n (expect 10); split files: $s (expect 10)"
[[ "$n" == 10 && "$s" == 10 ]] || { echo "[fetch_rmd17] unexpected layout under $DEST/rmd17 (the tarball may extract one level deeper)" >&2; exit 1; }
echo "[fetch_rmd17] OK: $DEST/rmd17"
