#!/bin/sh
# Download the FLEURS ur_pk files used by Awaaz-Nano (CC BY 4.0) and
# verify their SHA-256 hashes. Source: the google/fleurs dataset
# repository on Hugging Face (the dataset's official distribution).
# Fetched originally on 2026-10-29 (Asia/Karachi).
set -eu
DEST="${1:-./fleurs_ur_pk}"
mkdir -p "$DEST"
BASE="https://huggingface.co/datasets/google/fleurs/resolve/main/data/ur_pk"
export no_proxy='localhost,127.0.0.1,::1' NO_PROXY='localhost,127.0.0.1,::1'
for split in train dev test; do
  curl -sL "$BASE/audio/$split.tar.gz" -o "$DEST/$split.tar.gz"
  curl -sL "$BASE/$split.tsv"        -o "$DEST/$split.tsv"
done
cat > "$DEST/SHA256SUMS.expected" <<'EOF'
00fbfdd4b98fa4c264b843355aa7450a3922642b73cb13e54ffa8950c4cdd93a  train.tar.gz
7fda1bd8c7e852547cf2c0a8d06fa45cbc37b09633c067056b593cbb88e94162  train.tsv
ee0a7a18242a70ea9088e9618337e4835ce1972557bdc2cd3d500555ce89520f  dev.tar.gz
80fe94d485d78e6277af6ca308c5ceb40f9a19352907cfada9ed07006ac20eed  dev.tsv
618840ad5fb21a59cc179ec535ddbc8e756e2fb40624689c6d59c93e019cd745  test.tar.gz
39cd4fd14ddcf730a9f2ddfd590181255555ba060a53506dcff598a8b9ecc476  test.tsv
EOF
cd "$DEST" && sha256sum -c SHA256SUMS.expected
