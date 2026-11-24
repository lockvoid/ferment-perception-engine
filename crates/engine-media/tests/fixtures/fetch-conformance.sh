#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$SCRIPT_DIR/generated"
TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ferment-hevc-conformance.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT HUP INT TERM

BASE_URL=https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1

fetch_and_mux() {
    archive=$1
    member=$2
    output=$3
    curl -L --fail --silent --show-error -o "$TEMP_DIR/$archive.zip" \
        "$BASE_URL/$archive.zip"
    unzip -q "$TEMP_DIR/$archive.zip" "$member" -d "$TEMP_DIR"
    ffmpeg -hide_banner -loglevel error -y -f hevc -i "$TEMP_DIR/$member" \
        -c copy -tag:v hvc1 -movflags +faststart "$OUTPUT_DIR/$output"
}

mkdir -p "$OUTPUT_DIR"
fetch_and_mux AMVP_A_MTK_4 AMVP_A_MTK_4/AMVP_A_MTK_4.bit \
    conformance-hevc-amvp.mp4
fetch_and_mux RAP_A_docomo_6 RAP_A_docomo_6.bit \
    conformance-hevc-rap.mp4
