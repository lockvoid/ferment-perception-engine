#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$SCRIPT_DIR/generated"
mkdir -p "$OUTPUT_DIR"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required to regenerate video fixtures" >&2
    exit 1
fi

source_args="-f lavfi -i testsrc2=size=64x48:rate=12:duration=1"
common_args="-an -vf setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709"

container_color_args() {
    case $1 in
        *.mp4 | *.mov) echo "-movflags +write_colr" ;;
        *) echo "" ;;
    esac
}

make_h264() {
    output=$1
    key_interval=$2
    color_mux_args=$(container_color_args "$output")
    # shellcheck disable=SC2086
    ffmpeg -hide_banner -loglevel error -y $source_args $common_args \
        -c:v libx264 -pix_fmt yuv420p -preset veryslow \
        -g "$key_interval" -keyint_min "$key_interval" -sc_threshold 0 -bf 2 \
        $color_mux_args "$output"
}

make_hevc() {
    output=$1
    key_interval=$2
    color_mux_args=$(container_color_args "$output")
    # shellcheck disable=SC2086
    ffmpeg -hide_banner -loglevel error -y $source_args $common_args \
        -c:v libx265 -pix_fmt yuv420p \
        -x265-params "pools=1:frame-threads=1:log-level=error:keyint=$key_interval:min-keyint=$key_interval:scenecut=0:bframes=2" \
        $color_mux_args "$output"
}

make_prores() {
    output=$1
    muxer=$2
    color_mux_args=$(container_color_args "$output")
    # shellcheck disable=SC2086
    ffmpeg -hide_banner -loglevel error -y $source_args $common_args \
        -c:v prores_ks -profile:v 2 -pix_fmt yuv422p10le -vendor apl0 \
        -strict experimental $color_mux_args -f "$muxer" "$output"
}

make_h264 "$OUTPUT_DIR/h264-gop4.mp4" 4
make_h264 "$OUTPUT_DIR/h264-gop5.mov" 5
make_h264 "$OUTPUT_DIR/h264-gop6.mkv" 6

make_hevc "$OUTPUT_DIR/hevc-gop4.mp4" 4
make_hevc "$OUTPUT_DIR/hevc-gop5.mov" 5
make_hevc "$OUTPUT_DIR/hevc-gop6.mkv" 6

make_prores "$OUTPUT_DIR/prores-standard.mp4" mov
make_prores "$OUTPUT_DIR/prores-standard.mov" mov
make_prores "$OUTPUT_DIR/prores-standard.mkv" matroska

make_aac() {
    output=$1
    sample_rate=$2
    channels=$3
    muxer=$4
    if [ "$channels" -eq 1 ]; then
        layout=mono
        bit_rate=96000
        source="aevalsrc=0.35*sin(2*PI*997*t)+0.1*cos(2*PI*233*t):s=$sample_rate:d=1:c=mono"
    else
        layout=stereo
        bit_rate=160000
        source="aevalsrc=0.35*sin(2*PI*997*t)+0.1*cos(2*PI*233*t)|0.3*sin(2*PI*613*t)+0.08*cos(2*PI*173*t):s=$sample_rate:d=1:c=stereo"
    fi
    case $muxer in
        mp4) actual_muxer=mp4; mux_args="-movflags +faststart" ;;
        mov) actual_muxer=mp4; mux_args="-movflags +faststart -brand qt" ;;
        *) actual_muxer=$muxer; mux_args="" ;;
    esac
    # shellcheck disable=SC2086
    ffmpeg -hide_banner -loglevel error -y -f lavfi -i "$source" \
        -map 0:a:0 -c:a aac -profile:a aac_low -b:a "$bit_rate" \
        -ar "$sample_rate" -ac "$channels" -channel_layout "$layout" \
        $mux_args -f "$actual_muxer" "$output"
}

for sample_rate in 44100 48000; do
    for channel_name in mono stereo; do
        case $channel_name in
            mono) channels=1 ;;
            stereo) channels=2 ;;
        esac
        make_aac "$OUTPUT_DIR/aac-lc-$sample_rate-$channel_name.m4a" "$sample_rate" "$channels" mp4
        make_aac "$OUTPUT_DIR/aac-lc-$sample_rate-$channel_name.mp4" "$sample_rate" "$channels" mp4
        make_aac "$OUTPUT_DIR/aac-lc-$sample_rate-$channel_name.mov" "$sample_rate" "$channels" mov
        make_aac "$OUTPUT_DIR/aac-lc-$sample_rate-$channel_name.mkv" "$sample_rate" "$channels" matroska
    done
done

# HE-AAC (SBR) fixtures. FFmpeg cannot encode HE-AAC without libfdk, so these
# come from Apple's afconvert and regenerate on macOS only. Two container
# flavors are covered deliberately:
#   - afconvert m4a: dual-rate (stsd/mdhd at the 22050 Hz AAC core rate,
#     1024-sample packets, no edit list) — the Apple-encoder shape;
#   - ffmpeg -c copy remux: effective-rate (stsd/mdhd at 44100 Hz,
#     2048-sample packets, priming edit list) — the FFmpeg-muxer shape.
# `aacp` is HE-AACv2: a mono AAC core with implicit parametric stereo.
if command -v afconvert >/dev/null 2>&1; then
    he_source="$OUTPUT_DIR/tmp-he-source.wav"
    ffmpeg -hide_banner -loglevel error -y -f lavfi \
        -i "aevalsrc=0.35*sin(2*PI*997*t)+0.1*cos(2*PI*233*t)|0.3*sin(2*PI*613*t)+0.08*cos(2*PI*173*t):s=44100:d=1:c=stereo" \
        -c:a pcm_s16le "$he_source"
    afconvert -f m4af -d aach -b 64000 "$he_source" "$OUTPUT_DIR/aac-he-44100-stereo.m4a"
    afconvert -f m4af -d aacp -b 48000 "$he_source" "$OUTPUT_DIR/aac-he-v2-44100-stereo.m4a"
    # QuickTime-family muxers write channelcount=2 in the sample entry even
    # for mono AAC; this fixture pins channel derivation to the ASC.
    lc_mono_source="$OUTPUT_DIR/tmp-lc-mono-source.wav"
    ffmpeg -hide_banner -loglevel error -y -f lavfi \
        -i "aevalsrc=0.35*sin(2*PI*997*t)+0.1*cos(2*PI*233*t):s=22050:d=1:c=mono" \
        -c:a pcm_s16le "$lc_mono_source"
    afconvert -f m4af -d aac -b 48000 "$lc_mono_source" "$OUTPUT_DIR/aac-lc-22050-mono-apple.m4a"
    rm -f "$lc_mono_source"
    ffmpeg -hide_banner -loglevel error -y -i "$OUTPUT_DIR/aac-he-44100-stereo.m4a" \
        -c copy -movflags +faststart "$OUTPUT_DIR/aac-he-44100-stereo.mp4"
    rm -f "$he_source"
else
    echo "afconvert not found; skipping HE-AAC fixture regeneration (macOS only)" >&2
fi

# Parser-hardening fixtures. These remain deliberately small while exercising
# timeline and container structures that do not appear in the codec matrix.
ffmpeg -hide_banner -loglevel error -y $source_args $common_args \
    -c:v libx264 -pix_fmt yuv420p -preset veryslow -g 4 -bf 2 \
    -movflags +frag_keyframe+empty_moov+default_base_moof \
    "$OUTPUT_DIR/synthetic-fragmented.mp4"

ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i testsrc2=size=64x48:rate=12:duration=1 \
    -an -vf "setpts='if(lt(N,6),N/(12*TB),(0.5+(N-6)/6)/TB)',setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709" \
    -fps_mode vfr -c:v libx264 -pix_fmt yuv420p -g 6 -bf 2 \
    -movflags +faststart+write_colr "$OUTPUT_DIR/synthetic-vfr.mp4"

ffmpeg -hide_banner -loglevel error -y $source_args $common_args \
    -c:v libx264 -pix_fmt yuv420p -x264-params "open-gop=1:keyint=6:min-keyint=6:scenecut=0:bframes=3" \
    -movflags +faststart+write_colr "$OUTPUT_DIR/synthetic-open-gop.mp4"

ffmpeg -hide_banner -loglevel error -y -display_rotation:v:0 90 \
    -i "$OUTPUT_DIR/synthetic-open-gop.mp4" -map 0:v:0 -c copy \
    -movflags +faststart "$OUTPUT_DIR/synthetic-rotation.mov"

for timescale in 1000 90000; do
    ffmpeg -hide_banner -loglevel error -y $source_args $common_args \
        -c:v libx264 -pix_fmt yuv420p -g 4 -bf 2 \
        -video_track_timescale "$timescale" -movflags +faststart+write_colr \
        "$OUTPUT_DIR/synthetic-timescale-$timescale.mp4"
done

ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=black:size=64x48:rate=1:duration=600 \
    -an -c:v libx264 -pix_fmt yuv420p -preset ultrafast -g 30 -bf 0 \
    -video_track_timescale 1000 -movflags +faststart \
    "$OUTPUT_DIR/synthetic-long-duration.mp4"

ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i testsrc2=size=64x48:rate=12:duration=1 \
    -an -vf setparams=range=limited:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc \
    -c:v libx265 -pix_fmt yuv420p \
    -x265-params "pools=1:frame-threads=1:log-level=error:keyint=4:min-keyint=4:scenecut=0:bframes=2" \
    -movflags +faststart+write_colr "$OUTPUT_DIR/synthetic-hdr-nclx.mp4"

make_multitrack() {
    output=$1
    muxer=$2
    ffmpeg -hide_banner -loglevel error -y \
        -f lavfi -i testsrc2=size=64x48:rate=12:duration=1 \
        -f lavfi -i sine=frequency=440:sample_rate=48000:duration=1 \
        -f lavfi -i sine=frequency=880:sample_rate=44100:duration=1 \
        -map 0:v -map 0:v -map 1:a -map 2:a \
        -c:v libx264 -pix_fmt yuv420p -g 4 -bf 2 \
        -c:a aac -profile:a aac_low -shortest -f "$muxer" "$output"
}

make_multitrack "$OUTPUT_DIR/synthetic-multitrack.mp4" mp4
make_multitrack "$OUTPUT_DIR/synthetic-multitrack.mkv" matroska

# Force an ISO edit list unrelated to AAC priming: a 250 ms empty timeline
# edit followed by the one-second media segment. This gives the parser a
# delayed start and multiple `elst` entries without re-encoding the payload.
ffmpeg -hide_banner -loglevel error -y -itsoffset 0.25 \
    -i "$OUTPUT_DIR/synthetic-open-gop.mp4" -map 0:v:0 -c copy \
    -use_editlist 1 -movflags +faststart \
    "$OUTPUT_DIR/synthetic-offset-edit-list.mp4"

make_corruptions() {
    source=$1
    truncated=$2
    bitflipped=$3
    length=$(wc -c < "$source" | tr -d ' ')
    half=$((length / 2))
    dd if="$source" of="$truncated" bs=1 count="$half" 2>/dev/null
    cp "$source" "$bitflipped"
    printf '\000' | dd of="$bitflipped" bs=1 seek=0 conv=notrunc 2>/dev/null
}

make_corruptions "$OUTPUT_DIR/synthetic-open-gop.mp4" \
    "$OUTPUT_DIR/corrupt-truncated.mp4" "$OUTPUT_DIR/corrupt-bitflipped.mp4"
make_corruptions "$OUTPUT_DIR/synthetic-multitrack.mkv" \
    "$OUTPUT_DIR/corrupt-truncated.mkv" "$OUTPUT_DIR/corrupt-bitflipped.mkv"

# Bare MP3 elementary streams. Four deliberate shapes: CBR without a Xing
# frame (raw frame walk, no encoder timing), VBR with the Xing/LAME header
# (delay/padding trim), an ID3v2-tagged CBR file (tag skipping), and an
# MPEG-2 low-rate file (576-sample frames).
make_mp3() {
    output=$1
    sample_rate=$2
    channels=$3
    shift 3
    if [ "$channels" -eq 1 ]; then
        source="aevalsrc=0.35*sin(2*PI*997*t)+0.1*cos(2*PI*233*t):s=$sample_rate:d=1:c=mono"
    else
        source="aevalsrc=0.35*sin(2*PI*997*t)+0.1*cos(2*PI*233*t)|0.3*sin(2*PI*613*t)+0.08*cos(2*PI*173*t):s=$sample_rate:d=1:c=stereo"
    fi
    # shellcheck disable=SC2086
    ffmpeg -hide_banner -loglevel error -y -f lavfi -i "$source" \
        -map 0:a:0 -c:a libmp3lame -ar "$sample_rate" -ac "$channels" \
        "$@" -f mp3 "$output"
}

make_mp3 "$OUTPUT_DIR/mp3-cbr-44100-stereo.mp3" 44100 2 -b:a 128k -write_xing 0
make_mp3 "$OUTPUT_DIR/mp3-vbr-44100-stereo.mp3" 44100 2 -q:a 4
make_mp3 "$OUTPUT_DIR/mp3-id3-48000-mono.mp3" 48000 1 -b:a 96k \
    -metadata title="Ferment MP3 fixture" -metadata artist="engine-media" \
    -id3v2_version 3 -write_xing 0
make_mp3 "$OUTPUT_DIR/mp3-cbr-22050-mono.mp3" 22050 1 -b:a 48k -write_xing 0
