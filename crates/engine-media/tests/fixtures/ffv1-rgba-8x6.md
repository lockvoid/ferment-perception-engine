# Lossless recording regression

`ffv1-rgba-8x6.mkv` was encoded independently with managed FFmpeg 8.1.3-Jellyfin on
2026-11-09. It contains two identical 8×6 RGBA frames at 10 FPS. Four 4×3
quadrants are literal RGBA `[255,0,0,255]`, `[0,255,0,128]`, `[0,0,255,64]`,
`[17,33,65,0]`, in top-left, top-right, bottom-left, bottom-right order.

The encoder received packed raw RGBA on stdin with:

```text
-f rawvideo -pixel_format rgba -video_size 8x6 -framerate 10 -i pipe:0
-frames:v 2 -an -c:v ffv1 -level 3 -coder 1 -context 1 -slices 4
-slicecrc 1 -threads 2 -pix_fmt bgra -color_range pc
-color_primaries bt709 -color_trc iec61966-2-1 -colorspace rgb -f matroska
```

SHA-256: `0b990c71da48ba061e400bd6de25bc238216314c4f8496e5c0ce1814be4e324a`.
No input asset or third-party artwork is embedded. The regression grades
actual decoded bytes against the literal quadrants, including transparent RGB.
