# Exact-clock lossless recording fixtures

Historical defect: Ferment recording produced all frames, but Matroska1ms PTS
made native source replay at60 and30000/1001FPS select preceding frames.
These AVI files were independently encoded with managed FFmpeg8.1.3-1 Jellyfin
on2026-11-10. Each contains31 frames at16×8. Input RGBA for frame `i` and
column `x` is `[7*i,255-7*i,3*i,85*(x%4)]`, repeated over eight rows. No art,
external media or expected values derived through the decoder are embedded.

The original Matroska alpha fixture remains an independent regression for
previous saved media. AVI preserves the exact rational frame grid; the test
compares native timestamps by integer cross-multiplication and literal pixels.

Encoder args for each `RATE`:

```text
-f rawvideo -pixel_format rgba -video_size 16x8 -framerate RATE -i pipe:0
-an -c:v ffv1 -level 3 -coder 1 -context 1 -slices 4 -slicecrc 1 -threads 2
-pix_fmt bgra -color_range pc -color_primaries bt709
-color_trc iec61966-2-1 -colorspace rgb -f avi
```

- `ffv1-clock-60-1.avi`: SHA-256 `2a075080b70a7c16bea0ca467230c38f29bb2c5837a292dcea27a68ffb027e5e`, 9258 bytes.
- `ffv1-clock-30000-1001.avi`: SHA-256 `743b2655f64eb0f794e564ce29162ccafd298e1899796732e8a4f5ee8d7d53da`, 9258 bytes.
