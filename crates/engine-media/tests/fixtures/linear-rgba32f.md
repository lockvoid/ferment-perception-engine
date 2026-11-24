# Literal float recording fixtures

Independently encoded with the Homebrew FFmpeg CLI on2026-11-10. No engine
encoder or decoder generated the expected values. These files contain
uncompressed planar `gbrapf32le` rawvideo in NUT,16×8 at30000/1001FPS.

For frame `n`, pixel `(x,y)` selects RGBA from this literal array at
`(x+y+n)%4`:

```text
[2, -0.5, 8, 0.25]
[-2, 4, 0.125, 0]
[0, 0.25, 16, 1]
[0.03125, 3, -0.25, 0.5]
```

The input file writes each frame's G,B,R,A planes as little-endian float32.
CLI arguments (use31 frames for the first file,1 for the second input):

```text
-f rawvideo -pixel_format gbrapf32le -video_size 16x8
-framerate 30000/1001 -i INPUT.raw -map 0:v:0 -an
-c:v rawvideo -pix_fmt gbrapf32le -f nut OUTPUT.nut
```

| File | Input SHA-256 | NUT SHA-256 |
| --- | --- | --- |
| `linear-rgba32f-16x8.nut` | `3f1e6d663bccc813c5c223f3470ea90b26ab263288007f66ebb88db816b1330a` | `f8e7e647c08272fc79a423c82660fcd1effaa73beb598e2583a1bddd5a28a899` |
| `linear-rgba32f-one.nut` | `d1a0da5874c15bdd8fb41f8842a10fc69a4b6990e70aba4f0025d8ed556eff36` | `c4d01f197effe5525436a0b1c7905f988b86d899641b1a1c787cfa83aad45d4e` |

`libav_float` grades every sample bit, exact rational PTS and EOF on the file
and checked MediaInput paths. The complete presentation extent is31 or1
frame intervals; the NUT index's last PTS alone omits the final interval.
The one-frame file catches that edge with an index duration of0. Container
metadata bytes can differ across CLI versions; the literal pixel/PTS oracle
is independent of those bytes.
