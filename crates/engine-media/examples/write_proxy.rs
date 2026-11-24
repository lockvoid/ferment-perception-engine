//! write_proxy SOURCE OUT.mov [SHORT_EDGE]: an editing proxy, timed.
use ferment_engine_media::MediaInput;
use ferment_engine_media::platform::libav::write_proxy;
use std::io::{self, Read, Seek, SeekFrom};

struct File(std::fs::File, u64);
impl Read for File {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.0.read(bytes)
    }
}
impl Seek for File {
    fn seek(&mut self, at: SeekFrom) -> io::Result<u64> {
        self.0.seek(at)
    }
}
impl MediaInput for File {
    fn byte_len(&self) -> u64 {
        self.1
    }
    fn validate(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [source, output, rest @ ..] = args.as_slice() else {
        return Err("write_proxy SOURCE OUT.mov [SHORT_EDGE]".into());
    };
    let short_edge = rest.first().map_or(Ok(1080), |edge| edge.parse())?;
    let file = std::fs::File::open(source)?;
    let length = file.metadata()?.len();
    let started = std::time::Instant::now();
    let report = write_proxy(
        Box::new(File(file, length)),
        0,
        std::path::Path::new(output),
        short_edge,
        &mut |_| true,
    )?;
    println!("{report:?} in {:.2} s", started.elapsed().as_secs_f64());
    Ok(())
}
