//! Ephemeral analysis arrays. The host supplies fresh scratch files (and owns
//! their disk budget/lifetime); no pathname or whole-file mapping is required.
use ferment_engine_core::{EngineError, Result};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

const PAGE_BYTES: usize = 32 * 1024;
pub(super) type Scratch<'a> = dyn FnMut() -> Result<File> + 'a;

pub(super) struct Array<const WIDTH: usize> {
    file: File,
    length: usize,
    flushed: u64,
    tail: Vec<u8>,
    page: Vec<u8>,
    page_start: usize,
}
impl<const WIDTH: usize> Array<WIDTH> {
    pub fn new(mut file: File) -> Result<Self> {
        file.set_len(0).map_err(io)?;
        file.seek(SeekFrom::Start(0)).map_err(io)?;
        Ok(Self {
            file,
            length: 0,
            flushed: 0,
            tail: Vec::with_capacity(PAGE_BYTES),
            page: Vec::with_capacity(PAGE_BYTES),
            page_start: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn append(&mut self, bytes: &[u8]) -> Result<()> {
        if !bytes.len().is_multiple_of(WIDTH) {
            return Err(EngineError::internal("Invalid scratch array record"));
        }
        let length = self
            .length
            .checked_add(bytes.len() / WIDTH)
            .filter(|n| n.checked_mul(WIDTH).is_some())
            .ok_or_else(|| EngineError::internal("Analysis array overflow"))?;
        for mut input in bytes.chunks(PAGE_BYTES) {
            while !input.is_empty() {
                let count = input.len().min(PAGE_BYTES - self.tail.len());
                self.tail.extend_from_slice(&input[..count]);
                input = &input[count..];
                if self.tail.len() == PAGE_BYTES {
                    self.flush()?;
                }
            }
        }
        self.length = length;
        self.page.clear();
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        if !self.tail.is_empty() {
            self.file.seek(SeekFrom::Start(self.flushed)).map_err(io)?;
            self.file.write_all(&self.tail).map_err(io)?;
            self.flushed += self.tail.len() as u64;
            self.tail.clear();
        }
        Ok(())
    }
    pub fn get(&mut self, index: usize) -> Result<[u8; WIDTH]> {
        if index >= self.length {
            return Err(EngineError::internal("Analysis array range"));
        }
        if index < self.page_start || index >= self.page_start + self.page.len() / WIDTH {
            self.flush()?;
            let per_page = PAGE_BYTES / WIDTH;
            self.page_start = index / per_page * per_page;
            self.page
                .resize((self.length - self.page_start).min(per_page) * WIDTH, 0);
            self.file
                .seek(SeekFrom::Start((self.page_start * WIDTH) as u64))
                .map_err(io)?;
            self.file.read_exact(&mut self.page).map_err(io)?;
        }
        let offset = (index - self.page_start) * WIDTH;
        Ok(self.page[offset..offset + WIDTH]
            .try_into()
            .expect("record width"))
    }
    #[cfg(test)]
    fn resident_bytes(&self) -> usize {
        self.tail.capacity() + self.page.capacity()
    }
}

pub(super) struct Logits(Array<8>);
impl Logits {
    pub fn new(file: File) -> Result<Self> {
        Array::new(file).map(Self)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn append(&mut self, beat: &[f32], down: &[f32]) -> Result<()> {
        if beat.len() != down.len()
            || beat.len() > 4096
            || beat.iter().chain(down).any(|v| !v.is_finite())
        {
            return Err(EngineError::InvalidOutput);
        }
        let mut bytes = Vec::with_capacity(beat.len() * 8);
        for (&b, &d) in beat.iter().zip(down) {
            bytes.extend_from_slice(&b.to_le_bytes());
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        self.0.append(&bytes)
    }
    pub fn get(&mut self, at: usize) -> Result<[f32; 2]> {
        let bytes = self.0.get(at)?;
        Ok([
            f32::from_le_bytes(bytes[..4].try_into().expect("f32")),
            f32::from_le_bytes(bytes[4..].try_into().expect("f32")),
        ])
    }
    #[cfg(test)]
    pub fn resident_bytes(&self) -> usize {
        self.0.resident_bytes()
    }
}
fn io(error: std::io::Error) -> EngineError {
    EngineError::internal(format!("Analysis scratch: {error}"))
}
