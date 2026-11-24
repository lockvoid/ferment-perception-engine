use crate::{MediaTime, Packetization, Result, TimeBase, VideoError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketFormat {
    Native,
    AnnexB,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPacket {
    pub stream_index: u32,
    pub pts: Option<MediaTime>,
    pub dts: Option<MediaTime>,
    pub duration: Option<MediaTime>,
    pub is_sync: bool,
    pub data: Vec<u8>,
    pub packetization: Packetization,
}

impl EncodedPacket {
    pub fn data_in_format(&self, format: PacketFormat) -> Result<Vec<u8>> {
        match (format, self.packetization) {
            (PacketFormat::Native, _) | (PacketFormat::AnnexB, Packetization::AnnexB) => {
                Ok(self.data.clone())
            }
            (PacketFormat::AnnexB, Packetization::LengthPrefixed { nal_length_size }) => {
                length_prefixed_to_annex_b(&self.data, nal_length_size)
            }
            (PacketFormat::AnnexB, Packetization::Elementary) => Err(VideoError::InvalidData(
                "this codec does not use Annex-B packetization".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleInfo {
    pub pts: Option<MediaTime>,
    pub dts: Option<MediaTime>,
    pub duration: Option<MediaTime>,
    pub is_sync: bool,
    pub byte_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleTable {
    pub stream_index: u32,
    pub time_base: TimeBase,
    pub samples: Vec<SampleInfo>,
}

fn length_prefixed_to_annex_b(data: &[u8], nal_length_size: u8) -> Result<Vec<u8>> {
    if !(1..=4).contains(&nal_length_size) {
        return Err(VideoError::InvalidData(format!(
            "invalid NAL length field width {nal_length_size}"
        )));
    }
    let width = usize::from(nal_length_size);
    let mut cursor = 0usize;
    let mut output = Vec::with_capacity(data.len().saturating_add(16));
    while cursor < data.len() {
        let length_bytes = data
            .get(cursor..cursor + width)
            .ok_or_else(|| VideoError::InvalidData("truncated NAL length prefix".into()))?;
        cursor += width;
        let mut length = 0usize;
        for byte in length_bytes {
            length = length
                .checked_mul(256)
                .and_then(|value| value.checked_add(usize::from(*byte)))
                .ok_or_else(|| VideoError::InvalidData("NAL length overflow".into()))?;
        }
        let end = cursor
            .checked_add(length)
            .ok_or_else(|| VideoError::InvalidData("NAL payload length overflow".into()))?;
        let payload = data
            .get(cursor..end)
            .ok_or_else(|| VideoError::InvalidData("truncated NAL payload".into()))?;
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(payload);
        cursor = end;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_avcc_packets_to_annex_b() {
        let packet = EncodedPacket {
            stream_index: 0,
            pts: None,
            dts: None,
            duration: None,
            is_sync: true,
            data: vec![0, 0, 0, 2, 0x65, 0xaa, 0, 0, 0, 1, 0x06],
            packetization: Packetization::LengthPrefixed { nal_length_size: 4 },
        };
        assert_eq!(
            packet.data_in_format(PacketFormat::AnnexB).unwrap(),
            vec![0, 0, 0, 1, 0x65, 0xaa, 0, 0, 0, 1, 0x06]
        );
    }

    #[test]
    fn rejects_a_truncated_nal() {
        let packet = EncodedPacket {
            stream_index: 0,
            pts: None,
            dts: None,
            duration: None,
            is_sync: false,
            data: vec![0, 0, 0, 3, 0x41],
            packetization: Packetization::LengthPrefixed { nal_length_size: 4 },
        };
        assert!(packet.data_in_format(PacketFormat::AnnexB).is_err());
    }
}
