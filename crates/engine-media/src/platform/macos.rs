//! VideoToolbox decode and encode backend.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::Mutex;

use async_trait::async_trait;
use objc2_core_foundation::{
    CFBoolean, CFData, CFDictionary, CFMutableDictionary, CFNumber, CFRetained, CFString, CFType,
};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime, CMTimeFlags,
    CMVideoFormatDescriptionCreate, CMVideoFormatDescriptionCreateFromH264ParameterSets,
    CMVideoFormatDescriptionCreateFromHEVCParameterSets, kCMTimeInvalid,
    kCMVideoCodecType_AppleProRes422, kCMVideoCodecType_AppleProRes422HQ,
    kCMVideoCodecType_AppleProRes422LT, kCMVideoCodecType_AppleProRes422Proxy,
    kCMVideoCodecType_AppleProRes4444, kCMVideoCodecType_AppleProRes4444XQ, kCMVideoCodecType_H264,
    kCMVideoCodecType_HEVC,
};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    kCVImageBufferColorPrimaries_ITU_R_709_2, kCVImageBufferColorPrimaries_ITU_R_2020,
    kCVImageBufferColorPrimaries_P3_D65, kCVImageBufferTransferFunction_ITU_R_709_2,
    kCVImageBufferTransferFunction_ITU_R_2100_HLG, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,
    kCVImageBufferTransferFunction_sRGB, kCVImageBufferYCbCrMatrix_ITU_R_601_4,
    kCVImageBufferYCbCrMatrix_ITU_R_709_2, kCVImageBufferYCbCrMatrix_ITU_R_2020,
    kCVPixelBufferHeightKey, kCVPixelBufferIOSurfacePropertiesKey,
    kCVPixelBufferMetalCompatibilityKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey,
    kCVPixelFormatType_32BGRA,
};
use objc2_video_toolbox::{
    VTCompressionOutputCallback, VTCompressionSession, VTDecodeFrameFlags, VTDecodeInfoFlags,
    VTDecompressionOutputCallbackRecord, VTDecompressionSession, VTEncodeInfoFlags,
    VTPixelTransferSession, VTSessionSetProperty, kVTCompressionPropertyKey_AllowFrameReordering,
    kVTCompressionPropertyKey_AverageBitRate, kVTCompressionPropertyKey_ColorPrimaries,
    kVTCompressionPropertyKey_HDRMetadataInsertionMode,
    kVTCompressionPropertyKey_MaxKeyFrameInterval,
    kVTCompressionPropertyKey_PreserveDynamicHDRMetadata, kVTCompressionPropertyKey_ProfileLevel,
    kVTCompressionPropertyKey_TransferFunction, kVTCompressionPropertyKey_YCbCrMatrix,
    kVTDownsamplingMode_Average, kVTHDRMetadataInsertionMode_Auto,
    kVTPixelTransferPropertyKey_DownsamplingMode, kVTPixelTransferPropertyKey_ScalingMode,
    kVTProfileLevel_H264_Baseline_AutoLevel, kVTProfileLevel_H264_High_AutoLevel,
    kVTProfileLevel_H264_Main_AutoLevel, kVTProfileLevel_HEVC_Main_AutoLevel,
    kVTProfileLevel_HEVC_Main10_AutoLevel, kVTScalingMode_Normal,
    kVTVideoDecoderReferenceMissingErr,
};

use crate::platform::libav::PacketReader;
use crate::{
    Codec, DecodeOptions, DecodeSession, DecoderConfiguration, EncodeSession, EncodedPacket,
    EncodedVideo, EncoderSettings, FrameBuffer, MacPixelBuffer, MediaInfo, MediaTime, PacketFormat,
    Packetization, ProResProfile, Result, TimeBase, VideoError, VideoFrame, VideoIo,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct VideoToolboxVideoIo;

#[async_trait(?Send)]
impl VideoIo for VideoToolboxVideoIo {
    async fn probe(&self, source: &Path) -> Result<MediaInfo> {
        crate::demux::probe(source)
    }

    async fn open_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn DecodeSession>> {
        Ok(Box::new(VideoToolboxDecoder::open(source, stream_index)?))
    }

    async fn open_decoder_with(
        &self,
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        if options.acceleration == crate::DecodeAcceleration::Software {
            return crate::platform::libav::LibavVideoIo
                .open_decoder_with(source, stream_index, options)
                .await;
        }
        Ok(Box::new(VideoToolboxDecoder::open_with(
            source,
            stream_index,
            options,
        )?))
    }

    async fn open_encoder(&self, settings: EncoderSettings) -> Result<Box<dyn EncodeSession>> {
        Ok(Box::new(VideoToolboxEncoder::new(settings)?))
    }
}

struct DecodedOutput {
    buffer: CFRetained<CVPixelBuffer>,
    pts: MediaTime,
    duration: Option<MediaTime>,
    width: u32,
    height: u32,
}

#[derive(Default)]
struct DecoderCallbackState {
    frames: Mutex<VecDeque<DecodedOutput>>,
    error: Mutex<Option<i32>>,
}

impl std::fmt::Debug for DecoderCallbackState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("DecoderCallbackState { .. }")
    }
}

unsafe extern "C-unwind" fn decompression_callback(
    output_refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: i32,
    _info_flags: VTDecodeInfoFlags,
    image_buffer: *mut CVPixelBuffer,
    presentation_time_stamp: CMTime,
    presentation_duration: CMTime,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(state_pointer) = NonNull::new(output_refcon.cast::<DecoderCallbackState>()) else {
            return;
        };
        // SAFETY: The boxed callback state outlives the VT session and is not
        // moved. Drop waits for callbacks before releasing it.
        let state = unsafe { state_pointer.as_ref() };
        if status != 0 {
            *state
                .error
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(status);
            return;
        }
        let Some(buffer_pointer) = NonNull::new(image_buffer) else {
            return;
        };
        // SAFETY: VideoToolbox owns a live CVPixelBuffer during the callback;
        // retain it before returning so the public frame can outlive the call.
        let buffer = unsafe { CFRetained::retain(buffer_pointer) };
        let pts = media_time_from_cm(presentation_time_stamp).unwrap_or(MediaTime::ZERO);
        let duration = media_time_from_cm(presentation_duration);
        let width = CVPixelBufferGetWidth(&buffer) as u32;
        let height = CVPixelBufferGetHeight(&buffer) as u32;
        state
            .frames
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push_back(DecodedOutput {
                buffer,
                pts,
                duration,
                width,
                height,
            });
    }));
}

/// A VideoToolbox decompression session fed by libavformat.
///
/// The session stays hand-rolled because libavcodec's VideoToolbox hwaccel
/// waits for every frame it submits, which measures 2.8x slower than
/// submitting asynchronously and draining — see `platform::libav::packets`.
/// The DEMUXER is not hand-rolled: packets come from libavformat like
/// everything else's do.
pub struct VideoToolboxDecoder {
    demuxer: PacketReader,
    stream_index: u32,
    stream_time_base: TimeBase,
    color: crate::ColorInfo,
    format_description: CFRetained<CMFormatDescription>,
    options: DecodeOptions,
    destination: Destination,
    callback_state: Box<DecoderCallbackState>,
    session: CFRetained<VTDecompressionSession>,
    seek_target: Option<MediaTime>,
    discard_before_target: bool,
    seek_access_start: Option<MediaTime>,
    seek_retry_before: Option<MediaTime>,
    /// Times a jump went back a GOP for a missing reference.
    rewinds: u32,
    /// HEVC: the RASL pictures of the CRA or BLA decoding starts at are
    /// skipped (`NoRaslOutputFlag`), as libavcodec does — they need the GOP
    /// before, and nothing after them refers to them.
    hevc: bool,
    /// Decoding (re)started and has not met its first picture yet.
    starting: bool,
    skip_rasl: bool,
    last_submitted_dts: Option<MediaTime>,
    has_distinct_dts: bool,
    reached_eof: bool,
}

impl std::fmt::Debug for VideoToolboxDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VideoToolboxDecoder")
            .field("stream_index", &self.stream_index)
            .field("stream_time_base", &self.stream_time_base)
            .field("seek_target", &self.seek_target)
            .field("reached_eof", &self.reached_eof)
            .finish_non_exhaustive()
    }
}

impl VideoToolboxDecoder {
    pub fn open(path: &Path, stream_index: u32) -> Result<Self> {
        Self::open_with(path, stream_index, DecodeOptions::default())
    }

    /// `open` with [`DecodeOptions`]: an `output_size` becomes the session's
    /// destination image buffer attributes, so VideoToolbox scales every
    /// frame on the way out and the callback never sees the source size.
    pub fn open_with(path: &Path, stream_index: u32, options: DecodeOptions) -> Result<Self> {
        let demuxer = PacketReader::open(path, stream_index)?;
        Self::from_packets(demuxer, stream_index, options)
    }

    pub(crate) fn from_packets(
        demuxer: PacketReader,
        stream_index: u32,
        mut options: DecodeOptions,
    ) -> Result<Self> {
        options.validate()?;
        if options.acceleration == crate::DecodeAcceleration::Software {
            return Err(VideoError::BackendUnavailable(
                "VideoToolbox cannot fulfill a software decode request".into(),
            ));
        }
        let stream = demuxer
            .info()
            .stream(stream_index)
            .ok_or_else(|| VideoError::InvalidData(format!("stream {stream_index} not found")))?
            .clone();
        if !stream.codec.is_video() {
            return Err(VideoError::UnsupportedCodec(format!(
                "stream {stream_index} is not video"
            )));
        }
        options.validate_color(stream.color, false)?;
        // CPU residency needs a BGRA destination even without a resize. Keep
        // the original coded extent; a consumer never has to fake a resize.
        if options.output == crate::DecodeOutput::CpuBgra8 && options.output_size.is_none() {
            options.output_size = Some(stream.width.zip(stream.height).ok_or_else(|| {
                VideoError::InvalidData("CPU decode source has no coded extent".into())
            })?);
        }
        options.validate()?;
        let destination = match (options.output, options.output_size) {
            (crate::DecodeOutput::GpuPlanes, _) => Destination::Planes(gpu_planes_format(&stream)?),
            (_, Some((width, height))) => Destination::Bgra { width, height },
            (_, None) => Destination::Native,
        };
        let format_description = make_format_description(&stream)?;
        let mut callback_state = Box::<DecoderCallbackState>::default();
        let session =
            make_decompression_session(&format_description, &mut callback_state, destination)?;
        Ok(Self {
            demuxer,
            stream_index,
            stream_time_base: stream.time_base,
            color: stream.color,
            format_description,
            options,
            destination,
            callback_state,
            session,
            seek_target: None,
            discard_before_target: true,
            seek_access_start: None,
            seek_retry_before: None,
            rewinds: 0,
            hevc: stream.codec == Codec::Hevc,
            starting: true,
            skip_rasl: false,
            last_submitted_dts: None,
            has_distinct_dts: false,
            reached_eof: false,
        })
    }

    fn recreate_session(&mut self) -> Result<()> {
        // SAFETY: Waiting establishes that the callback state is no longer in
        // use before invalidation and queue clearing.
        let status = unsafe { self.session.wait_for_asynchronous_frames() };
        check_status("VTDecompressionSessionWaitForAsynchronousFrames", status)?;
        // SAFETY: The retained session is valid and is replaced immediately.
        unsafe { self.session.invalidate() };
        self.callback_state
            .frames
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        *self
            .callback_state
            .error
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        self.session = make_decompression_session(
            &self.format_description,
            &mut self.callback_state,
            self.destination,
        )?;
        self.last_submitted_dts = None;
        self.has_distinct_dts = false;
        self.reached_eof = false;
        Ok(())
    }

    fn pop_frame(&mut self) -> Result<Option<VideoFrame>> {
        if let Some(status) = self
            .callback_state
            .error
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            return Err(VideoError::platform(
                "VTDecompressionOutputCallback",
                status,
            ));
        }
        loop {
            let mut frames = self
                .callback_state
                .frames
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let output_position = frames
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| left.pts.seconds().total_cmp(&right.pts.seconds()))
                .map(|(position, _)| position);
            let output = output_position.and_then(|position| {
                let candidate = &frames[position];
                let presentation_is_decoded = self
                    .last_submitted_dts
                    .is_some_and(|dts| candidate.pts.rescale(dts.time_base).value <= dts.value);
                (self.reached_eof || presentation_is_decoded || frames.len() > 16)
                    .then(|| frames.remove(position))
                    .flatten()
            });
            drop(frames);
            let Some(output) = output else {
                return Ok(None);
            };
            if self.discard_before_target
                && self
                    .seek_target
                    .is_some_and(|target| output.pts.rescale(target.time_base).value < target.value)
            {
                continue;
            }
            self.seek_target = None;
            self.seek_access_start = None;
            self.seek_retry_before = None;
            if let Destination::Planes(format) = self.destination {
                let written = CVPixelBufferGetPixelFormatType(&output.buffer);
                if written != format {
                    return Err(VideoError::InvalidData(format!(
                        "VideoToolbox wrote pixel format {written:#010x}, not the GPU planes \
                         {format:#010x} it was asked for"
                    )));
                }
            }
            let buffer = MacPixelBuffer::new(output.buffer);
            return Ok(Some(VideoFrame {
                width: output.width,
                height: output.height,
                pts: output.pts,
                duration: output.duration,
                color: self.color,
                buffer: if self.options.output == crate::DecodeOutput::CpuBgra8 {
                    FrameBuffer::Cpu(buffer.copy_bgra8()?)
                } else {
                    FrameBuffer::MacOS(buffer)
                },
            }));
        }
    }

    fn submit(&self, packet: &EncodedPacket, discard: bool) -> Result<()> {
        let sample = make_sample_buffer(packet, &self.format_description)?;
        let mut flags = VTDecodeFrameFlags::Frame_EnableTemporalProcessing;
        if discard {
            // Decode edit-list dependencies but never surface them to the frame timeline.
            flags |= VTDecodeFrameFlags::Frame_DoNotOutputFrame;
        }
        // SAFETY: `sample` and session are retained for the call; callback
        // storage remains alive for the entire session.
        let status = unsafe {
            self.session
                .decode_frame(&sample, flags, ptr::null_mut(), ptr::null_mut())
        };
        check_status("VTDecompressionSessionDecodeFrame", status)
    }

    /// A demux key flag can name an open-GOP CRA whose leading pictures
    /// still depend on the previous GOP. Only while waiting for the first
    /// sought presentation, recover the precise reference-missing condition
    /// by rewinding to the sync before the actual access packet's DTS.
    fn recover_seek_reference(&mut self, error: &VideoError) -> Result<bool> {
        if !matches!(error, VideoError::Platform { status, .. }
            if *status == kVTVideoDecoderReferenceMissingErr)
        {
            return Ok(false);
        }
        let (Some(target), Some(start)) = (self.seek_target, self.seek_access_start) else {
            return Ok(false);
        };
        let before = previous_sync_probe(start, self.seek_retry_before)?;
        self.rewinds += 1;
        log::debug!(
            target: "ferment_engine_media::videotoolbox",
            "jump to {target:?} rewinds before {start:?} for a missing reference ({} so far)",
            self.rewinds
        );
        self.recreate_session()?;
        self.demuxer.seek(before)?;
        self.seek_target = Some(target);
        self.seek_access_start = None;
        self.seek_retry_before = Some(start);
        self.starting = true;
        Ok(true)
    }

    /// Whether `packet` is a RASL picture of the IRAP decoding started at.
    fn skips_leading(&mut self, packet: &EncodedPacket) -> bool {
        let Some(kind) = hevc_picture_type(packet) else {
            return false;
        };
        if (HEVC_BLA_W_LP..=HEVC_IRAP_LAST).contains(&kind) {
            // Only the IRAP decoding starts at loses its leading pictures; a
            // later one has the GOP before it decoded.
            self.skip_rasl =
                std::mem::take(&mut self.starting) && (kind <= HEVC_BLA_N_LP || kind == HEVC_CRA);
            return false;
        }
        if std::mem::take(&mut self.starting) {
            self.skip_rasl = false;
        }
        self.skip_rasl && matches!(kind, HEVC_RASL_N | HEVC_RASL_R)
    }

    fn finish(&mut self) -> Result<()> {
        if self.reached_eof {
            return Ok(());
        }
        // SAFETY: The session is valid and waiting synchronizes all callbacks.
        let status = unsafe { self.session.wait_for_asynchronous_frames() };
        check_status("VTDecompressionSessionWaitForAsynchronousFrames", status)?;
        self.reached_eof = true;
        Ok(())
    }

    fn seek_from_sync(&mut self, target: MediaTime, discard_before_target: bool) -> Result<()> {
        self.demuxer.validate()?;
        self.recreate_session()?;
        self.demuxer.seek(target)?;
        self.seek_target = Some(target.rescale(self.stream_time_base));
        self.discard_before_target = discard_before_target;
        self.seek_access_start = None;
        self.seek_retry_before = None;
        self.starting = true;
        Ok(())
    }
}

#[async_trait(?Send)]
impl DecodeSession for VideoToolboxDecoder {
    async fn seek_sync(&mut self, target: MediaTime) -> Result<()> {
        self.seek_from_sync(target, false)
    }

    async fn seek(&mut self, target: MediaTime) -> Result<()> {
        self.seek_from_sync(target, true)
    }

    async fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        loop {
            self.demuxer.validate()?;
            match self.pop_frame() {
                Ok(Some(frame)) => {
                    self.demuxer.validate()?;
                    return Ok(Some(frame));
                }
                Ok(None) => {}
                Err(error) => {
                    if self.recover_seek_reference(&error)? {
                        continue;
                    }
                    return Err(error);
                }
            }
            if self.reached_eof {
                return Ok(None);
            }
            match self.demuxer.next_packet_with_discard()? {
                Some((packet, discard)) if packet.stream_index == self.stream_index => {
                    if self.hevc && self.skips_leading(&packet) {
                        continue;
                    }
                    if self.seek_target.is_some() && self.seek_access_start.is_none() {
                        let start = packet.dts.or(packet.pts).ok_or_else(|| {
                            VideoError::InvalidData("seek access packet has no timestamp".into())
                        })?;
                        // The demuxer must actually move back after a retry; a
                        // damaged first GOP must fail, never loop indefinitely.
                        if let Some(previous) = self.seek_retry_before {
                            previous_sync_probe(start, Some(previous))?;
                        }
                        self.seek_access_start = Some(start);
                    }
                    // Matroska often reports DTS equal to PTS even for
                    // reordered pictures. Trust the decode clock only after
                    // the stream demonstrates a distinct DTS; otherwise the
                    // bounded reorder queue below is used.
                    self.has_distinct_dts |= packet
                        .dts
                        .zip(packet.pts)
                        .is_some_and(|(dts, pts)| dts != pts);
                    self.last_submitted_dts = self.has_distinct_dts.then_some(packet.dts).flatten();
                    if let Err(error) = self.submit(&packet, discard) {
                        if self.recover_seek_reference(&error)? {
                            continue;
                        }
                        return Err(error);
                    }
                }
                Some(_) => {}
                None => self.finish()?,
            }
        }
    }
}

const HEVC_RASL_N: u8 = 8;
const HEVC_RASL_R: u8 = 9;
const HEVC_BLA_W_LP: u8 = 16;
const HEVC_BLA_N_LP: u8 = 18;
const HEVC_CRA: u8 = 21;
const HEVC_IRAP_LAST: u8 = 23;

/// The NAL unit type of an HEVC access unit's picture: its first VCL unit,
/// read past each length prefix or start code. `None` without one.
fn hevc_picture_type(packet: &EncodedPacket) -> Option<u8> {
    let data = packet.data.as_slice();
    let kind = |header: u8| (header >> 1) & 0x3f;
    match packet.packetization {
        Packetization::LengthPrefixed { nal_length_size } => {
            let size = usize::from(nal_length_size);
            let mut at = 0;
            while at + size < data.len() {
                let length = data[at..at + size]
                    .iter()
                    .fold(0usize, |length, byte| length << 8 | usize::from(*byte));
                let unit = kind(data[at + size]);
                if unit < 32 {
                    return Some(unit);
                }
                at = at.checked_add(size + length)?;
            }
            None
        }
        Packetization::AnnexB => data
            .windows(4)
            .filter(|window| window[..3] == [0, 0, 1])
            .map(|window| kind(window[3]))
            .find(|unit| *unit < 32),
        Packetization::Elementary => None,
    }
}

fn previous_sync_probe(start: MediaTime, previous: Option<MediaTime>) -> Result<MediaTime> {
    if let Some(previous) = previous
        && (start.time_base != previous.time_base || start.value >= previous.value)
    {
        return Err(VideoError::InvalidData(
            "reference recovery did not reach an earlier sync sample".into(),
        ));
    }
    let value = start
        .value
        .checked_sub(1)
        .ok_or_else(|| VideoError::InvalidData("reference recovery timestamp underflow".into()))?;
    Ok(MediaTime::new(value, start.time_base))
}

impl Drop for VideoToolboxDecoder {
    fn drop(&mut self) {
        // SAFETY: Best-effort orderly teardown. Waiting prevents callbacks
        // from observing callback_state after this object is dropped.
        let _ = unsafe { self.session.wait_for_asynchronous_frames() };
        unsafe { self.session.invalidate() };
    }
}

#[derive(Debug)]
struct EncoderCallbackState {
    codec: Codec,
    time_base: TimeBase,
    packets: Mutex<Vec<EncodedPacket>>,
    configuration: Mutex<Option<Vec<u8>>>,
    error: Mutex<Option<i32>>,
}

unsafe extern "C-unwind" fn compression_callback(
    output_refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: i32,
    info_flags: VTEncodeInfoFlags,
    sample_buffer: *mut CMSampleBuffer,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(state_pointer) = NonNull::new(output_refcon.cast::<EncoderCallbackState>()) else {
            return;
        };
        // SAFETY: The callback state is boxed and lives until after
        // VTCompressionSessionCompleteFrames and invalidation.
        let state = unsafe { state_pointer.as_ref() };
        if status != 0 {
            *state
                .error
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(status);
            return;
        }
        if info_flags.contains(VTEncodeInfoFlags::FrameDropped) {
            return;
        }
        let Some(sample_pointer) = NonNull::new(sample_buffer) else {
            return;
        };
        // SAFETY: The sample buffer remains live for the callback duration.
        let sample = unsafe { sample_pointer.as_ref() };
        let Some(block) = (unsafe { sample.data_buffer() }) else {
            *state
                .error
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(-1);
            return;
        };
        let length = unsafe { sample.total_sample_size() };
        let mut data = vec![0u8; length];
        if let Some(destination) = NonNull::new(data.as_mut_ptr().cast::<c_void>()) {
            let copy_status = unsafe { block.copy_data_bytes(0, length, destination) };
            if copy_status != 0 {
                *state
                    .error
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(copy_status);
                return;
            }
        }

        let packetization = match state.codec {
            Codec::H264 | Codec::Hevc => Packetization::LengthPrefixed { nal_length_size: 4 },
            _ => Packetization::Elementary,
        };
        let is_sync = encoded_packet_is_sync(&state.codec, &data, packetization);
        let pts = media_time_from_cm(unsafe { sample.presentation_time_stamp() });
        let dts = media_time_from_cm(unsafe { sample.decode_time_stamp() });
        let duration = media_time_from_cm(unsafe { sample.duration() });
        state
            .packets
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(EncodedPacket {
                stream_index: 0,
                pts: pts.map(|time| time.rescale(state.time_base)),
                dts: dts.map(|time| time.rescale(state.time_base)),
                duration: duration.map(|time| time.rescale(state.time_base)),
                is_sync,
                data,
                packetization,
            });

        let mut configuration = state
            .configuration
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if configuration.is_none() {
            *configuration = unsafe { sample.format_description() }
                .and_then(|description| sample_description_atom(&description, &state.codec));
        }
    }));
}

pub struct VideoToolboxEncoder {
    settings: EncoderSettings,
    callback_state: Box<EncoderCallbackState>,
    session: CFRetained<VTCompressionSession>,
    finalized: bool,
}

impl std::fmt::Debug for VideoToolboxEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VideoToolboxEncoder")
            .field("settings", &self.settings)
            .field("finalized", &self.finalized)
            .finish_non_exhaustive()
    }
}

impl VideoToolboxEncoder {
    pub fn new(settings: EncoderSettings) -> Result<Self> {
        let codec_type = video_toolbox_codec(&settings.codec)?;
        let mut callback_state = Box::new(EncoderCallbackState {
            codec: settings.codec.clone(),
            time_base: settings.time_base,
            packets: Mutex::new(Vec::new()),
            configuration: Mutex::new(None),
            error: Mutex::new(None),
        });
        let mut raw_session = ptr::null_mut();
        let callback: VTCompressionOutputCallback = Some(compression_callback);
        // SAFETY: All pointers are valid out-pointers/refcons, and the boxed
        // callback state is stable until session completion and invalidation.
        let status = unsafe {
            VTCompressionSession::create(
                None,
                i32::try_from(settings.width)
                    .map_err(|_| VideoError::InvalidData("encoder width exceeds i32".into()))?,
                i32::try_from(settings.height)
                    .map_err(|_| VideoError::InvalidData("encoder height exceeds i32".into()))?,
                codec_type,
                None,
                None,
                None,
                callback,
                (&mut *callback_state as *mut EncoderCallbackState).cast(),
                NonNull::from(&mut raw_session),
            )
        };
        check_status("VTCompressionSessionCreate", status)?;
        let raw_session = NonNull::new(raw_session).ok_or_else(|| {
            VideoError::InvalidData("VideoToolbox returned a null encoder session".into())
        })?;
        // SAFETY: Create returned a +1 retained object.
        let session = unsafe { CFRetained::from_raw(raw_session) };
        apply_encoder_settings(&session, &settings)?;
        // SAFETY: Session is fully configured and valid.
        check_status("VTCompressionSessionPrepareToEncodeFrames", unsafe {
            session.prepare_to_encode_frames()
        })?;
        Ok(Self {
            settings,
            callback_state,
            session,
            finalized: false,
        })
    }

    fn finish(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        // SAFETY: kCMTimeInvalid requests all pending frames.
        let status = unsafe { self.session.complete_frames(kCMTimeInvalid) };
        check_status("VTCompressionSessionCompleteFrames", status)?;
        self.finalized = true;
        if let Some(status) = self
            .callback_state
            .error
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            return Err(VideoError::platform("VTCompressionOutputCallback", status));
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl EncodeSession for VideoToolboxEncoder {
    async fn push_frame(&mut self, frame: VideoFrame) -> Result<()> {
        if self.finalized {
            return Err(VideoError::InvalidData(
                "cannot push a frame after encoder finalization".into(),
            ));
        }
        if frame.width != self.settings.width || frame.height != self.settings.height {
            return Err(VideoError::InvalidData(format!(
                "encoder expected {}x{}, got {}x{}",
                self.settings.width, self.settings.height, frame.width, frame.height
            )));
        }
        let FrameBuffer::MacOS(buffer) = frame.buffer else {
            return Err(VideoError::UnsupportedCodec(
                "VideoToolbox encoding requires a CVPixelBuffer-backed frame".into(),
            ));
        };
        let pts = cm_time(frame.pts)?;
        let duration = match frame.duration {
            Some(duration) => cm_time(duration)?,
            None => unsafe { kCMTimeInvalid },
        };
        // SAFETY: The CVPixelBuffer is retained for the call; VideoToolbox
        // retains it further if asynchronous work outlives this method.
        let status = unsafe {
            self.session.encode_frame(
                buffer.as_cv_pixel_buffer(),
                pts,
                duration,
                None,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        check_status("VTCompressionSessionEncodeFrame", status)
    }

    async fn finalize(mut self: Box<Self>) -> Result<EncodedVideo> {
        self.finish()?;
        let packets = std::mem::take(
            &mut *self
                .callback_state
                .packets
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        let description = self
            .callback_state
            .configuration
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .unwrap_or_default();
        let configuration =
            DecoderConfiguration::from_codec_private(self.settings.codec.clone(), description)?;
        Ok(EncodedVideo {
            configuration,
            packets,
        })
    }
}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        let _ = self.finish();
        // SAFETY: Completion above drains all callbacks before invalidation.
        unsafe { self.session.invalidate() };
    }
}

fn make_format_description(stream: &crate::StreamInfo) -> Result<CFRetained<CMFormatDescription>> {
    let configuration = &stream.decoder_configuration;
    let mut raw_description: *const CMFormatDescription = ptr::null();
    let output = NonNull::from(&mut raw_description);
    let status = match &configuration.codec {
        Codec::H264 => {
            if configuration.parameter_sets.len() < 2 {
                return Err(VideoError::InvalidData(
                    "H.264 decode requires SPS and PPS codec-private data".into(),
                ));
            }
            let mut pointers = parameter_set_pointers(configuration)?;
            let mut sizes: Vec<usize> = configuration
                .parameter_sets
                .iter()
                .map(|set| set.bytes.len())
                .collect();
            unsafe {
                CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    None,
                    pointers.len(),
                    NonNull::new(pointers.as_mut_ptr()).expect("parameter sets are non-empty"),
                    NonNull::new(sizes.as_mut_ptr()).expect("parameter sets are non-empty"),
                    nal_length_size(configuration),
                    output,
                )
            }
        }
        Codec::Hevc => {
            if configuration.parameter_sets.len() < 3 {
                return Err(VideoError::InvalidData(
                    "HEVC decode requires VPS, SPS, and PPS codec-private data".into(),
                ));
            }
            let mut pointers = parameter_set_pointers(configuration)?;
            let mut sizes: Vec<usize> = configuration
                .parameter_sets
                .iter()
                .map(|set| set.bytes.len())
                .collect();
            unsafe {
                CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                    None,
                    pointers.len(),
                    NonNull::new(pointers.as_mut_ptr()).expect("parameter sets are non-empty"),
                    NonNull::new(sizes.as_mut_ptr()).expect("parameter sets are non-empty"),
                    nal_length_size(configuration),
                    None,
                    output,
                )
            }
        }
        Codec::ProRes(_) => unsafe {
            CMVideoFormatDescriptionCreate(
                None,
                video_toolbox_codec(&configuration.codec)?,
                i32::try_from(stream.width.unwrap_or(0))
                    .map_err(|_| VideoError::InvalidData("video width exceeds i32".into()))?,
                i32::try_from(stream.height.unwrap_or(0))
                    .map_err(|_| VideoError::InvalidData("video height exceeds i32".into()))?,
                None,
                output,
            )
        },
        codec => {
            return Err(VideoError::UnsupportedCodec(format!(
                "VideoToolbox decoder does not support {codec:?}"
            )));
        }
    };
    check_status("CMVideoFormatDescriptionCreate", status)?;
    let pointer = NonNull::new(raw_description.cast_mut()).ok_or_else(|| {
        VideoError::InvalidData("CoreMedia returned a null format description".into())
    })?;
    // SAFETY: CoreMedia Create functions return a +1 retained object.
    decoder_color_description(unsafe { CFRetained::from_raw(pointer) }, stream)
}

fn decoder_color_description(
    description: CFRetained<CMFormatDescription>,
    stream: &crate::StreamInfo,
) -> Result<CFRetained<CMFormatDescription>> {
    use objc2_core_media::{
        CMVideoFormatDescriptionGetDimensions, kCMFormatDescriptionExtension_ColorPrimaries,
        kCMFormatDescriptionExtension_FullRangeVideo,
        kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionExtension_YCbCrMatrix,
    };
    // CoreMedia extensions are CFString -> property-list objects. Preserve
    // codec-private atoms and parsed dimensions while supplying the same
    // stream color interpretation used by the software conversion owner.
    let source = unsafe { description.extensions() };
    let empty = CFDictionary::<CFString, CFType>::empty();
    let source = source.as_deref().unwrap_or(empty.as_opaque());
    let extensions = unsafe { CFMutableDictionary::new_copy(None, 0, Some(source)) }
        .ok_or_else(|| VideoError::InvalidData("Cannot copy decoder format extensions".into()))?;
    // SAFETY: CoreMedia guarantees the types above; the copy retains CFType callbacks.
    let typed = unsafe { extensions.cast_unchecked::<CFString, CFType>() };
    typed.set(
        unsafe { kCMFormatDescriptionExtension_YCbCrMatrix },
        ycbcr_matrix_value(stream.color.matrix.for_rgb())?,
    );
    typed.set(
        unsafe { kCMFormatDescriptionExtension_FullRangeVideo },
        CFBoolean::new(stream.color.range == crate::ColorRange::Full),
    );
    if stream.color.primaries != crate::ColorPrimaries::Unspecified {
        typed.set(
            unsafe { kCMFormatDescriptionExtension_ColorPrimaries },
            color_primaries_value(stream.color.primaries)?,
        );
    }
    if stream.color.transfer != crate::TransferCharacteristic::Unspecified {
        typed.set(
            unsafe { kCMFormatDescriptionExtension_TransferFunction },
            transfer_function_value(stream.color.transfer)?,
        );
    }
    let dimensions = unsafe { CMVideoFormatDescriptionGetDimensions(&description) };
    let mut raw = ptr::null();
    // SAFETY: retained extension dictionary, parsed coded dimensions and a
    // valid out-pointer; Create copies/retains its property-list contents.
    let status = unsafe {
        CMVideoFormatDescriptionCreate(
            None,
            video_toolbox_codec(&stream.codec)?,
            dimensions.width,
            dimensions.height,
            Some(extensions.as_ref()),
            NonNull::from(&mut raw),
        )
    };
    check_status("CMVideoFormatDescriptionCreate colors", status)?;
    let raw = NonNull::new(raw.cast_mut())
        .ok_or_else(|| VideoError::InvalidData("Missing colored decoder format".into()))?;
    // SAFETY: successful Create returned a +1 retained format description.
    Ok(unsafe { CFRetained::from_raw(raw) })
}

fn make_decompression_session(
    format_description: &CMFormatDescription,
    callback_state: &mut DecoderCallbackState,
    destination: Destination,
) -> Result<CFRetained<VTDecompressionSession>> {
    let callback_record = VTDecompressionOutputCallbackRecord {
        decompressionOutputCallback: Some(decompression_callback),
        decompressionOutputRefCon: (callback_state as *mut DecoderCallbackState).cast(),
    };
    let destination_attributes = match destination {
        Destination::Native => None,
        Destination::Bgra { width, height } => {
            Some(destination_image_buffer_attributes(width, height)?)
        }
        Destination::Planes(format) => Some(gpu_planes_attributes(format)?),
    };
    let mut raw_session = ptr::null_mut();
    // SAFETY: Callback record, attributes and out-pointer are valid for the
    // call; the pointed-to state remains stable for the entire returned
    // session.
    let status = unsafe {
        VTDecompressionSession::create(
            None,
            format_description,
            None,
            destination_attributes
                .as_deref()
                .map(CFDictionary::as_opaque),
            &callback_record,
            NonNull::from(&mut raw_session),
        )
    };
    check_status("VTDecompressionSessionCreate", status)?;
    let pointer = NonNull::new(raw_session).ok_or_else(|| {
        VideoError::InvalidData("VideoToolbox returned a null decoder session".into())
    })?;
    // SAFETY: VideoToolbox Create returned a +1 retained object.
    Ok(unsafe { CFRetained::from_raw(pointer) })
}

/// What a decompression session asks VideoToolbox to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Destination {
    /// The decoder's own layout, which may be compressed (`&8v0`, `&xv0`).
    Native,
    /// 32BGRA of exactly this size, for a CPU read.
    Bgra { width: u32, height: u32 },
    /// Metal-compatible bi-planar Y'CbCr of this pixel format at the source
    /// size, for a GPU that reads the planes in place.
    Planes(u32),
}

/// The uncompressed bi-planar layout a GPU reads a stream's pictures from:
/// its own depth and range; 4:2:0 for H.264 and HEVC, 4:2:2 for ProRes 422.
/// ProRes 4444 may carry alpha, which bi-planar Y'CbCr has no room for.
fn gpu_planes_format(stream: &crate::StreamInfo) -> Result<u32> {
    use crate::{Codec, ColorRange, ProResProfile};
    use objc2_core_video::{
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
        kCVPixelFormatType_422YpCbCr10BiPlanarFullRange,
        kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange,
    };
    let full = stream.color.range == ColorRange::Full;
    let deep = stream.color.bits_per_channel.is_some_and(|bits| bits > 8);
    Ok(match &stream.codec {
        Codec::ProRes(ProResProfile::P4444 | ProResProfile::P4444Xq) => {
            return Err(VideoError::BackendUnavailable(
                "ProRes 4444 keeps its alpha on the CPU path".into(),
            ));
        }
        Codec::ProRes(_) if full => kCVPixelFormatType_422YpCbCr10BiPlanarFullRange,
        Codec::ProRes(_) => kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange,
        _ => match (deep, full) {
            (false, false) => kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            (false, true) => kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            (true, false) => kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            (true, true) => kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        },
    })
}

/// Destination attributes for GPU planes: `format`, IOSurface-backed and
/// laid out for Metal textures over its planes.
fn gpu_planes_attributes(format: u32) -> Result<CFRetained<CFDictionary<CFString, CFType>>> {
    let format = CFNumber::new_i32(
        i32::try_from(format)
            .map_err(|_| VideoError::InvalidData("pixel format tag exceeds i32".into()))?,
    );
    let io_surface_properties: CFRetained<CFDictionary<CFString, CFString>> = CFDictionary::empty();
    // SAFETY: the CoreVideo key statics are immutable, initialized strings.
    let (format_key, io_surface_key, metal_key) = unsafe {
        (
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelBufferIOSurfacePropertiesKey,
            kCVPixelBufferMetalCompatibilityKey,
        )
    };
    let metal = CFBoolean::new(true);
    Ok(CFDictionary::from_slices(
        &[format_key, io_surface_key, metal_key],
        &[
            format.as_ref() as &CFType,
            io_surface_properties.as_ref() as &CFType,
            metal.as_ref() as &CFType,
        ],
    ))
}

/// Destination image buffer attributes asking VideoToolbox for scaled,
/// IOSurface-backed output of exactly `width`×`height` — as 32BGRA. Left to
/// choose, the decoder hands back its lossy-compressed native layout
/// (`&8v0`), which no CPU reader can open; a scaled picture exists to be
/// read on the CPU, and BGRA is the one format that read is a byte shuffle.
fn destination_image_buffer_attributes(
    width: u32,
    height: u32,
) -> Result<CFRetained<CFDictionary<CFString, CFType>>> {
    if width == 0 || height == 0 {
        return Err(VideoError::InvalidData(
            "decoder output size must be non-zero".into(),
        ));
    }
    let width = CFNumber::new_i32(
        i32::try_from(width)
            .map_err(|_| VideoError::InvalidData("output width exceeds i32".into()))?,
    );
    let height = CFNumber::new_i32(
        i32::try_from(height)
            .map_err(|_| VideoError::InvalidData("output height exceeds i32".into()))?,
    );
    let format = CFNumber::new_i32(
        i32::try_from(kCVPixelFormatType_32BGRA)
            .map_err(|_| VideoError::InvalidData("pixel format tag exceeds i32".into()))?,
    );
    let io_surface_properties: CFRetained<CFDictionary<CFString, CFString>> = CFDictionary::empty();
    // SAFETY: the CoreVideo key statics are immutable, initialized strings.
    let (width_key, height_key, format_key, io_surface_key) = unsafe {
        (
            kCVPixelBufferWidthKey,
            kCVPixelBufferHeightKey,
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelBufferIOSurfacePropertiesKey,
        )
    };
    Ok(CFDictionary::from_slices(
        &[width_key, height_key, format_key, io_surface_key],
        &[
            width.as_ref() as &CFType,
            height.as_ref() as &CFType,
            format.as_ref() as &CFType,
            io_surface_properties.as_ref() as &CFType,
        ],
    ))
}

thread_local! {
    /// One pixel-transfer session per engine thread: a VideoToolbox session is
    /// not for concurrent use, and every engine operation already runs on a
    /// thread of its own. Created on first use, kept for the thread's life.
    static PIXEL_TRANSFER: RefCell<Option<CFRetained<VTPixelTransferSession>>> =
        const { RefCell::new(None) };
}

fn make_pixel_transfer_session() -> Result<CFRetained<VTPixelTransferSession>> {
    let mut raw_session = ptr::null_mut();
    // SAFETY: the out-pointer is valid for the call.
    let status = unsafe { VTPixelTransferSession::create(None, NonNull::from(&mut raw_session)) };
    check_status("VTPixelTransferSessionCreate", status)?;
    let pointer = NonNull::new(raw_session).ok_or_else(|| {
        VideoError::InvalidData("VideoToolbox returned a null pixel transfer session".into())
    })?;
    // SAFETY: VideoToolbox Create returned a +1 retained object.
    let session = unsafe { CFRetained::from_raw(pointer) };
    // Sizes are the caller's (aspect already fitted), so stretch to the
    // destination; average when shrinking, so a thumbnail is not decimated.
    // SAFETY: the property keys and values are immutable statics; the session
    // is live.
    check_status("VTSessionSetProperty(ScalingMode)", unsafe {
        VTSessionSetProperty(
            &session,
            kVTPixelTransferPropertyKey_ScalingMode,
            Some(kVTScalingMode_Normal),
        )
    })?;
    check_status("VTSessionSetProperty(DownsamplingMode)", unsafe {
        VTSessionSetProperty(
            &session,
            kVTPixelTransferPropertyKey_DownsamplingMode,
            Some(kVTDownsamplingMode_Average),
        )
    })?;
    Ok(session)
}

/// `source` resampled to `width`×`height` in `format` by VideoToolbox's
/// pixel-transfer session — scaling and pixel-format conversion in one
/// hardware pass into a fresh IOSurface-backed buffer. This is the seam that
/// keeps a decoded frame off the CPU on its way to an encoder or a thumbnail.
pub(crate) fn transfer_scaled(
    source: &CVPixelBuffer,
    width: u32,
    height: u32,
    format: u32,
) -> Result<CFRetained<CVPixelBuffer>> {
    let destination = crate::frame::allocate_pixel_buffer(width, height, format)?;
    PIXEL_TRANSFER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let session = match slot.as_ref() {
            Some(session) => session,
            None => slot.insert(make_pixel_transfer_session()?),
        };
        // SAFETY: both buffers are retained for the call; the session is live
        // and used from this thread only.
        let status = unsafe { session.transfer_image(source, &destination) };
        check_status("VTPixelTransferSessionTransferImage", status)
    })?;
    Ok(destination)
}

fn make_sample_buffer(
    packet: &EncodedPacket,
    format_description: &CMFormatDescription,
) -> Result<CFRetained<CMSampleBuffer>> {
    let data = match packet.packetization {
        Packetization::AnnexB => annex_b_to_length_prefixed(&packet.data)?,
        _ => packet.data.clone(),
    };
    if data.is_empty() {
        return Err(VideoError::InvalidData(
            "cannot decode an empty packet".into(),
        ));
    }
    let mut raw_block = ptr::null_mut();
    // SAFETY: A null memory block asks CoreMedia to allocate `data.len()`
    // bytes. The out-pointer and all sizes are valid.
    let status = unsafe {
        CMBlockBuffer::create_with_memory_block(
            None,
            ptr::null_mut(),
            data.len(),
            None,
            ptr::null(),
            0,
            data.len(),
            0,
            NonNull::from(&mut raw_block),
        )
    };
    check_status("CMBlockBufferCreateWithMemoryBlock", status)?;
    let raw_block = NonNull::new(raw_block)
        .ok_or_else(|| VideoError::InvalidData("CoreMedia returned a null block buffer".into()))?;
    // SAFETY: Create returned a +1 retained object.
    let block = unsafe { CFRetained::from_raw(raw_block) };
    let source = NonNull::new(data.as_ptr().cast_mut().cast::<c_void>())
        .expect("non-empty packet has a non-null pointer");
    check_status("CMBlockBufferReplaceDataBytes", unsafe {
        CMBlockBuffer::replace_data_bytes(source, &block, 0, data.len())
    })?;

    let timing = CMSampleTimingInfo {
        duration: packet
            .duration
            .map(cm_time)
            .transpose()?
            .unwrap_or(unsafe { kCMTimeInvalid }),
        presentationTimeStamp: packet
            .pts
            .map(cm_time)
            .transpose()?
            .unwrap_or(unsafe { kCMTimeInvalid }),
        decodeTimeStamp: packet
            .dts
            .map(cm_time)
            .transpose()?
            .unwrap_or(unsafe { kCMTimeInvalid }),
    };
    let sample_size = data.len();
    let mut raw_sample = ptr::null_mut();
    // SAFETY: Arrays each contain the one entry declared in their counts;
    // block and format description stay retained for creation.
    let status = unsafe {
        CMSampleBuffer::create_ready(
            None,
            Some(&block),
            Some(format_description),
            1,
            1,
            &timing,
            1,
            &sample_size,
            NonNull::from(&mut raw_sample),
        )
    };
    check_status("CMSampleBufferCreateReady", status)?;
    let pointer = NonNull::new(raw_sample)
        .ok_or_else(|| VideoError::InvalidData("CoreMedia returned a null sample buffer".into()))?;
    // SAFETY: Create returned a +1 retained object.
    Ok(unsafe { CFRetained::from_raw(pointer) })
}

fn parameter_set_pointers(configuration: &DecoderConfiguration) -> Result<Vec<NonNull<u8>>> {
    configuration
        .parameter_sets
        .iter()
        .map(|set| {
            NonNull::new(set.bytes.as_ptr().cast_mut()).ok_or_else(|| {
                VideoError::InvalidData(
                    "codec configuration contains an empty parameter set".into(),
                )
            })
        })
        .collect()
}

fn nal_length_size(configuration: &DecoderConfiguration) -> i32 {
    match configuration.packetization {
        Packetization::LengthPrefixed { nal_length_size } => i32::from(nal_length_size),
        _ => 4,
    }
}

fn video_toolbox_codec(codec: &Codec) -> Result<u32> {
    match codec {
        Codec::H264 => Ok(kCMVideoCodecType_H264),
        Codec::Hevc => Ok(kCMVideoCodecType_HEVC),
        Codec::ProRes(ProResProfile::Proxy) => Ok(kCMVideoCodecType_AppleProRes422Proxy),
        Codec::ProRes(ProResProfile::Lt) => Ok(kCMVideoCodecType_AppleProRes422LT),
        Codec::ProRes(ProResProfile::Standard | ProResProfile::Unknown) => {
            Ok(kCMVideoCodecType_AppleProRes422)
        }
        Codec::ProRes(ProResProfile::Hq) => Ok(kCMVideoCodecType_AppleProRes422HQ),
        Codec::ProRes(ProResProfile::P4444) => Ok(kCMVideoCodecType_AppleProRes4444),
        Codec::ProRes(ProResProfile::P4444Xq) => Ok(kCMVideoCodecType_AppleProRes4444XQ),
        codec => Err(VideoError::UnsupportedCodec(format!(
            "VideoToolbox does not support {codec:?}"
        ))),
    }
}

fn apply_encoder_settings(
    session: &VTCompressionSession,
    settings: &EncoderSettings,
) -> Result<()> {
    let is_inter_frame_codec = matches!(settings.codec, Codec::H264 | Codec::Hevc);
    if let Some(bit_rate) = settings.bit_rate.filter(|_| is_inter_frame_codec) {
        let value = CFNumber::new_i64(
            i64::try_from(bit_rate)
                .map_err(|_| VideoError::InvalidData("encoder bit rate exceeds i64".into()))?,
        );
        check_status("VTSessionSetProperty(AverageBitRate)", unsafe {
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_AverageBitRate,
                Some(&value),
            )
        })?;
    }
    if let Some(interval) = settings
        .max_keyframe_interval
        .filter(|_| is_inter_frame_codec)
    {
        let value = CFNumber::new_i32(
            i32::try_from(interval)
                .map_err(|_| VideoError::InvalidData("keyframe interval exceeds i32".into()))?,
        );
        check_status("VTSessionSetProperty(MaxKeyFrameInterval)", unsafe {
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_MaxKeyFrameInterval,
                Some(&value),
            )
        })?;
    }
    if let Some(profile) = settings.profile.as_deref() {
        let value = encoder_profile(&settings.codec, profile)?;
        check_status("VTSessionSetProperty(ProfileLevel)", unsafe {
            VTSessionSetProperty(session, kVTCompressionPropertyKey_ProfileLevel, Some(value))
        })?;
    }
    if is_inter_frame_codec {
        // Always stated: VideoToolbox documents the default as `true`, so a
        // bounded-GOP preset would silently regain B-frames and hand callers
        // that index by delivery position a reordered stream.
        let value = CFBoolean::new(settings.allow_frame_reordering);
        check_status("VTSessionSetProperty(AllowFrameReordering)", unsafe {
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_AllowFrameReordering,
                Some(value),
            )
        })?;
    }
    if let Some(color) = settings.color {
        for (key, value) in [
            (
                unsafe { kVTCompressionPropertyKey_ColorPrimaries },
                color_primaries_value(color.primaries)?,
            ),
            (
                unsafe { kVTCompressionPropertyKey_TransferFunction },
                transfer_function_value(color.transfer)?,
            ),
            (
                unsafe { kVTCompressionPropertyKey_YCbCrMatrix },
                ycbcr_matrix_value(color.matrix)?,
            ),
        ] {
            // A `kVTPropertyReadOnlyErr` here means the encoder enforces its
            // own colorimetry and our tags would be a lie. Surfaced, never
            // swallowed.
            check_status("VTSessionSetProperty(color)", unsafe {
                VTSessionSetProperty(session, key, Some(value))
            })?;
        }
    }
    if settings.insert_hdr_metadata {
        check_status("VTSessionSetProperty(HDRMetadataInsertionMode)", unsafe {
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_HDRMetadataInsertionMode,
                Some(kVTHDRMetadataInsertionMode_Auto),
            )
        })?;
        // The generated metadata describes what we encode; a source buffer's
        // dynamic metadata would describe the pre-scale original.
        let preserve = CFBoolean::new(false);
        check_status("VTSessionSetProperty(PreserveDynamicHDRMetadata)", unsafe {
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_PreserveDynamicHDRMetadata,
                Some(preserve),
            )
        })?;
    }
    Ok(())
}

/// CoreVideo colorimetry constants for the tags VideoToolbox writes onto the
/// compressed format description. Anything outside this table would be
/// mis-tagged rather than approximated, so it is refused.
fn color_primaries_value(primaries: crate::ColorPrimaries) -> Result<&'static CFString> {
    match primaries {
        crate::ColorPrimaries::Bt709 => Ok(unsafe { kCVImageBufferColorPrimaries_ITU_R_709_2 }),
        crate::ColorPrimaries::Bt2020 => Ok(unsafe { kCVImageBufferColorPrimaries_ITU_R_2020 }),
        crate::ColorPrimaries::DisplayP3 => Ok(unsafe { kCVImageBufferColorPrimaries_P3_D65 }),
        crate::ColorPrimaries::Smpte170M | crate::ColorPrimaries::Smpte240M => {
            Ok(unsafe { objc2_core_video::kCVImageBufferColorPrimaries_SMPTE_C })
        }
        crate::ColorPrimaries::Bt470Bg => {
            Ok(unsafe { objc2_core_video::kCVImageBufferColorPrimaries_EBU_3213 })
        }
        other => Err(VideoError::UnsupportedCodec(format!(
            "VideoToolbox color primaries {other:?}"
        ))),
    }
}

fn transfer_function_value(transfer: crate::TransferCharacteristic) -> Result<&'static CFString> {
    match transfer {
        crate::TransferCharacteristic::Bt709 | crate::TransferCharacteristic::Smpte170M => {
            Ok(unsafe { kCVImageBufferTransferFunction_ITU_R_709_2 })
        }
        crate::TransferCharacteristic::Hlg => {
            Ok(unsafe { kCVImageBufferTransferFunction_ITU_R_2100_HLG })
        }
        crate::TransferCharacteristic::Pq => {
            Ok(unsafe { kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ })
        }
        crate::TransferCharacteristic::Srgb => Ok(unsafe { kCVImageBufferTransferFunction_sRGB }),
        crate::TransferCharacteristic::Linear => {
            Ok(unsafe { objc2_core_video::kCVImageBufferTransferFunction_Linear })
        }
        other => Err(VideoError::UnsupportedCodec(format!(
            "VideoToolbox transfer function {other:?}"
        ))),
    }
}

fn ycbcr_matrix_value(matrix: crate::ColorMatrix) -> Result<&'static CFString> {
    match matrix {
        crate::ColorMatrix::Bt709 => Ok(unsafe { kCVImageBufferYCbCrMatrix_ITU_R_709_2 }),
        crate::ColorMatrix::Bt470Bg | crate::ColorMatrix::Smpte170M => {
            Ok(unsafe { kCVImageBufferYCbCrMatrix_ITU_R_601_4 })
        }
        crate::ColorMatrix::Bt2020NonConstant => {
            Ok(unsafe { kCVImageBufferYCbCrMatrix_ITU_R_2020 })
        }
        other => Err(VideoError::UnsupportedCodec(format!(
            "VideoToolbox YCbCr matrix {other:?}"
        ))),
    }
}

fn encoder_profile(codec: &Codec, profile: &str) -> Result<&'static CFString> {
    let normalized = profile.to_ascii_lowercase();
    match (codec, normalized.as_str()) {
        (Codec::H264, "baseline") => Ok(unsafe { kVTProfileLevel_H264_Baseline_AutoLevel }),
        (Codec::H264, "main") => Ok(unsafe { kVTProfileLevel_H264_Main_AutoLevel }),
        (Codec::H264, "high") => Ok(unsafe { kVTProfileLevel_H264_High_AutoLevel }),
        (Codec::Hevc, "main") => Ok(unsafe { kVTProfileLevel_HEVC_Main_AutoLevel }),
        (Codec::Hevc, "main10") | (Codec::Hevc, "main-10") => {
            Ok(unsafe { kVTProfileLevel_HEVC_Main10_AutoLevel })
        }
        _ => Err(VideoError::UnsupportedCodec(format!(
            "VideoToolbox profile {profile:?} for {codec:?}"
        ))),
    }
}

fn cm_time(time: MediaTime) -> Result<CMTime> {
    let timescale = i32::try_from(time.time_base.denominator)
        .map_err(|_| VideoError::InvalidData("CMTime scale exceeds i32".into()))?;
    let value = time
        .value
        .checked_mul(time.time_base.numerator)
        .ok_or_else(|| VideoError::InvalidData("CMTime value overflow".into()))?;
    // SAFETY: TimeBase validates positive terms and timescale fits i32.
    Ok(unsafe { CMTime::new(value, timescale) })
}

fn media_time_from_cm(time: CMTime) -> Option<MediaTime> {
    let flags = time.flags;
    let timescale = time.timescale;
    let value = time.value;
    if !flags.contains(CMTimeFlags::Valid) || timescale <= 0 {
        return None;
    }
    Some(MediaTime::new(
        value,
        TimeBase::new_unchecked(1, i64::from(timescale)),
    ))
}

fn check_status(operation: &'static str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(VideoError::platform(operation, status))
    }
}

fn annex_b_to_length_prefixed(data: &[u8]) -> Result<Vec<u8>> {
    let mut units = Vec::new();
    let mut cursor = find_start_code(data, 0);
    while let Some((start, width)) = cursor {
        let payload_start = start + width;
        let next = find_start_code(data, payload_start);
        let payload_end = next.map_or(data.len(), |(position, _)| position);
        if payload_end > payload_start {
            units.push(&data[payload_start..payload_end]);
        }
        cursor = next;
    }
    if units.is_empty() {
        return Err(VideoError::InvalidData(
            "Annex-B packet contains no start code".into(),
        ));
    }
    let mut output = Vec::with_capacity(data.len());
    for unit in units {
        let length = u32::try_from(unit.len())
            .map_err(|_| VideoError::InvalidData("NAL unit exceeds u32".into()))?;
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(unit);
    }
    Ok(output)
}

fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut index = from;
    while index + 3 <= data.len() {
        if data.get(index..index + 4) == Some(&[0, 0, 0, 1]) {
            return Some((index, 4));
        }
        if data.get(index..index + 3) == Some(&[0, 0, 1]) {
            return Some((index, 3));
        }
        index += 1;
    }
    None
}

fn encoded_packet_is_sync(codec: &Codec, data: &[u8], packetization: Packetization) -> bool {
    if matches!(codec, Codec::ProRes(_)) {
        return true;
    }
    let packet = EncodedPacket {
        stream_index: 0,
        pts: None,
        dts: None,
        duration: None,
        is_sync: false,
        data: data.to_vec(),
        packetization,
    };
    let Ok(annex_b) = packet.data_in_format(PacketFormat::AnnexB) else {
        return false;
    };
    let mut cursor = find_start_code(&annex_b, 0);
    while let Some((start, width)) = cursor {
        if let Some(header) = annex_b.get(start + width) {
            match codec {
                Codec::H264 if header & 0x1f == 5 => return true,
                Codec::Hevc if matches!((header >> 1) & 0x3f, 19..=21) => return true,
                _ => {}
            }
        }
        cursor = find_start_code(&annex_b, start + width);
    }
    false
}

fn sample_description_atom(description: &CMFormatDescription, codec: &Codec) -> Option<Vec<u8>> {
    let atom_name = match codec {
        Codec::H264 => "avcC",
        Codec::Hevc => "hvcC",
        _ => return Some(Vec::new()),
    };
    let extensions = unsafe { description.extensions() }?;
    // SAFETY: CoreMedia documents this property-list dictionary as
    // CFType-keyed and CFType-valued; the generated binding defaults its
    // otherwise-erased generic parameters to `Opaque`.
    let extensions = unsafe {
        CFRetained::cast_unchecked::<objc2_core_foundation::CFDictionary<CFType, CFType>>(
            extensions,
        )
    };
    let atoms_key: &CFType =
        unsafe { objc2_core_media::kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms };
    let atoms = extensions.get(atoms_key)?;
    let atoms = atoms
        .downcast::<objc2_core_foundation::CFDictionary>()
        .ok()?;
    // SAFETY: `SampleDescriptionExtensionAtoms` is specified as a property-list
    // dictionary, so its keys and values are Core Foundation objects.
    let atoms = unsafe { atoms.cast_unchecked::<CFType, CFType>() };
    let key = CFString::from_str(atom_name);
    let value = atoms.get(&*key)?;
    value.downcast::<CFData>().ok().map(|data| data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorInfo, CpuFrame, CpuPixelFormat};

    fn fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/generated/h264-gop4.mp4")
    }

    #[test]
    fn a_decoder_opened_with_an_output_size_delivers_frames_at_that_size() {
        let mut decoder = VideoToolboxDecoder::open_with(
            &fixture(),
            0,
            DecodeOptions {
                output_size: Some((64, 48)),
                ..Default::default()
            },
        )
        .unwrap();
        let frame = futures_executor::block_on(decoder.decode_next())
            .unwrap()
            .expect("the fixture has frames");
        assert_eq!((frame.width, frame.height), (64, 48));
        let FrameBuffer::MacOS(buffer) = &frame.buffer else {
            panic!("a VideoToolbox frame is native");
        };
        assert_eq!(
            objc2_core_video::CVPixelBufferGetPixelFormatType(buffer.as_cv_pixel_buffer()),
            kCVPixelFormatType_32BGRA,
            "scaled output is pinned to BGRA, never the decoder's lossy native layout"
        );
        let cpu = frame.to_cpu_rgb().unwrap();
        assert_eq!((cpu.width, cpu.height), (64, 48));
    }

    /// The two planes of a GPU-planes picture, row by row without padding.
    fn plane_bytes(frame: &VideoFrame) -> Vec<u8> {
        use objc2_core_video::{
            CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
            CVPixelBufferGetHeightOfPlane, CVPixelBufferGetWidthOfPlane,
            CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
        };
        let FrameBuffer::MacOS(buffer) = &frame.buffer else {
            panic!("not GPU planes");
        };
        let buffer = buffer.as_cv_pixel_buffer();
        let flags = CVPixelBufferLockFlags::ReadOnly;
        // SAFETY: `frame` retains the buffer through the read lock; rows are
        // read inside each plane's extent before the matching unlock.
        unsafe {
            assert_eq!(CVPixelBufferLockBaseAddress(buffer, flags), 0);
            let mut bytes = Vec::new();
            for (plane, per_sample) in [(0, 1), (1, 2)] {
                let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane).cast::<u8>();
                let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
                let row = CVPixelBufferGetWidthOfPlane(buffer, plane) * per_sample;
                for y in 0..CVPixelBufferGetHeightOfPlane(buffer, plane) {
                    bytes.extend_from_slice(std::slice::from_raw_parts(base.add(y * stride), row));
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, flags);
            bytes
        }
    }

    /// An open-GOP HEVC stream, as x265 and many cameras write it: every
    /// second a CRA whose RASL leading pictures need the GOP before it. A
    /// jump decodes from the CRA before the target, as libavcodec does, and
    /// presents the picture a decode from the start does. Kill: going back a
    /// GOP for each missing reference — down to the first picture of the
    /// file, so a jump costs more the later it lands.
    #[test]
    fn a_jump_in_an_open_gop_stream_starts_at_the_cra_before_it() {
        let root = std::env::temp_dir().join(format!("ferment-open-gop-{}", std::process::id()));
        std::fs::create_dir(&root).expect("new owned fixture directory");
        let path = root.join("open-gop.mp4");
        assert!(
            std::process::Command::new("ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=128x72:rate=30:duration=4",
                    "-c:v",
                    "libx265",
                    "-x265-params",
                    "keyint=30:min-keyint=30:scenecut=0:open-gop=1:bframes=3:log-level=error",
                    "-pix_fmt",
                    "yuv420p",
                ])
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let options = DecodeOptions {
            output: crate::DecodeOutput::GpuPlanes,
            ..Default::default()
        };
        let target = MediaTime::new(3_500, TimeBase::MILLISECONDS);
        let mut start = VideoToolboxDecoder::open_with(&path, 0, options).unwrap();
        let expected = loop {
            let frame = futures_executor::block_on(start.decode_next())
                .unwrap()
                .expect("a picture at the target");
            if frame.pts.seconds() >= target.seconds() {
                break frame;
            }
        };
        let mut jumping = VideoToolboxDecoder::open_with(&path, 0, options).unwrap();
        futures_executor::block_on(jumping.seek(target)).unwrap();
        let frame = futures_executor::block_on(jumping.decode_next())
            .unwrap()
            .expect("a picture at the target");
        assert_eq!(frame.pts.seconds(), expected.pts.seconds());
        assert_eq!(jumping.rewinds, 0, "the jump went back to an earlier GOP");
        assert!(
            plane_bytes(&frame) == plane_bytes(&expected),
            "another picture"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Kill: subtract a guessed frame period or let a failed seek retry the same access point.
    #[test]
    fn reference_recovery_moves_one_exact_tick_before_the_previous_access_point() {
        let base = TimeBase::new_unchecked(1, 15_360);
        let cra_dts = MediaTime::new(125_952, base);
        assert_eq!(
            previous_sync_probe(cra_dts, None).unwrap(),
            MediaTime::new(125_951, base)
        );
        assert!(previous_sync_probe(cra_dts, Some(cra_dts)).is_err());
        assert!(previous_sync_probe(MediaTime::new(125_953, base), Some(cra_dts)).is_err());
        assert_eq!(
            previous_sync_probe(MediaTime::new(-1_024, base), Some(cra_dts)).unwrap(),
            MediaTime::new(-1_025, base)
        );
        assert!(previous_sync_probe(MediaTime::new(i64::MIN, base), None).is_err());
    }

    /// Kill: reject a native layout which has no direct CPU reader rather than use pixel transfer.
    #[test]
    fn cpu_readback_converts_a_non_direct_native_layout_without_losing_channels() {
        let red = CpuFrame::new(
            8,
            8,
            32,
            CpuPixelFormat::Bgra8,
            [0u8, 0, 255, 255].repeat(64),
        )
        .unwrap();
        let source = MacPixelBuffer::from_cpu_bgra(&red).unwrap();
        let uncommon = transfer_scaled(
            source.as_cv_pixel_buffer(),
            8,
            8,
            objc2_core_video::kCVPixelFormatType_32ARGB,
        )
        .unwrap();
        let frame = VideoFrame {
            width: 8,
            height: 8,
            pts: MediaTime::ZERO,
            duration: None,
            color: ColorInfo::default(),
            buffer: FrameBuffer::MacOS(MacPixelBuffer::new(uncommon)),
        };
        let rgb = frame.to_cpu_rgb().unwrap().to_rgb24();
        assert_eq!(rgb.data.as_ref(), [255u8, 0, 0].repeat(64));
    }

    /// `FERMENT_BENCH_CLIP=/path/to.mp4 cargo test -p ferment-engine-media
    /// bench_decode -- --ignored --nocapture`: the decode floor of a clip,
    /// frames left on the GPU.
    #[test]
    #[ignore]
    fn bench_decode_on_env_clip() {
        let Ok(clip) = std::env::var("FERMENT_BENCH_CLIP") else {
            eprintln!("FERMENT_BENCH_CLIP not set");
            return;
        };
        // `FERMENT_BENCH_DECODE_SIZE=192x192` asks for scaled output.
        let output_size = std::env::var("FERMENT_BENCH_DECODE_SIZE")
            .ok()
            .and_then(|size| {
                let (width, height) = size.split_once('x')?;
                Some((width.parse().ok()?, height.parse().ok()?))
            });
        let mut decoder = VideoToolboxDecoder::open_with(
            Path::new(&clip),
            0,
            DecodeOptions {
                output_size,
                ..Default::default()
            },
        )
        .unwrap();
        let started = std::time::Instant::now();
        let mut frames = 0usize;
        while futures_executor::block_on(decoder.decode_next())
            .unwrap()
            .is_some()
        {
            frames += 1;
        }
        eprintln!(
            "BENCH decode only [{output_size:?}]: {:.2}s wall, {frames} frames",
            started.elapsed().as_secs_f64()
        );
    }

    #[test]
    fn the_pixel_transfer_scales_a_native_frame_without_a_cpu_round_trip() {
        // A solid red 8x8 BGRA frame, uploaded once; every scaled copy of it
        // must still be red, and the small read-back is the only download.
        let red = CpuFrame::new(
            8,
            8,
            32,
            CpuPixelFormat::Bgra8,
            [0u8, 0, 255, 255].repeat(64),
        )
        .unwrap();
        let native = VideoFrame {
            width: 8,
            height: 8,
            pts: MediaTime::ZERO,
            duration: None,
            color: ColorInfo::default(),
            buffer: FrameBuffer::MacOS(MacPixelBuffer::from_cpu_bgra(&red).unwrap()),
        };

        let encodable = native.encodable_scaled(4, 2, ColorInfo::default()).unwrap();
        assert_eq!((encodable.width, encodable.height), (4, 2));
        assert!(matches!(encodable.buffer, FrameBuffer::MacOS(_)));
        let rgb = encodable.to_cpu_rgb().unwrap();
        assert_eq!((rgb.width, rgb.height), (4, 2));
        let pixel = &rgb.data[..3];
        assert!(
            pixel[0] > 200 && pixel[1] < 40 && pixel[2] < 40,
            "not red: {pixel:?}"
        );

        let thumb = native.scaled_cpu_rgb(2, 2).unwrap();
        assert_eq!(
            (thumb.width, thumb.height, thumb.format),
            (2, 2, CpuPixelFormat::Rgb24)
        );
        assert!(
            thumb.data[0] > 200 && thumb.data[1] < 40,
            "not red: {:?}",
            &thumb.data[..3]
        );

        // Same size: the decoder's buffer is handed over as it is.
        let same = native.encodable_scaled(8, 8, ColorInfo::default()).unwrap();
        assert!(matches!(same.buffer, FrameBuffer::MacOS(_)));
    }

    #[test]
    fn annex_b_conversion_accepts_three_and_four_byte_start_codes() {
        let converted =
            annex_b_to_length_prefixed(&[0, 0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68, 0xbb]).unwrap();
        assert_eq!(
            converted,
            vec![0, 0, 0, 2, 0x67, 0xaa, 0, 0, 0, 2, 0x68, 0xbb]
        );
    }

    #[test]
    fn keyframe_probe_understands_h264_and_hevc() {
        assert!(encoded_packet_is_sync(
            &Codec::H264,
            &[0, 0, 0, 1, 0x65, 1],
            Packetization::AnnexB,
        ));
        assert!(encoded_packet_is_sync(
            &Codec::Hevc,
            &[0, 0, 0, 1, 19 << 1, 1],
            Packetization::AnnexB,
        ));
    }
}
