//! Shared vImage HQ RGB sampling; retained here for the DINO family boundary.

pub(super) fn resize_rgb(
    rgb: Vec<u8>,
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<u8> {
    crate::rgb_hq::resize_rgb(rgb, width, height, target_width, target_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kill: lose edge extension/DC normalization or overflow on a one-pixel input.
    #[test]
    fn constant_color_survives_upscale_downscale_and_single_pixel_edges() {
        for (width, height, target_width, target_height) in
            [(1, 1, 13, 7), (7, 13, 13, 7), (640, 360, 224, 126)]
        {
            let color = [19, 137, 241];
            let actual = resize_rgb(
                color.repeat(width * height),
                width,
                height,
                target_width,
                target_height,
            );
            assert_eq!(actual, color.repeat(target_width * target_height));
        }
    }

    /// Kill: filter an identity input or swap its packed channels.
    #[test]
    fn an_identity_resize_preserves_each_original_byte() {
        let rgb: Vec<_> = (0..13 * 7 * 3).map(|i| (i * 73) as u8).collect();
        assert_eq!(resize_rgb(rgb.clone(), 13, 7, 13, 7), rgb);
    }
}
