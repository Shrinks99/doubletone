#!/usr/bin/env python

import argparse
import imageio.v3 as iio
import logging as log
import numpy as np
import re
import scipy as sp
import scipy.fft


def hex_color(hex):
    m = re.fullmatch(r"#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})", hex)
    if not m:
        raise ValueError("Failed to parse hex color: {hex}")
    return np.array([int(m[i], base=16) for i in [1, 2, 3]], dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(
        description="Filters a halftone pattern to produce an image more suitable for digital displays"
    )
    parser.add_argument("image", help="Image to filter")
    parser.add_argument(
        "-l",
        "--log-level",
        help="Verbosity of logging",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        type=str.upper,
        default=log.INFO,
    )
    parser.add_argument(
        "-c",
        "--cyan-in",
        help="Color of cyan in input image",
        type=hex_color,
        default="#00ffff",
    )
    parser.add_argument(
        "-m",
        "--magenta-in",
        help="Color of magenta in input image",
        type=hex_color,
        default="#ff00ff",
    )
    parser.add_argument(
        "-y",
        "--yellow-in",
        help="Color of yellow in input image",
        type=hex_color,
        default="#ffff00",
    )
    parser.add_argument(
        "-k",
        "--black-in",
        help="Color of black in input image",
        type=hex_color,
        default="#000000",
    )
    parser.add_argument(
        "-C",
        "--cyan-out",
        help="Color of cyan in output image",
        type=hex_color,
        default=None,
    )
    parser.add_argument(
        "-M",
        "--magenta-out",
        help="Color of magenta in output image",
        type=hex_color,
        default=None,
    )
    parser.add_argument(
        "-Y",
        "--yellow-out",
        help="Color of yellow in output image",
        type=hex_color,
        default=None,
    )
    parser.add_argument(
        "-K",
        "--black-out",
        help="Color of black in output image",
        type=hex_color,
        default=None,
    )
    parser.add_argument(
        "-t",
        "--black-threshold",
        help="Threshold to consider pixel black",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path",
        type=str,
        default="filtered.png",
    )
    parser.add_argument(
        "--notch-radius",
        help="Sigma of Gaussian notch filters in frequency bins (larger = more aggressive)",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--lowpass",
        help="Apply a Butterworth low-pass at this fraction of the detected screen frequency "
        "(e.g. 0.9 cuts just below the halftone fundamental; 0 disables). "
        "Catches all halftone energy including cross-products and harmonics.",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--lowpass-order",
        help="Steepness of the low-pass rolloff (higher = sharper cutoff)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-harmonics",
        help="Number of harmonic orders to suppress beyond fundamentals",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--detection-threshold",
        help="Peak detection sensitivity in standard deviations above mean",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--screen-freq",
        help="Override auto-detected screen frequency (cycles/pixel)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--no-auto-detect",
        help="Disable auto-detection; requires --screen-freq and angle arguments",
        action="store_true",
    )
    parser.add_argument(
        "--cyan-angle",
        help="Angle of halftone screen for cyan in turns (used with --no-auto-detect)",
        type=float,
        default=3 / 16,
    )
    parser.add_argument(
        "--magenta-angle",
        help="Angle of halftone screen for magenta in turns (used with --no-auto-detect)",
        type=float,
        default=2 / 16,
    )
    parser.add_argument(
        "--yellow-angle",
        help="Angle of halftone screen for yellow in turns (used with --no-auto-detect)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--black-angle",
        help="Angle of halftone screen for black in turns (used with --no-auto-detect)",
        type=float,
        default=1 / 16,
    )
    parser.add_argument(
        "--debug-spectrum",
        help="Save debug images of the frequency spectrum with detected peaks",
        action="store_true",
    )

    args = parser.parse_args()
    handle_default_colors_out(args)
    log.basicConfig(level=args.log_level)

    log.info("loading image")
    image = load_image(args.image)

    log.info("converting to linear RGB")
    rgb_intensity = intensity_from_srgb(image)
    del image

    # detect halftone parameters
    screen_freq = 0.0
    if args.no_auto_detect:
        if args.screen_freq is None:
            log.critical("--screen-freq is required when using --no-auto-detect")
            exit(1)
        log.info("using manual halftone parameters")
        screen_freq = args.screen_freq
        angles = [args.cyan_angle, args.magenta_angle, args.yellow_angle, args.black_angle]
        peak_locations = build_peaks_from_manual_params(
            screen_freq, angles, rgb_intensity.shape[:2], args.max_harmonics
        )
    else:
        log.info("detecting halftone parameters")
        screen_freq, peak_locations = auto_detect_screen(
            rgb_intensity, args.detection_threshold, args.max_harmonics
        )
        if screen_freq > 0:
            log.info(f"detected screen frequency: {screen_freq:.4f} cycles/pixel")
        log.info(f"detected {len(peak_locations)} frequency peaks to suppress")
        if len(peak_locations) < 2:
            log.warning("few halftone peaks detected; image may not be halftoned")

    if args.debug_spectrum:
        luminance = (
            0.2126 * rgb_intensity[:, :, 0]
            + 0.7152 * rgb_intensity[:, :, 1]
            + 0.0722 * rgb_intensity[:, :, 2]
        )
        save_debug_spectrum(luminance, peak_locations, "debug_spectrum.png")
        log.info("saved debug spectrum to debug_spectrum.png")

    log.info("converting to CMYK")
    cmy = cmy_from_rgb(rgb_intensity, args.cyan_in, args.magenta_in, args.yellow_in)
    black_intensity = intensity_from_srgb(args.black_in.copy())
    k = (
        np.linalg.norm(rgb_intensity - black_intensity, axis=2) < args.black_threshold
    ).astype(np.float32)
    del rgb_intensity

    # compute low-pass cutoff from detected screen frequency
    lowpass_cutoff = 0.0
    if args.lowpass > 0 and screen_freq > 0:
        lowpass_cutoff = args.lowpass * screen_freq
        log.info(f"low-pass cutoff: {lowpass_cutoff:.4f} cycles/pixel "
                 f"({args.lowpass:.0%} of screen frequency)")

    log.info("descreening channels via FFT filtering")
    c = descreen_channel_fft(cmy[:, :, 0], peak_locations, args.notch_radius,
                             lowpass_cutoff, args.lowpass_order)
    m = descreen_channel_fft(cmy[:, :, 1], peak_locations, args.notch_radius,
                             lowpass_cutoff, args.lowpass_order)
    y = descreen_channel_fft(cmy[:, :, 2], peak_locations, args.notch_radius,
                             lowpass_cutoff, args.lowpass_order)
    k = descreen_channel_fft(k, peak_locations, args.notch_radius,
                             lowpass_cutoff, args.lowpass_order)
    k **= 2.0  # reduce darkening from black being "double counted"
    del cmy

    log.info("re-combining CMYK")
    filtered = np.stack([c, m, y], axis=2)
    del c, m, y

    combined_intensity = rgb_from_cmy(
        filtered, args.cyan_out, args.magenta_out, args.yellow_out
    )
    black_out_intensity = intensity_from_srgb(args.black_out.copy())
    combined_intensity = (
        np.expand_dims(1 - k, axis=2) * combined_intensity
        + np.expand_dims(k, axis=2) * np.expand_dims(black_out_intensity, axis=(0, 1))
    )
    del k

    combined = srgb_from_intensity(combined_intensity)
    del combined_intensity

    log.info("writing out filtered image")
    save_image(args.output, combined)


def intensity_from_srgb(image):
    gamma = 2.4
    A = 0.055
    phi = 12.92
    X = 0.04045
    linear_region = image < X
    image[linear_region] /= phi
    image_non_linear_region = image[np.logical_not(linear_region)]
    del linear_region
    image_non_linear_region += A
    image_non_linear_region /= 1.0 + A
    image_non_linear_region **= gamma
    return image


def srgb_from_intensity(intensity):
    gamma = 2.4
    A = 0.055
    phi = 12.92
    X = 0.04045
    linear_region = intensity < X / phi
    intensity[linear_region] *= phi
    intensity_non_linear_region = intensity[np.logical_not(linear_region)]
    del linear_region
    intensity_non_linear_region **= 1.0 / gamma
    intensity_non_linear_region *= 1.0 + A
    intensity_non_linear_region -= A

    return intensity


def cmy_from_rgb(rgb_intensity, cyan, magenta, yellow):
    cmy_srgb = np.array(
        [
            cyan,
            magenta,
            yellow,
        ]
    )
    cmy_intensity = intensity_from_srgb(cmy_srgb)
    white_intensity = np.array([1.0, 1.0, 1.0])
    basis = white_intensity - cmy_intensity

    cmy = white_intensity - rgb_intensity
    cmy = cmy @ np.linalg.inv(basis)
    return cmy


def rgb_from_cmy(cmy, cyan, magenta, yellow):
    cmy_srgb = np.array(
        [
            cyan,
            magenta,
            yellow,
        ]
    )
    cmy_intensity = intensity_from_srgb(cmy_srgb)
    white_intensity = np.array([1.0, 1.0, 1.0])
    basis = white_intensity - cmy_intensity

    rgb_intensity = cmy @ basis
    rgb_intensity = white_intensity - rgb_intensity
    return rgb_intensity


def detect_halftone_params(channel, detection_threshold=4.0, max_harmonics=4):
    """Detect halftone screen frequency and peak locations from a 2D channel.

    Returns (screen_freq, peak_locations) where screen_freq is in cycles/pixel
    and peak_locations is a list of (row, col) indices into the unshifted FFT.
    """
    H, W = channel.shape

    F = scipy.fft.fft2(channel.astype(np.float32), workers=-1)
    F_shifted = scipy.fft.fftshift(F)
    magnitude = np.abs(F_shifted)
    log_mag = np.log1p(magnitude)

    cy, cx = H // 2, W // 2

    # suppress DC and low frequencies — use a generous radius to avoid
    # detecting image content as halftone peaks
    dc_radius = max(15, int(0.02 * min(H, W)))
    yy, xx = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dc_mask = dist_from_center <= dc_radius
    log_mag[dc_mask] = 0

    # also suppress the Nyquist edges (top/bottom/left/right 2 rows/cols)
    # which can have spurious energy
    log_mag[:2, :] = 0
    log_mag[-2:, :] = 0
    log_mag[:, :2] = 0
    log_mag[:, -2:] = 0

    # detect peaks as local maxima with high prominence
    # use a local maximum filter to find candidate peaks
    neighborhood_size = max(7, int(0.01 * min(H, W)))
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
    local_max = sp.ndimage.maximum_filter(log_mag, size=neighborhood_size)
    is_local_max = (log_mag == local_max) & (log_mag > 0)

    # threshold: peak must be well above the background
    for sigma_mult in [detection_threshold, 3.0, 2.5]:
        # compute background stats excluding DC-suppressed region
        bg_values = log_mag[~dc_mask & (log_mag > 0)]
        if len(bg_values) == 0:
            continue
        bg_mean = bg_values.mean()
        bg_std = bg_values.std()
        thresh = bg_mean + sigma_mult * bg_std

        candidates = is_local_max & (log_mag > thresh)
        peak_rows, peak_cols = np.where(candidates)

        if len(peak_rows) >= 4:
            break
    else:
        log.warning("no halftone peaks detected; image may not contain a halftone pattern")
        return 0.0, []

    # convert to polar and filter
    peaks_polar = []
    for r, c in zip(peak_rows, peak_cols):
        fy = (r - cy) / H
        fx = (c - cx) / W
        radius = np.sqrt(fy ** 2 + fx ** 2)
        angle = np.arctan2(fy, fx)
        # skip anything still too close to DC
        if radius < dc_radius / min(H, W) * 1.5:
            continue
        # skip anything too close to Nyquist
        if radius > 0.45:
            continue
        peaks_polar.append((radius, angle, float(r), float(c), log_mag[r, c]))

    if len(peaks_polar) < 2:
        log.warning("too few halftone peaks found after filtering")
        return 0.0, []

    # sort by strength (brightest first) and take the strongest peaks
    peaks_polar.sort(key=lambda p: p[4], reverse=True)

    # find the fundamental frequency by looking for a cluster of peaks
    # at the same radius (the CMYK screens share a frequency)
    radii = np.array([p[0] for p in peaks_polar])
    sorted_by_radius = np.argsort(radii)

    # use the cluster of nearest-to-center peaks as fundamentals
    fundamental_radius = radii[sorted_by_radius[0]]
    fundamental_peaks = []
    for idx in sorted_by_radius:
        r = radii[idx]
        if r <= fundamental_radius * 1.4:
            fundamental_peaks.append(peaks_polar[idx])
        elif len(fundamental_peaks) >= 4:
            break
        else:
            # if we haven't found enough fundamentals, widen the search
            fundamental_radius = r
            fundamental_peaks.append(peaks_polar[idx])

    screen_freq = np.median([p[0] for p in fundamental_peaks])
    log.debug(f"fundamental peaks at radii: {[f'{p[0]:.4f}' for p in fundamental_peaks]}")

    # collect all detected peaks (fundamentals + higher-order)
    all_peaks_shifted = [(p[2], p[3]) for p in peaks_polar]

    # add harmonics that may have been missed by initial detection
    harmonic_thresh = bg_mean + 2.0 * bg_std
    for fp in fundamental_peaks:
        f_radius, f_angle = fp[0], fp[1]
        for n in range(2, max_harmonics + 1):
            harm_fy = n * f_radius * np.sin(f_angle)
            harm_fx = n * f_radius * np.cos(f_angle)
            harm_r = int(round(harm_fy * H + cy))
            harm_c = int(round(harm_fx * W + cx))
            if 0 <= harm_r < H and 0 <= harm_c < W:
                if log_mag[harm_r, harm_c] > harmonic_thresh:
                    all_peaks_shifted.append((float(harm_r), float(harm_c)))
            conj_r = int(round(-harm_fy * H + cy))
            conj_c = int(round(-harm_fx * W + cx))
            if 0 <= conj_r < H and 0 <= conj_c < W:
                if log_mag[conj_r, conj_c] > harmonic_thresh:
                    all_peaks_shifted.append((float(conj_r), float(conj_c)))

    # convert shifted coordinates to unshifted FFT coordinates
    peak_locations = []
    for (sr, sc) in all_peaks_shifted:
        ur = int(round(sr - cy)) % H
        uc = int(round(sc - cx)) % W
        peak_locations.append((ur, uc))

    peak_locations = list(set(peak_locations))
    return screen_freq, peak_locations


def auto_detect_screen(image_linear_rgb, detection_threshold=4.0, max_harmonics=4):
    """Detect halftone parameters from the full image using luminance."""
    # luminance from linear RGB
    if image_linear_rgb.ndim == 3:
        luminance = (
            0.2126 * image_linear_rgb[:, :, 0]
            + 0.7152 * image_linear_rgb[:, :, 1]
            + 0.0722 * image_linear_rgb[:, :, 2]
        )
    else:
        luminance = image_linear_rgb

    screen_freq, peak_locations = detect_halftone_params(
        luminance, detection_threshold, max_harmonics
    )
    return screen_freq, peak_locations


def descreen_channel_fft(channel, peak_locations, notch_radius=3.0,
                         lowpass_cutoff=0.0, lowpass_order=4):
    """Remove halftone pattern from a channel using FFT notch + low-pass filtering.

    Args:
        peak_locations: list of (row, col) in unshifted FFT coords for notch filters
        notch_radius: sigma of Gaussian notch suppression (frequency bins)
        lowpass_cutoff: cutoff frequency in cycles/pixel for Butterworth low-pass
                        (0 disables the low-pass)
        lowpass_order: steepness of the Butterworth rolloff
    """
    if not peak_locations and lowpass_cutoff <= 0:
        return channel

    H, W = channel.shape
    F = scipy.fft.fft2(channel.astype(np.float32), workers=-1)

    mask = np.ones((H, W), dtype=np.float32)

    # notch filters for detected peaks
    if peak_locations:
        sigma = notch_radius
        patch_radius = int(np.ceil(4 * sigma))

        for (py, px) in peak_locations:
            r_min = py - patch_radius
            r_max = py + patch_radius + 1
            c_min = px - patch_radius
            c_max = px + patch_radius + 1

            rows = np.arange(r_min, r_max) % H
            cols = np.arange(c_min, c_max) % W

            dr = np.arange(r_min, r_max) - py
            dc = np.arange(c_min, c_max) - px
            dist_sq = dr[:, None] ** 2 + dc[None, :] ** 2

            notch = 1.0 - np.exp(-dist_sq / (2 * sigma ** 2))
            mask[np.ix_(rows, cols)] *= notch.astype(np.float32)

    # Butterworth low-pass to catch all halftone energy above the cutoff
    if lowpass_cutoff > 0:
        # frequency grid in cycles/pixel (unshifted: DC at corners)
        freq_y = np.fft.fftfreq(H).astype(np.float32)
        freq_x = np.fft.fftfreq(W).astype(np.float32)
        freq_radius = np.sqrt(freq_y[:, None] ** 2 + freq_x[None, :] ** 2)

        # Butterworth: 1 / (1 + (r/cutoff)^(2*order))
        # smooth rolloff — preserves detail below cutoff, attenuates above
        ratio = freq_radius / lowpass_cutoff
        butterworth = 1.0 / (1.0 + ratio ** (2 * lowpass_order))
        mask *= butterworth

    F_filtered = F * mask
    result = np.real(scipy.fft.ifft2(F_filtered))
    return np.clip(result, 0.0, None)


def build_peaks_from_manual_params(screen_freq, angles_turns, image_shape, max_harmonics=4):
    """Generate peak locations from manually specified screen frequency and angles."""
    H, W = image_shape[:2]
    peaks = []
    for angle_turns in angles_turns:
        angle_rad = angle_turns * 2 * np.pi
        for n in range(1, max_harmonics + 1):
            freq = n * screen_freq
            fy = freq * np.sin(angle_rad)
            fx = freq * np.cos(angle_rad)
            row = int(round(fy * H)) % H
            col = int(round(fx * W)) % W
            peaks.append((row, col))
            peaks.append(((-row) % H, (-col) % W))
    return list(set(peaks))


def save_debug_spectrum(channel, peak_locations, path):
    """Save a visualization of the frequency spectrum with detected peaks marked."""
    H, W = channel.shape
    F = scipy.fft.fft2(channel.astype(np.float32), workers=-1)
    F_shifted = scipy.fft.fftshift(F)
    log_mag = np.log1p(np.abs(F_shifted))

    # normalize to 0-255
    log_mag = log_mag / log_mag.max() * 255.0
    debug_img = np.stack([log_mag, log_mag, log_mag], axis=2).astype(np.uint8)

    # mark peaks in red
    cy, cx = H // 2, W // 2
    for (ur, uc) in peak_locations:
        sr = (ur + cy) % H  # back to shifted
        sc = (uc + cx) % W
        r_min = max(0, sr - 3)
        r_max = min(H, sr + 4)
        c_min = max(0, sc - 3)
        c_max = min(W, sc + 4)
        debug_img[r_min:r_max, c_min:c_max] = [255, 0, 0]

    iio.imwrite(path, debug_img)


def handle_default_colors_out(args):
    for color in "cyan", "magenta", "yellow", "black":
        if vars(args)[f"{color}_out"] is None:
            vars(args)[f"{color}_out"] = vars(args)[f"{color}_in"]


def load_image(path):
    try:
        props = iio.improps(path)
        image = iio.imread(path)
        if props.shape[2] == 4:
            image = image[:, :, 0:3]
        if np.issubdtype(props.dtype, np.integer):
            iinfo = np.iinfo(props.dtype)
            image = (image.astype(np.float32) - iinfo.min) / (iinfo.max - iinfo.min)
        return image
    except Exception as e:
        log.critical(f"Failed to open image: {e}")
        exit(1)


def save_image(path, image):
    iio.imwrite(path, (image * 255.0).round().clip(0.0, 255.0).astype(np.uint8))


if __name__ == "__main__":
    main()
