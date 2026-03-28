Doubletone removes the halftone pattern from a scan of an image printed with offset halftone.

## Usage

```
uv run doubletone.py image.jpg
```

The script auto-detects the halftone screen frequency and removes it. Output is saved to `filtered.png` by default (override with `-o`).

### Tuning

- `--lowpass 0.9` — cutoff as a fraction of the detected screen frequency. Lower values (e.g. 0.75) remove more halftone at the cost of some softness. Set to 0 to rely on notch filters only.
- `--lowpass-order 4` — steepness of the low-pass rolloff.
- `--notch-radius 3.0` — width of the per-peak notch filters.
- `--debug-spectrum` — saves `debug_spectrum.png` showing the frequency spectrum with detected peaks marked in red.

Run `uv run doubletone.py --help` for all options, including ink color overrides and manual screen frequency/angle specification.

## Algorithm

- Convert the image from sRGB to linear RGB, then decompose into CMYK channels
- Auto-detect the halftone screen frequency by finding peaks in the 2D FFT magnitude spectrum of the luminance channel
- For each CMYK channel:
  - Compute the 2D FFT
  - Apply Gaussian notch filters at detected halftone peak locations
  - Apply a Butterworth low-pass filter with cutoff just below the screen frequency to catch all halftone energy (harmonics, cross-channel modulation products)
  - Inverse FFT back to spatial domain
- Re-combine the CMYK channels and convert back to sRGB
