
import warnings
from pathlib import Path
from itertools import combinations

import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch, csd

# ============================================================
# PARAMETERS
# ============================================================

NFFT_WELCH = 512

FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 60),
}

# Resolve paths relative to this script, not the terminal's cwd.
ROOT = Path(__file__).resolve().parent
ASD_DIR = ROOT / "subjects" / "asd"
TD_DIR = ROOT / "subjects" / "td"
OUTPUT_DIR = ROOT / "sheets"

REGIONS = {
    "Frontal":   ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"],
    "Central":   ["C3", "C4", "CZ"],
    "Temporal":  ["T3", "T4", "T5", "T6"],
    "Parietal":  ["P3", "P4", "PZ"],
    "Occipital": ["O1", "O2"],
}

ALL_ELECTRODES = list(dict.fromkeys(
    ch for channels in REGIONS.values() for ch in channels
))
ALL_PAIRS = list(combinations(ALL_ELECTRODES, 2))
PAIR_COLUMNS = [f"{a}-{b}" for a, b in ALL_PAIRS]

# ============================================================
# DISCOVER SUBJECTS AND RECORDINGS
# ============================================================

def discover_subjects():
    """Return (group, subject, EDF paths) for all discovered subjects."""
    discovered = []

    for group, directory, prefix in [
        ("ASD", ASD_DIR, "pre-"),
        ("TD", TD_DIR, "na-"),
    ]:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Subject directory not found: {directory}"
            )

        for subject_dir in sorted(directory.iterdir()):
            if not subject_dir.is_dir():
                continue

            # Case-insensitive prefix matching handles NA-YF1.edf,
            # NA-yf1.edf, etc.
            files = sorted(
                p for p in subject_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".edf"
                and p.name.lower().startswith(prefix.lower())
            )

            if not files:
                warnings.warn(
                    f"No {prefix} EDF files found for "
                    f"{group}/{subject_dir.name}"
                )
                continue

            discovered.append(
                (group, subject_dir.name, files)
            )

    return discovered

# ============================================================
# LOAD AND PREPROCESS EDF
# ============================================================

def load_preprocessed_data(edf_path):
    """
    Return channel data in the predefined electrode order.
    Missing channels remain absent and their pairs become NaN.
    """
    raw = mne.io.read_raw_edf(
        edf_path, preload=True, verbose="ERROR"
    )

    # Standardize names and handle duplicate names safely.
    raw.rename_channels(
        {name: name.upper() for name in raw.ch_names}
    )

    available = set(raw.ch_names)
    picks = [ch for ch in ALL_ELECTRODES if ch in available]

    if len(picks) < 2:
        warnings.warn(f"Insufficient EEG channels: {edf_path}")
        return None, None, None

    raw.pick(picks)

    raw.notch_filter(freqs=[60], verbose="ERROR")
    raw.filter(
        0.5, 60.0,
        fir_design="firwin",
        verbose="ERROR"
    )

    data = detrend(raw.get_data(), axis=1)
    return data, raw.info["sfreq"], list(raw.ch_names)

# ============================================================
# FAST WELCH COHERENCE
# ============================================================

def compute_all_pair_coherence(data, ch_names, sfreq):
    """
    Compute all 171 pairwise coherence features for one EDF.

    Each channel's PSD is calculated once. Cross-spectral density
    is then calculated once per pair, avoiding repeated PSD work.
    """
    results = {pair: np.nan for pair in PAIR_COLUMNS}

    n_times = data.shape[1]
    nperseg = min(NFFT_WELCH, n_times)

    if nperseg < 2:
        return results

    # Match SciPy's default Welch/CSD segmenting and detrending.
    # Calculate each channel's PSD only once.
    psd = {}
    freqs = None

    for i, ch in enumerate(ch_names):
        freqs, power = welch(
            data[i],
            fs=sfreq,
            nperseg=nperseg,
        )
        psd[ch] = power

    # Cache frequency-band masks for this recording.
    band_masks = {
        band_name: (freqs >= low) & (freqs <= high)
        for band_name, (low, high) in FREQ_BANDS.items()
    }

    for ch1, ch2 in ALL_PAIRS:
        if ch1 not in psd or ch2 not in psd:
            continue

        _, pxy = csd(
            data[ch_names.index(ch1)],
            data[ch_names.index(ch2)],
            fs=sfreq,
            nperseg=nperseg,
        )

        denominator = psd[ch1] * psd[ch2]

        # Protect against zero-power frequency bins.
        coherence_spectrum = np.full(
            denominator.shape, np.nan, dtype=float
        )

        valid = denominator > 0
        coherence_spectrum[valid] = (
            np.abs(pxy[valid]) ** 2 / denominator[valid]
        )

        pair_name = f"{ch1}-{ch2}"

        for band_name, mask in band_masks.items():
            if np.any(mask):
                value = np.nanmean(coherence_spectrum[mask])
                results[(band_name, pair_name)] = value

    return results

# ============================================================
# SUBJECT-LEVEL FEATURE EXTRACTION
# ============================================================

def extract_subject(group, subject, edf_files):
    """
    Average coherence across a subject's valid EDF recordings.
    Returns one feature dictionary per frequency band.
    """
    band_recordings = {band: [] for band in FREQ_BANDS}

    for edf_path in edf_files:
        print(f"[{group}] {subject}: {edf_path.name}")

        try:
            data, sfreq, ch_names = load_preprocessed_data(edf_path)

            if data is None:
                continue

            features = compute_all_pair_coherence(
                data, ch_names, sfreq
            )

            for band in FREQ_BANDS:
                row = {
                    pair: features[(band, pair)]
                    for pair in PAIR_COLUMNS
                }
                band_recordings[band].append(row)

        except Exception as exc:
            warnings.warn(
                f"Failed to process {edf_path}: {exc}"
            )

    subject_rows = {}

    for band in FREQ_BANDS:
        recordings = band_recordings[band]

        if recordings:
            # Equal-weight average across available EDF recordings.
            averaged = (
                pd.DataFrame(recordings, columns=PAIR_COLUMNS)
                .mean(axis=0, skipna=True)
            )
        else:
            averaged = pd.Series(
                np.nan, index=PAIR_COLUMNS, dtype=float
            )

        subject_rows[band] = {
            "subject": subject,
            "group": group,
            **averaged.to_dict(),
        }

    return subject_rows

# ============================================================
# MAIN: WRITE FIVE CSV FILES
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    discovered = discover_subjects()

    if not discovered:
        raise RuntimeError(
            "No subjects found. Check the subjects/asd and "
            "subjects/td directories and EDF filename prefixes."
        )

    print(f"Found {len(discovered)} subjects.")
    print(f"Electrode pairs per subject: {len(ALL_PAIRS)}")

    rows_by_band = {band: [] for band in FREQ_BANDS}

    for group, subject, edf_files in discovered:
        subject_rows = extract_subject(
            group, subject, edf_files
        )

        for band, row in subject_rows.items():
            rows_by_band[band].append(row)

    for band, rows in rows_by_band.items():
        # Fixed feature order for every workbook/CSV.
        df = pd.DataFrame(rows)

        df = df[["subject", "group"] + PAIR_COLUMNS]

        output_path = OUTPUT_DIR / f"{band}.csv"
        df.to_csv(output_path, index=False, float_format="%.8f")

        print(
            f"[SUCCESS] {output_path} | "
            f"{len(df)} subjects x {len(PAIR_COLUMNS)} features"
        )

    print("\nFinished exporting all five frequency bands.")


if __name__ == "__main__":
    main()