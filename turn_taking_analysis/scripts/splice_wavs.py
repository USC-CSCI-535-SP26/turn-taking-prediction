#!/usr/bin/env python3
"""
splice_wavs.py

Slice each .wav in --in-dir into fixed-length sliding windows and write each
window to --out-dir/<stem>/<start>-<end>_<stem>.wav.

Purpose: produce uniformly-sized audio windows so downstream feature
extractors (e.g. CPC) operate on bounded context per window instead of
bleeding an entire multi-minute interaction's context forward. Applying the
same windowing across every modality keeps features time-aligned for
multimodal stacking.

Partial trailing windows (shorter than --window-len) are skipped so every
output file is exactly --window-len seconds.

By default existing output files are left alone so a terminated run can
resume on rerun; pass --overwrite to force re-splicing.

Example
-------
    python splice_wavs.py \\
        --in-dir ../subset/audio \\
        --out-dir ../subset/audio_sliced \\
        --window-len 2 \\
        --stride 0.5

Output layout
-------------
    out-dir/
        V03_S1930_I00000105_P5055/
            0000.00-0002.00_V03_S1930_I00000105_P5055.wav
            0000.50-0002.50_V03_S1930_I00000105_P5055.wav
            0001.00-0003.00_V03_S1930_I00000105_P5055.wav
            ...
"""

import argparse
from pathlib import Path

import soundfile as sf


def _fmt_time(t: float) -> str:
    """Format a time value for filenames as zero-padded seconds with 2 decimal
    places (width 7, NNNN.NN). Fixed width ensures lexicographic sort order
    matches temporal order. 4 whole-second digits cover durations up to
    9999.99 s (> the 20-minute dataset cap); 2 fractional digits cover any
    stride >= 0.01 s."""
    return f"{t:07.2f}"


def splice_wav(
    src: Path,
    out_dir: Path,
    window_len: float,
    stride: float,
    overwrite: bool,
) -> tuple[int, int]:
    """Slice one wav. Returns (num_written, num_skipped_because_existing)."""
    info = sf.info(str(src))
    sr = info.samplerate
    duration = info.frames / sr

    stem = src.stem
    sub_dir = out_dir / stem
    sub_dir.mkdir(parents=True, exist_ok=True)

    data, _ = sf.read(str(src), always_2d=False)
    window_samples = int(round(window_len * sr))

    written = 0
    skipped = 0
    i = 0
    while True:
        start_sec = i * stride
        end_sec = start_sec + window_len
        # Tolerance guards against float drift making end_sec exceed duration
        # by a fraction of a sample when it shouldn't.
        if end_sec > duration + 1e-9:
            break

        start_sample = int(round(start_sec * sr))
        end_sample = start_sample + window_samples
        if end_sample > len(data):
            break

        fname = f"{_fmt_time(start_sec)}-{_fmt_time(end_sec)}_{stem}.wav"
        out_path = sub_dir / fname

        if out_path.exists() and not overwrite:
            skipped += 1
        else:
            sf.write(
                str(out_path),
                data[start_sample:end_sample],
                sr,
                subtype=info.subtype,
            )
            written += 1

        i += 1

    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="flat directory of input .wav files (not recursed)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="directory to write spliced wavs into (one subdirectory per input wav)",
    )
    ap.add_argument(
        "--window-len",
        required=True,
        type=float,
        help="window duration in seconds (each output wav is exactly this long)",
    )
    ap.add_argument(
        "--stride",
        required=True,
        type=float,
        help="sliding window stride in seconds between successive window starts",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing spliced wavs (default: skip-if-exists so a "
        "terminated run can resume on rerun)",
    )
    args = ap.parse_args()

    if not args.in_dir.is_dir():
        ap.error(f"--in-dir does not exist or is not a directory: {args.in_dir}")
    if args.window_len <= 0:
        ap.error("--window-len must be > 0")
    if args.stride <= 0:
        ap.error("--stride must be > 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.in_dir.glob("*.wav"))
    if not wavs:
        ap.error(f"no .wav files found in {args.in_dir}")

    total_written = 0
    total_skipped = 0
    for i, src in enumerate(wavs, 1):
        written, skipped = splice_wav(
            src, args.out_dir, args.window_len, args.stride, args.overwrite
        )
        print(f"[{i}/{len(wavs)}] {src.name}: {written} written, {skipped} skipped")
        total_written += written
        total_skipped += skipped

    print(
        f"\nDone. {total_written} windows written, "
        f"{total_skipped} skipped across {len(wavs)} files."
    )


if __name__ == "__main__":
    main()
