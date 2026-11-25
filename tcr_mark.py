#!/usr/bin/env python3
"""
tcr_mark.py - first-pass TCR IN / TCR OUT marking from a dialogue stem.

Idea 1 (ffmpeg-only baseline): ffmpeg's silencedetect filter finds the silent
stretches of the stem; everything between two silences is one sound event.
Each event is written as one entry in an SRT file. The SRT timestamps are in
milliseconds (the SRT standard), snapped to frame boundaries; the text of each
entry carries the same IN / OUT as frame timecode, HH:MM:SS:FF.

Requires: Python 3.8+, and ffmpeg + ffprobe on the PATH. No pip packages.

Example:
    python tcr_mark.py episode_dx.wav --fps 25 --start-tc 01:00:00:00

Tuning knobs (see --help):
    --noise        how quiet counts as silence, in dB (lower = more sensitive)
    --min-silence  a gap must be at least this long (s) to split two events
    --min-event    events shorter than this (s) are dropped as blips
    --pad-frames   frames of padding added before IN and after OUT
    --clean        filter rumble, hiss and steady noise before detecting
    --min-duration shortest time a subtitle stays up (extended if there is room)
    --tail         extra time added after every OUT
    --gap-frames   minimum frames between one subtitle and the next

"""

import argparse
import math
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

# Frame rates that are really N*1000/1001. Timecode still counts at N.
NTSC_RATES = {
    "23.976": Fraction(24000, 1001),
    "23.98": Fraction(24000, 1001),
    "29.97": Fraction(30000, 1001),
    "59.94": Fraction(60000, 1001),
}


# ----------------------------------------------------------------------------
# Timecode
# ----------------------------------------------------------------------------

class Timecode:
    """Converts between seconds, frame counts and HH:MM:SS:FF strings."""

    def __init__(self, fps_text, drop_frame=False):
        fps_text = fps_text.strip()
        self.rate = NTSC_RATES.get(fps_text) or Fraction(fps_text)
        if self.rate <= 0:
            raise ValueError("Frame rate must be positive")
        self.base = int(math.ceil(self.rate))  # frames counted per TC second
        self.drop = drop_frame
        if self.drop and self.base not in (30, 60):
            raise ValueError("Drop-frame only applies to 29.97 and 59.94 fps")
        # frames skipped at the start of each minute (except every 10th)
        self.drop_count = self.base // 15 if self.drop else 0
        self.sep = ";" if self.drop else ":"

    def tc_to_frames(self, tc):
        parts = re.split(r"[:;.\s]", tc.strip())
        if len(parts) != 4 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Timecode must look like HH:MM:SS:FF, got {tc!r}")
        h, m, s, f = (int(p) for p in parts)
        if m > 59 or s > 59 or f >= self.base:
            raise ValueError(f"Timecode {tc!r} is out of range for {self.base} fps")
        frames = (h * 3600 + m * 60 + s) * self.base + f
        if self.drop:
            total_min = h * 60 + m
            frames -= self.drop_count * (total_min - total_min // 10)
        return frames

    def frames_to_tc(self, frames):
        frames = max(0, int(frames))
        if self.drop:
            per_10min = self.base * 600 - self.drop_count * 9
            per_min = self.base * 60 - self.drop_count
            tens, rem = divmod(frames, per_10min)
            frames += self.drop_count * 9 * tens
            if rem > self.drop_count:
                frames += self.drop_count * ((rem - self.drop_count) // per_min)
        f = frames % self.base
        total_s = frames // self.base
        return "{:02d}:{:02d}:{:02d}{}{:02d}".format(
            total_s // 3600, (total_s // 60) % 60, total_s % 60, self.sep, f
        )

    def seconds_to_frames(self, seconds, round_up=False):
        exact = Fraction(seconds).limit_denominator(1_000_000) * self.rate
        return math.ceil(exact) if round_up else math.floor(exact)

    def frames_to_seconds(self, frames):
        return float(Fraction(frames) / self.rate)


# ----------------------------------------------------------------------------
# ffmpeg
# ----------------------------------------------------------------------------

def probe_duration(path):
    """Length of the audio in seconds, from ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"ffprobe failed:\n{out.stderr.strip()}")
    return float(out.stdout.strip())


# Applied before detection when --clean is given. Detection only: the audio
# file itself is never changed.
#   highpass / lowpass : keep the band where voices live, removing rumble,
#                        hum, and high hiss
#   afftdn             : ffmpeg's denoiser, lowers steady background noise
CLEAN_FILTERS = "highpass=f=100,lowpass=f=6000,afftdn=nr=12:nf=-35"


def detect_silences(path, noise_db, min_silence, clean=False):
    """Run ffmpeg silencedetect and return a list of (start, end) silences."""
    chain = f"silencedetect=noise={noise_db}dB:d={min_silence}"
    if clean:
        chain = CLEAN_FILTERS + "," + chain
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-sn",
        "-af", chain,
        "-f", "null", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{out.stderr.strip()[-2000:]}")

    silences, start = [], None
    for line in out.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if m and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    if start is not None:               # silence runs to the end of the file
        silences.append((start, float("inf")))
    return silences


def silences_to_events(silences, duration, min_event):
    """Everything between two silences is one sound event."""
    events, cursor = [], 0.0
    for s_start, s_end in silences:
        if s_start > cursor:
            events.append((cursor, min(s_start, duration)))
        cursor = min(s_end, duration)
    if cursor < duration:
        events.append((cursor, duration))
    return [(a, b) for a, b in events if b - a >= min_event]


def srt_time(seconds):
    """Seconds -> SRT timestamp HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    return "{:02d}:{:02d}:{:02d},{:03d}".format(
        ms // 3600000, (ms // 60000) % 60, (ms // 1000) % 60, ms % 1000)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Mark TCR IN / TCR OUT for every sound event in a dialogue stem.")
    ap.add_argument("audio", help="dialogue stem (wav, mp3, m4a, mov ...)")
    ap.add_argument("--fps", required=True,
                    help="frame rate: 24, 25, 30, 23.976, 29.97, 50, 59.94 ...")
    ap.add_argument("--start-tc", default="00:00:00:00",
                    help="timecode of the first sample of the file (default 00:00:00:00)")
    ap.add_argument("--drop-frame", action="store_true",
                    help="use drop-frame timecode (29.97 / 59.94 only)")
    ap.add_argument("--noise", type=float, default=-35.0,
                    help="silence threshold in dB (default -35)")
    ap.add_argument("--min-silence", type=float, default=0.30,
                    help="shortest gap in seconds that splits two events (default 0.30)")
    ap.add_argument("--min-event", type=float, default=0.10,
                    help="drop events shorter than this many seconds (default 0.10)")
    ap.add_argument("--pad-frames", type=int, default=1,
                    help="frames of padding before IN and after OUT (default 1)")
    ap.add_argument("--min-duration", type=float, default=1.0,
                    help="shortest time a subtitle stays up, in seconds; shorter "
                         "ones are extended if there is room (default 1.0)")
    ap.add_argument("--tail", type=float, default=0.25,
                    help="extra seconds added after every OUT, if there is room (default 0.25)")
    ap.add_argument("--gap-frames", type=int, default=2,
                    help="minimum frames between one subtitle's OUT and the next IN (default 2)")
    ap.add_argument("--clean", action="store_true",
                    help="filter rumble, hiss and steady noise before detecting")
    ap.add_argument("--short-flag", type=float, default=0.25,
                    help="flag events shorter than this many seconds (default 0.25)")
    ap.add_argument("--long-flag", type=float, default=15.0,
                    help="flag events longer than this many seconds (default 15)")
    ap.add_argument("-o", "--output", help="SRT path (default: <audio name>_tcr.srt)")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        sys.exit(f"File not found: {audio}")
    try:
        tc = Timecode(args.fps, args.drop_frame)
        start_frames = tc.tc_to_frames(args.start_tc)
    except (ValueError, ZeroDivisionError) as e:
        sys.exit(f"Error: {e}")

    duration = probe_duration(audio)
    silences = detect_silences(audio, args.noise, args.min_silence, args.clean)
    events = silences_to_events(silences, duration, args.min_event)

    # Segments are (start, end). Each one becomes one subtitle.
    segments = list(events)

    # ---- Timing rules, all in frames -------------------------------------
    #  1. The speech itself: IN rounds down, OUT rounds up.
    #  2. IN moves earlier by --pad-frames, never closer than --gap-frames
    #     to the previous subtitle's speech.
    #  3. OUT moves later by --pad-frames + --tail, and further if needed so
    #     the subtitle lasts at least --min-duration.
    #  4. No OUT may come closer than --gap-frames to the next IN. Where there
    #     is no room, the subtitle stays shorter and is flagged.
    last_frame = tc.seconds_to_frames(duration, round_up=True)
    gap = max(0, args.gap_frames)
    tail_f = tc.seconds_to_frames(args.tail, round_up=True)
    min_f = tc.seconds_to_frames(args.min_duration, round_up=True)

    raw = [[tc.seconds_to_frames(a), tc.seconds_to_frames(b, round_up=True)]
           for a, b in segments]
    for i in range(1, len(raw)):                  # keep pieces in order
        raw[i][0] = max(raw[i][0], raw[i - 1][1])
        raw[i][1] = max(raw[i][1], raw[i][0] + 1)

    ins = []
    for i, (r_in, _) in enumerate(raw):
        earliest = raw[i - 1][1] + gap if i > 0 else 0
        ins.append(min(r_in, max(r_in - args.pad_frames, earliest)))

    outs, extended, no_room = [], 0, []
    for i, (r_in, r_out) in enumerate(raw):
        want = max(r_out + args.pad_frames + tail_f, ins[i] + min_f)
        limit = ins[i + 1] - gap if i + 1 < len(raw) else last_frame
        out = max(min(want, limit), ins[i] + 1)
        outs.append(out)
        if r_out - r_in < min_f and out - ins[i] > r_out - r_in:
            extended += 1
        no_room.append(out - ins[i] < min_f)

    # ---- Write the SRT ---------------------------------------------------
    out_path = Path(args.output) if args.output else audio.with_name(audio.stem + "_tcr.srt")
    flagged = 0
    blocks = []
    for n, ((a, b), f_in, f_out, tight) in enumerate(
            zip(segments, ins, outs, no_room), start=1):
        text = []
        # The frame-accurate timecodes go in the text, because SRT timestamps
        # themselves are in milliseconds and cannot hold frames.
        text.append("IN {}  OUT {}".format(tc.frames_to_tc(start_frames + f_in),
                                           tc.frames_to_tc(start_frames + f_out)))
        checks = []
        if b - a < args.short_flag:
            checks.append("very short, may be a click or breath")
        elif b - a > args.long_flag:
            checks.append("very long, may be several lines merged")
        if tight:
            checks.append(f"under {args.min_duration:g} s, no room to extend")
        if checks:
            text.append("CHECK: " + "; ".join(checks))
            flagged += 1
        blocks.append("{}\n{} --> {}\n{}\n".format(
            n,
            srt_time(tc.frames_to_seconds(start_frames + f_in)),
            srt_time(tc.frames_to_seconds(start_frames + f_out)),
            "\n".join(text)))
    with open(out_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("\n".join(blocks))

    total = sum(b - a for a, b in events)
    print(f"File length      : {duration:.2f} s")
    print(f"Settings         : noise {args.noise} dB, min silence {args.min_silence} s, "
          f"min event {args.min_event} s, pad {args.pad_frames} frame(s), "
          f"clean {'on' if args.clean else 'off'}")
    print(f"Timing           : min duration {args.min_duration} s, tail {args.tail} s, "
          f"gap {gap} frames")
    print(f"Subtitles        : {len(segments)}  ({flagged} flagged CHECK)")
    print(f"Extended to min  : {extended}  (no room for {sum(no_room)})")
    print(f"Sound / silence  : {total:.1f} s / {duration - total:.1f} s")
    print(f"SRT written to   : {out_path}")


if __name__ == "__main__":
    main()
