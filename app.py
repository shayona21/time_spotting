"""
Zibanka TCR Spotting: a simple web interface for tcr_mark.py.

Run locally:
    pip install -r requirements.txt
    python app.py
then open http://127.0.0.1:5050 in a browser.

On a server, run it with gunicorn instead:
    gunicorn -w 2 -b 0.0.0.0:5050 --timeout 1800 app:app
"""

import re
import subprocess
import sys
import uuid
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
SCRIPT = BASE / "tcr_mark.py"
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".aif", ".aiff",
               ".ogg", ".mov", ".mp4", ".mxf"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3   # 2 GB upload limit

# The settings shown in the table: (key, command-line flag, default, kind,
# label, unit, explanation). kind decides how the value is checked.
PARAMS = [
    ("fps", "--fps", "25", "fps", "Frame rate", "fps",
     "Frame rate of the film, e.g. 24, 25, 29.97."),
    ("start_tc", "--start-tc", "00:00:00:00", "tc", "Start timecode", "HH:MM:SS:FF",
     "Timecode of the first frame. Keep 00:00:00:00 to line up with the audio in Subtitle Edit."),
    ("noise", "--noise", "-30", "float", "Noise level", "dB",
     "How loud a sound must be to count. Closer to 0 (e.g. -25) ignores more background noise; "
     "lower (e.g. -40) catches quieter sounds."),
    ("min_silence", "--min-silence", "0.3", "float", "Pause to split lines", "seconds",
     "A pause must be at least this long to split two lines. Lower splits more."),
    ("min_event", "--min-event", "0.1", "float", "Ignore sounds shorter than", "seconds",
     "Sounds shorter than this are dropped as clicks or bumps."),
    ("pad_frames", "--pad-frames", "1", "int", "Padding", "frames",
     "Extra frames added before TCR IN and after TCR OUT."),
    ("min_duration", "--min-duration", "1.0", "float", "Minimum subtitle length", "seconds",
     "Shorter lines are stretched to this length if there is room."),
    ("tail", "--tail", "0.25", "float", "Buffer after speech", "seconds",
     "Extra time added after every TCR OUT, if there is room."),
    ("gap_frames", "--gap-frames", "2", "int", "Gap between subtitles", "frames",
     "Minimum frames between one line's TCR OUT and the next TCR IN."),
    ("clean", "--clean", True, "bool", "Clean background noise", "",
     "Filter out rumble, hiss and steady noise before detecting."),
]


def defaults():
    return {p[0]: p[2] for p in PARAMS}


def build_args(form):
    """Check the submitted settings and turn them into command-line arguments.

    Returns (values, args, errors).
    """
    values, args, errors = {}, [], []
    for key, flag, _default, kind, label, _unit, _help in PARAMS:
        if kind == "bool":
            on = form.get(key) == "on"
            values[key] = on
            if on:
                args.append(flag)
            continue
        raw = (form.get(key) or "").strip()
        values[key] = raw
        if kind == "fps":
            ok = re.fullmatch(r"\d{1,3}(\.\d{1,3})?", raw) is not None
        elif kind == "tc":
            ok = re.fullmatch(r"\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2}", raw) is not None
        elif kind == "int":
            ok = re.fullmatch(r"-?\d+", raw) is not None
        else:
            ok = re.fullmatch(r"-?\d+(\.\d+)?", raw) is not None
        if ok:
            args += [flag, raw]
        else:
            errors.append(f"{label}: '{raw}' is not a valid value.")
    return values, args, errors


def preview(srt_path, n=10):
    blocks = srt_path.read_text(encoding="utf-8").strip().split("\n\n")
    return blocks[:n], len(blocks)


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"params": PARAMS, "values": defaults()}
    if request.method == "GET":
        return render_template("index.html", **ctx)

    values, args, errors = build_args(request.form)
    ctx["values"] = values

    upload = request.files.get("audio")
    if not upload or not upload.filename:
        errors.append("Please choose an audio file.")
    else:
        name = secure_filename(upload.filename) or "audio"
        if Path(name).suffix.lower() not in AUDIO_TYPES:
            errors.append(f"'{upload.filename}' does not look like an audio file.")
    if errors:
        ctx["errors"] = errors
        return render_template("index.html", **ctx)

    job = uuid.uuid4().hex
    audio_path = UPLOADS / f"{job}_{name}"
    out_dir = OUTPUTS / job
    out_dir.mkdir()
    srt_name = Path(name).stem + "_tcr.srt"
    upload.save(audio_path)

    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(audio_path), *args,
             "-o", str(out_dir / srt_name)],
            capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        result = None
    finally:
        audio_path.unlink(missing_ok=True)   # don't keep client audio

    if result is None or result.returncode != 0:
        detail = "It took longer than 30 minutes." if result is None else \
            (result.stderr or result.stdout).strip()[-2000:]
        ctx["errors"] = ["Processing failed.", detail]
        return render_template("index.html", **ctx)

    blocks, total = preview(out_dir / srt_name)
    ctx.update(job=job, srt_name=srt_name, summary=result.stdout.strip(),
               preview=blocks, total=total, filename=upload.filename)
    return render_template("index.html", **ctx)


@app.route("/download/<job>/<path:filename>")
def download(job, filename):
    if not re.fullmatch(r"[0-9a-f]{32}", job):
        abort(404)
    return send_from_directory(OUTPUTS / job, filename, as_attachment=True)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5080 in your browser. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5080, debug=False)
