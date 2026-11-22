"""
The 3-minute submission video: intro (on camera, 20 s) + footage with voice-over (2:20) + outro (on camera,
20 s), as in docs/VIDEO_SCRIPT.md. Needs ffmpeg on PATH and demo_replays/broll.mp4 (python make_broll.py).

Put your recordings in video_parts/ (any of these may be missing; a text card with the script's words
replaces a missing camera clip, and the footage stays silent without a voice-over):
  video_parts/intro.mp4      (or .mov)   one team member on camera, about 20 s
  video_parts/voiceover.m4a  (or .mp3, .wav, .aac)   the narration, up to 2:20
  video_parts/outro.mp4      (or .mov)   on camera, about 20 s

  python make_final_video.py          # -> demo_replays/final_video.mp4 (3:00, 1280x720)
"""

import glob
import os
import subprocess
import tempfile

from make_broll import encode
from make_pitch_video import CYAN, MUTED, TEXT, caption_frames

PARTS, BROLL, OUT = "video_parts", "demo_replays/broll.mp4", "demo_replays/final_video.mp4"
CLIP_S, BROLL_S = 20, 140
VIDEO = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25,format=yuv420p"
INTRO_CARD = [("HACKLAB BERLIN × TABPFN · THE BIG GANG", "label", CYAN, 26),
              ("How far can a table take a prosthetic arm?", "h2", TEXT, 34),
              ("Minh Nguyen · Quan Vo · Kazuki Dozono", "body", TEXT, 30),
              ("Someone who has lost an arm at the shoulder is handed a cup. A prosthetic arm should", "body", MUTED, 6),
              ("take it safely, and only when they ask. We built that arm in simulation,", "body", MUTED, 6),
              ("with TabPFN as its spatial awareness.", "body", MUTED, 0)]
OUTRO_CARD = [("WHAT’S NEXT", "label", CYAN, 26),
              ("A real depth camera instead of the simulator’s object tracking,", "body", TEXT, 6),
              ("and real hand-over recordings as TabPFN’s context.", "body", TEXT, 40),
              ("github.com/nhatminhnguyen0419/Hacklab-Hackathon---Prior-Labs", "h3", CYAN, 16),
              ("The Big Gang · Minh Nguyen, Quan Vo, Kazuki Dozono · thanks for watching!", "body", TEXT, 0)]


def find(name, exts):
    hits = [p for e in exts for p in glob.glob(os.path.join(PARTS, f"{name}.{e}"))]
    return hits[0] if hits else None


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def segment(video, out, seconds, audio=None):
    """One normalised piece: 1280x720 25 fps H.264 + stereo AAC 48 kHz, exactly `seconds` long. Audio from
    `audio` (a separate file), else the clip's own sound, else silence."""
    args = ["-i", video]
    if audio:
        args += ["-i", audio]
        amap = "1:a"
    elif has_audio(video):
        amap = "0:a"
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = "1:a"
    ffmpeg(*args, "-map", "0:v", "-map", amap, "-vf", f"{VIDEO},tpad=stop_mode=clone:stop_duration={seconds}",
           "-af", "aresample=48000,apad", "-t", str(seconds), "-c:v", "libx264", "-crf", "20", "-c:a", "aac",
           "-ac", "2", "-ar", "48000", out)


def main():
    assert os.path.exists(BROLL), f"{BROLL} missing: run python make_broll.py first"
    with tempfile.TemporaryDirectory() as tmp:
        pieces = []
        for name, card in (("intro", INTRO_CARD), ("outro", OUTRO_CARD)):
            clip = find(name, ["mp4", "mov", "MP4", "MOV"])
            if clip is None:  # no recording: a text card with the script's words
                clip = os.path.join(tmp, f"{name}_card.mp4")
                encode(caption_frames(card, CLIP_S), clip)
                print(f"{name}: no video_parts/{name}.mp4, using a text card")
            else:
                print(f"{name}: {clip}")
            out = os.path.join(tmp, f"{name}.mp4")
            segment(clip, out, CLIP_S)
            pieces.append(out)
        voice = find("voiceover", ["m4a", "mp3", "wav", "aac", "M4A", "MP3", "WAV"])
        print(f"voice-over: {voice or 'none (silent footage)'}")
        middle = os.path.join(tmp, "middle.mp4")
        segment(BROLL, middle, BROLL_S, audio=voice)
        pieces.insert(1, middle)
        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{p}'\n" for p in pieces)
        ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", OUT)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                OUT], capture_output=True, text=True).stdout)
    assert abs(dur - (2 * CLIP_S + BROLL_S)) < 1.0, f"unexpected length {dur:.1f} s"
    print(f"saved {OUT}: {dur:.0f} s, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
