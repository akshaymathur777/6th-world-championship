"""Render a ≤10 minute championship video from slides + demo transcript."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "tmp" / "video_slides"
VIDEO = ROOT / "docs" / "championship_video.mp4"
FFMPEG = (
    Path(r"C:\Users\aksha\AppData\Local\Microsoft\WinGet\Packages")
    / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "ffmpeg-8.1.1-full_build"
    / "bin"
    / "ffmpeg.exe"
)
W, H = 1920, 1080
BG = "#0B1F2A"
PANEL = "#0E2A38"
ACCENT = "#3BA3A0"
GOLD = "#F0C75E"
TEXT = "#E8F4F3"
MUTED = "#9BB7C3"
SOFT = "#8FCDCB"


def font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "calibri.ttf"):
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE = font(64)
HEAD = font(48)
BODY = font(34)
SMALL = font(26)


def new_slide() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    for x in range(0, W, 48):
        draw.line([(x, 0), (x, H)], fill="#123447")
    for y in range(0, H, 48):
        draw.line([(0, y), (W, y)], fill="#123447")
    return img, draw


def wrap(draw, text: str, x: int, y: int, max_chars: int, fill, fnt, gap: int = 44):
    for line in textwrap.wrap(text, width=max_chars):
        draw.text((x, y), line, fill=fill, font=fnt)
        y += gap
    return y


def save(img: Image.Image, name: str, seconds: float, frames: list[tuple[Path, float]]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{len(frames):02d}_{name}.png"
    img.save(path)
    frames.append((path, seconds))


def part_banner(draw, part: str):
    draw.rounded_rectangle([60, 40, 420, 110], radius=16, fill=PANEL, outline=ACCENT, width=2)
    draw.text((90, 55), part, fill=GOLD, font=BODY)


def build_slides() -> list[tuple[Path, float]]:
    frames: list[tuple[Path, float]] = []
    winner = json.loads((ROOT / "data" / "winner.json").read_text(encoding="utf-8"))
    experiments = (ROOT / "experiments.csv").read_text(encoding="utf-8").strip().splitlines()

    # 0 cover
    img, draw = new_slide()
    draw.rectangle([0, 0, 760, H], fill=PANEL)
    draw.rectangle([70, 120, 690, 960], outline=ACCENT, width=3)
    draw.text((110, 200), "Local CDC", fill=TEXT, font=TITLE)
    draw.text((110, 280), "Guideline Retrieval", fill=TEXT, font=TITLE)
    draw.text((110, 400), "Privacy-first agentic RAG", fill=SOFT, font=HEAD)
    draw.text((110, 480), "Question stays on-device", fill=MUTED, font=BODY)
    draw.text((110, 540), "Public guidance only in the index", fill=MUTED, font=BODY)
    draw.text((110, 640), "Chroma + YankeeAI Desktop", fill=GOLD, font=BODY)
    draw.text((110, 820), "6th World Championship in Data Science & AI", fill=MUTED, font=SMALL)
    for i, label in enumerate(
        ["CDC public guidance", "Local Chroma index", "Yankee chat + citations"]
    ):
        y0 = 180 + i * 220
        draw.rounded_rectangle([900, y0, 1780, y0 + 140], radius=18, fill="#143948", outline=ACCENT, width=2)
        draw.text((960, y0 + 45), label, fill=TEXT, font=HEAD)
        if i < 2:
            draw.line([(1340, y0 + 140), (1340, y0 + 220)], fill=GOLD, width=5)
    save(img, "cover", 8, frames)

    # Part 1 opener
    img, draw = new_slide()
    part_banner(draw, "Part 1 — Vector database")
    draw.text((80, 180), "A clinician question never leaves the machine.", fill=TEXT, font=HEAD)
    y = wrap(
        draw,
        "Retrieval runs over public CDC guidance only. The published artifact is the index and the original guidelines — not any patient text.",
        80,
        280,
        70,
        MUTED,
        BODY,
    )
    draw.text((80, y + 40), "Canonical metric: Recall@5 on a frozen 200-question set.", fill=GOLD, font=BODY)
    save(img, "part1_open", 25, frames)

    # Privacy flow
    img, draw = new_slide()
    part_banner(draw, "Part 1 — Privacy boundary")
    boxes = [
        (120, 220, "CDC guidance"),
        (700, 220, "Section chunks"),
        (1280, 220, "Local Chroma"),
        (700, 520, "Recall@5"),
        (1280, 520, "Yankee ZIP"),
        (1280, 780, "Cited local answer"),
    ]
    for x, y, label in boxes:
        draw.rounded_rectangle([x, y, x + 420, y + 120], radius=16, fill=PANEL, outline=ACCENT, width=2)
        draw.text((x + 40, y + 35), label, fill=TEXT, font=BODY)
    for a, b in [((540, 280), (700, 280)), ((1120, 280), (1280, 280)), ((1490, 340), (1490, 520)), ((1490, 640), (1490, 780)), ((910, 340), (910, 520))]:
        draw.line([a, b], fill=GOLD, width=4)
    draw.rounded_rectangle([120, 780, 640, 960], radius=16, fill="#1A3A2A", outline=GOLD, width=3)
    draw.text((160, 820), "Patient question", fill=GOLD, font=BODY)
    draw.text((160, 880), "stays local → Yankee", fill=SOFT, font=BODY)
    save(img, "privacy", 35, frames)

    # Corpus
    img, draw = new_slide()
    part_banner(draw, "Part 1 — Corpus")
    draw.text((80, 180), "epfl-llm/guidelines · source=cdc", fill=TEXT, font=HEAD)
    stats = [
        "616 documents kept (6.77M words)",
        "5 dropped (courtesy / third-party / too short)",
        "Titles recovered from headings; first cdc.gov URL when present",
        "U.S. government works only — no NICE / textbook bleed-in",
    ]
    y = 280
    for line in stats:
        draw.ellipse([90, y + 12, 110, y + 32], fill=ACCENT)
        draw.text((140, y), line, fill=MUTED, font=BODY)
        y += 70
    save(img, "corpus", 40, frames)

    # Chunking
    img, draw = new_slide()
    part_banner(draw, "Part 1 — Chunking")
    draw.text((80, 180), "Families timeboxed; section-aware won", fill=TEXT, font=HEAD)
    rows = [
        ("fixed_256 + nomic", "0.530", "worse"),
        ("fixed_128 + nomic", "0.550", "worse"),
        ("section + nomic", "0.645", "replaced later"),
        ("recommendation + nomic", "0.645", "within noise"),
        ("section + mxbai-embed-large", "0.760", "KEPT"),
    ]
    y = 280
    draw.text((140, y), "Config", fill=GOLD, font=SMALL)
    draw.text((980, y), "Recall@5", fill=GOLD, font=SMALL)
    draw.text((1280, y), "Decision", fill=GOLD, font=SMALL)
    y += 50
    for cfg, score, decision in rows:
        color = GOLD if decision == "KEPT" else MUTED
        draw.text((140, y), cfg, fill=color, font=BODY)
        draw.text((980, y), score, fill=color, font=BODY)
        draw.text((1280, y), decision, fill=color, font=BODY)
        y += 60
    draw.text((80, 880), "Every chunk carries url, title, section, chunk_id.", fill=SOFT, font=BODY)
    save(img, "chunking", 45, frames)

    # Metric / experiments
    img, draw = new_slide()
    part_banner(draw, "Part 1 — One number")
    draw.text((80, 180), f"Winner Recall@5 = {winner['recall_at_5']:.3f}", fill=GOLD, font=TITLE)
    y = wrap(
        draw,
        "A top-5 hit is either the gold URL+section or the gold evidence quote. A challenger replaces the incumbent only when the gain clears 2√(se_a²+se_b²). Latency never overrides Recall@5.",
        80,
        300,
        70,
        MUTED,
        BODY,
    )
    draw.text((80, y + 40), f"SE ≈ {winner['recall_se']:.3f} · noise floor ≈ {winner['noise_floor']:.3f}", fill=SOFT, font=BODY)
    draw.text((80, y + 100), f"p50 latency {winner['p50_latency_ms']:.0f} ms · {winner['n_chunks']} chunks", fill=MUTED, font=BODY)
    draw.text((80, 920), f"experiments.csv has {len(experiments) - 1} scored runs", fill=MUTED, font=SMALL)
    save(img, "metric", 45, frames)

    # Embeddings
    img, draw = new_slide()
    part_banner(draw, "Part 1 — Embeddings")
    draw.text((80, 200), "Yankee-loadable Ollama models only", fill=TEXT, font=HEAD)
    for i, (name, note) in enumerate(
        [
            ("nomic-embed-text", "staff pick · early baseline"),
            ("all-minilm", "within noise of section+nomic"),
            ("mxbai-embed-large", "WINNER · ollama:mxbai-embed-large"),
        ]
    ):
        y = 320 + i * 140
        outline = GOLD if "WINNER" in note else ACCENT
        draw.rounded_rectangle([120, y, 1800, y + 110], radius=16, fill=PANEL, outline=outline, width=2)
        draw.text((160, y + 30), name, fill=TEXT, font=HEAD)
        draw.text((900, y + 40), note, fill=SOFT, font=BODY)
    save(img, "embeddings", 25, frames)

    # Part 2 opener
    img, draw = new_slide()
    part_banner(draw, "Part 2 — YankeeAI integration")
    draw.text((80, 200), "Import the v3 collection ZIP", fill=TEXT, font=HEAD)
    steps = [
        "1. My Files → Collections → Import",
        "2. Select cdc-guidelines-yankee.zip",
        "3. Embedding model: ollama:mxbai-embed-large",
        "4. Chat with collection — Simple or Agent RAG",
    ]
    y = 320
    for step in steps:
        draw.text((120, y), step, fill=MUTED, font=BODY)
        y += 70
    draw.text((80, 720), "format = yankeeai-collection-export · version = 3", fill=GOLD, font=BODY)
    draw.text((80, 800), "No Chroma Cloud. No cloud LLM.", fill=SOFT, font=BODY)
    save(img, "part2_import", 40, frames)

    # Demo questions from transcript
    transcript = (ROOT / "docs" / "demo_transcript.md").read_text(encoding="utf-8")
    blocks = [b.strip() for b in transcript.split("## ") if b.strip().startswith("q")]
    for block in blocks[:3]:
        lines = block.splitlines()
        qid = lines[0].strip()
        question = next((ln for ln in lines[1:] if ln.strip() and not ln.startswith("|")), "")
        table_rows = [ln for ln in lines if ln.startswith("|") and "Rank" not in ln and "---" not in ln][:3]
        answer_lines = []
        seen_table = False
        for ln in lines:
            if ln.startswith("|"):
                seen_table = True
                continue
            if seen_table and ln.strip():
                answer_lines.append(ln.strip())
        answer = " ".join(answer_lines)[:420]
        img, draw = new_slide()
        part_banner(draw, f"Part 2 — Demo {qid}")
        wrap(draw, question, 80, 170, 68, TEXT, HEAD, gap=56)
        y = 360
        draw.text((80, y), "Top retrieved sections", fill=GOLD, font=BODY)
        y += 60
        for row in table_rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) >= 3:
                draw.text((100, y), f"{cells[0]}. {cells[1][:70]}", fill=MUTED, font=SMALL)
                y += 40
        y += 20
        draw.text((80, y), "Local answer (cited)", fill=GOLD, font=BODY)
        wrap(draw, answer, 80, y + 50, 78, SOFT, SMALL, gap=36)
        save(img, f"demo_{qid}", 55, frames)

    # Close
    img, draw = new_slide()
    part_banner(draw, "Close")
    draw.text((80, 200), "Sensitive question stays local.", fill=TEXT, font=HEAD)
    draw.text((80, 300), "Public CDC guidance powers retrieval.", fill=TEXT, font=HEAD)
    links = [
        "Code: github.com/akshaymathur777/6th-world-championship",
        "Dataset: huggingface.co/datasets/akshaymathur777/cdc-clinical-guidelines-privacy-first",
        "Import: cdc-guidelines-yankee.zip · ollama:mxbai-embed-large",
        "Recall@5 = 0.760 · section chunks · mxbai-embed-large",
    ]
    y = 420
    for link in links:
        draw.text((100, y), link, fill=MUTED, font=BODY)
        y += 70
    draw.text((80, 860), "Not a medical device. Not clinical advice.", fill=SOFT, font=SMALL)
    save(img, "close", 30, frames)

    return frames


def stitch(frames: list[tuple[Path, float]]) -> None:
    list_file = OUT_DIR / "concat.txt"
    lines = []
    for path, seconds in frames:
        # ffmpeg concat demuxer needs escaped paths on Windows
        safe = path.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{safe}'")
        lines.append(f"duration {seconds}")
    # repeat last frame for concat demuxer
    last = frames[-1][0].resolve().as_posix().replace("'", r"'\''")
    lines.append(f"file '{last}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FFMPEG),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        "fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO),
    ]
    subprocess.run(cmd, check=True)
    total = sum(seconds for _, seconds in frames)
    print(f"Wrote {VIDEO} ({total:.0f}s, {len(frames)} slides)")


def main() -> None:
    if OUT_DIR.exists():
        for old in OUT_DIR.glob("*"):
            old.unlink()
    frames = build_slides()
    stitch(frames)


if __name__ == "__main__":
    main()
