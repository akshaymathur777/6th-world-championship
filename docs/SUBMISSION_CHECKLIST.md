# Submission checklist

Championship page: https://www.kaggle.com/competitions/privacy-first-agentic-ai  
ISODS registration: https://web.isods.org/events/world-championship-2026  
Hard deadline: **31 March 2027, 23:59 UTC**  
Internal buffer: **28 March 2027**

## Eligibility (pass/fail)

- [ ] Registered on ISODS **and** Kaggle
- [ ] Team 1–3 members in an eligible category
- [ ] Kaggle Writeup submitted (not left as draft) — paste from `docs/WRITEUP.md`
- [ ] Cover image in Media Gallery (`docs/cover.png`)
- [ ] YouTube video ≤10 min, public, Part 1 + Part 2 labeled — upload `docs/championship_video.mp4` (~7.6 min slideshow; optionally re-record Part 2 live in Yankee)
- [x] Public code repository: https://github.com/akshaymathur777/6th-world-championship
- [x] Public vector DB + original data: https://huggingface.co/datasets/akshaymathur777/cdc-clinical-guidelines-privacy-first
- [ ] All links open without login

## Qualitative scoring targets

| Criterion | Points | Where it lives |
| --- | --- | --- |
| Research impact & problem vision | 40 | `docs/WRITEUP.md` story + privacy boundary |
| Video presentation | 30 | `docs/VIDEO_SCRIPT.md` |
| Technical depth / Chroma optimization | 30 | `experiments.csv`, `data/winner.json`, code |

## Publish steps (after ablation)

1. Upload `data/corpus/cdc.jsonl` + Yankee zip + winner index to Hugging Face or Kaggle Dataset.
2. Push this repo public.
3. Record the video from `docs/VIDEO_SCRIPT.md` using `docs/demo_transcript.md`.
4. Paste writeup into Kaggle; attach video, repo, dataset, cover.
5. Submit; keep one edit buffer before 28 March 2027.
