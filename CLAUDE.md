# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Facebook Page comment sentiment dashboard. A classical ML pipeline (TF-IDF +
Logistic Regression, trained on Sentiment140) classifies comments pulled from
the Graph API; Flask serves the results; a Vue 3 SPA charts them.

The project's stated purpose is **learning classical ML and dataset handling**,
not production infrastructure. Prefer the simple, legible option over the
scalable one, and keep the teaching comments in `train.py`, `dataset.py` and
`predict.py` intact when editing them — they are the deliverable, not noise.

## Toolchain — neither binary is on PATH

Both bite in non-interactive shells:

```bash
export PATH="$HOME/.local/bin:$PATH"                          # uv
export PATH="/home/minhaz/.nvm/versions/node/v23.5.0/bin:$PATH"  # node + npm
```

`npm` silently "succeeded" once because `npm install 2>&1 | tail` masked the
failure exit code. Use `set -o pipefail` when piping build commands.

## Commands

```bash
# Backend (from backend/)
uv sync
uv run python download_data.py            # Sentiment140 from Stanford, no Kaggle account
uv run python train.py                    # 200k sample, ~1 min
uv run python train.py --full             # all 1.6M rows
uv run python train.py --sample 50000 --cv 5
uv run python predict.py                  # smoke-test the classifier
uv run python get_token.py                # mint a Facebook Page token into .env
uv run python app.py                      # http://127.0.0.1:5000

# Frontend (from frontend/)
npm install
npm run dev                               # :5173, proxies /api to Flask
npm run build                             # builds into ../backend/static/
```

There is no test suite and no linter configured. Verify changes by running the
app and exercising the endpoints.

## Architecture

Layers, bottom-up. Each depends only on those below it:

```
dataset.py    all data decisions: cleaning, label mapping, splitting
train.py      all modelling decisions; writes models/*.joblib + model_meta.json
predict.py    loads the pipeline, applies the neutral band  <- the web app uses this
db.py         SQLite; idempotent ingest keyed on Facebook's comment id
facebook.py   Graph API client
app.py        Flask: JSON API + serves the built SPA
frontend/     Vue 3 SFCs; components read colors from CSS vars, never hardcode
```

`train.py` must never be imported by the web app. `predict.py` is the boundary.

### Build output lives in backend/static/

`frontend/vite.config.js` sets `outDir: '../backend/static'`. **`backend/static/`
is generated — never edit it by hand**, and it is gitignored. After changing
anything under `frontend/src/`, run `npm run build` or the served app will not
reflect it.

## Constraints that will bite you

**Never `render_template()` the SPA.** Jinja and Vue both use `{{ }}`, so Jinja
tries to evaluate every Vue expression and raises `TemplateSyntaxError`. The
route uses `send_static_file()`. Nothing on the page needs server-side
templating.

**Flask is constructed with `static_url_path=""`.** Vite emits root-relative
`/assets/...` URLs; Flask's default `/static/` prefix would 404 them. Changing
this means adding a matching `base` to the Vite config.

**Graph API tokens go in an `Authorization: Bearer` header, never the query
string.** Query strings land in tracebacks, access logs and proxy logs, and a
Page access token does not expire — one leak is permanent until manually
revoked. `facebook._redact()` strips `EAA…` patterns from error text as a
second line of defence, and `_get()` raises `from None` to keep the
token-bearing exception chain out of logs.

**No sample, demo, or placeholder data anywhere.** An earlier demo mode wrote
fabricated posts into the same SQLite table as real Page data, and the two were
indistinguishable afterwards. When Facebook is unconfigured the app shows an
empty state and says so. Do not reintroduce fixtures that reach the database.

**`train.py` always overwrites `models/sentiment_pipeline.joblib`.** A quick
`--sample 20000` experiment silently replaces a good model with a worse one.
Retrain at the intended size afterwards.

**The raw dataset is sorted by label** — rows 0–799,999 are all negative.
`df.head(n)` yields a 100% negative dataset. `dataset.load_raw()` samples
randomly for this reason.

**Cleaning and prediction must use the same `clean_text()`.** `predict.py`
imports it from `dataset.py` rather than redefining it; if they drift, the
model sees a different distribution than it was fitted on and accuracy drops
with no visible error.

## Modelling conventions

- **Always a `sklearn.Pipeline`**, never a bare classifier. Saving the model
  without its vectorizer is the classic failure: `fit_transform()` on new text
  builds a fresh vocabulary, so feature columns stop meaning what they meant at
  training time. Nothing raises; accuracy just collapses.
- **Three splits.** Compare configurations on `val` freely; `test` is evaluated
  once, at the end. Tuning against `test` invalidates the reported number.
- **The neutral band** (`predict.NEUTRAL_LOW/HIGH`, currently 0.40–0.60) is a
  calibration choice, not a trained one. Sentiment140 is binary, so neutral
  comments are recovered by reading the probability rather than the label.
  Widening it trades coverage for confidence.
- Feature-engineering decisions interact: the bigram configs drop stop-word
  removal because sklearn's English list contains `not`/`no`/`cannot`, which
  would delete the negation bigrams the config exists to capture.

**English only.** Sentiment140 contains no Bangla. `clean_text()` keeps `[a-z]`,
so Bangla-script text cleans to an empty string; `predict.classify()` detects
this and returns neutral with `confidence: 0.0` rather than inventing a score.
Romanized Banglish survives cleaning but is absent from the vocabulary, so those
scores are unreliable too.

## Chart colors

Sentiment is **polarity, not identity**, so the charts use a diverging scale —
two opposed hues with a neutral gray midpoint — not three categorical colors.
Tokens are defined once in `frontend/src/style.css` and read via
`getComputedStyle`; components never hardcode hex.

Both themes were validated for colorblind separation (worst adjacent pair ΔE 8.7
light, 10.3 dark; target ≥ 8). Dark mode uses `#b5b3a8` for neutral rather than
the light-mode `#898781`, because at `#898781` the dark-mode red and gray
collapse to ΔE 3.9 under protanopia. Re-validate if you change these.

Charts bake colors in at construction, so they subscribe to
`theme.onThemeChange()` and redraw rather than picking up the CSS change.

## Facebook specifics

- Development Mode is sufficient for Pages the developer administers; App Review
  is only needed for other people's Pages.
- Required permissions: `pages_show_list`, `pages_read_engagement`,
  `pages_read_user_content`. Missing the third yields posts with **zero
  comments** and no error.
- The `from` field on a comment is absent unless the commenter authorized the
  app, so most authors are `Unknown`. Never assume the key exists.
- Page tokens derived from a long-lived user token do not expire, but die on
  password change or loss of admin.
- `backend/.env` holds the Page token and is gitignored.
