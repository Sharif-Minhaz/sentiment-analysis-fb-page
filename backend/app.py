"""Flask API + dashboard.

Run:
    uv run python app.py
    -> http://127.0.0.1:5000

Flask serves both the JSON API and the single-page Vue front end. Serving
them from one origin means no CORS configuration and no second dev server --
one command and the whole thing is up.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Must run before facebook.py reads the environment at import time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import Flask, jsonify, request  # noqa: E402

import db  # noqa: E402
import facebook  # noqa: E402
import predict  # noqa: E402

# static_url_path="" mounts the static folder at the site root, so Vite's
# root-relative /assets/... URLs resolve without a custom `base` in
# vite.config.js. Dev (Vite proxy) and production then use identical paths.
app = Flask(__name__, static_url_path="")
db.init_db()


@app.get("/")
def index():
    # Served as a static file, NOT render_template(): Jinja and Vue both use
    # {{ }} delimiters, so Jinja tries to evaluate every Vue expression and
    # dies on the first one. Nothing here needs server-side templating.
    return app.send_static_file("index.html")


@app.get("/api/analytics")
def get_analytics():
    """Everything the charts need. Reads only - never touches Facebook."""
    return jsonify(db.analytics())


@app.post("/api/refresh")
def refresh():
    """The 'View latest expression' button.

    Three steps, in order:
      1. pull the newest posts + comments from Facebook
      2. store them, skipping any comment id we already hold
      3. classify only the rows that have no sentiment yet

    Step 3 is why this stays fast as the table grows: a refresh costs time
    proportional to the *new* comments, not the total.
    """
    try:
        posts, comments = facebook.fetch_latest()
    except facebook.GraphError as exc:
        # A network blip or a bad token is an expected condition, not a
        # crash -- return it as JSON so the UI can show it.
        return jsonify({"error": str(exc)}), 502

    db.save_posts(posts)
    new_count = db.save_comments(comments)

    pending = db.unclassified()
    classified = 0
    if pending:
        # One batched call for the whole backlog - far faster than looping.
        results = predict.classify([row["message"] for row in pending])
        db.save_classifications([
            {
                "id": row["id"],
                "sentiment": result["sentiment"],
                "score": result["score"],
                "confidence": result["confidence"],
            }
            for row, result in zip(pending, results)
        ])
        classified = len(results)

    return jsonify({
        "posts_seen": len(posts),
        "comments_seen": len(comments),
        "new_comments": new_count,
        "newly_classified": classified,
        "analytics": db.analytics(),
    })


@app.get("/api/comments")
def get_comments():
    sentiment = request.args.get("sentiment")
    limit = min(int(request.args.get("limit", 100)), 500)
    return jsonify(db.list_comments(sentiment=sentiment, limit=limit))


@app.post("/api/classify")
def classify_text():
    """Try-it-yourself box: type a sentence, see what the model thinks."""
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    return jsonify(predict.classify_one(text))


@app.get("/api/model")
def model_info():
    """Training metrics, surfaced in the UI.

    Putting the model's own accuracy and top features on screen keeps the
    thing you built visible, instead of hiding it behind the charts.
    """
    meta = predict.load_meta()
    return jsonify({
        "trained": bool(meta),
        "configured": facebook.is_configured(),
        "neutral_band": [predict.NEUTRAL_LOW, predict.NEUTRAL_HIGH],
        **meta,
    })


if __name__ == "__main__":
    if not predict.MODEL_PATH.exists():
        print("\n  No trained model found.")
        print("  Run this first:  uv run python train.py\n")
        raise SystemExit(1)

    if facebook.is_configured():
        print(f"\n  Facebook page: {facebook.PAGE_ID}")
    else:
        print("\n  Facebook not configured - the dashboard will load but have")
        print("  nothing to show. Run:  uv run python get_token.py")
    print("  Dashboard:     http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
