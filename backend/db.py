"""SQLite storage for fetched comments and their classifications.

SQLite is the right call here: one file, no server, no credentials, and it
handles far more than a Facebook page will ever produce. Swap to Postgres if
this ever becomes multi-user; nothing above this layer needs to change.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id            TEXT PRIMARY KEY,
    message       TEXT,
    created_time  TEXT,
    permalink     TEXT
);

CREATE TABLE IF NOT EXISTS comments (
    id            TEXT PRIMARY KEY,   -- Facebook's comment id
    post_id       TEXT,
    message       TEXT NOT NULL,
    author        TEXT,
    created_time  TEXT,
    sentiment     TEXT,               -- positive | negative | neutral
    score         REAL,               -- P(positive), 0..1
    confidence    REAL,
    classified_at TEXT,
    FOREIGN KEY (post_id) REFERENCES posts(id)
);

CREATE INDEX IF NOT EXISTS idx_comments_sentiment ON comments(sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_created   ON comments(created_time);
CREATE INDEX IF NOT EXISTS idx_comments_post      ON comments(post_id);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row      # rows behave like dicts
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def save_posts(posts: list[dict]) -> None:
    with connect() as conn:
        conn.executemany(
            """INSERT INTO posts (id, message, created_time, permalink)
               VALUES (:id, :message, :created_time, :permalink)
               ON CONFLICT(id) DO UPDATE SET message = excluded.message""",
            posts,
        )


def save_comments(comments: list[dict]) -> int:
    """Insert comments, skipping any we already hold. Returns the new count.

    The ON CONFLICT DO NOTHING is what makes the refresh button safe to mash:
    Facebook hands back the same comments every time you ask, and because the
    primary key is Facebook's own comment id, re-inserting is a no-op. Without
    it you would double-count every comment on every refresh.
    """
    if not comments:
        return 0
    with connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        conn.executemany(
            """INSERT INTO comments
                   (id, post_id, message, author, created_time)
               VALUES (:id, :post_id, :message, :author, :created_time)
               ON CONFLICT(id) DO NOTHING""",
            comments,
        )
        after = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    return after - before


def unclassified(limit: int = 1000) -> list[sqlite3.Row]:
    """Comments we have stored but not yet scored.

    Classification is idempotent and cheap, but skipping already-scored rows
    is what keeps a refresh O(new comments) instead of O(everything).
    """
    with connect() as conn:
        return conn.execute(
            "SELECT id, message FROM comments WHERE sentiment IS NULL LIMIT ?",
            (limit,),
        ).fetchall()


def save_classifications(rows: list[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["classified_at"] = now
    with connect() as conn:
        conn.executemany(
            """UPDATE comments
                  SET sentiment = :sentiment, score = :score,
                      confidence = :confidence, classified_at = :classified_at
                WHERE id = :id""",
            rows,
        )


def analytics() -> dict:
    """Everything the dashboard needs, in one round trip."""
    with connect() as conn:
        totals = {r["sentiment"]: r["n"] for r in conn.execute(
            """SELECT sentiment, COUNT(*) AS n FROM comments
                WHERE sentiment IS NOT NULL GROUP BY sentiment"""
        )}

        total = sum(totals.values())

        timeline = [dict(r) for r in conn.execute(
            """SELECT DATE(created_time) AS day, sentiment, COUNT(*) AS n
                 FROM comments
                WHERE sentiment IS NOT NULL AND created_time IS NOT NULL
             GROUP BY day, sentiment
             ORDER BY day"""
        )]

        per_post = [dict(r) for r in conn.execute(
            """SELECT p.id, p.message, p.permalink,
                      COUNT(c.id) AS total,
                      SUM(c.sentiment = 'positive') AS positive,
                      SUM(c.sentiment = 'negative') AS negative,
                      SUM(c.sentiment = 'neutral')  AS neutral
                 FROM posts p JOIN comments c ON c.post_id = p.id
                WHERE c.sentiment IS NOT NULL
             GROUP BY p.id
             ORDER BY total DESC"""
        )]

        # Most-negative comments first - in practice this is the screen you
        # actually act on, since these are the ones needing a reply.
        # Capped at 50: this is a worklist, not an archive. The UI pages
        # through these client-side, so the cap is what bounds the payload.
        attention = [dict(r) for r in conn.execute(
            """SELECT id, message, author, created_time, sentiment, score
                 FROM comments
                WHERE sentiment = 'negative'
             ORDER BY score ASC LIMIT 50"""
        )]

    return {
        "totals": {
            "positive": totals.get("positive", 0),
            "negative": totals.get("negative", 0),
            "neutral": totals.get("neutral", 0),
            "total": total,
        },
        "percentages": {
            k: round(totals.get(k, 0) / total * 100, 1) if total else 0.0
            for k in ("positive", "negative", "neutral")
        },
        "timeline": timeline,
        "per_post": per_post,
        "needs_attention": attention,
    }


def list_comments(sentiment: str | None = None, limit: int = 100) -> list[dict]:
    query = """SELECT id, post_id, message, author, created_time,
                      sentiment, score, confidence
                 FROM comments WHERE sentiment IS NOT NULL"""
    params: list = []
    if sentiment:
        query += " AND sentiment = ?"
        params.append(sentiment)
    query += " ORDER BY created_time DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return [dict(r) for r in conn.execute(query, params)]
