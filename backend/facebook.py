"""Facebook Graph API client.

Configure in backend/.env:
    FB_PAGE_ID=...
    FB_PAGE_ACCESS_TOKEN=...

Get those by running:  uv run python get_token.py

There is deliberately no sample/demo data here. Fabricated comments mixed with
real ones in the same database are impossible to tell apart after the fact --
an empty dashboard is always clearer than a plausible-looking fake one.
"""
from __future__ import annotations

import os
import re

import requests

GRAPH_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"

PAGE_ID = os.getenv("FB_PAGE_ID", "").strip()
PAGE_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "").strip()


class GraphError(RuntimeError):
    """Anything that went wrong talking to Facebook, with no token in the text."""


def is_configured() -> bool:
    return bool(PAGE_ID and PAGE_TOKEN)


def _redact(text: str) -> str:
    """Strip anything that looks like an access token out of a message.

    Belt and braces. The token no longer travels in the URL, but Facebook
    sometimes echoes input back, and an error string can end up in a log file,
    a bug report, or a chat window. A leaked Page token does not expire.
    """
    return re.sub(r"EAA[A-Za-z0-9_\-]{20,}", "<redacted-token>", text)


def _get(path: str, **params) -> dict:
    """GET a Graph API path, with the token in a header rather than the URL.

    The token goes in `Authorization: Bearer`, NOT in the query string. Query
    strings end up in exception messages, access logs, proxy logs and browser
    history -- and a Page access token does not expire, so one leak is
    permanent until it is manually revoked. Facebook accepts either form.
    """
    try:
        response = requests.get(
            f"{GRAPH_BASE}/{path}",
            params=params,
            headers={"Authorization": f"Bearer {PAGE_TOKEN}"},
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        # DNS failure, no route, TLS problem, timeout. Not a bug -- the network
        # is just unavailable -- so report it as such instead of a 500.
        raise GraphError(
            "Could not reach graph.facebook.com. Check your internet "
            f"connection and try again. ({type(exc).__name__})"
        ) from None   # `from None` keeps the token-bearing chain out of the log

    if not response.ok:
        # Graph API puts the useful detail in the body, not the status line.
        raise GraphError(
            f"Graph API {response.status_code}: {_redact(response.text[:400])}"
        )
    return response.json()


def fetch_posts(limit: int = 10) -> list[dict]:
    payload = _get(f"{PAGE_ID}/posts",
                   fields="id,message,created_time,permalink_url",
                   limit=limit)
    return [
        {
            "id": post["id"],
            "message": post.get("message", ""),
            "created_time": post.get("created_time"),
            "permalink": post.get("permalink_url"),
        }
        for post in payload.get("data", [])
    ]


def fetch_comments(post_id: str, limit: int = 100) -> list[dict]:
    """Fetch comments for one post, following pagination cursors.

    filter=stream includes replies as well as top-level comments; drop it if
    you only want top-level. Each page hands back an 'after' cursor in
    paging.cursors -- keep passing it until Facebook stops returning one.
    """
    comments: list[dict] = []
    params = dict(fields="id,message,created_time,from", limit=limit,
                  filter="stream")
    path = f"{post_id}/comments"

    while True:
        payload = _get(path, **params)
        for comment in payload.get("data", []):
            if not comment.get("message"):
                continue                     # sticker/photo-only comment
            comments.append({
                "id": comment["id"],
                "post_id": post_id,
                "message": comment["message"],
                # 'from' is absent unless the commenter granted access -- very
                # common. Never assume this key exists.
                "author": (comment.get("from") or {}).get("name"),
                "created_time": comment.get("created_time"),
            })

        cursor = payload.get("paging", {}).get("cursors", {}).get("after")
        if not cursor or not payload.get("data"):
            break
        params["after"] = cursor

    return comments


def fetch_latest(post_limit: int = 10) -> tuple[list[dict], list[dict]]:
    """Return (posts, comments) for the newest activity on the page."""
    if not is_configured():
        raise GraphError(
            "Facebook is not configured. Create backend/.env with FB_PAGE_ID "
            "and FB_PAGE_ACCESS_TOKEN -- run `uv run python get_token.py` to "
            "generate them."
        )

    posts = fetch_posts(limit=post_limit)
    comments: list[dict] = []
    for post in posts:
        comments.extend(fetch_comments(post["id"]))
    return posts, comments
