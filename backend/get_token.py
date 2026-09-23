"""Turn a short-lived token from the Graph API Explorer into a permanent Page token.

Run:
    uv run python get_token.py

You will be asked for three things, all from the Meta developer portal:
    App ID, App Secret, and a short-lived User token.

What it does (the part that is fiddly to do by hand):

  1. short-lived user token  --(needs your app secret)-->  long-lived user token
  2. long-lived user token   --> the Page tokens for every page you administer

Step 1 must happen server-side because it needs the app secret, which is why
this is a script and not something you can do in the browser.

The Page token that comes out of step 2 does NOT expire -- that is the one you
put in .env.
"""
from __future__ import annotations

from getpass import getpass
from pathlib import Path
import sys

import requests

GRAPH = "https://graph.facebook.com/v21.0"
ENV_PATH = Path(__file__).resolve().parent / ".env"


def die(message: str, payload: dict | None = None) -> None:
    print(f"\n  ERROR: {message}")
    if payload:
        error = payload.get("error", {})
        if error:
            print(f"  Facebook says: {error.get('message')}")
            print(f"  (type={error.get('type')} code={error.get('code')})")
    sys.exit(1)


def exchange_for_long_lived(app_id: str, app_secret: str, short_token: str) -> str:
    """Step 1: short-lived user token -> long-lived user token (~60 days)."""
    print("\n[1/3] Exchanging for a long-lived user token...")
    response = requests.get(
        f"{GRAPH}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": short_token,
        },
        timeout=30,
    )
    payload = response.json()
    if "access_token" not in payload:
        die("could not exchange the token", payload)
    expires = payload.get("expires_in")
    print(f"      got it" + (f" (expires in ~{int(expires) // 86400} days)" if expires else ""))
    return payload["access_token"]


def list_pages(long_token: str) -> list[dict]:
    """Step 2: long-lived user token -> a Page token per page you administer.

    These Page tokens do not expire, as long as you stay an admin of the page
    and do not change your Facebook password.
    """
    print("[2/3] Fetching your Pages and their tokens...")
    response = requests.get(
        f"{GRAPH}/me/accounts",
        params={"access_token": long_token, "fields": "id,name,access_token"},
        timeout=30,
    )
    payload = response.json()
    if "data" not in payload:
        die("could not list your Pages", payload)
    pages = payload["data"]
    if not pages:
        die("no Pages found. Are you an admin of at least one Facebook Page, "
            "and did you grant the pages_show_list permission?")
    return pages


def verify(page_token: str, page_id: str) -> None:
    """Step 3: prove the token actually works before we write it to disk."""
    print("[3/3] Verifying the Page token can read posts...")
    response = requests.get(
        f"{GRAPH}/{page_id}/posts",
        params={"access_token": page_token, "limit": 1, "fields": "id"},
        timeout=30,
    )
    payload = response.json()
    if "data" not in payload:
        die("the Page token did not work for reading posts", payload)
    print(f"      works ({len(payload['data'])} post(s) visible)")


def write_env(page_id: str, page_token: str) -> None:
    if ENV_PATH.exists():
        answer = input(f"\n  {ENV_PATH.name} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("\n  Not written. Add these lines yourself:\n")
            print(f"    FB_PAGE_ID={page_id}")
            print(f"    FB_PAGE_ACCESS_TOKEN={page_token}")
            return
    ENV_PATH.write_text(
        f"FB_PAGE_ID={page_id}\nFB_PAGE_ACCESS_TOKEN={page_token}\n"
    )
    print(f"\n  Wrote {ENV_PATH}")
    print("  Restart the app and click View latest expression to pull your comments.")


def main() -> None:
    print(__doc__)
    app_id = input("App ID: ").strip()
    app_secret = getpass("App Secret (hidden): ").strip()
    short_token = getpass("Short-lived User token (hidden): ").strip()

    if not (app_id and app_secret and short_token):
        die("all three values are required")

    long_token = exchange_for_long_lived(app_id, app_secret, short_token)
    pages = list_pages(long_token)

    print(f"\n  Found {len(pages)} page(s):")
    for i, page in enumerate(pages, 1):
        print(f"    {i}. {page['name']}  (id {page['id']})")

    if len(pages) == 1:
        chosen = pages[0]
    else:
        raw = input(f"\n  Which page? [1-{len(pages)}]: ").strip()
        try:
            chosen = pages[int(raw) - 1]
        except (ValueError, IndexError):
            die("that was not one of the listed numbers")

    verify(chosen["access_token"], chosen["id"])
    print(f"\n  Page:  {chosen['name']}")
    print(f"  ID:    {chosen['id']}")
    print(f"  Token: {chosen['access_token'][:24]}... ({len(chosen['access_token'])} chars)")
    write_env(chosen["id"], chosen["access_token"])


if __name__ == "__main__":
    main()
