"""Download Overwatch hero portrait images to data/images/.

Usage:
    python scripts/download_images.py
    python scripts/download_images.py --heroes-path data/heroes.json --out-dir data/images
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# Overfast API — open-source community OW API, returns hero portraits
_OVERFAST_API = "https://overfast-api.tekrop.fr/heroes"
# Blizzard CDN fallback pattern
_BLIZZARD_CDN = "https://d15f34w2p8l1cc.cloudfront.net/overwatch/{slug}.png"


def _fetch_overfast_portraits() -> dict[str, str]:
    """Fetches hero portrait URLs from the Overfast API.

    Returns:
        Dict mapping hero key (slug-like) to portrait URL.
        Empty dict if the request fails.
    """
    req = urllib.request.Request(
        _OVERFAST_API,
        headers={"User-Agent": "ow-rag-multimodal/0.1 (image downloader)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return {hero["key"]: hero["portrait"] for hero in data if "portrait" in hero}
    except Exception as exc:
        print(f"  Warning: Overfast API unavailable ({exc}). Falling back to CDN pattern.")
        return {}


def _download(url: str, dest: Path) -> bool:
    """Downloads a URL to dest. Returns True on success."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ow-rag-multimodal/0.1 (image downloader)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"    Failed ({exc})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OW hero portrait images.")
    parser.add_argument(
        "--heroes-path",
        type=Path,
        default=Path("data/heroes.json"),
        help="Path to heroes.json (default: data/heroes.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/images"),
        help="Output directory for images (default: data/images)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists.",
    )
    args = parser.parse_args()

    if not args.heroes_path.exists():
        print(f"Error: heroes file not found at {args.heroes_path}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.heroes_path.open(encoding="utf-8") as f:
        heroes = json.load(f)

    slugs = list(heroes.keys())
    print(f"Fetching portrait URLs from Overfast API...")
    portrait_map = _fetch_overfast_portraits()

    ok = 0
    skipped = 0
    failed: list[str] = []

    for slug in slugs:
        dest = args.out_dir / f"{slug}.png"

        if dest.exists() and not args.force:
            skipped += 1
            continue

        # Try Overfast URL first, then CDN fallback
        url = portrait_map.get(slug) or _BLIZZARD_CDN.format(slug=slug)
        print(f"  {slug} -> {url[:70]}...")

        if _download(url, dest):
            ok += 1
        else:
            # Second attempt: CDN fallback with dash-stripped slug
            alt_slug = slug.replace("-", "")
            if alt_slug != slug:
                alt_url = _BLIZZARD_CDN.format(slug=alt_slug)
                print(f"    Retrying with {alt_url[:70]}...")
                if _download(alt_url, dest):
                    ok += 1
                    continue
            failed.append(slug)

        time.sleep(0.1)  # be polite

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print(f"Failed slugs: {', '.join(failed)}")
        print("You can manually place PNG files at data/images/{slug}.png")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
