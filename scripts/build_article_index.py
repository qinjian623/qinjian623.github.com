#!/usr/bin/env python3
import json
import re
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "_posts"
BIT_ASS_DIR = ROOT / "bit_ass"
OUTPUT_PATH = ROOT / "_data" / "article_index.json"

POST_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.+)\.(?P<ext>html|md)$")
BIT_ASS_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.+)\.md$")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.S)
TITLE_RE = re.compile(r"^title\s*:\s*(?P<title>.+?)\s*$", re.M)


def read_title(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return None

    title_match = TITLE_RE.search(match.group("body"))
    if not title_match:
        return None

    title = title_match.group("title").strip()
    if len(title) >= 2 and title[0] == title[-1] and title[0] in ("'", '"'):
        title = title[1:-1]
    return title or None


def title_from_slug(slug: str) -> str:
    return slug.replace("_", " ")


def post_url(date: str, slug: str) -> str:
    year, month, day = date.split("-")
    return f"/{year}/{month}/{day}/{quote(slug)}.html"


def bit_ass_url(slug_with_date: str) -> str:
    return f"/bit_ass/{quote(slug_with_date)}.html"


def normalized_key(date: str, title: str) -> str:
    compact_title = re.sub(r"\s+", "", title).lower()
    return f"{date}:{compact_title}"


def collect_articles() -> tuple[list[dict], list[str]]:
    candidates: list[dict] = []
    skipped: list[str] = []

    for path in sorted(POSTS_DIR.glob("*")):
        if path.suffix not in (".html", ".md"):
            continue

        match = POST_RE.match(path.name)
        if not match:
            skipped.append(str(path.relative_to(ROOT)))
            continue

        date = match.group("date")
        slug = match.group("slug")
        ext = match.group("ext")
        title = read_title(path) or title_from_slug(slug)
        candidates.append(
            {
                "date": date,
                "title": title,
                "url": post_url(date, slug),
                "source": f"post_{ext}",
                "source_path": str(path.relative_to(ROOT)),
                "priority": 1 if ext == "md" else 0,
            }
        )

    for path in sorted(BIT_ASS_DIR.glob("*.md")):
        match = BIT_ASS_RE.match(path.name)
        if not match:
            skipped.append(str(path.relative_to(ROOT)))
            continue

        date = match.group("date")
        slug = match.group("slug")
        title = read_title(path) or title_from_slug(slug)
        candidates.append(
            {
                "date": date,
                "title": title,
                "url": bit_ass_url(path.stem),
                "source": "bit_ass",
                "source_path": str(path.relative_to(ROOT)),
                "priority": 2,
            }
        )

    chosen: dict[str, dict] = {}
    for article in candidates:
        key = normalized_key(article["date"], article["title"])
        current = chosen.get(key)
        if current is None or article["priority"] > current["priority"]:
            chosen[key] = article

    articles = sorted(
        chosen.values(),
        key=lambda item: (item["date"], item["title"]),
        reverse=True,
    )
    for article in articles:
        article.pop("priority", None)

    return articles, skipped


def main() -> None:
    articles, skipped = collect_articles()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(articles)} articles to {OUTPUT_PATH.relative_to(ROOT)}")
    if skipped:
        print("skipped non-Jekyll post filenames:")
        for path in skipped:
            print(f"  {path}")


if __name__ == "__main__":
    main()
