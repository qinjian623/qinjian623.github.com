#!/usr/bin/env python3
import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote


ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "_posts"
BIT_ASS_DIR = ROOT / "bit_ass"
OUTPUT_PATH = ROOT / "_data" / "article_index.json"
TOPIC_OUTPUT_PATH = ROOT / "_data" / "topic_index.json"

POST_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.+)\.(?P<ext>html|md)$")
BIT_ASS_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.+)\.md$")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.S)
TITLE_RE = re.compile(r"^title\s*:\s*(?P<title>.+?)\s*$", re.M)
MARKDOWN_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.M)
HTML_H1_RE = re.compile(r"<h1[^>]*>(?P<title>.*?)</h1>", re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ORIGINAL_URL_RE = re.compile(r"\[\*?Link:\*?\]\((?P<url>https?://[^)]+)\)")
INLINE_HEADING_RE = re.compile(r"\)\s*(#{2,6}\s+\S)")

TOPICS = [
    {
        "name": "自动驾驶 / BEV",
        "slug": "ad-bev",
        "description": "自动驾驶感知、BEV、车道线、传感器、规划和数据闭环。",
        "pattern": re.compile(
            r"自动驾驶|ADAS|BEV|Tesla|FSD|AI Day|车道|HDMap|规划|规控|"
            r"摄像头|多相机|鱼眼|Lidar|Camera|D Space|单目视觉|路面|"
            r"感知|Freespace|Occupancy",
            re.I,
        ),
    },
    {
        "name": "深度学习工程",
        "slug": "deep-learning-engineering",
        "description": "模型训练、CV、PyTorch、目标检测、量化、蒸馏和部署经验。",
        "pattern": re.compile(
            r"深度学习|DNN|CNN|PyTorch|Pytorch|Focal|FlowNet|Challenger|YOLO|"
            r"MLP|RNN|LSTM|im2col|gradient|Feature Visualization|ShuffleNet|"
            r"Small object|RePr|Group Conv|Quantization|BatchNorm|ROI Align|"
            r"distill|过拟合|网络裁剪|模型伪量化|目标检测|图像压缩|Attention|"
            r"CS231n|Neural|Machine Learning|生成模型|判别模型|分类器|ONNX",
            re.I,
        ),
    },
    {
        "name": "向量搜索 / GPU / 系统",
        "slug": "vector-gpu-systems",
        "description": "向量检索、GPU/CPU 性能、系统、OS、网络和底层工具。",
        "pattern": re.compile(
            r"向量搜索|Product Quantization|IVF|K-means|GPU|CUDA|CPU cache|"
            r"Operating Systems|Baking Pi|Linux|Arch|Unix|VPN|goagent|WebQQ|"
            r"Volatile|寄存器|栈|虚拟机|Tanenbaum|Torvalds|pass管理密码",
            re.I,
        ),
    },
    {
        "name": "编程语言 / 工程管理",
        "slug": "programming-engineering",
        "description": "编程语言、代码审美、API、工具、Tech Leader 和工程组织。",
        "pattern": re.compile(
            r"Clojure|JVM|Java|lambda|Python|yield|Dart|Atom|Vim|Emacs|"
            r"org-mode|Jekyll|Unit Test|Clean Code|代码|API|Tech Leader|"
            r"技术能力|技术审美|工程Tips|Coding|创新|程序员|编程|架构|团队|"
            r"CPP|C\+\+",
            re.I,
        ),
    },
]
DEFAULT_TOPIC = {
    "name": "读书 / 个人思考",
    "slug": "reading-notes",
    "description": "读书、影视、个人选择、技术观和长期自我记录。",
}
FEATURED_KEYS = {
    "2026-05-05:新的工作流水账（23-26）",
    "2023-04-10:规划模型的PoC及难点",
    "2023-03-15:自己小组的一篇 CVPR23 车道线检测论文",
    "2023-01-04:BEV 感知模型实用的一些经验",
    "2022-08-01:最近一年自动驾驶工作的总结与流水账",
    "2019-12-14:水文4理想主义Tech Leader",
    "2018-07-09:一些有关数据的人生教训",
    "2020-07-29:向量搜索短期工作总结",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_front_matter(text: str) -> tuple[str, str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return "", text
    return match.group("body"), text[match.end() :]


def read_title(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    front_matter, body = split_front_matter(text)

    title_match = TITLE_RE.search(front_matter)
    if title_match:
        title = clean_title(title_match.group("title"))
        if title:
            return title

    markdown_match = MARKDOWN_H1_RE.search(body)
    if markdown_match:
        title = clean_title(markdown_match.group("title"))
        if title:
            return title

    html_match = HTML_H1_RE.search(body)
    if html_match:
        title = clean_title(HTML_TAG_RE.sub("", html_match.group("title")))
        if title:
            return title

    return None


def clean_title(title: str) -> str:
    title = html.unescape(title).strip()
    if len(title) >= 2 and title[0] == title[-1] and title[0] in ("'", '"'):
        title = title[1:-1]
    return re.sub(r"\s+", " ", title).strip()


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


def featured_key(date: str, title: str) -> str:
    return f"{date}:{title}"


def clean_excerpt(text: str, title: str) -> str:
    _, body = split_front_matter(text)
    body = ORIGINAL_URL_RE.sub(" ", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = HTML_TAG_RE.sub(" ", body)

    lines = []
    for line in body.splitlines():
        stripped = html.unescape(line).strip()
        if not stripped:
            continue
        if stripped == title or stripped == f"# {title}":
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("[*Link:*]"):
            continue
        if stripped.startswith("[http"):
            continue
        lines.append(stripped)

    excerpt = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(excerpt) > 180:
        excerpt = excerpt[:177].rstrip() + "..."
    return excerpt


def original_url(text: str) -> str | None:
    match = ORIGINAL_URL_RE.search(text)
    return match.group("url") if match else None


def choose_topic(title: str, source_path: str, text: str) -> dict:
    haystack = " ".join([title, source_path, text[:4000]])
    for topic in TOPICS:
        if topic["pattern"].search(haystack):
            return topic
    return DEFAULT_TOPIC


def article_payload(date: str, title: str, url: str, source: str, source_path: str, text: str, priority: int) -> dict:
    topic = choose_topic(title, source_path, text)
    payload = {
        "date": date,
        "title": title,
        "url": url,
        "source": source,
        "source_path": source_path,
        "topic": topic["name"],
        "topic_slug": topic["slug"],
        "excerpt": clean_excerpt(text, title),
        "priority": priority,
    }
    url_from_source = original_url(text)
    if url_from_source:
        payload["original_url"] = url_from_source
    if featured_key(date, title) in FEATURED_KEYS:
        payload["featured"] = True
    return payload


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
        text = read_text(path)
        title = read_title(path) or title_from_slug(slug)
        rel_path = str(path.relative_to(ROOT))
        candidates.append(
            article_payload(
                date=date,
                title=title,
                url=post_url(date, slug),
                source=f"post_{ext}",
                source_path=rel_path,
                text=text,
                priority=1 if ext == "md" else 0,
            )
        )

    for path in sorted(BIT_ASS_DIR.glob("*.md")):
        match = BIT_ASS_RE.match(path.name)
        if not match:
            skipped.append(str(path.relative_to(ROOT)))
            continue

        date = match.group("date")
        slug = match.group("slug")
        text = read_text(path)
        title = read_title(path) or title_from_slug(slug)
        rel_path = str(path.relative_to(ROOT))
        candidates.append(
            article_payload(
                date=date,
                title=title,
                url=bit_ass_url(path.stem),
                source="bit_ass",
                source_path=rel_path,
                text=text,
                priority=2,
            )
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


def build_topic_index(articles: list[dict]) -> list[dict]:
    topic_defs = [*TOPICS, DEFAULT_TOPIC]
    topic_index = []
    for topic in topic_defs:
        topic_articles = [article for article in articles if article["topic_slug"] == topic["slug"]]
        if not topic_articles:
            continue
        topic_index.append(
            {
                "name": topic["name"],
                "slug": topic["slug"],
                "description": topic["description"],
                "count": len(topic_articles),
                "articles": [
                    {
                        "date": article["date"],
                        "title": article["title"],
                        "url": article["url"],
                        "excerpt": article["excerpt"],
                    }
                    for article in topic_articles
                ],
            }
        )
    return topic_index


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def git_tracks(path: str) -> bool:
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def site_output_exists(site_dir: Path, url: str) -> bool:
    relative = url.lstrip("/")
    candidates = [site_dir / relative, site_dir / unquote(relative)]
    if url.endswith("/"):
        candidates.extend([path / "index.html" for path in list(candidates)])
    return any(path.exists() for path in candidates)


def lint_inline_headings(paths: list[Path]) -> list[str]:
    errors = []
    for path in paths:
        if path.suffix != ".md":
            continue
        for lineno, line in enumerate(read_text(path).splitlines(), start=1):
            if INLINE_HEADING_RE.search(line):
                errors.append(f"{path.relative_to(ROOT)}:{lineno}: heading is attached to previous link/text")
    return errors


def check_index(articles: list[dict], topic_index: list[dict], site_dir: Path | None) -> int:
    errors = []
    if not OUTPUT_PATH.exists():
        errors.append(f"{OUTPUT_PATH.relative_to(ROOT)} does not exist")
    elif read_json(OUTPUT_PATH) != articles:
        errors.append(f"{OUTPUT_PATH.relative_to(ROOT)} is stale; run scripts/build_article_index.py")

    if not TOPIC_OUTPUT_PATH.exists():
        errors.append(f"{TOPIC_OUTPUT_PATH.relative_to(ROOT)} does not exist")
    elif read_json(TOPIC_OUTPUT_PATH) != topic_index:
        errors.append(f"{TOPIC_OUTPUT_PATH.relative_to(ROOT)} is stale; run scripts/build_article_index.py")

    seen_urls = set()
    for article in articles:
        source_path = article["source_path"]
        if article["url"] in seen_urls:
            errors.append(f"duplicate url in article index: {article['url']}")
        seen_urls.add(article["url"])

        if not (ROOT / source_path).exists():
            errors.append(f"indexed source is missing: {source_path}")
        elif not git_tracks(source_path):
            errors.append(f"indexed source is not tracked by git: {source_path}")

        if site_dir and site_dir.exists() and not site_output_exists(site_dir, article["url"]):
            errors.append(f"built output missing for {article['url']} from {source_path}")

    source_paths = [ROOT / article["source_path"] for article in articles]
    errors.extend(lint_inline_headings(source_paths))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"article index check passed: {len(articles)} articles, {len(topic_index)} topics")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated data, tracked sources, built URLs, and migrated Markdown")
    parser.add_argument("--site-dir", default="_site", help="built Jekyll output directory for URL checks")
    args = parser.parse_args()

    articles, skipped = collect_articles()
    topic_index = build_topic_index(articles)

    if args.check:
        site_dir = (ROOT / args.site_dir).resolve()
        return_code = check_index(articles, topic_index, site_dir)
        raise SystemExit(return_code)

    write_json(OUTPUT_PATH, articles)
    write_json(TOPIC_OUTPUT_PATH, topic_index)

    print(f"wrote {len(articles)} articles to {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"wrote {len(topic_index)} topics to {TOPIC_OUTPUT_PATH.relative_to(ROOT)}")
    if skipped:
        print("skipped non-Jekyll post filenames:")
        for path in skipped:
            print(f"  {path}")


if __name__ == "__main__":
    main()
