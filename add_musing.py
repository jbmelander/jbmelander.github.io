#!/usr/bin/env python3
import os
import re
import sys
import tempfile
import subprocess
import yaml
from datetime import datetime

SITE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSINGS_HTML = os.path.join(SITE_DIR, "musings.html")
MUSINGS_DIR = os.path.join(SITE_DIR, "musings")


def slugify(title):
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    return re.sub(r'-+', '-', slug)


def text_to_html(text):
    paragraphs = []
    current = []
    for line in text.split('\n'):
        if line.strip() == '':
            if current:
                paragraphs.append('<p>' + ' '.join(current) + '</p>')
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append('<p>' + ' '.join(current) + '</p>')
    return '\n  '.join(paragraphs) if paragraphs else ''


def create_musing_file(slug, title, date_str, content_html):
    os.makedirs(MUSINGS_DIR, exist_ok=True)
    path = os.path.join(MUSINGS_DIR, f"{slug}.html")
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{title} // Joshua Melander</title>
  <style>
    body {{
      background: #000;
      color: #ccc;
      font-family: "Courier New", monospace;
      font-size: 14px;
      margin: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}

    a {{ color: #0ff; }}
    a:visited {{ color: #f9f; }}

    h2 {{
      color: #ff0;
      font-size: 16px;
      margin-top: 30px;
    }}

    hr {{
      border: none;
      border-top: 1px dashed #444;
      margin: 20px 0;
      width: 100%;
      max-width: 600px;
    }}

    .content {{
      max-width: 600px;
      width: 100%;
    }}

    .back {{
      margin-bottom: 10px;
      display: block;
    }}

    .date {{
      color: #555;
      font-size: 12px;
      margin-bottom: 20px;
    }}

    p {{ line-height: 1.6; }}
  </style>
</head>
<body>
<div class="content">
  <a class="back" href="../musings.html">&larr; back</a>
  <hr>
  <h2>// {title}</h2>
  <div class="date">{date_str}</div>
  {content_html}
  <hr>
</div>
</body>
</html>
"""
    with open(path, 'w') as f:
        f.write(html)
    return path


def sync_musings_index(new_slug=None, new_title=None, new_date_str=None):
    with open(MUSINGS_HTML, 'r') as f:
        content = f.read()

    # Remove entries whose html files no longer exist
    li_pattern = re.compile(r'    <li><a href="musings/([^"]+)\.html">.*?</li>\n')
    removed = []
    for match in li_pattern.finditer(content):
        slug = match.group(1)
        if not os.path.exists(os.path.join(MUSINGS_DIR, f"{slug}.html")):
            content = content.replace(match.group(0), '')
            removed.append(slug)

    if new_slug and new_title and new_date_str:
        new_li = f'    <li><a href="musings/{new_slug}.html">{new_title}</a><span class="date">// {new_date_str}</span></li>\n'
        content = content.replace('<ul>\n', '<ul>\n' + new_li, 1)

    with open(MUSINGS_HTML, 'w') as f:
        f.write(content)

    return removed


def main():
    print("// add musing")
    title = input("title: ").strip()
    if not title:
        print("no title, exiting.")
        sys.exit(1)

    slug = slugify(title)
    date_str = datetime.now().strftime('%Y-%m-%d')

    yaml_content = f"title: {title}\ndate: {date_str}\ncontent: |\n  \n"

    tmp_path = os.path.join(tempfile.gettempdir(), f"{slug}.yaml")
    with open(tmp_path, 'w') as f:
        f.write(yaml_content)

    subprocess.call(['vim', tmp_path])

    with open(tmp_path, 'r') as f:
        raw = f.read()

    os.unlink(tmp_path)

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        print(f"yaml parse error: {e}")
        sys.exit(1)

    content = (data.get('content') or '').strip()
    if not content:
        print("no content, exiting.")
        sys.exit(1)

    title = data.get('title') or title
    date_str = str(data.get('date') or date_str)
    content_html = text_to_html(content)

    path = create_musing_file(slug, title, date_str, content_html)
    removed = sync_musings_index(slug, title, date_str)

    print(f"created:  {os.path.relpath(path, SITE_DIR)}")
    print(f"updated:  musings.html")
    for r in removed:
        print(f"removed:  musings/{r}.html (file not found)")


if __name__ == '__main__':
    main()
