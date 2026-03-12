#!/usr/bin/env python3
import curses
import os
import re
import sys
from datetime import datetime

SITE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSINGS_HTML = os.path.join(SITE_DIR, "musings.html")
MUSINGS_DIR = os.path.join(SITE_DIR, "musings")


def slugify(title):
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    return re.sub(r'-+', '-', slug)


def run_editor(stdscr, title):
    curses.curs_set(1)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)   # box
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)    # status
    curses.init_pair(3, curses.COLOR_WHITE, -1)                   # background

    stdscr.bkgd(' ', curses.color_pair(3))

    lines = ['']
    cursor_row = 0
    cursor_col = 0
    scroll = 0

    while True:
        term_h, term_w = stdscr.getmaxyx()
        box_w = min(80, term_w - 4)
        box_h = term_h - 6
        box_y = (term_h - box_h) // 2
        box_x = (term_w - box_w) // 2
        text_h = box_h - 2
        text_w = box_w - 2
        text_y = box_y + 1
        text_x = box_x + 1

        stdscr.erase()

        # Title above box
        label = f" // {title} "
        try:
            stdscr.addstr(box_y - 2, box_x, label, curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass

        # Box border
        box_win = curses.newwin(box_h, box_w, box_y, box_x)
        box_win.bkgd(' ', curses.color_pair(1))
        box_win.border()
        box_win.refresh()

        # Text area
        text_win = curses.newwin(text_h, text_w, text_y, text_x)
        text_win.bkgd(' ', curses.color_pair(1))
        for i in range(text_h):
            idx = scroll + i
            if idx < len(lines):
                seg = lines[idx][:text_w]
                try:
                    text_win.addstr(i, 0, seg, curses.color_pair(1))
                except curses.error:
                    pass
        text_win.refresh()

        # Status bar
        status = "  ^S  submit    ^C  cancel  "
        try:
            stdscr.addstr(term_h - 1, 0, status.ljust(term_w - 1), curses.color_pair(2))
        except curses.error:
            pass

        # Cursor
        cy = text_y + (cursor_row - scroll)
        cx = text_x + cursor_col
        try:
            stdscr.move(cy, cx)
        except curses.error:
            pass

        stdscr.refresh()

        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            return None

        if ch == 19:  # Ctrl+S — submit
            return '\n'.join(lines)

        elif ch == 3:  # Ctrl+C — cancel
            return None

        elif ch in (curses.KEY_ENTER, 10, 13):
            rest = lines[cursor_row][cursor_col:]
            lines[cursor_row] = lines[cursor_row][:cursor_col]
            lines.insert(cursor_row + 1, rest)
            cursor_row += 1
            cursor_col = 0
            if cursor_row - scroll >= text_h:
                scroll += 1

        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if cursor_col > 0:
                l = lines[cursor_row]
                lines[cursor_row] = l[:cursor_col - 1] + l[cursor_col:]
                cursor_col -= 1
            elif cursor_row > 0:
                prev_len = len(lines[cursor_row - 1])
                lines[cursor_row - 1] += lines[cursor_row]
                lines.pop(cursor_row)
                cursor_row -= 1
                cursor_col = prev_len
                if scroll > cursor_row:
                    scroll = cursor_row

        elif ch == curses.KEY_UP:
            if cursor_row > 0:
                cursor_row -= 1
                cursor_col = min(cursor_col, len(lines[cursor_row]))
                if cursor_row < scroll:
                    scroll -= 1

        elif ch == curses.KEY_DOWN:
            if cursor_row < len(lines) - 1:
                cursor_row += 1
                cursor_col = min(cursor_col, len(lines[cursor_row]))
                if cursor_row - scroll >= text_h:
                    scroll += 1

        elif ch == curses.KEY_LEFT:
            if cursor_col > 0:
                cursor_col -= 1
            elif cursor_row > 0:
                cursor_row -= 1
                cursor_col = len(lines[cursor_row])
                if cursor_row < scroll:
                    scroll -= 1

        elif ch == curses.KEY_RIGHT:
            if cursor_col < len(lines[cursor_row]):
                cursor_col += 1
            elif cursor_row < len(lines) - 1:
                cursor_row += 1
                cursor_col = 0
                if cursor_row - scroll >= text_h:
                    scroll += 1

        elif ch == curses.KEY_HOME:
            cursor_col = 0

        elif ch == curses.KEY_END:
            cursor_col = len(lines[cursor_row])

        elif 32 <= ch < 256:
            l = lines[cursor_row]
            lines[cursor_row] = l[:cursor_col] + chr(ch) + l[cursor_col:]
            cursor_col += 1


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


def update_musings_index(slug, title, date_str):
    with open(MUSINGS_HTML, 'r') as f:
        content = f.read()
    new_li = f'    <li><a href="musings/{slug}.html">{title}</a><span class="date">// {date_str}</span></li>\n'
    # Insert newest at top of list
    content = content.replace('<ul>\n', '<ul>\n' + new_li, 1)
    with open(MUSINGS_HTML, 'w') as f:
        f.write(content)


def main():
    print("// add musing")
    title = input("title: ").strip()
    if not title:
        print("no title, exiting.")
        sys.exit(1)

    text = curses.wrapper(run_editor, title)

    if text is None:
        print("cancelled.")
        sys.exit(0)

    slug = slugify(title)
    date_str = datetime.now().strftime('%Y-%m-%d')
    content_html = text_to_html(text)

    path = create_musing_file(slug, title, date_str, content_html)
    update_musings_index(slug, title, date_str)

    print(f"created:  {os.path.relpath(path, SITE_DIR)}")
    print(f"updated:  musings.html")


if __name__ == '__main__':
    main()
