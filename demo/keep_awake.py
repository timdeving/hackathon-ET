"""Keep the live demo awake. Streamlit Community Cloud puts an app to sleep after about 12 hours
without a visitor, and only a browser counts as one. This opens the demo in a headless browser,
presses the wake-up button if the app is asleep, and waits until the page has drawn its title.
.github/workflows/keep-demo-awake.yml runs it every 6 hours.

    python demo/keep_awake.py https://<app>.streamlit.app [--screenshot demo.png]

Needs Playwright with its Chromium: pip install playwright && playwright install chromium
"""
from __future__ import annotations

import argparse
import time

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

TITLE = "Traffic events from a junction camera"  # the demo's title (demo/streamlit_app.py)
WAKE_BUTTON = "Yes, get this app back up!"  # what Streamlit Cloud shows on a sleeping app
WAIT_SEC = 300  # waking up and loading the model take a minute or two


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="the demo's address")
    parser.add_argument("--screenshot", help="save a picture of the page once it is up")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        try:
            page.get_by_role("button", name=WAKE_BUTTON).click(timeout=20_000)
            print("the demo was asleep: woken up")
        except PlaywrightTimeout:
            print("the demo was awake")
        up = wait_for_title(page, WAIT_SEC)
        if args.screenshot:
            page.screenshot(path=args.screenshot, full_page=True)
        browser.close()
    print("the demo is up" if up else f"the demo's title didn't appear within {WAIT_SEC} s")
    return 0 if up else 1


def wait_for_title(page: Page, seconds: float) -> bool:
    """Whether the demo's title shows, in the page or the frame Streamlit Cloud runs it in."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for frame in page.frames:
            try:
                if frame.get_by_text(TITLE).count():
                    return True
            except Exception:  # a frame that navigated away while we looked
                continue
        page.wait_for_timeout(3_000)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
