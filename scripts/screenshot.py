#!/usr/bin/env python3
"""Take a screenshot of an HTML file using Playwright."""

import argparse
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright


def screenshot(
    html_path: str,
    output_path: str = "screenshots/screenshot.png",
    wait_ms: int = 0,
    verbose: bool = False,
) -> None:
    """Take a screenshot of an HTML file.

    Args:
        html_path: Path to the HTML file to screenshot.
        output_path: Path where the screenshot will be saved.
        wait_ms: Additional milliseconds to wait after page load.
        verbose: If True, print console logs from the browser.
    """
    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    # Serve files from the HTML file's directory via HTTP
    serve_dir = html_path.parent
    html_filename = html_path.name

    # Create a simple HTTP server in a thread
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress server logs

    port = 8765
    server = socketserver.TCPServer(("", port), QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    console_logs: list[str] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            if verbose:
                page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
                page.on("pageerror", lambda err: console_logs.append(f"[PAGE ERROR] {err}"))

            page.goto(f"http://localhost:{port}/{html_filename}")
            page.wait_for_load_state("networkidle")
            if wait_ms > 0:
                page.wait_for_timeout(wait_ms)
            page.screenshot(path=output_path, full_page=True)
            browser.close()
    finally:
        server.shutdown()

    print(f"Screenshot saved to {output_path}")

    if verbose and console_logs:
        print("\nConsole logs:")
        for log in console_logs:
            print(f"  {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Take a screenshot of an HTML file")
    parser.add_argument("html_path", help="Path to the HTML file")
    parser.add_argument(
        "-o", "--output", default="screenshots/screenshot.png", help="Output path for screenshot"
    )
    parser.add_argument(
        "-w", "--wait", type=int, default=0, help="Additional ms to wait after page load"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print browser console logs")
    args = parser.parse_args()

    screenshot(args.html_path, args.output, args.wait, args.verbose)


if __name__ == "__main__":
    main()
