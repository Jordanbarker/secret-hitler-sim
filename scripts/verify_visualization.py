#!/usr/bin/env python3
"""Verify game visualization by taking screenshots at specific rounds."""

import argparse
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright


def verify_visualization(
    html_path: str = "index.html",
    output_dir: str = "screenshots/verify_visualization",
    verbose: bool = False,
) -> None:
    """Take screenshots at specific rounds for two games.

    Args:
        html_path: Path to the visualization HTML file.
        output_dir: Directory where screenshots will be saved.
        verbose: If True, print console logs from the browser.
    """
    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Serve files from the HTML file's directory via HTTP
    serve_dir = html_path.parent
    html_filename = html_path.name

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress server logs

    port = 8765
    socketserver.TCPServer.allow_reuse_address = True
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

            # Dismiss tutorial overlay if present
            tutorial_skip = page.locator(".tutorial-btn-skip")
            if tutorial_skip.is_visible(timeout=5000):
                tutorial_skip.click()
                page.wait_for_timeout(200)

            for game_num in [1, 2]:
                # Screenshot at initial state (Round 0)
                screenshot_path = output_dir / f"game{game_num}_round_0.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"Screenshot saved to {screenshot_path}")

                # Click next button 4 times to reach Round 4
                for _ in range(4):
                    page.click("#btn-next")
                    page.wait_for_timeout(100)

                # Screenshot at Round 4
                screenshot_path = output_dir / f"game{game_num}_round_4.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"Screenshot saved to {screenshot_path}")

                # Click until final round (button becomes disabled)
                while not page.is_disabled("#btn-next"):
                    page.click("#btn-next")
                    page.wait_for_timeout(100)

                # Screenshot at final round
                screenshot_path = output_dir / f"game{game_num}_round_final.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"Screenshot saved to {screenshot_path}")

                # Generate new game for next iteration (skip on last game)
                if game_num < 2:
                    page.click("#btn-new-game")
                    page.wait_for_timeout(500)  # Wait for Pyodide to generate new game

            browser.close()
    finally:
        server.shutdown()

    if verbose and console_logs:
        print("\nConsole logs:")
        for log in console_logs:
            print(f"  {log}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify game visualization by taking screenshots at specific rounds"
    )
    parser.add_argument(
        "html_path",
        nargs="?",
        default="index.html",
        help="Path to the visualization HTML file (default: index.html)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="screenshots/verify_visualization",
        help="Output directory for screenshots (default: screenshots/verify_visualization)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print browser console logs")
    args = parser.parse_args()

    verify_visualization(args.html_path, args.output_dir, args.verbose)


if __name__ == "__main__":
    main()
