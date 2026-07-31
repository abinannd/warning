"""
Main entry point for the Early Outbreak Detection System Dashboard.
"""
import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time

def start_server(port, directory):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass  # Suppress detailed HTTP request logging for clean console output

    handler = QuietHandler
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Server successfully started at http://localhost:{port}/outbreak_dashboard.html")
            httpd.serve_forever()
    except OSError as e:
        print(f"Port {port} already in use or server active: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "outputs")
    dashboard_file = os.path.join(outputs_dir, "outbreak_dashboard.html")

    if not os.path.exists(dashboard_file):
        print(f"Error: Dashboard file not found at {dashboard_file}")
        sys.exit(1)

    port = 8050
    url = f"http://localhost:{port}/outbreak_dashboard.html"

    print("=" * 60)
    print("      Early Outbreak Detection System — Interactive Dashboard")
    print("=" * 60)
    print(f"Dashboard File: {dashboard_file}")
    print(f"Serving URL:    {url}")
    print("Opening web browser...")
    print("=" * 60)

    # Launch web browser
    webbrowser.open(url)

    # Start server if not running
    start_server(port, outputs_dir)

if __name__ == "__main__":
    main()
