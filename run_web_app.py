#!/usr/bin/env python3
"""
run_web_app.py
==============
Starts a local web server to run the interactive Space Debris Analyzer
and opens it in your default web browser automatically.

Usage:
    python run_web_app.py
"""

import http.server
import socketserver
import webbrowser
import threading
import time

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence console log spam for cleaner output
        pass

def start_server():
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"\n[INFO] Web App Server started at port {PORT}")
        print(f"[INFO] Click this link to open: http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server.\n")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in a separate thread so we can print the link
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to initialize
    time.sleep(1)
    
    # Open the browser
    webbrowser.open(f"http://localhost:{PORT}")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Web App Server...")
