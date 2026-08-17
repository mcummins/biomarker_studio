"""
One-time Google Health API authorization. Run it yourself in a terminal:

    .venv/bin/python google_health_login.py

Before the first run, create the OAuth client (once):

1. Go to https://console.cloud.google.com/ and create a project
   (e.g. "biomarker-studio").
2. "APIs & Services" -> "Library" -> search "Google Health API" -> Enable.
   (If it is not listed, enable it via
   https://console.cloud.google.com/apis/library/health.googleapis.com)
3. "APIs & Services" -> "OAuth consent screen":
   - User type: External, then add your own Google account under
     "Test users". Leave the app in Testing mode — no verification needed
     for personal use.
4. "APIs & Services" -> "Credentials" -> "Create credentials" ->
   "OAuth client ID" -> Application type: "Desktop app".
5. Copy the Client ID and Client Secret; this script asks for both on the
   first run and stores them in google_health_config.json.

The script starts a temporary local web server on port 8765, opens the
Google consent page in your browser, and captures the redirect. Tokens are
stored in google_health_config.json; your Google password never touches
this machine.
"""

import http.server
import threading
import urllib.parse
import webbrowser

import google_health_client as ghc

REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}"

_auth_code = {}


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        if "code" in params:
            _auth_code["code"] = params["code"][0]
            message = "Authorization complete. You can close this tab."
        else:
            _auth_code["error"] = params.get("error", ["unknown"])[0]
            message = f"Authorization failed: {_auth_code['error']}"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(f"<html><body><p>{message}</p></body></html>".encode())

    def log_message(self, *args):
        pass


def main() -> None:
    config = ghc.load_config()
    if not config.get("client_id"):
        print("First run — paste the OAuth client details (created per the")
        print("instructions at the top of this file).")
        config["client_id"] = input("Client ID: ").strip()
        config["client_secret"] = input("Client Secret: ").strip()
        ghc.save_config(config)

    server = http.server.HTTPServer(("localhost", REDIRECT_PORT), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    url = ghc.build_auth_url(config["client_id"], REDIRECT_URI)
    print("\nOpening Google consent page in your browser...")
    print(f"If it does not open, visit:\n\n{url}\n")
    webbrowser.open(url)

    thread.join(timeout=300)
    server.server_close()

    if "code" not in _auth_code:
        print("No authorization code received (timed out or denied).")
        print("Re-run this script to try again.")
        return

    ghc.exchange_code(_auth_code["code"], REDIRECT_URI)
    print("Authorization OK — tokens saved to google_health_config.json.")
    print("You can now sync via the Google Health API.")


if __name__ == "__main__":
    main()
