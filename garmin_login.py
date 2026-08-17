"""
One-time Garmin Connect login. Run this yourself in a terminal:

    .venv/bin/python garmin_login.py

Prompts for your Garmin email/password (and MFA code if enabled), then stores
OAuth tokens in .garmin_tokens. After this, garmin_client.py works silently;
your password is never stored.
"""

import getpass

from garminconnect import Garmin

from garmin_client import TOKENSTORE


def main() -> None:
    email = input("Garmin email: ").strip()
    password = getpass.getpass("Garmin password: ")
    api = Garmin(
        email=email,
        password=password,
        prompt_mfa=lambda: input("Garmin MFA code: ").strip(),
    )
    api.login()
    api.client.dump(TOKENSTORE)
    print(f"Login OK — tokens saved to {TOKENSTORE}")
    print("You can now close this terminal; the app will use the tokens.")


if __name__ == "__main__":
    main()
