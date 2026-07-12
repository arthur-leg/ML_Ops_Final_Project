"""Generate a demo JWT for the external API.

Usage:
    python -m backend.scripts.generate_external_token demo-client
"""
import sys

from backend.auth import generate_external_token

if __name__ == "__main__":
    client_name = sys.argv[1] if len(sys.argv) > 1 else "demo-client"
    token = generate_external_token(client_name)
    print(token)
