"""Load secrets from a .env file."""
import os
from pathlib import Path

REPO_ROOT = 'uw-central'
DEFAULT_SECRETS_FILE = '.env/secrets.txt'

def get_default_secret_file_path() -> str:
    """Get the path to the secrets file."""
    secret_file = os.path.join(
        str(Path(os.getcwd())).split(REPO_ROOT)[0],
        REPO_ROOT,
        DEFAULT_SECRETS_FILE
    )
    return secret_file

class Secrets:
    """Class to access secrets."""

    def __init__(self) :
        """Initialize the Secrets class."""
        secrets_file = get_default_secret_file_path()
        self._parse_secrets(secrets_file)

    def _parse_secrets(self, secrets_file: str) -> None:
        """Parse the secrets file."""
        self._secrets = {}

        with open(secrets_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    self._secrets[key.strip()] = value.strip()

    def __getitem__(self, key: str) -> str:
        """Get a secret by key."""
        return self._secrets.get(key, None)
