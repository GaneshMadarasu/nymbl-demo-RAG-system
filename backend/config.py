import os
import sys
from dotenv import load_dotenv

load_dotenv()


class _Settings:
    gemini_api_key: str
    database_url: str

    def __init__(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.database_url = os.getenv("DATABASE_URL", "")
        missing = [
            k for k in self.__class__.__annotations__ if not getattr(self, k, "")
        ]
        if missing:
            print(
                f"ERROR: missing required env vars: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)


# Tests must set GEMINI_API_KEY and DATABASE_URL via os.environ before importing this module.
settings = _Settings()
