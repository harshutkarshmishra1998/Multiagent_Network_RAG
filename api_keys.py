from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from same directory
load_dotenv(Path(__file__).parent / ".env")

def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing environment variable: {key}")
    return value


# ---- REQUIRED ENV VARS ----
ENV_KEYS = {
    "OPENAI_API_KEY": "OPENAI_API",
    "LANGCHAIN_API_KEY": "LANGCHAIN_API",
    "HUGGING_FACE_API": "HUGGING_FACE_API",
    "GROQ_API_KEY": "GROQ_API",
    "ASTRA_DB_API_ENDPOINT": "ASTRA_DB_API_ENDPOINT",
    "ASTRA_DB_APPLICATION_TOKEN": "ASTRA_DB_APPLICATION_TOKEN",
    "ASTRA_DB_KEYSPACE": "ASTRA_DB_KEYSPACE",
    "WB_API": "WB_API",
    "SERPAI_API": "SERPAI_API",
}

# ---- LOAD + EXPORT ----
_loaded = {}

for env_name, source_key in ENV_KEYS.items():
    value = require_env(source_key)
    os.environ[env_name] = value
    _loaded[env_name] = value

# ---- OPTIONAL: SAFE DEBUG (NO LEAKS) ----
if __name__ == "__main__":
    for k in _loaded:
        print(f"{k}: loaded")

os.environ["LANGCHAIN_TRACING_V2"] = "1"
os.environ["LANGCHAIN_PROJECT"] = "QA RAG Chatbot"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USER_AGENT"] = "qa-search-engine/1.0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"