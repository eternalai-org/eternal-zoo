import os
from pathlib import Path

ETERNAL_ZOO_MODEL_DIR = os.environ.get("ETERNAL_ZOO_MODEL_DIR", None)
DEFAULT_MODEL_DIR = (Path.home() / ".eternal-zoo" / "models") if not ETERNAL_ZOO_MODEL_DIR else ETERNAL_ZOO_MODEL_DIR
POSTFIX_MODEL_PATH = ".gguf"
GATEWAY_URLS = [
    # Lighthouse's own gateway (recommended for files stored on Lighthouse)
    "https://gateway.lighthouse.storage/ipfs/",
    # Mesh3 public gateway
    "https://gateway.mesh3.network/ipfs/",
    # IPFS official gateway
    "https://ipfs.io/ipfs/",
    # Cloudflare IPFS gateway
    # "https://cloudflare-ipfs.com/ipfs/",
    # Cloudflare alternative
    # "https://cf-ipfs.com/ipfs/",
    # Pinata-backed gateway
    "https://dweb.link/ipfs/",
    # 4everland Filecoin ecosystem gateway
    # "https://4everland.io/ipfs/",
    # Infura IPFS gateway
    "https://infura-ipfs.io/ipfs/",
]

ETERNAL_AI_METADATA_GW = "https://agent.api.eternalai.org/api/vibe/model/"
HF_LOCK_DIR = str(Path.home() / ".cache" / "huggingface" / "hub" / ".locks")

PREFIX_DOWNLOAD_LOG = "[CRYPTOAGENTS_LOGGER] [MODEL_INSTALL]"