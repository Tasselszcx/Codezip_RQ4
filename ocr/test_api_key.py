import os


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return key[0:2] + "..." + key[-2:]
    return key[:6] + "..." + key[-6:]


def _read_key_from_env_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != "AIHUBMIX_API_KEY":
                    continue
                value = v.strip().strip('"').strip("'")
                if value:
                    return value
    except Exception:
        return ""
    return ""


def load_aihubmix_api_key() -> tuple[str, str]:
    """返回 (api_key, source)"""
    key = os.getenv("AIHUBMIX_API_KEY")
    if key:
        return key, "env:AIHUBMIX_API_KEY"

    script_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(repo_dir, ".env"),
        os.path.join(script_dir, ".env"),
    ]

    for p in candidates:
        if not os.path.exists(p):
            continue
        key = _read_key_from_env_file(p)
        if key:
            return key, f"file:{p}"

    return "", "not-found"


if __name__ == "__main__":
    key, source = load_aihubmix_api_key()
    print("AIHUBMIX_API_KEY source:", source)
    if key:
        print("AIHUBMIX_API_KEY (masked):", _mask(key))
        print("OK: key loaded")
    else:
        print("ERROR: key not loaded")
        print("Checked env var: AIHUBMIX_API_KEY")
        print("Checked files: .env (cwd), .env (repo root), ocr/.env")
