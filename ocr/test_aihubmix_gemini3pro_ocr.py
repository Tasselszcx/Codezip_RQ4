import base64
import json
import os
import re
import time
from pathlib import Path


AIHUBMIX_BASE_URL_DEFAULT = "https://aihubmix.com/v1"
MODEL_DEFAULT = "gemini-3-pro-preview"

OCR_SYSTEM_PROMPT_DEFAULT = "You are an OCR engine for code images."
OCR_USER_PROMPT_DEFAULT = (
    "Transcribe the code in these images exactly.\n"
    "- These images are consecutive pages of the SAME code file, in order.\n"
    "- The page may start mid-block (e.g., indented lines without a visible 'def' header). Keep the indentation exactly as shown.\n"
    "- Do NOT invent missing context. Do NOT add wrapper code such as 'def', 'class', imports, or any extra lines.\n"
    "- Output plain text only (no Markdown, no code fences).\n"
    "- Preserve all whitespace, indentation, and newlines.\n"
    "- Do not add, remove, or rename anything.\n"
)


def _encode_image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_page_num_from_filename(path: Path) -> int:
    m = re.search(r"page_(\d+)", path.stem)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _pick_images(images_root: Path, max_images: int) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    all_imgs = [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    all_imgs.sort(key=lambda p: (str(p.parent), _extract_page_num_from_filename(p), p.name))
    return all_imgs[:max_images]


def _extract_text(message_content) -> str:
    # Some OpenAI-compatible providers return either a string or a list of parts.
    if message_content is None:
        return ""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, dict):
        # Some providers may return {"text": "..."} or nested objects.
        if isinstance(message_content.get("text"), str):
            return message_content["text"]
        if isinstance(message_content.get("content"), str):
            return message_content["content"]
        text_obj = message_content.get("text")
        if isinstance(text_obj, dict) and isinstance(text_obj.get("value"), str):
            return text_obj["value"]
    if isinstance(message_content, list):
        parts: list[str] = []
        for part in message_content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                    continue
                if isinstance(part.get("content"), str):
                    parts.append(part["content"])
                    continue
                text_obj = part.get("text")
                if isinstance(text_obj, dict) and isinstance(text_obj.get("value"), str):
                    parts.append(text_obj["value"])
        return "".join(parts)
    try:
        return json.dumps(message_content, ensure_ascii=False)
    except Exception:
        return str(message_content)


def _clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    return cleaned.strip("\n")


def main() -> int:
    # Required
    api_key = os.getenv("AIHUBMIX_API_KEY", "").strip()
    if not api_key:
        print("Missing env AIHUBMIX_API_KEY")
        print('PowerShell example: $env:AIHUBMIX_API_KEY="sk-..."')
        return 2

    # Configurable
    base_url = os.getenv("AIHUBMIX_BASE_URL", AIHUBMIX_BASE_URL_DEFAULT).strip() or AIHUBMIX_BASE_URL_DEFAULT
    model = os.getenv("GEMINI_MODEL", MODEL_DEFAULT).strip() or MODEL_DEFAULT

    images_root = Path(os.getenv("AIHUBMIX_IMAGES_ROOT", "./experiment_output/images_gemini")).resolve()
    max_images = int(os.getenv("AIHUBMIX_MAX_IMAGES", "6"))

    system_prompt = os.getenv("OCR_SYSTEM_PROMPT", OCR_SYSTEM_PROMPT_DEFAULT)
    user_prompt = os.getenv("OCR_USER_PROMPT", OCR_USER_PROMPT_DEFAULT)

    max_tokens = int(os.getenv("OCR_MAX_TOKENS", "2048"))
    temperature = float(os.getenv("OCR_TEMPERATURE", "0"))
    retries = int(os.getenv("OCR_MAX_RETRIES", "4"))
    use_max_completion_tokens = os.getenv("USE_MAX_COMPLETION_TOKENS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    debug = os.getenv("AIHUBMIX_DEBUG", "0").strip().lower() in ("1", "true", "yes", "y")

    if not images_root.exists():
        print(f"Images root not found: {images_root}")
        return 2

    images = _pick_images(images_root, max_images=max_images)
    if not images:
        print(f"No images found under: {images_root}")
        return 2

    content = [{"type": "text", "text": user_prompt}]
    for p in images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(p)}})

    try:
        from openai import OpenAI
    except Exception as e:
        print("Missing dependency: openai")
        print("Install: pip install openai")
        print(f"Import error: {e}")
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("=== AIHubMix Gemini OCR test ===")
    print(f"base_url: {base_url}")
    print(f"model: {model}")
    print(f"images_root: {images_root}")
    print(f"sending_images: {len(images)}")
    print("first_images:")
    for p in images[:10]:
        print(f"- {p}")

    last_err: str | None = None
    resp = None

    for attempt in range(1, retries + 1):
        try:
            req = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            }
            if use_max_completion_tokens:
                req["max_completion_tokens"] = max_tokens
            else:
                req["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**req)
            last_err = None
            break
        except Exception as e:
            last_err = str(e)
            backoff = min(20.0, float(2 ** (attempt - 1)))
            print(f"attempt {attempt}/{retries} failed: {last_err}")
            if attempt < retries:
                time.sleep(backoff)

    if last_err is not None or resp is None:
        print("\n--- request failed ---")
        print(last_err or "unknown error")
        return 3

    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    text = _clean_ocr_text(_extract_text(choice.message.content))

    if debug:
        try:
            raw_path = Path(
                os.getenv(
                    "AIHUBMIX_RAW_OUTPUT",
                    "./experiment_output/aihubmix_gemini3pro_ocr_test.raw_response.json",
                )
            ).resolve()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw = resp.model_dump() if hasattr(resp, "model_dump") else resp
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[debug] saved raw response: {raw_path}")
            mc = choice.message.content
            print(f"[debug] message.content type: {type(mc)}")
            print(f"[debug] message.content repr: {repr(mc)[:500]}")
        except Exception as e:
            print(f"[debug] failed to dump raw response: {e}")

    print("\n--- result ---")
    print(f"finish_reason: {finish_reason}")
    print(f"text_len: {len(text)}")
    print("preview:")
    print(text[:800])

    if len(text) == 0:
        print("\n提示：返回为空通常有几种可能：")
        print("1) 响应文本不在 message.content（需要开启 AIHUBMIX_DEBUG=1 看 raw response）")
        print("2) 模型/渠道触发了内容过滤但没显式报错（raw 里可能有 safety 字段）")
        print("3) 参数兼容问题：试试 USE_MAX_COMPLETION_TOKENS=1 或把 OCR_MAX_TOKENS 调大")
        print("4) 发送图片太多/太大：先用 AIHUBMIX_MAX_IMAGES=1 做单图 sanity check")

    out_path = Path(os.getenv("AIHUBMIX_OUTPUT", "./experiment_output/aihubmix_gemini3pro_ocr_test.json")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "base_url": base_url,
        "model": model,
        "images": [str(p) for p in images],
        "finish_reason": finish_reason,
        "text": text,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
