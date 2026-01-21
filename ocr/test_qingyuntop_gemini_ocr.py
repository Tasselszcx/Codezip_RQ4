import base64
import json
import os
from pathlib import Path


def _encode_image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _pick_some_images(images_root: Path, max_images: int = 6) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    all_imgs = [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    all_imgs.sort(key=lambda p: (str(p.parent), p.name))
    return all_imgs[:max_images]


def _extract_text(message_content) -> str:
    # OpenAI-compatible providers may return either a string or a list of parts.
    if message_content is None:
        return ""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts = []
        for part in message_content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                # common shapes: {type: 'text', text: '...'}
                if "text" in part and isinstance(part["text"], str):
                    parts.append(part["text"])
        return "".join(parts)
    # last resort
    try:
        return json.dumps(message_content, ensure_ascii=False)
    except Exception:
        return str(message_content)


def main() -> int:
    # Qingyun OpenAI-compatible endpoint (per their docs):
    # curl https://api.qingyuntop.top/v1/chat/completions ...
    base_url = os.getenv("QINGYUN_BASE_URL", "https://api.qingyuntop.top/v1").strip()
    api_key = os.getenv("QINGYUN_API_KEY", "").strip()
    model = os.getenv("QINGYUN_MODEL", "gemini-3-pro-preview").strip()

    if not api_key:
        print("Missing env QINGYUN_API_KEY")
        print('PowerShell example: $env:QINGYUN_API_KEY="sk-..."')
        return 2

    # Use images from your existing dataset folder.
    images_root = Path(os.getenv("QINGYUN_IMAGES_ROOT", "./experiment_output/images_gemini")).resolve()
    if not images_root.exists():
        print(f"Images root not found: {images_root}")
        return 2

    images = _pick_some_images(images_root, max_images=int(os.getenv("QINGYUN_MAX_IMAGES", "6")))
    if not images:
        print(f"No images found under: {images_root}")
        return 2

    prompt = (
        "You are an OCR engine for code images.\n"
        "Transcribe the code in these images exactly.\n"
        "- Output plain text only (no Markdown, no code fences).\n"
        "- Preserve whitespace, indentation, and newlines.\n"
        "- Do NOT add or invent any code." 
    )

    content = [{"type": "text", "text": prompt}]
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

    print("=== Qingyun Gemini OCR test ===")
    print(f"base_url: {base_url}")
    print(f"model: {model}")
    print(f"images_root: {images_root}")
    print(f"sending_images: {len(images)}")
    print("first_images:")
    for p in images[:3]:
        print(f"- {p}")

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=int(os.getenv("QINGYUN_MAX_TOKENS", "2048")),
            messages=[
                {"role": "user", "content": content},
            ],
        )
    except Exception as e:
        msg = str(e)
        print("\n--- request failed ---")
        print(msg)
        if "No available channels" in msg and "group default" in msg:
            print("\n这表示：你的 Key 当前在 default 分组下，没有 gemini-3-pro-preview 的可用通道/权限。")
            print("解决：到青云控制台 -> 令牌(Token) -> 编辑该 Key -> 添加备选分组或创建新 Key。")
            print("建议分组：优质gemini 或 官转gemini（文档：分组详细表格）。")
            print("如果你只想先验证连通性，也可以把 QINGYUN_MODEL 换成 default 分组支持的模型再试。")
        return 3

    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    text = _extract_text(choice.message.content)

    print("\n--- result ---")
    print(f"finish_reason: {finish_reason}")
    print(f"text_len: {len(text)}")
    preview = text[:800]
    print("preview:")
    print(preview)

    out_path = Path(os.getenv("QINGYUN_OUTPUT", "./experiment_output/qingyun_gemini3pro_ocr_test.json")).resolve()
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
