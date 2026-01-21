import os
import json
import shutil
import time
import base64
import re
import difflib
from collections import Counter
from PIL import Image  # 用于手动实现视觉压缩
import text_to_image_compact 

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ================= 配置区 =================
OUTPUT_DIR = "./experiment_output"
IMAGES_DIR_DEFAULT = os.path.join(OUTPUT_DIR, "images_gemini")  # 🌟 Gemini 专用目录
# 是否使用“已有图片集”直接 OCR + judge（用于跨模型公平对比）。
# - USE_EXISTING_IMAGES=1：跳过模块1/2，不清理 images；直接用 EXISTING_IMAGES_DIR（或默认 IMAGES_DIR_DEFAULT）里的图片。
# - DATASET_FILENAME：指定同一份 GT 数据集文件名（放在 OUTPUT_DIR 下），两种模型跑同一张表即可对比。
USE_EXISTING_IMAGES = os.getenv("USE_EXISTING_IMAGES", "0").strip().lower() in ("1", "true", "yes", "y")
EXISTING_IMAGES_DIR = os.getenv("EXISTING_IMAGES_DIR", "").strip()
IMAGES_DIR = EXISTING_IMAGES_DIR or IMAGES_DIR_DEFAULT
DEFAULT_DATASET_FILENAME = "dataset_gemini.json"
DATASET_FILENAME = os.getenv("DATASET_FILENAME", DEFAULT_DATASET_FILENAME).strip() or DEFAULT_DATASET_FILENAME
TARGET_RATIOS = [1, 2, 4, 6, 8]  # 我们的压缩目标


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")

# ================= 模块三配置（Inference Engine）=================
# 使用 Gemini（通过 aihubmix OpenAI-compat 接口）
RUN_MODULE_3 = _env_bool("RUN_MODULE_3", True)
AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"
GEMINI_MODEL_NAME = "gemini-3-pro-preview"  # 🌟 修改为 Gemini 模型
OCR_SYSTEM_PROMPT = "You are an OCR engine for code images."
OCR_USER_PROMPT = (
    "Transcribe the code in these images exactly.\n"
    "- These images are consecutive pages of the SAME code file, in order.\n"
    "- The page may start mid-block (e.g., indented lines without a visible 'def' header). Keep the indentation exactly as shown.\n"
    "- Do NOT invent missing context. Do NOT add wrapper code such as 'def', 'class', imports, or any extra lines.\n"
    "- Output plain text only (no Markdown, no code fences).\n"
    "- Preserve all whitespace, indentation, and newlines.\n"
    "- Do not add, remove, or rename anything.\n"
)

# Gemini Safety Settings（默认关闭，以避免改变原有行为；需要时通过环境变量开启）
# 说明：不同 OpenAI-compat 中转对该字段支持不一，开启后若报参数错误，可关闭该开关。
GEMINI_ENABLE_SAFETY_SETTINGS = _env_bool("GEMINI_ENABLE_SAFETY_SETTINGS", False)
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Prompt 可选增强/覆盖（默认不启用，以避免改变原有行为）
OCR_PROMPT_PERSONAL_OFFLINE = _env_bool("OCR_PROMPT_PERSONAL_OFFLINE", False)
OCR_USER_PROMPT_OVERRIDE = os.getenv("OCR_USER_PROMPT_OVERRIDE", "").strip()


def _get_ocr_user_prompt() -> str:
    """获取 OCR user prompt。

    优先级：
    1) OCR_USER_PROMPT_OVERRIDE（完全覆盖）
    2) OCR_PROMPT_PERSONAL_OFFLINE=1（在不改变约束的情况下，增加用途说明）
    3) 默认 OCR_USER_PROMPT
    """
    if OCR_USER_PROMPT_OVERRIDE:
        return OCR_USER_PROMPT_OVERRIDE
    if OCR_PROMPT_PERSONAL_OFFLINE:
        return (
            "Transcribe the code in these images exactly as it appears. "
            "This is for a personal offline syntax check project.\n"
            "- These images are consecutive pages of the SAME code file, in order.\n"
            "- The page may start mid-block (e.g., indented lines without a visible 'def' header). Keep the indentation exactly as shown.\n"
            "- Do NOT invent missing context. Do NOT add wrapper code such as 'def', 'class', imports, or any extra lines.\n"
            "- Output plain text only (no Markdown, no code fences).\n"
            "- Preserve all whitespace, indentation, and newlines.\n"
            "- Do not add, remove, or rename anything.\n"
        )
    return OCR_USER_PROMPT
OCR_MAX_TOKENS = 16384  # Gemini 支持更大上下文，这里设置为较大值
OCR_TEMPERATURE = 0.0
OCR_SLEEP_SECONDS = 0.2
OCR_MAX_RETRIES = 5
# OCR 并行配置：默认不改变行为（=1 串行）。
# 设置环境变量 OCR_CONCURRENCY=4 可显著提速；如遇到服务限流，可设置 OCR_PARALLEL_MIN_INTERVAL_SECONDS 做全局节流。
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))
OCR_PARALLEL_MIN_INTERVAL_SECONDS = float(os.getenv("OCR_PARALLEL_MIN_INTERVAL_SECONDS", "0"))
# =========================================

# ================= 模块四配置（Auto-Judge）=================
RUN_MODULE_4 = _env_bool("RUN_MODULE_4", True)  # 是否运行评估模块
JUDGE_LLM_MODEL = "gpt-5-mini"  # 用于soft taxonomy分类的模型

# 错误分类体系 (8类)
ERROR_TAXONOMY = [
    "Visual_Typo",          # 视觉相似字符替换，如 O/0, l/1
    "Symbol_Loss",          # 标点符号丢失，如缺少括号、冒号
    "Indentation_Error",    # 缩进错误
    "Line_Skipped",         # 跳行漏读
    "Variable_Hallucination", # 变量名幻觉，如将 'data' 读成 'date'
    "Code_Invention",       # 凭空捏造不存在的代码
    "Repetition",           # 重复输出某些行
    "Comment_Loss"          # 注释内容丢失
]
# =========================================


def _mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return key[:2] + "..." + key[-2:]
    return key[:6] + "..." + key[-6:]


def _try_load_api_key_from_env_files() -> str:
    """从 .env 文件中尝试读取 AIHUBMIX_API_KEY。

    优先级：环境变量（调用方处理）> 工作区根目录 .env > ocr/.env
    """
    script_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    candidates = [
        os.path.join(os.getcwd(), ".env"),   # 取决于你从哪里启动
        os.path.join(repo_dir, ".env"),      # 仓库根目录（更稳）
        os.path.join(script_dir, ".env"),    # ocr/.env
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
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
            continue

    return ""


def _safe_filename_component(text: str) -> str:
    """将模型名等字符串转换为可用于文件名的安全片段。"""
    value = (text or "").strip()
    if not value:
        return "model"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value[:80]


def _remove_file_if_exists(path: str) -> bool:
    try:
        if path and os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        return False
    return False


def _dataset_filename_for_model(model_name: str) -> str:
    """为不同大模型生成隔离的数据集文件名，避免互相覆盖。"""
    model_tag = _safe_filename_component(model_name)
    return f"dataset_{model_tag}.json"


def _iter_image_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                yield os.path.join(dirpath, fn)


def _load_done_set(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("image_path"):
                        done.add(obj["image_path"])
                    if obj.get("code_id") and ("ratio" in obj):
                        done.add(f"{obj.get('code_id')}|{obj.get('ratio')}")
                except Exception:
                    continue
    except Exception:
        return done
    return done


def _encode_image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    # 上游（含 GLM/部分中转）可能带的包围标记
    cleaned = cleaned.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    return cleaned.strip("\n")


def _extract_response_diagnostics(resp) -> dict:
    """从 OpenAI-compat 响应对象中尽量提取可用于排障的字段。

    注意：不同中转/SDK 版本字段形状可能不同；这里尽量容错，不影响原有流程。
    """
    diag: dict = {}
    try:
        resp_id = getattr(resp, "id", None)
        if resp_id:
            diag["response_id"] = resp_id
    except Exception:
        pass

    try:
        model = getattr(resp, "model", None)
        if model:
            diag["response_model"] = model
    except Exception:
        pass

    finish_reason = None
    try:
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
    except Exception:
        finish_reason = None
    if finish_reason is not None:
        diag["finish_reason"] = finish_reason

    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            # usage 可能是对象或 dict
            if hasattr(usage, "model_dump"):
                diag["usage"] = usage.model_dump()
            elif isinstance(usage, dict):
                diag["usage"] = usage
    except Exception:
        pass

    # 某些实现可能提供 refusal / safety 信息（尽量抓取，不做强依赖）
    try:
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            msg = getattr(resp.choices[0], "message", None)
            refusal = getattr(msg, "refusal", None) if msg is not None else None
            if refusal:
                diag["refusal"] = refusal
    except Exception:
        pass

    return diag


def _parse_ratio_from_filename(image_path: str) -> int:
    # e.g. page_001_ratio2.png -> 2 ; page_001.png -> 1
    stem = os.path.splitext(os.path.basename(image_path))[0]
    marker = "_ratio"
    if marker in stem:
        try:
            return int(stem.split(marker, 1)[1])
        except Exception:
            return 1
    return 1


def _extract_page_num_from_filename(image_path: str) -> int:
    """page_001_ratio2.png -> 1；提取不到则返回 0。"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    m = re.search(r"page_(\d+)", stem)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def run_module_3_gemini(images_dir: str, output_dir: str):
    print("\n" + "=" * 40)
    print(f"🚀 Running Module 3: Inference Engine ({GEMINI_MODEL_NAME})")
    print("=" * 40)

    if OpenAI is None:
        print("❌ Missing dependency: openai. Run: pip install openai")
        return

    api_key = os.getenv("AIHUBMIX_API_KEY")
    api_key_source = "env:AIHUBMIX_API_KEY" if api_key else ""
    if not api_key:
        api_key = _try_load_api_key_from_env_files()
        if api_key:
            api_key_source = "file:.env"
    if not api_key:
        searched = [
            os.path.join(os.getcwd(), ".env"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".env")),
            os.path.join(os.path.dirname(__file__), ".env"),
        ]
        print("❌ Missing AIHUBMIX_API_KEY.")
        print("   - 方式1：PowerShell 临时设置：$env:AIHUBMIX_API_KEY=\"sk-...\"")
        print("   - 方式2：写入 .env 文件（不要提交到仓库）")
        print("     格式：AIHUBMIX_API_KEY=sk-...")
        print("     查找路径：")
        for p in searched:
            print(f"       - {p}")
        return

    print(f"🔑 AIHUBMIX_API_KEY loaded ({api_key_source}): {_mask_api_key(api_key)}")

    os.makedirs(output_dir, exist_ok=True)
    out_jsonl = os.path.join(output_dir, "gemini_ocr.jsonl")  # 🌟 修改输出文件名
    done = _load_done_set(out_jsonl)

    client = OpenAI(api_key=api_key, base_url=AIHUBMIX_BASE_URL)

    total = 0
    skipped = 0
    errors = 0

    # single-turn：按 (code_id, ratio) 分组，把同一个样本的所有 pages 在同一次请求中发送
    image_paths = list(_iter_image_files(images_dir))
    from collections import defaultdict

    grouped_images = defaultdict(list)  # (code_id, ratio) -> [image_path...]
    for image_path in image_paths:
        parent_dir = os.path.dirname(image_path)
        code_id_dir = os.path.dirname(parent_dir)
        code_id = os.path.basename(code_id_dir)
        ratio = _parse_ratio_from_filename(image_path)
        grouped_images[(code_id, ratio)].append(image_path)

    cases = []  # [(code_id, ratio, [paths...])]
    for (code_id, ratio), paths in grouped_images.items():
        paths.sort(key=lambda p: (_extract_page_num_from_filename(p), os.path.basename(p)))
        cases.append((code_id, ratio, paths))
    cases.sort(key=lambda x: (x[0], x[1]))

    print(f"🧩 Total cases to OCR (single-turn): {len(cases)}")

    if OCR_CONCURRENCY <= 1:
        for i, (code_id, ratio, page_paths) in enumerate(cases, start=1):
            case_key = f"{code_id}|{ratio}"
            if case_key in done:
                skipped += 1
                continue
            print(
                f"[{i}/{len(cases)}] OCR(single-turn): {code_id} @ ratio {ratio}x ({len(page_paths)} pages)"
            )

            content = [{"type": "text", "text": _get_ocr_user_prompt()}]
            for p in page_paths:
                data_url = _encode_image_to_data_url(p)
                content.append({"type": "image_url", "image_url": {"url": data_url}})

            last_err = None
            text = ""
            diagnostics = {}

            for attempt in range(1, OCR_MAX_RETRIES + 1):
                try:
                    extra_body = {"safety_settings": GEMINI_SAFETY_SETTINGS} if GEMINI_ENABLE_SAFETY_SETTINGS else None
                    resp = client.chat.completions.create(
                        model=GEMINI_MODEL_NAME,  # 🌟 使用 Gemini 模型
                        temperature=OCR_TEMPERATURE,
                        max_tokens=OCR_MAX_TOKENS,
                        messages=[
                            {"role": "system", "content": OCR_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": content,
                            },
                        ],
                        extra_body=extra_body,
                    )
                    text = _clean_ocr_text(resp.choices[0].message.content or "")
                    diagnostics = _extract_response_diagnostics(resp)
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    # exponential backoff: 1,2,4,8,... capped at 30s
                    backoff = min(30.0, float(2 ** (attempt - 1)))
                    time.sleep(backoff)

            rec = {
                "code_id": code_id,
                "ratio": ratio,
                "num_pages": len(page_paths),
                "image_paths": page_paths,
                "image_path": page_paths[0] if page_paths else "",
                "model": GEMINI_MODEL_NAME,  # 🌟 记录模型名称
            }

            if diagnostics:
                rec.update(diagnostics)

            if last_err is None:
                rec["text"] = text
                rec["text_len"] = len(text)
                if rec.get("finish_reason") in ("content_filter", "safety"):
                    rec["blocked_by_safety"] = True
                total += 1
            else:
                rec["error"] = last_err
                errors += 1

            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            time.sleep(OCR_SLEEP_SECONDS)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        pending_cases = [(code_id, ratio, page_paths) for (code_id, ratio, page_paths) in cases if f"{code_id}|{ratio}" not in done]
        skipped = len(cases) - len(pending_cases)

        print(
            f"⚡ Parallel OCR enabled: workers={OCR_CONCURRENCY}, "
            f"global_min_interval={OCR_PARALLEL_MIN_INTERVAL_SECONDS}s"
        )

        client_local = threading.local()
        write_lock = threading.Lock()
        rate_lock = threading.Lock()
        next_allowed_time = 0.0

        def _get_client():
            c = getattr(client_local, "client", None)
            if c is None:
                c = OpenAI(api_key=api_key, base_url=AIHUBMIX_BASE_URL)
                client_local.client = c
            return c

        def _rate_limit_wait():
            nonlocal next_allowed_time
            interval = float(OCR_PARALLEL_MIN_INTERVAL_SECONDS)
            if interval <= 0:
                return
            with rate_lock:
                now = time.monotonic()
                if now < next_allowed_time:
                    wait_s = next_allowed_time - now
                    next_allowed_time = next_allowed_time + interval
                else:
                    wait_s = 0.0
                    next_allowed_time = now + interval
            if wait_s > 0:
                time.sleep(wait_s)

        def _ocr_one_case(code_id: str, ratio: int, page_paths: list[str]):
            content = [{"type": "text", "text": _get_ocr_user_prompt()}]
            for p in page_paths:
                data_url = _encode_image_to_data_url(p)
                content.append({"type": "image_url", "image_url": {"url": data_url}})

            last_err = None
            text = ""
            diagnostics = {}

            for attempt in range(1, OCR_MAX_RETRIES + 1):
                try:
                    _rate_limit_wait()
                    extra_body = {"safety_settings": GEMINI_SAFETY_SETTINGS} if GEMINI_ENABLE_SAFETY_SETTINGS else None
                    resp = _get_client().chat.completions.create(
                        model=GEMINI_MODEL_NAME,
                        temperature=OCR_TEMPERATURE,
                        max_tokens=OCR_MAX_TOKENS,
                        messages=[
                            {"role": "system", "content": OCR_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": content,
                            },
                        ],
                        extra_body=extra_body,
                    )
                    text = _clean_ocr_text(resp.choices[0].message.content or "")
                    diagnostics = _extract_response_diagnostics(resp)
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    backoff = min(30.0, float(2 ** (attempt - 1)))
                    time.sleep(backoff)

            rec = {
                "code_id": code_id,
                "ratio": ratio,
                "num_pages": len(page_paths),
                "image_paths": page_paths,
                "image_path": page_paths[0] if page_paths else "",
                "model": GEMINI_MODEL_NAME,
            }
            if diagnostics:
                rec.update(diagnostics)
            if last_err is None:
                rec["text"] = text
                rec["text_len"] = len(text)
                if rec.get("finish_reason") in ("content_filter", "safety"):
                    rec["blocked_by_safety"] = True
                return rec, True
            rec["error"] = last_err
            return rec, False

        completed = 0
        total_jobs = len(pending_cases)

        with ThreadPoolExecutor(max_workers=OCR_CONCURRENCY) as ex:
            futures = {ex.submit(_ocr_one_case, code_id, ratio, page_paths): (code_id, ratio, page_paths) for (code_id, ratio, page_paths) in pending_cases}
            for fut in as_completed(futures):
                code_id, ratio, page_paths = futures[fut]
                try:
                    rec, ok = fut.result()
                except Exception as e:
                    rec = {
                        "code_id": code_id,
                        "ratio": ratio,
                        "num_pages": len(page_paths),
                        "image_paths": page_paths,
                        "image_path": page_paths[0] if page_paths else "",
                        "model": GEMINI_MODEL_NAME,
                        "error": f"worker_exception: {e}",
                    }
                    ok = False

                with write_lock:
                    with open(out_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                completed += 1
                if ok:
                    total += 1
                else:
                    errors += 1
                print(
                    f"[{completed}/{total_jobs}] OCR done: {code_id} @ ratio {ratio}x "
                    f"({'ok' if ok else 'error'})"
                )

    print(f"✅ Module 3 finished. ok={total}, skipped={skipped}, error={errors}")
    print(f"📄 Output: {os.path.abspath(out_jsonl)}")


# ============================================================
# 🟠 模块四: Auto-Judge (评估器)
# ============================================================

def normalize_code(text: str) -> str:
    """
    代码规范化：压缩空行 + 去除行尾空格 + tab→4空格
    用于计算 CER/WER/BLEU 等指标时减少格式噪声
    """
    lines = text.splitlines()
    
    # 1. Tab → 4 spaces
    lines = [line.replace('\t', '    ') for line in lines]
    
    # 2. 去除行尾空格（trailing spaces）
    lines = [line.rstrip() for line in lines]
    
    # 3. 压缩连续空行为单个空行
    normalized = []
    prev_blank = False
    for line in lines:
        is_blank = (line.strip() == '')
        if is_blank:
            if not prev_blank:  # 只保留第一个空行
                normalized.append('')
            prev_blank = True
        else:
            normalized.append(line)
            prev_blank = False
    
    # 4. 去除首尾空行
    while normalized and normalized[0] == '':
        normalized.pop(0)
    while normalized and normalized[-1] == '':
        normalized.pop()
    
    return '\n'.join(normalized)


def _split_nonblank_lines_for_diff(text: str) -> list[str]:
    """用于 codediff 的预处理：tab->4空格、去行尾空格、删除所有空行（不动行首缩进）。"""
    lines = text.splitlines()
    lines = [line.replace('\t', '    ').rstrip() for line in lines]
    return [line for line in lines if line.strip() != ""]


def _compute_codediff_metrics_no_blank(reference: str, hypothesis: str) -> dict:
    ref_lines = _split_nonblank_lines_for_diff(reference)
    hyp_lines = _split_nonblank_lines_for_diff(hypothesis)

    sm = difflib.SequenceMatcher(a=ref_lines, b=hyp_lines, autojunk=False)
    opcodes = sm.get_opcodes()

    added = 0
    removed = 0
    replaced = 0
    hunks = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            hunks += 1
        if tag == "insert":
            added += (j2 - j1)
        elif tag == "delete":
            removed += (i2 - i1)
        elif tag == "replace":
            replaced += max(i2 - i1, j2 - j1)

    ref_line_count = len(ref_lines)
    change = added + removed + replaced

    return {
        "line_similarity": round(sm.ratio(), 4),
        "added_lines": added,
        "removed_lines": removed,
        "replaced_lines": replaced,
        "diff_hunks": hunks,
        "change_rate": round(change / max(1, ref_line_count), 4),
        "ref_nonblank_lines": ref_line_count,
        "hyp_nonblank_lines": len(hyp_lines),
    }


def _compute_cer(reference: str, hypothesis: str) -> float:
    """
    计算字符错误率 (Character Error Rate)
    CER = (S + D + I) / N
    使用 Levenshtein 编辑距离
    """
    ref = list(reference)
    hyp = list(hypothesis)
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return 1.0 if m > 0 else 0.0

    # DP 矩阵
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    edit_distance = dp[n][m]
    return edit_distance / n


def _check_ast_parsable(code: str) -> bool:
    """检查代码是否可被 Python AST 解析"""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _compute_keyword_f1(reference: str, hypothesis: str) -> float:
    """计算语言关键字的 F1（CodeBLEU 的 keyword-match 子项，轻量实现）。"""
    import keyword

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    keywords = set(keyword.kwlist)
    ref_kw = [t for t in tokenize(reference) if t in keywords]
    hyp_kw = [t for t in tokenize(hypothesis) if t in keywords]

    if not ref_kw and not hyp_kw:
        return 1.0
    if not ref_kw or not hyp_kw:
        return 0.0

    from collections import Counter

    ref_c = Counter(ref_kw)
    hyp_c = Counter(hyp_kw)
    overlap = sum(min(ref_c[k], hyp_c.get(k, 0)) for k in ref_c)

    precision = overlap / max(1, sum(hyp_c.values()))
    recall = overlap / max(1, sum(ref_c.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_codebleu(reference: str, hypothesis: str) -> float:
    """轻量版 CodeBLEU：ngram-match(token BLEU) + keyword-match(F1)。

    说明：完整 CodeBLEU 还包含 syntax-match / dataflow-match，通常需要 tree-sitter 与数据流提取。
    这里先实现无额外依赖的版本，便于工程稳定落地。
    """
    ngram = _compute_token_bleu(reference, hypothesis)
    kw_f1 = _compute_keyword_f1(reference, hypothesis)
    return 0.8 * ngram + 0.2 * kw_f1


def _compute_token_bleu(reference: str, hypothesis: str) -> float:
    """
    计算基于 token 的简化 BLEU (CodeBLEU 核心)
    使用 1-gram 到 4-gram 的几何平均
    """
    import re
    from collections import Counter
    import math

    def tokenize(text):
        # 简单的代码分词：按非字母数字字符分割
        return re.findall(r'\w+|[^\w\s]', text)

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if len(hyp_tokens) == 0:
        return 0.0

    # 计算 n-gram precision
    def ngram_precision(ref_toks, hyp_toks, n):
        if len(hyp_toks) < n:
            return 0.0
        ref_ngrams = Counter(tuple(ref_toks[i:i+n]) for i in range(len(ref_toks) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_toks[i:i+n]) for i in range(len(hyp_toks) - n + 1))

        match = 0
        total = 0
        for ng, cnt in hyp_ngrams.items():
            match += min(cnt, ref_ngrams.get(ng, 0))
            total += cnt
        return match / total if total > 0 else 0.0

    # 计算 1-gram 到 4-gram 的 precision
    precisions = []
    for n in range(1, 5):
        p = ngram_precision(ref_tokens, hyp_tokens, n)
        precisions.append(p)

    # 过滤掉 0 值（避免 log(0)）
    non_zero = [p for p in precisions if p > 0]
    if not non_zero:
        return 0.0

    # 几何平均
    log_avg = sum(math.log(p) for p in non_zero) / len(non_zero)
    bleu = math.exp(log_avg)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0

    return bp * bleu


def _compute_wer(reference: str, hypothesis: str) -> float:
    """
    计算词错误率 (Word Error Rate)
    WER = (S + D + I) / N，以词为单位
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 1.0 if m > 0 else 0.0

    # DP 矩阵
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    edit_distance = dp[n][m]
    return edit_distance / n


def _compute_exact_match_rate(reference: str, hypothesis: str) -> float:
    """
    计算精确匹配率 (Exact Match Rate)
    完全匹配的行数 / 总行数
    """
    ref_lines = reference.split('\n')
    hyp_lines = hypothesis.split('\n')
    
    if len(ref_lines) == 0:
        return 1.0 if len(hyp_lines) == 0 else 0.0
    
    # 对齐比较：取两者中较短的长度
    matched = 0
    for i, ref_line in enumerate(ref_lines):
        if i < len(hyp_lines) and ref_line == hyp_lines[i]:
            matched += 1
    
    return matched / len(ref_lines)


# ============================================================
# 🏷️ 八大错误谱系检测函数（基于规则，不依赖 LLM）
# ============================================================

def _detect_visual_typo(reference: str, hypothesis: str) -> int:
    """
    检测形近字混淆 (1 vs l, 0 vs O, etc.)
    返回: 1 = 检测到, 0 = 未检测到
    """
    # 形近字对（双向检测）
    confusable_pairs = [
        ('1', 'l'), ('1', 'I'), ('l', 'I'),  # 1/l/I
        ('0', 'O'), ('0', 'o'), ('O', 'o'),  # 0/O/o
        ('5', 'S'), ('5', 's'),              # 5/S
        ('8', 'B'),                          # 8/B
        ('2', 'Z'), ('2', 'z'),              # 2/Z
        ('6', 'G'),                          # 6/G
        ('rn', 'm'),                         # rn/m (连字)
        ("'", '`'), ('"', "''"),             # 引号混淆
    ]
    
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()
    
    for a, b in confusable_pairs:
        # 检查是否发生了替换：ref 中有 a，hyp 中对应位置变成了 b
        ref_count_a = reference.count(a)
        hyp_count_a = hypothesis.count(a)
        ref_count_b = reference.count(b)
        hyp_count_b = hypothesis.count(b)
        
        # 如果 a 在 ref 中更多，但 b 在 hyp 中更多，可能发生了混淆
        if ref_count_a > hyp_count_a and hyp_count_b > ref_count_b:
            return 1
        if ref_count_b > hyp_count_b and hyp_count_a > ref_count_a:
            return 1
    
    return 0


def _detect_symbol_loss(reference: str, hypothesis: str) -> int:
    """
    检测符号丢失 (_, :, ;, 括号等)
    返回: 1 = 检测到, 0 = 未检测到
    """
    critical_symbols = ['_', ':', ';', '(', ')', '[', ']', '{', '}', ',', '.', '=', '+', '-', '*', '/']
    
    for sym in critical_symbols:
        ref_count = reference.count(sym)
        hyp_count = hypothesis.count(sym)
        
        # 如果 ref 中的符号比 hyp 中多 20% 以上，认为发生了丢失
        if ref_count > 0 and hyp_count < ref_count * 0.8:
            return 1
    
    return 0


def _detect_indentation_error(reference: str, hypothesis: str) -> int:
    """
    检测缩进错误
    返回: 1 = 检测到, 0 = 未检测到
    """
    ref_lines = reference.split('\n')
    hyp_lines = hypothesis.split('\n')
    
    # 计算每行开头的空格数
    def get_indent(line):
        return len(line) - len(line.lstrip(' \t'))
    
    # 取两者较短的长度进行比较
    min_len = min(len(ref_lines), len(hyp_lines))
    
    indent_errors = 0
    for i in range(min_len):
        ref_indent = get_indent(ref_lines[i])
        hyp_indent = get_indent(hyp_lines[i])
        
        # 如果缩进差异超过 2 个空格，认为是错误
        if abs(ref_indent - hyp_indent) >= 2:
            indent_errors += 1
    
    # 如果超过 5% 的行有缩进错误，标记为 1
    if min_len > 0 and indent_errors / min_len > 0.05:
        return 1
    
    return 0


def _detect_line_skipped(reference: str, hypothesis: str) -> int:
    """
    检测整行丢失
    返回: 1 = 检测到, 0 = 未检测到
    """
    ref_lines = [line.strip() for line in reference.split('\n') if line.strip()]
    hyp_lines = [line.strip() for line in hypothesis.split('\n') if line.strip()]
    
    # 如果 hyp 的行数少于 ref 的 90%，认为发生了行丢失
    if len(hyp_lines) < len(ref_lines) * 0.9:
        return 1
    
    # 检查是否有 ref 中的行完全不在 hyp 中
    hyp_set = set(hyp_lines)
    missing_lines = 0
    for line in ref_lines:
        if len(line) > 10 and line not in hyp_set:  # 只检查有意义的行
            missing_lines += 1
    
    # 如果超过 10% 的有意义行丢失，标记为 1
    if len(ref_lines) > 0 and missing_lines / len(ref_lines) > 0.1:
        return 1
    
    return 0


def _detect_variable_hallucination(reference: str, hypothesis: str) -> int:
    """
    检测变量名幻觉（OCR 中出现了 reference 中不存在的标识符）
    返回: 1 = 检测到, 0 = 未检测到
    """
    import re
    
    # 提取标识符（变量名、函数名等）
    def extract_identifiers(code):
        # 匹配 Python 标识符
        identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code))
        # 排除 Python 关键字和常见内置函数
        keywords = {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 
                   'import', 'from', 'as', 'try', 'except', 'finally', 'with', 
                   'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is',
                   'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
                   'self', 'cls', 'args', 'kwargs'}
        return identifiers - keywords
    
    ref_ids = extract_identifiers(reference)
    hyp_ids = extract_identifiers(hypothesis)
    
    # 找出 hyp 中有但 ref 中没有的标识符（幻觉）
    hallucinated = hyp_ids - ref_ids
    
    # 过滤掉长度小于 3 的（可能是误判）
    hallucinated = {h for h in hallucinated if len(h) >= 3}
    
    # 如果有超过 3 个幻觉标识符，标记为 1
    if len(hallucinated) >= 3:
        return 1
    
    return 0


def _detect_code_invention(reference: str, hypothesis: str) -> int:
    """
    检测代码捏造（OCR 中出现了完全不存在的代码段）
    返回: 1 = 检测到, 0 = 未检测到
    """
    hyp_lines = [line.strip() for line in hypothesis.split('\n') if line.strip()]
    
    invented_lines = 0
    for line in hyp_lines:
        # 只检查有意义的行（长度 > 15）
        if len(line) > 15:
            # 如果这行在 reference 中完全找不到任何相似片段
            if line not in reference and line[:20] not in reference:
                invented_lines += 1
    
    # 如果有超过 5% 的行是捏造的，标记为 1
    if len(hyp_lines) > 0 and invented_lines / len(hyp_lines) > 0.05:
        return 1
    
    return 0


def _detect_repetition(reference: str, hypothesis: str) -> int:
    """
    检测重复输出（复读机现象）
    返回: 1 = 检测到, 0 = 未检测到
    """
    hyp_lines = [line for line in hypothesis.split('\n') if line.strip()]
    
    if len(hyp_lines) < 3:
        return 0
    
    # 检测连续重复的行
    consecutive_repeats = 0
    for i in range(1, len(hyp_lines)):
        if hyp_lines[i] == hyp_lines[i-1] and len(hyp_lines[i].strip()) > 5:
            consecutive_repeats += 1
    
    # 如果有 2 行以上连续重复，标记为 1
    if consecutive_repeats >= 2:
        return 1
    
    # 检测非连续的大量重复
    from collections import Counter
    line_counts = Counter(line for line in hyp_lines if len(line.strip()) > 10)
    
    # 如果有任何行重复超过 3 次，标记为 1
    for line, count in line_counts.items():
        if count >= 3:
            return 1
    
    return 0


def _detect_comment_loss(reference: str, hypothesis: str) -> int:
    """
    检测注释丢失或乱码
    返回: 1 = 检测到, 0 = 未检测到
    """
    # 提取注释行
    def extract_comments(code):
        comments = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#'):
                comments.append(stripped)
            # 也检测行内注释
            if '#' in line:
                comment_part = line.split('#', 1)[1].strip()
                if comment_part:
                    comments.append('#' + comment_part)
        return comments
    
    ref_comments = extract_comments(reference)
    hyp_comments = extract_comments(hypothesis)
    
    if not ref_comments:
        return 0  # 原代码没有注释
    
    # 检查注释数量是否大幅减少
    if len(hyp_comments) < len(ref_comments) * 0.7:
        return 1
    
    # 检查注释内容是否严重变形
    matched = 0
    for ref_c in ref_comments:
        for hyp_c in hyp_comments:
            # 简单的相似度检测
            if ref_c in hyp_c or hyp_c in ref_c:
                matched += 1
                break
            # 或者超过 70% 的字符匹配
            common_chars = sum(1 for c in ref_c if c in hyp_c)
            if len(ref_c) > 0 and common_chars / len(ref_c) > 0.7:
                matched += 1
                break
    
    # 如果少于 70% 的注释被正确保留，标记为 1
    if len(ref_comments) > 0 and matched / len(ref_comments) < 0.7:
        return 1
    
    return 0


def _detect_all_taxonomy_errors(reference: str, hypothesis: str) -> dict:
    """
    检测所有八大错误谱系，返回 0/1 标签字典
    """
    return {
        "Visual_Typo": _detect_visual_typo(reference, hypothesis),
        "Symbol_Loss": _detect_symbol_loss(reference, hypothesis),
        "Indentation_Error": _detect_indentation_error(reference, hypothesis),
        "Line_Skipped": _detect_line_skipped(reference, hypothesis),
        "Variable_Hallucination": _detect_variable_hallucination(reference, hypothesis),
        "Code_Invention": _detect_code_invention(reference, hypothesis),
        "Repetition": _detect_repetition(reference, hypothesis),
        "Comment_Loss": _detect_comment_loss(reference, hypothesis),
    }


def _call_llm_for_taxonomy(client, reference: str, hypothesis: str) -> list:
    """
    调用 LLM 进行错误分类
    返回检测到的错误类型列表
    """
    prompt = f"""You are an expert code quality evaluator. 
Compare the reference code with the OCR output and identify error types.

Reference code:
```
{reference[:3000]}
```

OCR output:
```
{hypothesis[:3000]}
```

Analyze the differences and return ONLY a JSON array of error types from this list:
{ERROR_TAXONOMY}

Rules:
- Return an empty array [] if the output matches the reference perfectly
- Only include error types that are clearly present
- Return ONLY the JSON array, no other text

Example response: ["Visual_Typo", "Symbol_Loss"]
"""

    try:
        resp = client.chat.completions.create(
            model=JUDGE_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=256,
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        # 解析 JSON
        import re
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except Exception as e:
        print(f"   ⚠️ LLM taxonomy call failed: {e}")
        return []


def run_module_4_judge(
    output_dir: str,
    ocr_jsonl_filename: str,
    ocr_model_name: str,
    dataset_json_filename: str | None = None,
):
    """
    模块4: Auto-Judge 评估器
    读取 OCR 结果，与原始代码对比，输出评估指标
    """
    model_tag = _safe_filename_component(ocr_model_name)

    print("\n" + "=" * 40)
    print(f"🚀 Running Module 4: Auto-Judge ({ocr_model_name})")
    print("=" * 40)

    # 读取 dataset (按模型隔离，避免跨脚本覆盖)
    preferred_dataset = dataset_json_filename or _dataset_filename_for_model(ocr_model_name)
    dataset_path = os.path.join(output_dir, preferred_dataset)
    if not os.path.exists(dataset_path):
        legacy_path = os.path.join(output_dir, "dataset.json")
        if os.path.exists(legacy_path):
            dataset_path = legacy_path
        else:
            print(f"❌ {preferred_dataset} not found (and dataset.json not found), skipping Module 4")
            return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 建立 code_id -> code 的映射
    code_map = {item["id"]: item["code"] for item in dataset}

    # 读取 OCR 结果
    ocr_path = os.path.join(output_dir, ocr_jsonl_filename)
    if not os.path.exists(ocr_path):
        print(f"❌ {ocr_jsonl_filename} not found, skipping Module 4")
        return

    ocr_results = []
    with open(ocr_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ocr_results.append(json.loads(line))

    # 🌟 按 (code_id, ratio) 分组：
    # - 旧格式：每页一条记录（image_path/text），需要合并多页
    # - 新格式（single-turn）：每样本一条记录（image_paths/num_pages/text），无需再拼页
    from collections import defaultdict
    grouped = defaultdict(list)  # (code_id, ratio) -> [{"text": ..., "image_path": ..., "num_pages": ...?}, ...]

    for rec in ocr_results:
        ratio = rec.get("ratio", 1)
        ocr_text = rec.get("text", "")

        # 跳过错误结果
        if not ocr_text or "error" in rec:
            continue

        # 优先使用显式 code_id；否则从 image_path 回推
        code_id = rec.get("code_id", "")
        img_path = rec.get("image_path", "")
        if not code_id and img_path:
            parts = img_path.replace("\\", "/").split("/")
            code_id = parts[-3] if len(parts) >= 3 else ""

        if not code_id:
            continue

        # 清理特殊标记（关键！）
        ocr_text = ocr_text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '').strip()

        if rec.get("image_paths"):
            grouped[(code_id, ratio)].append({
                "text": ocr_text,
                "image_path": img_path,
                "num_pages": int(rec.get("num_pages") or len(rec.get("image_paths") or [])),
            })
        else:
            grouped[(code_id, ratio)].append({
                "text": ocr_text,
                "image_path": img_path,
            })

    # 评估结果
    detail_path = os.path.join(output_dir, f"judge_results_detail_{model_tag}.jsonl")
    summary_path = os.path.join(output_dir, f"judge_summary_{model_tag}.json")

    # 按 ratio 分组统计
    stats_by_ratio = {r: {
        "cer_sum": 0, "wer_sum": 0, "bleu_sum": 0, "codebleu_sum": 0,
        "exact_match_sum": 0,
        "codediff_line_sim_sum": 0.0,
        "codediff_change_rate_sum": 0.0,
        "codediff_added_sum": 0,
        "codediff_removed_sum": 0,
        "codediff_replaced_sum": 0,
        "codediff_hunks_sum": 0,
        "count": 0,
        "errors": Counter(), "taxonomy_sums": Counter()
    } for r in TARGET_RATIOS}
    # 清空 detail 文件
    open(detail_path, "w").close()

    total = len(grouped)
    evaluated = 0
    
    # 🌟 现在按 (code_id, ratio) 组合进行评估
    for idx, ((code_id, ratio), pages) in enumerate(grouped.items(), 1):
        reference = code_map.get(code_id, "")
        if not reference:
            print(f"[{idx}/{total}] ⚠️ No ground truth for {code_id}, skipping")
            continue

        # 🌟 合并多页 OCR 结果：
        # - single-turn：pages 只有一条（整段文本），不再拼页
        # - legacy：按 image_path 排序后拼接
        pages.sort(key=lambda x: x.get("image_path", ""))
        if len(pages) == 1 and ("num_pages" in pages[0]):
            merged_ocr = pages[0]["text"]
            num_pages = int(pages[0].get("num_pages") or 1)
        else:
            merged_ocr = '\n'.join([p["text"] for p in pages])
            num_pages = len(pages)

        evaluated += 1
        print(f"[{idx}/{total}] Evaluating: {code_id} @ ratio {ratio}x ({num_pages} pages)")

        # 🌟 规范化处理（用于 hard metrics）
        ref_normalized = normalize_code(reference)
        ocr_normalized = normalize_code(merged_ocr)

        # 1. Hard metrics（使用规范化后的文本）
        cer = _compute_cer(ref_normalized, ocr_normalized)
        wer = _compute_wer(ref_normalized, ocr_normalized)
        bleu = _compute_token_bleu(ref_normalized, ocr_normalized)
        codebleu = _compute_codebleu(ref_normalized, ocr_normalized)
        exact_match = _compute_exact_match_rate(ref_normalized, ocr_normalized)

        # 2. Soft taxonomy（使用原始文本，保留缩进/符号/空行信息）
        taxonomy_labels = _detect_all_taxonomy_errors(reference, merged_ocr)
        detected_error_types = [k for k, v in taxonomy_labels.items() if v == 1]

        # 记录详情 
        detail_rec = {
            "code_id": code_id,
            "ratio": ratio,
            "num_pages": num_pages,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "token_bleu": round(bleu, 4),
            "codebleu": round(codebleu, 4),
            "exact_match_rate": round(exact_match, 4),
            "taxonomy_labels": taxonomy_labels,
            "detected_errors": detected_error_types,
        }

        codediff_no_blank = _compute_codediff_metrics_no_blank(reference, merged_ocr)
        detail_rec["codediff_no_blank"] = codediff_no_blank

        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(detail_rec, ensure_ascii=False) + "\n")

        # 更新统计
        if ratio in stats_by_ratio:
            stats_by_ratio[ratio]["cer_sum"] += cer
            stats_by_ratio[ratio]["wer_sum"] += wer
            stats_by_ratio[ratio]["bleu_sum"] += bleu
            stats_by_ratio[ratio]["codebleu_sum"] += codebleu
            stats_by_ratio[ratio]["exact_match_sum"] += exact_match
            stats_by_ratio[ratio]["codediff_line_sim_sum"] += float(codediff_no_blank.get("line_similarity", 0.0))
            stats_by_ratio[ratio]["codediff_change_rate_sum"] += float(codediff_no_blank.get("change_rate", 0.0))
            stats_by_ratio[ratio]["codediff_added_sum"] += int(codediff_no_blank.get("added_lines", 0))
            stats_by_ratio[ratio]["codediff_removed_sum"] += int(codediff_no_blank.get("removed_lines", 0))
            stats_by_ratio[ratio]["codediff_replaced_sum"] += int(codediff_no_blank.get("replaced_lines", 0))
            stats_by_ratio[ratio]["codediff_hunks_sum"] += int(codediff_no_blank.get("diff_hunks", 0))
            stats_by_ratio[ratio]["count"] += 1
            
            # 记录该样本中出现的错误（taxonomy）
            for err_type, val in taxonomy_labels.items():
                if val == 1:
                    stats_by_ratio[ratio]["errors"][err_type] += 1
                stats_by_ratio[ratio]["taxonomy_sums"][err_type] += val

    # 生成汇总
    summary = {}
    for ratio, s in stats_by_ratio.items():
        if s["count"] == 0:
            continue
        # 计算每种错误类型的检出率
        error_rates = {et: round(cnt / s["count"], 4) 
                      for et, cnt in s["errors"].items()}

        # 8类错误 -> 3大类聚合统计
        error_groups = {
            "Recognition": ["Visual_Typo", "Symbol_Loss", "Comment_Loss"],
            "Structure": ["Indentation_Error", "Line_Skipped", "Repetition"],
            "Hallucination": ["Variable_Hallucination", "Code_Invention"],
        }
        error_group_counts = {
            group: int(sum(s["errors"].get(k, 0) for k in members))
            for group, members in error_groups.items()
        }
        error_group_rates = {
            group: round(cnt / s["count"], 4)
            for group, cnt in error_group_counts.items()
        }
        summary[f"ratio_{ratio}x"] = {
            "count": s["count"],
            "avg_cer": round(s["cer_sum"] / s["count"], 4),
            "avg_wer": round(s["wer_sum"] / s["count"], 4),
            "avg_token_bleu": round(s["bleu_sum"] / s["count"], 4),
            "avg_codebleu": round(s["codebleu_sum"] / s["count"], 4),
            "avg_exact_match_rate": round(s["exact_match_sum"] / s["count"], 4),
            "avg_codediff_line_similarity_no_blank": round(s["codediff_line_sim_sum"] / s["count"], 4),
            "avg_codediff_change_rate_no_blank": round(s["codediff_change_rate_sum"] / s["count"], 4),
            "avg_codediff_added_lines_no_blank": round(s["codediff_added_sum"] / s["count"], 4),
            "avg_codediff_removed_lines_no_blank": round(s["codediff_removed_sum"] / s["count"], 4),
            "avg_codediff_replaced_lines_no_blank": round(s["codediff_replaced_sum"] / s["count"], 4),
            "avg_codediff_diff_hunks_no_blank": round(s["codediff_hunks_sum"] / s["count"], 4),
            "error_counts": dict(s["errors"]),  # 原始计数
            "error_rates": error_rates,  # 检出率
            "error_group_counts": error_group_counts,
            "error_group_rates": error_group_rates,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Module 4 finished.")
    print(f"📄 Detail: {os.path.abspath(detail_path)}")
    print(f"📊 Summary: {os.path.abspath(summary_path)}")

    # 打印简要汇总
    print("\n📈 Quick Summary:")
    for ratio_key, data in summary.items():
        print(f"   {ratio_key}:")
        print(f"      ├─ CER={data['avg_cer']:.2%}, WER={data['avg_wer']:.2%}")
        print(
            f"      ├─ CodeBLEU={data.get('avg_codebleu', 0.0):.4f}, "
            f"BLEU={data['avg_token_bleu']:.4f}, Exact Match={data['avg_exact_match_rate']:.2%}"
        )
        if "avg_codediff_line_similarity_no_blank" in data:
            print(
                "      ├─ CodeDiff(no-blank): "
                f"line_sim={data.get('avg_codediff_line_similarity_no_blank', 0.0):.4f}, "
                f"change_rate={data.get('avg_codediff_change_rate_no_blank', 0.0):.4f}, "
                f"hunks={data.get('avg_codediff_diff_hunks_no_blank', 0.0):.2f}, "
                f"replaced={data.get('avg_codediff_replaced_lines_no_blank', 0.0):.2f}"
            )
        if "error_group_counts" in data:
            egc = data.get("error_group_counts") or {}
            print(
                "      ├─ ErrorGroups(count): "
                f"Recognition={egc.get('Recognition', 0)}, "
                f"Structure={egc.get('Structure', 0)}, "
                f"Hallucination={egc.get('Hallucination', 0)}"
            )
        if "error_group_rates" in data:
            egr = data.get("error_group_rates") or {}
            print(
                "      ├─ ErrorGroups(rate): "
                f"Recognition={float(egr.get('Recognition', 0.0)):.2%}, "
                f"Structure={float(egr.get('Structure', 0.0)):.2%}, "
                f"Hallucination={float(egr.get('Hallucination', 0.0)):.2%}"
            )
        if data['error_counts']:
            err_str = ", ".join([f"{k}:{v}" for k, v in data['error_counts'].items()])
            print(f"      └─ Errors: {err_str}")
        else:
            print(f"      └─ Errors: (none detected)")


def apply_visual_corruption(image_path, ratio):
    """
    手动实现视觉干扰器：读取原图，先按比例缩小再放大回原尺寸（保持尺寸一致）
    约定：无论 ratio 是 1/2/4/6/8，都生成一个带 _ratio{ratio} 后缀的新文件。
    """
    try:
        with Image.open(image_path) as img:
            # 构造新文件名: page_001.png -> page_001_ratio1.png
            dir_name = os.path.dirname(image_path)
            base_name = os.path.basename(image_path)
            name_part, ext = os.path.splitext(base_name)
            new_filename = f"{name_part}_ratio{ratio}{ext}"
            new_path = os.path.join(dir_name, new_filename)
            
            if ratio == 1:
                # ratio=1: 直接保存原图（不压缩），但重命名为 _ratio1
                img.save(new_path)
                return new_path
            
            # ratio>1: 执行压缩处理
            original_w, original_h = img.size
            new_w = max(1, int(original_w / ratio))
            new_h = max(1, int(original_h / ratio))
            
            # 执行压缩 (Downsampling) -> 再 Upsampling 回原尺寸
            small_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized_img = small_img.resize((original_w, original_h), Image.Resampling.BILINEAR)
            resized_img.save(new_path)
            return new_path
            
    except Exception as e:
        print(f"   ⚠️ Compression failed for ratio {ratio}: {e}")
        return None

def run_full_process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在

    # 0. 图片输入模式
    if USE_EXISTING_IMAGES:
        if RUN_MODULE_3 and (not os.path.exists(IMAGES_DIR)):
            print("❌ USE_EXISTING_IMAGES=1 but images directory not found:")
            print(f"   - {os.path.abspath(IMAGES_DIR)}")
            print("   你当前启用了 RUN_MODULE_3（需要图片做 OCR）。")
            print("   请设置 $env:EXISTING_IMAGES_DIR=\"...\" 指向已有图片目录，或先跑一次模块1/2生成图片。")
            return
        print("🧩 Using existing images (skip Module 1 & 2)")
        print(f"   - images_dir: {os.path.abspath(IMAGES_DIR)}")
        print(f"   - dataset: {os.path.abspath(os.path.join(OUTPUT_DIR, DATASET_FILENAME))}")
    else:
        # 0. 清理环境（只删除当前模型的 images 目录）
        if os.path.exists(IMAGES_DIR):
            try:
                shutil.rmtree(IMAGES_DIR)
                print(f"🧹 Cleaned up: {IMAGES_DIR}")
            except Exception as e:
                print(f"⚠️ Failed to clean {IMAGES_DIR}: {e}")

    # 🧹 清理当前模型上次运行残留的输出文件（避免 done-set 跳过 + 评估结果混淆）
    gemini_ocr_jsonl = os.path.join(OUTPUT_DIR, "gemini_ocr.jsonl")
    gemini_model_tag = _safe_filename_component(GEMINI_MODEL_NAME)
    gemini_dataset_json = os.path.join(OUTPUT_DIR, DEFAULT_DATASET_FILENAME)
    legacy_dataset_json = os.path.join(OUTPUT_DIR, "dataset.json")
    gemini_judge_detail = os.path.join(OUTPUT_DIR, f"judge_results_detail_{gemini_model_tag}.jsonl")
    gemini_judge_summary = os.path.join(OUTPUT_DIR, f"judge_summary_{gemini_model_tag}.json")
    removed = []
    # 使用已有图片集时：不要删除 dataset（否则 judge 没有 GT）。
    # 走全流程时：会重建 dataset，因此可安全清理掉旧的 dataset 及 legacy dataset.json。
    to_remove = [gemini_judge_detail, gemini_judge_summary]
    # 只有在要重新跑 OCR 时才删除 ocr.jsonl；只跑 Module 4 时保留现有 OCR 结果。
    if RUN_MODULE_3:
        to_remove.insert(0, gemini_ocr_jsonl)
    if not USE_EXISTING_IMAGES:
        to_remove.extend([gemini_dataset_json, legacy_dataset_json])

    for p in to_remove:
        if _remove_file_if_exists(p):
            removed.append(os.path.basename(p))
    if removed:
        print("🧹 Removed model artifacts: " + ", ".join(removed))

    if not USE_EXISTING_IMAGES:
        os.makedirs(IMAGES_DIR, exist_ok=True)

    dataset_filename = DATASET_FILENAME
    dataset = None
    if not USE_EXISTING_IMAGES:
        # -------------------------------------------------
        # 🟢 模块一: 数据挖掘 (Data Miner)
        # -------------------------------------------------
        print("\n" + "="*40)
        print("🚀 Running Module 1: Data Miner")
        print("="*40)

        try:
            from data_miner import fetch_fresh_code
        except Exception as e:
            print("❌ Failed to import data_miner.fetch_fresh_code.")
            print(f"   - error: {e}")
            print("   你当前如果只是想用已有图片集评测，请设置 USE_EXISTING_IMAGES=1。")
            print("   如果要走全流程（拉取 GitHub 代码），请先安装依赖：pip install PyGithub")
            return
        
        dataset = fetch_fresh_code()
        
        if not dataset:
            print("❌ No data found.")
            return

        dataset_path = os.path.join(OUTPUT_DIR, dataset_filename)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        # -------------------------------------------------
        # 🔵 模块二: 视觉干扰器 (Visual Corruptor)
        # -------------------------------------------------
        print("\n" + "="*40)
        print("🚀 Running Module 2: Visual Corruptor")
        print(f"🎯 Target Ratios: {TARGET_RATIOS}")
        print("="*40)

        total_images_generated = 0
        
        for idx, item in enumerate(dataset):
            code_id = item['id']
            source_code = item['code']
            
            print(f"[{idx+1}/{len(dataset)}] Processing: {code_id} ...")
            
            item_output_dir = os.path.join(IMAGES_DIR, code_id)
            os.makedirs(item_output_dir, exist_ok=True)
            
            temp_file_path = os.path.join(item_output_dir, "temp_source.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(source_code)
                
            try:
                # 1. 生成基准高清图 (1x)
                generated_paths = text_to_image_compact.generate_images_for_file(
                    filename=temp_file_path,
                    source_code=source_code,
                    base_output_dir=item_output_dir,
                    width=1024,
                    height=1024,  # 正方形
                    font_size=18,  # 稍微小一点适应正方形
                    line_height=1.2,
                    dpi=100,
                    preserve_newlines=True,
                    enable_syntax_highlight=True,
                    unique_id="base"
                )
                
                if not generated_paths:
                    print("   ❌ No base image generated.")
                    continue

                # 2. 执行视觉压缩循环 (1x, 2x, 4x, 8x)
                for original_path in generated_paths:
                    for ratio in TARGET_RATIOS:
                        new_path = apply_visual_corruption(original_path, ratio)
                        if new_path:
                            total_images_generated += 1
                    
                    # 🗑️ 删除原始图片（已生成所有 ratio 版本）
                    try:
                        if os.path.exists(original_path):
                            os.remove(original_path)
                    except Exception as e:
                        print(f"      ⚠️ Failed to remove original: {e}")

            except Exception as e:
                print(f"   ❌ Error processing {code_id}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        print("\n" + "="*40)
        print("🎉 Pipeline Stage 1 & 2 Completed!")
        print(f"📊 Summary:")
        print(f"   - Data Mined: {len(dataset)}")
        print(f"   - Total Variants Generated: {total_images_generated}")
        print(f"   - Output Location: {os.path.abspath(OUTPUT_DIR)}")
        print("="*40)

    # -------------------------------------------------
    # 🟣 模块三: 推理引擎 (Inference Engine)
    # -------------------------------------------------
    if RUN_MODULE_3:
        run_module_3_gemini(IMAGES_DIR, OUTPUT_DIR)

    # -------------------------------------------------
    # 🟠 模块四: 自动评估器 (Auto-Judge)
    # -------------------------------------------------
    if RUN_MODULE_4:
        run_module_4_judge(OUTPUT_DIR, "gemini_ocr.jsonl", GEMINI_MODEL_NAME, dataset_filename)

if __name__ == "__main__":
    run_full_process()
