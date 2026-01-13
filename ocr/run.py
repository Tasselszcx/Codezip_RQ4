import os
import json
import shutil
import time
import base64
from collections import Counter
from PIL import Image  # 用于手动实现视觉压缩
from data_miner import fetch_fresh_code 
import text_to_image_compact 

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ================= 配置区 =================
OUTPUT_DIR = "./experiment_output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
TARGET_RATIOS = [1, 2, 4, 8]  # 我们的压缩目标

# ================= 模块三配置（Inference Engine）=================
# 只跑 GLM-4.6V（通过 aihubmix OpenAI-compat 接口）
RUN_MODULE_3 = True
AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"
GLM_MODEL_NAME = "glm-4.6v"
OCR_SYSTEM_PROMPT = "You are an OCR engine for code images."
OCR_USER_PROMPT = (
    "Transcribe the code in this image exactly.\n"
    "- Output plain text only (no Markdown, no code fences).\n"
    "- Preserve all whitespace, indentation, and newlines.\n"
    "- Do not add, remove, or rename anything.\n"
)
OCR_MAX_TOKENS = 4096
OCR_TEMPERATURE = 0.0
OCR_SLEEP_SECONDS = 0.2
OCR_MAX_RETRIES = 5
# =========================================

# ================= 模块四配置（Auto-Judge）=================
RUN_MODULE_4 = True  # 是否运行评估模块
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


def run_module_3_glm46v(images_dir: str, output_dir: str):
    print("\n" + "=" * 40)
    print("🚀 Running Module 3: Inference Engine (GLM-4.6V)")
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
    out_jsonl = os.path.join(output_dir, "glm46v_ocr.jsonl")
    done = _load_done_set(out_jsonl)

    client = OpenAI(api_key=api_key, base_url=AIHUBMIX_BASE_URL)

    total = 0
    skipped = 0
    errors = 0

    image_paths = list(_iter_image_files(images_dir))
    print(f"🖼️  Total images to OCR: {len(image_paths)}")

    for i, image_path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] OCR: {os.path.basename(image_path)}")
        if image_path in done:
            skipped += 1
            continue

        # 路径结构: images/{code_id}/{variant_folder}/page_xxx.png
        # 需要取上两层目录才是 code_id
        parent_dir = os.path.dirname(image_path)            # .../1024x1500_hl_nl
        code_id_dir = os.path.dirname(parent_dir)           # .../{code_id}
        code_id = os.path.basename(code_id_dir)
        ratio = _parse_ratio_from_filename(image_path)

        data_url = _encode_image_to_data_url(image_path)

        last_err = None
        text = ""

        for attempt in range(1, OCR_MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=GLM_MODEL_NAME,
                    temperature=OCR_TEMPERATURE,
                    max_tokens=OCR_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": OCR_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": OCR_USER_PROMPT},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                )
                text = (resp.choices[0].message.content or "").rstrip()
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
            "image_path": image_path,
            "model": GLM_MODEL_NAME,
        }

        if last_err is None:
            rec["text"] = text
            total += 1
        else:
            rec["error"] = last_err
            errors += 1

        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        time.sleep(OCR_SLEEP_SECONDS)

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


def run_module_4_judge(output_dir: str):
    """
    模块4: Auto-Judge 评估器
    读取 OCR 结果，与原始代码对比，输出评估指标
    """
    print("\n" + "=" * 40)
    print("🚀 Running Module 4: Auto-Judge")
    print("=" * 40)

    # 读取 dataset.json (原始代码)
    dataset_path = os.path.join(output_dir, "dataset.json")
    if not os.path.exists(dataset_path):
        print("❌ dataset.json not found, skipping Module 4")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 建立 code_id -> code 的映射
    code_map = {item["id"]: item["code"] for item in dataset}

    # 读取 OCR 结果
    ocr_path = os.path.join(output_dir, "glm46v_ocr.jsonl")
    if not os.path.exists(ocr_path):
        print("❌ glm46v_ocr.jsonl not found, skipping Module 4")
        return

    ocr_results = []
    with open(ocr_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ocr_results.append(json.loads(line))

    # 🌟 核心修复：按 (code_id, ratio) 分组，合并多页结果
    from collections import defaultdict
    grouped = defaultdict(list)  # (code_id, ratio) -> [{"text": ..., "image_path": ...}, ...]
    
    for rec in ocr_results:
        # 提取 code_id
        img_path = rec.get("image_path", "")
        parts = img_path.replace("\\", "/").split("/")
        code_id = parts[-3] if len(parts) >= 3 else rec.get("code_id", "")
        
        ratio = rec.get("ratio", 1)
        ocr_text = rec.get("text", "")
        
        # 跳过错误结果
        if not ocr_text or "error" in rec:
            continue
        
        # 清理特殊标记（关键！）
        ocr_text = ocr_text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '').strip()
        
        # 按 (code_id, ratio) 分组
        grouped[(code_id, ratio)].append({
            "text": ocr_text,
            "image_path": img_path,
        })

    # 评估结果
    detail_path = os.path.join(output_dir, "judge_results_detail.jsonl")
    summary_path = os.path.join(output_dir, "judge_summary.json")

    # 按 ratio 分组统计
    stats_by_ratio = {r: {
        "cer_sum": 0, "wer_sum": 0, "bleu_sum": 0, 
        "exact_match_sum": 0,
        "ast_pass": 0, "count": 0, 
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

        # 🌟 合并多页 OCR 结果（按文件名排序确保顺序正确）
        pages.sort(key=lambda x: x["image_path"])
        merged_ocr = '\n'.join([p["text"] for p in pages])
        
        evaluated += 1
        num_pages = len(pages)
        print(f"[{idx}/{total}] Evaluating: {code_id} @ ratio {ratio}x ({num_pages} pages)")

        # 🌟 规范化处理（用于 hard metrics）
        ref_normalized = normalize_code(reference)
        ocr_normalized = normalize_code(merged_ocr)

        # 1. Hard metrics（使用规范化后的文本）
        cer = _compute_cer(ref_normalized, ocr_normalized)
        wer = _compute_wer(ref_normalized, ocr_normalized)
        bleu = _compute_token_bleu(ref_normalized, ocr_normalized)
        exact_match = _compute_exact_match_rate(ref_normalized, ocr_normalized)
        ast_ok = _check_ast_parsable(merged_ocr)  # AST 用原始文本

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
            "exact_match_rate": round(exact_match, 4),
            "ast_parsable": ast_ok,
            "taxonomy_labels": taxonomy_labels,
            "detected_errors": detected_error_types,
        }

        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(detail_rec, ensure_ascii=False) + "\n")

        # 更新统计
        if ratio in stats_by_ratio:
            stats_by_ratio[ratio]["cer_sum"] += cer
            stats_by_ratio[ratio]["wer_sum"] += wer
            stats_by_ratio[ratio]["bleu_sum"] += bleu
            stats_by_ratio[ratio]["exact_match_sum"] += exact_match
            stats_by_ratio[ratio]["ast_pass"] += (1 if ast_ok else 0)
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
        summary[f"ratio_{ratio}x"] = {
            "count": s["count"],
            "avg_cer": round(s["cer_sum"] / s["count"], 4),
            "avg_wer": round(s["wer_sum"] / s["count"], 4),
            "avg_token_bleu": round(s["bleu_sum"] / s["count"], 4),
            "avg_exact_match_rate": round(s["exact_match_sum"] / s["count"], 4),
            "ast_pass_rate": round(s["ast_pass"] / s["count"], 4),
            "error_counts": dict(s["errors"]),  # 原始计数
            "error_rates": error_rates,  # 检出率
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
        print(f"      ├─ BLEU={data['avg_token_bleu']:.4f}, Exact Match={data['avg_exact_match_rate']:.2%}")
        print(f"      ├─ AST Pass={data['ast_pass_rate']:.0%}")
        if data['error_counts']:
            err_str = ", ".join([f"{k}:{v}" for k, v in data['error_counts'].items()])
            print(f"      └─ Errors: {err_str}")
        else:
            print(f"      └─ Errors: (none detected)")


def apply_visual_corruption(image_path, ratio):
    """
    手动实现视觉干扰器：读取原图，先按比例缩小再放大回原尺寸（保持尺寸一致）
    """
    if ratio == 1:
        return image_path
    
    try:
        with Image.open(image_path) as img:
            # 计算新尺寸
            original_w, original_h = img.size
            new_w = max(1, int(original_w / ratio))
            new_h = max(1, int(original_h / ratio))
            
            # 执行压缩 (Downsampling) -> 再 Upsampling 回原尺寸
            # 这样可以保持尺寸一致，同时通过信息丢失制造“变糊”效果
            small_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized_img = small_img.resize((original_w, original_h), Image.Resampling.BILINEAR)
            
            # 构造新文件名: page_001.png -> page_001_ratio2.png
            dir_name = os.path.dirname(image_path)
            base_name = os.path.basename(image_path)
            name_part, ext = os.path.splitext(base_name)
            new_filename = f"{name_part}_ratio{ratio}{ext}"
            new_path = os.path.join(dir_name, new_filename)
            
            resized_img.save(new_path)
            return new_path
    except Exception as e:
        print(f"   ⚠️ Compression failed for ratio {ratio}: {e}")
        return None

def run_full_process():
    # 0. 清理环境
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
        except:
            pass # 有时候文件占用删不掉，忽略
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # -------------------------------------------------
    # 🟢 模块一: 数据挖掘 (Data Miner)
    # -------------------------------------------------
    print("\n" + "="*40)
    print("🚀 Running Module 1: Data Miner")
    print("="*40)
    
    dataset = fetch_fresh_code()
    
    if not dataset:
        print("❌ No data found.")
        return

    dataset_path = os.path.join(OUTPUT_DIR, "dataset.json")
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
            # 使用您提供的函数定义，注意参数名变化
            generated_paths = text_to_image_compact.generate_images_for_file(
                filename=temp_file_path,
                source_code=source_code,
                base_output_dir=item_output_dir,
                width=1024,
                height=1024,  # 正方形
                font_size=18,  # 稍微小一点适应正方形
                line_height=1.2,
                dpi=100,
                # 🌟 关键修改：改为 True，保持代码原样换行 🌟
                preserve_newlines=True,  
                enable_syntax_highlight=True,
                unique_id="base"
            )
            
            if not generated_paths:
                print("   ❌ No base image generated.")
                continue

            # 2. 执行视觉压缩循环 (1x, 2x, 4x, 8x)
            # 既然 generate_images_for_file 只能生成一种，我们在外面手动压缩
            for original_path in generated_paths:
                for ratio in TARGET_RATIOS:
                    if ratio == 1:
                        # 1x 就是原图，不用动，或者重命名一下方便统一
                        total_images_generated += 1
                        continue
                    
                    # 生成变糊的图
                    new_path = apply_visual_corruption(original_path, ratio)
                    if new_path:
                        # print(f"      -> Generated {ratio}x compressed: {os.path.basename(new_path)}")
                        total_images_generated += 1

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
    # 🟣 模块三: 推理引擎 (Inference Engine) - GLM-4.6V
    # -------------------------------------------------
    if RUN_MODULE_3:
        run_module_3_glm46v(IMAGES_DIR, OUTPUT_DIR)

    # -------------------------------------------------
    # 🟠 模块四: 自动评估器 (Auto-Judge)
    # -------------------------------------------------
    if RUN_MODULE_4:
        run_module_4_judge(OUTPUT_DIR)

if __name__ == "__main__":
    run_full_process()