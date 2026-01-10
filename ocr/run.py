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

    # 初始化 OpenAI 客户端 (用于 taxonomy)
    api_key = os.environ.get("AIHUBMIX_API_KEY") or _try_load_api_key_from_env_files()
    client = None
    if api_key and OpenAI:
        client = OpenAI(api_key=api_key, base_url=AIHUBMIX_BASE_URL)

    # 评估结果
    detail_path = os.path.join(output_dir, "judge_results_detail.jsonl")
    summary_path = os.path.join(output_dir, "judge_summary.json")

    # 按 ratio 分组统计
    stats_by_ratio = {r: {"cer_sum": 0, "bleu_sum": 0, "ast_pass": 0, "count": 0, "errors": Counter()} 
                      for r in TARGET_RATIOS}

    # 清空 detail 文件
    open(detail_path, "w").close()

    total = len(ocr_results)
    for idx, rec in enumerate(ocr_results):
        ratio = rec.get("ratio", 1)
        ocr_text = rec.get("text", "")

        # 从 image_path 重新提取正确的 code_id
        # 路径结构: images/{code_id}/{variant_folder}/page_xxx.png
        img_path = rec.get("image_path", "")
        parts = img_path.replace("\\", "/").split("/")
        code_id = parts[-3] if len(parts) >= 3 else rec.get("code_id", "")

        if not ocr_text or "error" in rec:
            continue

        reference = code_map.get(code_id, "")
        if not reference:
            continue

        print(f"[{idx + 1}/{total}] Evaluating: {code_id} @ ratio {ratio}")

        # 1. Hard metrics
        cer = _compute_cer(reference, ocr_text)
        ast_ok = _check_ast_parsable(ocr_text)
        bleu = _compute_token_bleu(reference, ocr_text)

        # 2. Soft taxonomy (LLM)
        error_types = []
        if client:
            error_types = _call_llm_for_taxonomy(client, reference, ocr_text)

        # 记录详情
        detail_rec = {
            "code_id": code_id,
            "ratio": ratio,
            "cer": round(cer, 4),
            "ast_parsable": ast_ok,
            "token_bleu": round(bleu, 4),
            "error_types": error_types
        }

        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(detail_rec, ensure_ascii=False) + "\n")

        # 更新统计
        if ratio in stats_by_ratio:
            stats_by_ratio[ratio]["cer_sum"] += cer
            stats_by_ratio[ratio]["bleu_sum"] += bleu
            stats_by_ratio[ratio]["ast_pass"] += int(ast_ok)
            stats_by_ratio[ratio]["count"] += 1
            for et in error_types:
                stats_by_ratio[ratio]["errors"][et] += 1

    # 生成汇总
    summary = {}
    for ratio, s in stats_by_ratio.items():
        if s["count"] == 0:
            continue
        summary[f"ratio_{ratio}x"] = {
            "count": s["count"],
            "avg_cer": round(s["cer_sum"] / s["count"], 4),
            "avg_token_bleu": round(s["bleu_sum"] / s["count"], 4),
            "ast_pass_rate": round(s["ast_pass"] / s["count"], 4),
            "error_distribution": dict(s["errors"])
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Module 4 finished.")
    print(f"📄 Detail: {os.path.abspath(detail_path)}")
    print(f"📊 Summary: {os.path.abspath(summary_path)}")

    # 打印简要汇总
    print("\n📈 Quick Summary:")
    for ratio_key, data in summary.items():
        print(f"   {ratio_key}: CER={data['avg_cer']:.2%}, BLEU={data['avg_token_bleu']:.4f}, "
              f"AST Pass={data['ast_pass_rate']:.0%}")


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
                height=1500,
                font_size=16, 
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