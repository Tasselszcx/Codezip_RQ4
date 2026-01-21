"""
简单的多页 OCR + 合并 + 对比脚本
用法: 
  python ocr_and_compare.py <图片目录> <code_id>  # 实时 OCR 模式
  python ocr_and_compare.py --from-cache <code_id> <ratio>  # 从缓存读取模式
"""
import os
import sys
import json
import glob
import base64
import time
#from openai import OpenAI
import difflib
import re


def normalize_code(text: str) -> str:
    """代码规范化（仅用于评估指标，不改变 OCR 原始输出的保存）：

    - Tab → 4 spaces：统一缩进风格
    - 去除行尾空格：消除 trailing spaces 噪声
    - 压缩连续空行：多个空行 → 单个空行
    - 去除首尾空行：统一文件头尾格式
    """
    lines = text.splitlines()

    # Tab → 4 spaces + 去除行尾空格
    lines = [line.replace('\t', '    ').rstrip() for line in lines]

    # 压缩连续空行
    normalized: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = (line.strip() == '')
        if is_blank:
            if not prev_blank:
                normalized.append('')
            prev_blank = True
        else:
            normalized.append(line)
            prev_blank = False

    # 去除首尾空行
    while normalized and normalized[0] == '':
        normalized.pop(0)
    while normalized and normalized[-1] == '':
        normalized.pop()

    return '\n'.join(normalized)


def smart_join_pages(ocr_pages):
    """智能拼接多页 OCR 结果，保留缩进上下文"""
    if not ocr_pages:
        return ''
    
    if len(ocr_pages) == 1:
        return ocr_pages[0]
    
    result = [ocr_pages[0]]
    
    for i in range(1, len(ocr_pages)):
        prev_page = ocr_pages[i-1]
        curr_page = ocr_pages[i]
        
        if not prev_page.strip() or not curr_page.strip():
            result.append(curr_page)
            continue
        
        prev_lines = prev_page.splitlines()
        curr_lines = curr_page.splitlines()
        
        # 获取上一页最后一个非空行的缩进
        prev_last_line = None
        for line in reversed(prev_lines):
            if line.strip():
                prev_last_line = line
                break
        
        # 获取当前页第一个非空行的缩进
        curr_first_line = None
        for line in curr_lines:
            if line.strip():
                curr_first_line = line
                break
        
        # 检测缩进差异并修正
        if prev_last_line and curr_first_line:
            prev_indent = len(prev_last_line) - len(prev_last_line.lstrip())
            curr_indent = len(curr_first_line) - len(curr_first_line.lstrip())
            
            # 如果当前页第一行缩进异常（比上一页少很多），可能是跨页缩进丢失
            # 这里我们保守处理：只在明显是类/函数延续时才调整
            indent_diff = prev_indent - curr_indent
            
            # 如果上一页最后一行有缩进，而当前页第一行缩进为0，且不是新的顶层定义
            # 很可能是缩进丢失
            if (prev_indent >= 4 and curr_indent == 0 and 
                not curr_first_line.strip().startswith(('def ', 'class ', 'import ', 'from '))):
                # 可能需要继承上一页的缩进
                # 但这个启发式可能不准确，所以暂时只记录，不强制修正
                pass
        
        result.append(curr_page)
    
    return '\n'.join(result)


def load_api_key():
    """加载 API Key"""
    api_key = os.getenv("AIHUBMIX_API_KEY")
    if not api_key:
        # 尝试从多个位置查找 .env 文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_paths = [
            ".env",  # 当前目录
            os.path.join(script_dir, ".env"),  # 脚本所在目录
            os.path.join(script_dir, "..", ".env"),  # 上级目录
        ]
        
        for env_file in env_paths:
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith("AIHUBMIX_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if api_key:
                    break
    return api_key


def ocr_image(image_path, api_key, max_retries=3):
    """对单张图片进行 OCR，带重试"""
    print(f"  [OCR] {os.path.basename(image_path)}", end='', flush=True)
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    
    client = OpenAI(api_key=api_key, base_url="https://aihubmix.com/v1")
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="glm-4.6v",
                temperature=0.0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": "You are an OCR engine for code images. Your output must preserve the exact formatting of the code."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the code EXACTLY as shown in the image. Preserve all blank lines, indentation, and formatting EXACTLY. Do not remove empty lines. Do not modify whitespace. Output only the raw code text without any markdown formatting."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            
            text = (resp.choices[0].message.content or "").strip()
            text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
            
            if text:
                print(f" -> {len(text)} chars")
                return text
            else:
                print(f" -> empty (retry {attempt+1})", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2)
        except Exception as e:
            print(f" -> error: {e} (retry {attempt+1})", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f" -> FAILED")
    return ""


def calculate_metrics(reference, hypothesis):
    """计算 CER, WER, BLEU"""
    def lev_dist(s1, s2):
        if len(s1) < len(s2):
            return lev_dist(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    # CER
    cer = lev_dist(reference, hypothesis) / len(reference) * 100
    
    # WER
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    wer = lev_dist(ref_words, hyp_words) / len(ref_words) * 100
    
    # BLEU
    from collections import Counter
    import math
    
    precisions = []
    for n in range(1, 5):
        ref_ngrams = Counter([tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_words[i:i+n]) for i in range(len(hyp_words)-n+1)])
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        precisions.append(matches / total if total > 0 else 0)
    
    if any(p == 0 for p in precisions):
        bleu = 0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        bp = 1.0 if len(hyp_words) >= len(ref_words) else math.exp(1 - len(ref_words) / len(hyp_words))
        bleu = bp * geo_mean * 100
    
    return cer, wer, bleu


def load_from_cache(code_id, ratio):
    """从 glm46v_ocr.jsonl 加载已有的 OCR 结果"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, "..", "experiment_output", "glm46v_ocr.jsonl")
    
    if not os.path.exists(cache_path):
        print(f"[ERROR] Cache file not found: {cache_path}")
        return None
    
    ocr_pages = []
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item['code_id'] == code_id and item['ratio'] == int(ratio):
                text = item['text'].replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '').strip()
                ocr_pages.append(text)
    
    if not ocr_pages:
        return None
    
    # 使用智能拼接而不是简单的 join
    return smart_join_pages(ocr_pages)


def print_diff_report(original_code, ocr_text):
    """打印详细的差异对比报告"""
    print("\n" + "=" * 80)
    print("DETAILED DIFF REPORT")
    print("=" * 80)
    
    ref_lines = original_code.splitlines()
    ocr_lines = ocr_text.splitlines()
    
    print(f"\n📊 Lines: Original={len(ref_lines)}, OCR={len(ocr_lines)}")
    
    # 使用 difflib 生成差异
    diff = list(difflib.unified_diff(
        ref_lines, 
        ocr_lines, 
        fromfile='Original', 
        tofile='OCR',
        lineterm=''
    ))
    
    if len(diff) <= 2:  # 只有头部信息，没有实际差异
        print("\n✅ No differences found (line-by-line match)")
        return
    
    print(f"\n❌ Found {len([d for d in diff if d.startswith('-') or d.startswith('+')])} diff lines")
    print("\n--- Unified Diff ---")
    for line in diff[:100]:  # 限制输出前100行
        if line.startswith('-'):
            print(f"\033[91m{line}\033[0m")  # 红色
        elif line.startswith('+'):
            print(f"\033[92m{line}\033[0m")  # 绿色
        elif line.startswith('@'):
            print(f"\033[94m{line}\033[0m")  # 蓝色
        else:
            print(line)
    
    if len(diff) > 100:
        print(f"\n... (truncated, {len(diff) - 100} more lines)")
    
    # 逐行对比并标注差异
    print("\n--- Line-by-Line Comparison (first 30 mismatches) ---")
    mismatch_count = 0
    for i in range(max(len(ref_lines), len(ocr_lines))):
        if mismatch_count >= 30:
            print(f"\n... (truncated, more mismatches exist)")
            break
        
        ref = ref_lines[i] if i < len(ref_lines) else ""
        ocr = ocr_lines[i] if i < len(ocr_lines) else ""
        
        if ref.strip() != ocr.strip():
            mismatch_count += 1
            print(f"\n🔴 Line {i+1}:")
            print(f"  REF: {repr(ref)}")
            print(f"  OCR: {repr(ocr)}")
            
            # 字符级差异
            if ref and ocr:
                s = difflib.SequenceMatcher(None, ref, ocr)
                char_diffs = []
                for tag, i1, i2, j1, j2 in s.get_opcodes():
                    if tag == 'replace':
                        char_diffs.append(f"pos {i1}-{i2}: '{ref[i1:i2]}' -> '{ocr[j1:j2]}'")
                    elif tag == 'delete':
                        char_diffs.append(f"pos {i1}-{i2}: deleted '{ref[i1:i2]}'")
                    elif tag == 'insert':
                        char_diffs.append(f"pos {i1}: inserted '{ocr[j1:j2]}'")
                if char_diffs:
                    print(f"  Diff: {'; '.join(char_diffs[:5])}")


def main_from_cache(code_id, ratio):
    """从缓存读取并对比的模式"""
    print(f"🔄 Loading from cache: {code_id} @ ratio {ratio}x")
    
    # Load OCR from cache
    ocr_text = load_from_cache(code_id, ratio)
    if ocr_text is None:
        print(f"[ERROR] No OCR results found for {code_id} @ ratio {ratio}")
        return
    
    print(f"✅ Loaded OCR: {len(ocr_text)} chars, {len(ocr_text.splitlines())} lines")
    
    # Load original code
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "experiment_output", "dataset.json")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] dataset.json not found at: {dataset_path}")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = {item['id']: item for item in json.load(f)}
    
    if code_id not in dataset:
        print(f"[ERROR] code_id not found: {code_id}")
        return
    
    original_code = dataset[code_id]['code']
    print(f"✅ Loaded original: {len(original_code)} chars, {len(original_code.splitlines())} lines")
    
    # 规范化代码后再计算指标（用于验证：tab/行尾空格/空行/首尾空行）
    print(f"\n⏳ Normalizing code (tabs/trailing spaces/blank lines/head-tail)...")
    original_normalized = normalize_code(original_code)
    ocr_normalized = normalize_code(ocr_text)
    
    print(f"After normalization:")
    print(f"  Original: {len(original_normalized)} chars, {len(original_normalized.splitlines())} lines")
    print(f"  OCR:      {len(ocr_normalized)} chars, {len(ocr_normalized.splitlines())} lines")
    
    # Calculate metrics (原始 & 规范化后)
    print(f"\n⏳ Calculating metrics...")
    cer_raw, wer_raw, bleu_raw = calculate_metrics(original_code, ocr_text)
    cer_norm, wer_norm, bleu_norm = calculate_metrics(original_normalized, ocr_normalized)
    
    # Output metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Code ID:                     {code_id}")
    print(f"Ratio:                       {ratio}x")
    print(f"\n📊 Raw (before normalization):")
    print(f"  CER (Character Error Rate):  {cer_raw:.2f}%")
    print(f"  WER (Word Error Rate):       {wer_raw:.2f}%")
    print(f"  BLEU Score:                  {bleu_raw:.2f}")
    print(f"\n✨ Normalized (after code normalization):")
    print(f"  CER (Character Error Rate):  {cer_norm:.2f}%")
    print(f"  WER (Word Error Rate):       {wer_norm:.2f}%")
    print(f"  BLEU Score:                  {bleu_norm:.2f}")
    print(f"\n📈 Improvement: CER {cer_raw - cer_norm:+.2f}%, WER {wer_raw - wer_norm:+.2f}%, BLEU {bleu_norm - bleu_raw:+.2f}")
    
    # Line matching (使用规范化后的版本)
    ref_lines_norm = original_normalized.splitlines()
    ocr_lines_norm = ocr_normalized.splitlines()
    matches_norm = sum(1 for i in range(min(len(ref_lines_norm), len(ocr_lines_norm))) 
                  if ref_lines_norm[i].strip() == ocr_lines_norm[i].strip())
    emr_norm = matches_norm / max(len(ref_lines_norm), len(ocr_lines_norm)) * 100
    print(f"\nExact Match Rate (normalized): {emr_norm:.2f}% ({matches_norm}/{max(len(ref_lines_norm), len(ocr_lines_norm))} lines)")
    print("=" * 80)
    
    # Print detailed diff (使用规范化后的版本)
    print("\n[Note: Showing diff after code normalization]")
    print_diff_report(original_normalized, ocr_normalized)
    
    # Save outputs (保存原始和规范化后的版本)
    output_ref_raw = f"compare_ref_{code_id}_ratio{ratio}_raw.txt"
    output_ocr_raw = f"compare_ocr_{code_id}_ratio{ratio}_raw.txt"
    output_ref_norm = f"compare_ref_{code_id}_ratio{ratio}_normalized.txt"
    output_ocr_norm = f"compare_ocr_{code_id}_ratio{ratio}_normalized.txt"
    
    with open(output_ref_raw, 'w', encoding='utf-8') as f:
        f.write(original_code)
    with open(output_ocr_raw, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    with open(output_ref_norm, 'w', encoding='utf-8') as f:
        f.write(original_normalized)
    with open(output_ocr_norm, 'w', encoding='utf-8') as f:
        f.write(ocr_normalized)
    
    print(f"\n💾 Saved:")
    print(f"   Reference (raw):        {output_ref_raw}")
    print(f"   OCR (raw):              {output_ocr_raw}")
    print(f"   Reference (normalized): {output_ref_norm}")
    print(f"   OCR (normalized):       {output_ocr_norm}")


def main():
    # 检查是否使用缓存模式
    if len(sys.argv) >= 2 and sys.argv[1] == "--from-cache":
        if len(sys.argv) >= 4:
            code_id = sys.argv[2]
            ratio = sys.argv[3]
            main_from_cache(code_id, ratio)
            return
        else:
            print("Usage: python ocr_and_compare.py --from-cache <code_id> <ratio>")
            print("\nExample from dataset:")
            print("  python ocr_and_compare.py --from-cache astrbot_plugin_lmarena_file_bed.py 1")
            print("  python ocr_and_compare.py --from-cache astrbot_plugin_lmarena_file_bed.py 2")
            sys.exit(1)
    
    # 示例路径
    example_dir = r"D:\llm_projects\CodeZip\experiment_output\images\crypto-trader-bot-with-AI-algo_indicator_calculator.py\1024x1024_hl_nl"
    example_code_id = "crypto-trader-bot-with-AI-algo_indicator_calculator.py"
    
    # 获取参数（支持命令行或交互式输入）
    if len(sys.argv) >= 3:
        image_dir = sys.argv[1]
        code_id = sys.argv[2]
    else:
        print("Usage: python ocr_and_compare.py <image_dir> <code_id>")
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("=" * 60)
        
        # 显示示例
        print(f"\nExample directory: {example_dir}")
        print(f"Example code_id: {example_code_id}")
        
        # 询问是否使用示例
        use_example = input("\nUse example path? (y/n, default=y): ").strip().lower()
        
        if use_example == '' or use_example == 'y':
            image_dir = example_dir
            code_id = example_code_id
            print(f"Using: {image_dir}")
            print(f"Code ID: {code_id}")
        else:
            # 交互式输入
            image_dir = input("\nEnter image directory: ").strip()
            code_id = input("Enter code_id: ").strip()
            
            if not image_dir or not code_id:
                print("[ERROR] Both parameters are required")
                sys.exit(1)
    
    # Load API Key
    api_key = load_api_key()
    if not api_key:
        print("[ERROR] AIHUBMIX_API_KEY not found")
        sys.exit(1)
    
    # Get ratio
    print("\nEnter compression ratio (1, 2, 4, 8): ", end='')
    ratio = input().strip()
    
    # Find images (ratio=1 means original images without ratio suffix)
    if ratio == '1':
        pattern = os.path.join(image_dir, "page_*.png")
        # 排除带 ratio 后缀的文件
        all_images = glob.glob(pattern)
        images = sorted([img for img in all_images if not any(f'_ratio{r}.png' in img for r in [2, 4, 8])])
    else:
        pattern = os.path.join(image_dir, f"page_*_ratio{ratio}.png")
        images = sorted(glob.glob(pattern))
    
    if not images:
        print(f"[ERROR] No images found: {pattern}")
        sys.exit(1)
    
    print(f"\nFound {len(images)} images")
    
    # OCR each image
    print(f"\nStarting OCR...")
    ocr_results = []
    for img in images:
        text = ocr_image(img, api_key, max_retries=3)
        ocr_results.append(text)
    
    # Merge (使用智能拼接)
    merged_ocr = smart_join_pages(ocr_results)
    print(f"\nMerged (smart join): {len(merged_ocr)} chars, {len(merged_ocr.splitlines())} lines")
    
    # Load original code - 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "experiment_output", "dataset.json")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] dataset.json not found at: {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = {item['id']: item for item in json.load(f)}
    
    if code_id not in dataset:
        print(f"[ERROR] code_id not found: {code_id}")
        sys.exit(1)
    
    original_code = dataset[code_id]['code']
    print(f"Original: {len(original_code)} chars, {len(original_code.splitlines())} lines")
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    cer, wer, bleu = calculate_metrics(original_code, merged_ocr)
    
    # Output
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"CER (Character Error Rate):  {cer:.2f}%")
    print(f"WER (Word Error Rate):       {wer:.2f}%")
    print(f"BLEU Score:                  {bleu:.2f}")
    
    # Line matching
    ref_lines = original_code.splitlines()
    ocr_lines = merged_ocr.splitlines()
    matches = sum(1 for i in range(min(len(ref_lines), len(ocr_lines))) 
                  if ref_lines[i].strip() == ocr_lines[i].strip())
    emr = matches / max(len(ref_lines), len(ocr_lines)) * 100
    print(f"Exact Match Rate:            {emr:.2f}% ({matches}/{max(len(ref_lines), len(ocr_lines))} lines)")
    
    print("=" * 60)
    
    # Save
    output_file = f"ocr_merged_ratio{ratio}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_ocr)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
