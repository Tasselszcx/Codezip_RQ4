import json
import time
from datetime import datetime
from github import Github
from tqdm import tqdm

# ================= 配置区 (根据您的要求修改) =================
# ⚠️ 请务必在此处填入您的 GitHub Token
GITHUB_TOKEN = "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" 

TARGET_DATE = "2025-08-01"  # 截止日期
TARGET_LANG = "python"      # 目标语言
MIN_STARS = 10              # 最小 Star 数
MAX_STARS = 200             # 最大 Star 数
MIN_LINES = 50              # 最小行数
MAX_LINES = 120             # 最大行数
LIMIT = 2                   # 抓取数量 (测试用)
OUTPUT_FILE = "dataset_fresh_2025.json"
# =========================================================

def fetch_fresh_code():
    # 简单的 Token 检查
    if "ghp_" not in GITHUB_TOKEN and "github_" not in GITHUB_TOKEN:
        print("⚠️ 警告: GitHub Token 可能未配置，请检查 data_miner.py")

    print(f"🚀 [Module 1] Data Miner Started...")
    print(f"📅 Filter: Created > {TARGET_DATE} | Lines: {MIN_LINES}-{MAX_LINES} | Limit: {LIMIT}")
    
    g = Github(GITHUB_TOKEN)
    query = f"language:{TARGET_LANG} created:>{TARGET_DATE} stars:{MIN_STARS}..{MAX_STARS}"
    
    try:
        repos = g.search_repositories(query, sort="stars", order="desc")
    except Exception as e:
        print(f"❌ GitHub API Error: {e}")
        return []

    dataset = []
    pbar = tqdm(total=LIMIT, desc="Mining Code")

    for repo in repos:
        if len(dataset) >= LIMIT:
            break
        try:
            contents = repo.get_contents("")
            files_to_check = []
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    if file_content.path in ['src', 'lib', 'core', 'app']:
                        try:
                            contents.extend(repo.get_contents(file_content.path))
                        except: pass
                elif file_content.path.endswith(".py"):
                    if "test" not in file_content.path and "__init__" not in file_content.path:
                        files_to_check.append(file_content)
            
            for file_node in files_to_check:
                if 1000 < file_node.size < 20000:
                    code_text = file_node.decoded_content.decode('utf-8')
                    lines = code_text.splitlines()
                    if MIN_LINES <= len(lines) <= MAX_LINES:
                        dataset.append({
                            "id": f"{repo.name}_{file_node.path}".replace("/", "_"), # 扁平化ID方便做文件名
                            "repo": repo.full_name,
                            "url": file_node.html_url,
                            "code": code_text,
                            "line_count": len(lines)
                        })
                        pbar.update(1)
                        break 
        except:
            continue
        time.sleep(0.1)

    pbar.close()
    
    # 保存文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"✅ [Module 1] Completed. Saved {len(dataset)} items to {OUTPUT_FILE}")
    return dataset

if __name__ == "__main__":
    fetch_fresh_code()