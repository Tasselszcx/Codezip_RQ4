import json
import os
import time
import random
import ast
from datetime import datetime
from github import Github
from tqdm import tqdm

# ================= 配置区 (根据您的要求修改) =================

# ✅ 建议使用环境变量而不是把 Token 写进代码：
# Windows PowerShell: $env:GITHUB_TOKEN="ghp_xxx" ; python data_miner.py

def _load_env_file() -> None:
    """最小版 dotenv：从若干位置读取 .env，并写入 os.environ（不覆盖已存在的环境变量）。"""
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    key = k.strip()
                    val = v.strip().strip('"').strip("'")
                    if key and (key not in os.environ):
                        os.environ[key] = val
        except Exception:
            # .env 读取失败不应阻断主流程
            pass


_load_env_file()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

TARGET_DATE = "2025-08-01"  # 截止日期
TARGET_LANG = "python"      # 目标语言
MIN_STARS = 50              # 最小 Star 数
MAX_STARS = 400             # 最大 Star 数
MIN_LINES = 50              # 最小行数
MAX_LINES = 120             # 最大行数
LIMIT = 100
                  # 抓取数量
OUTPUT_FILE = "dataset_fresh_2025.json"

# 扫描范围控制：递归扫描哪些目录
# 说明：原逻辑只会进入 src/lib/core/app 且不会继续深入子目录，容易漏掉大量 .py。
# 这里改为：只要顶层目录名在白名单里，就递归扫描其所有子目录。
SCAN_ROOT_DIRS = {
    "src", "lib", "core", "app",
    "python", "py",
    "backend", "server", "service", "services",
    "pkg", "package", "packages",
    "api", "apis",
    "project", "projects",
    "module", "modules",
    "script", "scripts",
}

# 保护阈值：避免单个 repo 目录过深/过大导致 API 调用爆炸
MAX_DIR_LISTINGS_PER_REPO = 80

# 文件大小过滤（字节）。不是必须，但能显著减少“极小碎片文件”和“超大文件”带来的噪声与耗时。
# 如需关闭下限可设为 0；如需放宽上限可调大。
MIN_FILE_BYTES = 500
MAX_FILE_BYTES = 3000

# 随机化设置
ENABLE_RANDOM = True        # 是否启用随机化
RANDOM_POOL_SIZE = LIMIT * 15       # 从前 N 个结果中随机抽取

# =============== 新增：代码结构过滤（class）===============
# 需求：优先采集“包含 class 的 Python 文件”（而不是只有零散 def/json）
REQUIRE_BIG_CLASS = True          # 是否强制文件中存在“较大的 class”
MIN_CLASS_METHODS = 3            # class 内最少方法数（def/async def）
MIN_CLASS_LINES = 25             # class 最少行数（基于 ast 的行号估算）
# =========================================================
# =========================================================


def _class_span_lines(node: ast.AST) -> int:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)

    if start is None:
        return 0
    if end is not None:
        return max(1, end - start + 1)

    max_end = start
    for child in ast.walk(node):
        ln = getattr(child, "lineno", None)
        if isinstance(ln, int):
            max_end = max(max_end, ln)
    return max(1, max_end - start + 1)


def _has_big_class(code_text: str) -> bool:
    # 快速剪枝
    if "class " not in code_text:
        return False

    try:
        tree = ast.parse(code_text)
    except Exception:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            method_count = 0
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_count += 1

            span_lines = _class_span_lines(node)
            if method_count >= MIN_CLASS_METHODS and span_lines >= MIN_CLASS_LINES:
                return True

    return False

def fetch_fresh_code():
    # 简单的 Token 检查
    if "ghp_" not in GITHUB_TOKEN and "github_" not in GITHUB_TOKEN:
        print("⚠️ 警告: GitHub Token 可能未配置。建议设置环境变量 GITHUB_TOKEN")

    print(f"🚀 [Module 1] Data Miner Started...")
    print(f"📅 Filter: Created > {TARGET_DATE} | Lines: {MIN_LINES}-{MAX_LINES} | Limit: {LIMIT}")
    
    g = Github(GITHUB_TOKEN)
    
    # 随机化查询参数
    if ENABLE_RANDOM:
        # 随机选择排序方式和顺序
        sort_options = ["stars", "forks", "updated"]
        order_options = ["desc", "asc"]
        sort_by = random.choice(sort_options)
        order_by = random.choice(order_options)
        
        # 随机偏移星星范围 (在 MIN_STARS~MAX_STARS 基础上随机偏移)
        star_offset = random.randint(0, 50)
        actual_min_stars = MIN_STARS + star_offset
        actual_max_stars = MAX_STARS + star_offset
        
        print(f"🎲 Random mode: sort={sort_by}, order={order_by}, stars={actual_min_stars}..{actual_max_stars}")
    else:
        sort_by = "stars"
        order_by = "desc"
        actual_min_stars = MIN_STARS
        actual_max_stars = MAX_STARS
    
    query = f"language:{TARGET_LANG} created:>{TARGET_DATE} stars:{actual_min_stars}..{actual_max_stars}"
    
    try:
        repos = g.search_repositories(query, sort=sort_by, order=order_by)
    except Exception as e:
        print(f"❌ GitHub API Error: {e}")
        return []

    # 收集候选仓库（先收集一个池子，再随机抽取）
    candidate_repos = []
    repo_count = 0
    
    print(f"📦 Building candidate pool (max {RANDOM_POOL_SIZE} repos)...")
    for repo in repos:
        if repo_count >= RANDOM_POOL_SIZE:
            break
        candidate_repos.append(repo)
        repo_count += 1
        time.sleep(0.05)  # 避免 API 限制
    
    # 随机打乱候选仓库顺序
    if ENABLE_RANDOM:
        random.shuffle(candidate_repos)
        print(f"🔀 Shuffled {len(candidate_repos)} candidate repos")

    dataset = []
    pbar = tqdm(total=LIMIT, desc="Mining Code")

    for repo in candidate_repos:
        if len(dataset) >= LIMIT:
            break
        try:
            # BFS 递归扫描：顶层目录在白名单内则继续深入（支持 src/**、backend/** 等）。
            contents = repo.get_contents("")
            dir_calls = 1
            files_to_check = []
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    top = (file_content.path.split("/", 1)[0] or "").lower()
                    if top in SCAN_ROOT_DIRS and dir_calls < MAX_DIR_LISTINGS_PER_REPO:
                        try:
                            contents.extend(repo.get_contents(file_content.path))
                            dir_calls += 1
                        except:
                            pass
                elif file_content.path.endswith(".py"):
                    path_lower = file_content.path.lower()
                    if "test" not in path_lower and "__init__" not in path_lower:
                        files_to_check.append(file_content)
            
            for file_node in files_to_check:
                if MIN_FILE_BYTES < file_node.size < MAX_FILE_BYTES:
                    code_text = file_node.decoded_content.decode('utf-8', errors='replace')
                    lines = code_text.splitlines()
                    if MIN_LINES <= len(lines) <= MAX_LINES:
                        if REQUIRE_BIG_CLASS and (not _has_big_class(code_text)):
                            continue
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
    
    print(f"✅ [Module 1] Completed. Saved {len(dataset)} items to {OUTPUT_FILE}")
    return dataset

if __name__ == "__main__":
    fetch_fresh_code()