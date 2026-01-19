import argparse
import difflib
import json
import os
import re
from typing import Any, Optional


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _basename_like(s: str) -> str:
    return s.replace("\\", "/").split("/")[-1]


def _load_reference_code(dataset_path: str, code_id: str) -> tuple[Optional[str], Optional[str]]:
    """返回 (matched_key, code_text)。

    兼容 dataset item 的 key: id/path/file_path。
    支持用 code_id 的 basename 去匹配。
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    code_map: dict[str, str] = {}
    for item in dataset:
        code = item.get("code")
        if not code:
            continue

        candidates: list[str] = []
        for k in ("id", "path", "file_path"):
            if item.get(k):
                v = str(item[k])
                candidates.append(v)
                candidates.append(_basename_like(v))

        for c in candidates:
            if c and c not in code_map:
                code_map[c] = code

    if code_id in code_map:
        return code_id, code_map[code_id]

    b = _basename_like(code_id)
    if b in code_map:
        return b, code_map[b]

    return None, None


def _extract_page_num(image_path: str) -> int:
    # .../page_001_ratio4.png -> 1
    m = re.search(r"page_(\d+)_ratio\d+\.(png|jpg|jpeg|webp)$", image_path.replace("\\", "/"), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _clean_ocr_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    # 去掉代码围栏行（``` 或 ```lang）
    s = re.sub(r"^\s*```[\w+-]*\s*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*```\s*$", "", s, flags=re.MULTILINE)
    return s.strip("\n")


def _normalize_no_blank_lines(s: str) -> str:
    """与 codediff_no_blank 对齐：tab->4空格、rstrip、删除所有空行（保留行首缩进）。"""
    lines = s.splitlines()
    lines = [ln.replace("\t", "    ").rstrip() for ln in lines]
    lines = [ln for ln in lines if ln.strip() != ""]
    return "\n".join(lines)


def _merge_ocr_pages(records: list[dict[str, Any]]) -> str:
    records = sorted(records, key=lambda r: _extract_page_num(str(r.get("image_path", ""))))
    parts: list[str] = []
    for r in records:
        parts.append(_clean_ocr_text(str(r.get("text", ""))))
    return "\n\n".join([p for p in parts if p.strip() != ""])


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a single judge case (reference vs OCR) with the same preprocessing used by codediff_no_blank.")
    ap.add_argument("--output-dir", default="./experiment_output")
    ap.add_argument("--model", required=True, help="glm46v | gemini | gpt51 | gpt52（决定默认文件名）")
    ap.add_argument("--code-id", required=True)
    ap.add_argument("--ratio", type=int, required=True)
    ap.add_argument("--dataset", default=None, help="可选：显式指定 dataset json 路径")
    ap.add_argument("--ocr-jsonl", default=None, help="可选：显式指定 ocr jsonl 路径")
    ap.add_argument("--save", action="store_true", help="是否把 ref/ocr/diff 落盘到 experiment_output/inspect/")
    args = ap.parse_args()

    model_key = args.model.strip().lower()
    default_ocr = {
        "glm46v": "glm46v_ocr.jsonl",
        "gemini": "gemini_ocr.jsonl",
        "gpt51": "gpt51_ocr.jsonl",
        "gpt52": "gpt52_ocr.jsonl",
    }.get(model_key)
    default_dataset = {
        "glm46v": "dataset_glm46.json",
        "gemini": "dataset_gemini.json",
        "gpt51": "dataset_gpt51.json",
        "gpt52": "dataset_gpt52.json",
    }.get(model_key)

    if not default_ocr or not default_dataset:
        raise SystemExit(f"Unknown --model: {args.model}. Use glm46v|gemini|gpt51|gpt52")

    ocr_path = args.ocr_jsonl or os.path.join(args.output_dir, default_ocr)
    dataset_path = args.dataset or os.path.join(args.output_dir, default_dataset)

    if not os.path.exists(ocr_path):
        raise SystemExit(f"OCR jsonl not found: {ocr_path}")
    if not os.path.exists(dataset_path):
        raise SystemExit(f"Dataset not found: {dataset_path}")

    matched_key, ref = _load_reference_code(dataset_path, args.code_id)
    if ref is None:
        raise SystemExit(f"No ground truth found for code_id={args.code_id} in {dataset_path}")

    rows = _read_jsonl(ocr_path)
    rows = [
        r
        for r in rows
        if str(r.get("code_id")) == args.code_id
        and int(r.get("ratio", -1)) == args.ratio
        and "error" not in r
    ]
    if not rows:
        raise SystemExit(f"No OCR records found for code_id={args.code_id}, ratio={args.ratio} in {ocr_path}")

    ocr_merged = _merge_ocr_pages(rows)

    ref_nb = _normalize_no_blank_lines(ref)
    ocr_nb = _normalize_no_blank_lines(ocr_merged)

    print("=" * 60)
    print(f"MODEL={args.model}  code_id={args.code_id}  ratio={args.ratio}")
    print(f"dataset_match_key={matched_key}")
    print(f"ocr_pages={len(rows)}")
    print("=" * 60 + "\n")

    print("----- REF (raw) -----")
    print(ref)
    print("\n----- OCR (merged raw) -----")
    print(ocr_merged)
    print("\n----- REF (no blank) -----")
    print(ref_nb)
    print("\n----- OCR (no blank) -----")
    print(ocr_nb)

    diff = difflib.unified_diff(
        ref_nb.splitlines(True),
        ocr_nb.splitlines(True),
        fromfile="ref_no_blank.py",
        tofile="ocr_no_blank.py",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    print("\n----- DIFF (unified, no blank) -----")
    print(diff_text if diff_text.strip() else "(no diff)")

    if args.save:
        out_dir = os.path.join(args.output_dir, "inspect", args.model, args.code_id, f"ratio{args.ratio}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "ref_raw.py"), "w", encoding="utf-8") as f:
            f.write(ref)
        with open(os.path.join(out_dir, "ocr_merged_raw.py"), "w", encoding="utf-8") as f:
            f.write(ocr_merged)
        with open(os.path.join(out_dir, "ref_no_blank.py"), "w", encoding="utf-8") as f:
            f.write(ref_nb)
        with open(os.path.join(out_dir, "ocr_no_blank.py"), "w", encoding="utf-8") as f:
            f.write(ocr_nb)
        with open(os.path.join(out_dir, "diff_no_blank.patch"), "w", encoding="utf-8") as f:
            f.write(diff_text)
        print(f"\n✅ Saved to: {out_dir}")


if __name__ == "__main__":
    main()
