import argparse
import os
import sys

# Ensure tasks module can be imported
sys.path.insert(0, os.path.dirname(__file__))

from tasks.OCR import run_ocr_task

def main():
    parser = argparse.ArgumentParser(description="Run OCR Task")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--output_dir", type=str, default="./ocr_results", help="Output directory")
    parser.add_argument("--limit", type=int, default=100, help="Number of items to test")
    parser.add_argument("--data_path", type=str, default="/workspace/data/liuzijun/research/long-code-understanding/final/qa_dataset_test_no_comments.json", help="Path to dataset")
    
    args = parser.parse_args()
    
    print(f"Running OCR Task with model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Limit: {args.limit}")
    print(f"Data path: {args.data_path}")
    
    run_ocr_task(
        model_name=args.model,
        output_dir=args.output_dir,
        data_path=args.data_path,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
