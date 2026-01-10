import json
import os
import sys
import shutil
from typing import List, Dict, Any
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from text_to_image_compact import text_to_image_compact
from llm_utils import create_client, call_llm_with_images

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

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

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate (CER)"""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = levenshtein_distance(reference, hypothesis)
    return dist / len(reference)

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score"""
    chencherry = SmoothingFunction()
    # Simple tokenization by characters or split
    # For code, character level might be too granular for BLEU, usually word/token level
    # But repoqa.py uses list(original_code) which implies character level tokenization for BLEU?
    # Let's check repoqa.py reference: "reference_tokens = [list(original_code)]"
    # Yes, it treats each character as a token.
    reference_tokens = [list(reference)]
    candidate_tokens = list(hypothesis)
    return sentence_bleu(
        reference_tokens,
        candidate_tokens,
        smoothing_function=chencherry.method1,
    )

def load_data(file_path: str, limit: int = 100) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter valid data
    valid_data = [item for item in data if item.get('context')]
    return valid_data[:limit]

def run_ocr_task(
    model_name: str,
    output_dir: str = "./ocr_results",
    data_path: str = "./qa_dataset_test_no_comments.json",
    limit: int = 100
):
    print(f"Loading data from {data_path}...")
    dataset = load_data(data_path, limit)
    print(f"Loaded {len(dataset)} items.")

    # Load prompt
    prompt_path = os.path.join(os.path.dirname(__file__), "../prompts/ocr_prompt.json")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    
    system_prompt = prompt_data.get('system', '')
    user_prompt_template = prompt_data.get('user', '')

    os.makedirs(output_dir, exist_ok=True)
    
    client = create_client()
    
    results = []
    
    # Metrics
    total_bleu = 0.0
    total_cer = 0.0
    count = 0

    for i, item in enumerate(tqdm(dataset, desc="Running OCR Task")):
        code_context = item['context']
        repo_id = item.get('repo_id', 'unknown')
        file_path = item.get('file_path', 'unknown')
        
        # Generate Image
        # We need a temporary path for the image
        temp_img_dir = os.path.join(output_dir, "temp_images")
        os.makedirs(temp_img_dir, exist_ok=True)
        img_filename = f"ocr_{i}.png"
        img_path = os.path.join(temp_img_dir, img_filename)
        
        # Using text_to_image_compact to generate image
        # Assuming we want a reasonable size, e.g., default or dynamic
        # Since user mentioned "resize image test", maybe we should use specific settings?
        # The prompt says "resize image test", implying maybe testing different sizes or just using the resizing logic.
        # I will use text_to_image_compact which handles resizing logic if text is long.
        # But wait, text_to_image_compact returns success bool.
        
        try:
            # We use a standard configuration for now
            success = text_to_image_compact(
                text=code_context,
                output_path=img_path,
                width=1024, # Standard width
                max_height=8000, # Allow tall images
                line_height=1.2,
                font_size=16
            )
            
            if not success:
                print(f"Failed to generate image for item {i}")
                continue
                
            # Call LLM
            response_text, token_info = call_llm_with_images(
                client,
                model_name,
                [img_path],
                system_prompt,
                user_prompt_template
            )
            
            # Clean response (remove markdown if present)
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```"):
                # Remove first line (```python or similar) and last line (```)
                lines = cleaned_response.splitlines()
                if len(lines) >= 2:
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned_response = "\n".join(lines)
            
            # Compute Metrics
            bleu = calculate_bleu(code_context, cleaned_response)
            cer = calculate_cer(code_context, cleaned_response)
            
            total_bleu += bleu
            total_cer += cer
            count += 1
            
            result_item = {
                "id": i,
                "repo_id": repo_id,
                "file_path": file_path,
                "bleu": bleu,
                "cer": cer,
                "original_length": len(code_context),
                "generated_length": len(cleaned_response),
                "generated_text": cleaned_response
            }
            results.append(result_item)
            
            # Save individual result
            with open(os.path.join(output_dir, f"result_{i}.json"), 'w', encoding='utf-8') as f:
                json.dump(result_item, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing item {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Clean up temp images
            if 'image_paths' in locals():
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        os.remove(img_path)
            # Also clean up the single path if it was created (backward compatibility in thought process)
            if 'img_path' in locals() and os.path.exists(img_path) and img_path not in locals().get('image_paths', []):
                os.remove(img_path)

    # Calculate averages
    avg_bleu = total_bleu / count if count > 0 else 0
    avg_cer = total_cer / count if count > 0 else 0
    
    summary = {
        "model": model_name,
        "total_items": len(dataset),
        "processed_items": count,
        "average_bleu": avg_bleu,
        "average_cer": avg_cer
    }
    
    print("\n=== Evaluation Results ===")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Save all results
    with open(os.path.join(output_dir, "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    return summary
