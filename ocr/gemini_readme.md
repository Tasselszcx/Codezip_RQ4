## 1. API Key 配置

脚本默认从环境变量或 `.env` 文件中读取 `AIHUBMIX_API_KEY`，可以让AI帮忙改一下api_key的调用，在`.env`文件中同步更改一下。

## 2. 打开一个powershell窗口，配置环境变量如下（直接复制粘贴进去）

### Win版本：

```powershell
# 1. 核心开关：使用已有图片，跳过数据挖掘和图片生成
$env:USE_EXISTING_IMAGES="1"

# 2. 指定已有图片的路径 (请根据实际情况调整相对路径)
$env:EXISTING_IMAGES_DIR=".\experiment_output\images_glm46v"

# 3. 指定对应的 Ground Truth 数据集文件 (位于 experiment_output 下)
$env:DATASET_FILENAME="dataset_glm46.json"

# 4. 模块开关：运行 OCR (Module 3) 和 评分 (Module 4)
$env:RUN_MODULE_3="1"
$env:RUN_MODULE_4="1"

# 5. 性能配置：并发数为 4 (建议根据 API 限流情况调整)
$env:OCR_CONCURRENCY="4"
# 如果遇到限流报错，可设置最小间隔(秒)，例如 0.5
$env:OCR_PARALLEL_MIN_INTERVAL_SECONDS="0"

# 6. Gemini特有配置
# 开启安全设置透传 (防止图片被误判为有害而被拦截)
$env:GEMINI_ENABLE_SAFETY_SETTINGS="1"
# 开启 Prompt 增强 (声明用于个人离线测试，降低拒答率)
$env:OCR_PROMPT_PERSONAL_OFFLINE="1"
```

### Mac/Linux版本：

```bash
export USE_EXISTING_IMAGES="1"
export EXISTING_IMAGES_DIR="./experiment_output/images_glm46v"
export DATASET_FILENAME="dataset_glm46.json"
export RUN_MODULE_3="1"
export RUN_MODULE_4="1"
export OCR_CONCURRENCY="4"
export OCR_PARALLEL_MIN_INTERVAL_SECONDS="0"
export GEMINI_ENABLE_SAFETY_SETTINGS="1"
export OCR_PROMPT_PERSONAL_OFFLINE="1"
```

## 3. 切换到对应路径，运行脚本

```bash
python run_gemini.py
```

## 4. 结果说明

脚本运行完成后，结果将保存在 `experiment_output/` 目录下：

- **OCR 结果**: `gemini_ocr.jsonl`
  
  包含每一张图片的识别文本、耗时及可能出现的错误信息。

- **评测详情**: `judge_results_detail_gemini-3-pro-preview.jsonl`
  
  包含每一条代码的详细打分（CER, WER, CodeBLEU, 错误类型分析）。

- **评测汇总**: `judge_summary_gemini-3-pro-preview.json`
  
  不同压缩倍率（1x, 2x... 8x）下的平均准确率统计。
