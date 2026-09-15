# DualEmotionBench Evaluation

Official evaluation code for **DualEmotionBench**, a bilingual benchmark for
intra-utterance emotion-transition text-to-speech synthesis.

This repository evaluates synthesized TTS audio with:

- **Emotion Matching (EM)**
- **Naturalness (NA)**
- **Transition Smoothness (TS)** for emotion-transition evaluation
- **Word Error Rate (WER)** using SenseVoiceSmall
- **DNSMOS P.835**: SIG, BAK, and OVR
- **DNSMOS P.808 MOS**

The LLM-based evaluator uses a Qwen-compatible audio model, with
`qwen3-omni-flash` as the default judge model.

## Repository Structure

```text
dualemotionbench-eval/
├── evaluate.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── dnsmos/
    └── README.md
```

DNSMOS checkpoints, evaluation audio, task JSON files, API keys, and generated
results are not included in this repository.

## Installation

```bash
git clone https://github.com/ikunikun-ass/dualemotionbench-eval.git
cd dualemotionbench-eval

python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

`pydub` also requires FFmpeg to be installed on your system.

## API Key Configuration

Create a local `.env` file in the repository root:

```bash
cp .env.example .env
```

For Windows, manually create a file named `.env`.

Then add your DashScope API key:

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

Do not commit `.env` or any API key to GitHub.

## DNSMOS Checkpoints

The evaluator requires two DNSMOS ONNX checkpoints:

```text
sig_bak_ovr.onnx
model_v8.onnx
```

Download them separately from the official Microsoft DNS Challenge resources.
They are intentionally not redistributed in this repository.

## Task JSON Format

The task JSON is a JSON object whose top-level keys are user-defined group or
folder names. These names do not determine the evaluation mode.

```json
{
  "my_transition_folder": [
    [
      {
        "audio_id": "1_example",
        "question": "/path/to/ground_truth.wav",
        "text": "Target transcript.",
        "emotions": "happiness -> sadness",
        "dialogue": {
          "Model-A": "/path/to/model_a.wav",
          "Model-B": "/path/to/model_b.wav"
        }
      }
    ]
  ]
}
```

Required fields for each sample:

- `audio_id`: unique sample identifier
- `text`: target transcript
- `emotions`: target emotion or ordered emotion path
- `dialogue`: dictionary mapping model names to generated audio paths

Optional fields:

- `question`: ground-truth reference audio path
- `language`: `zh` or `en`
- `evaluation_mode`: `single` or `transition`

## Evaluation Modes

The evaluator supports two modes:

- `single`: evaluates Emotion Matching and Naturalness
- `transition`: evaluates Transition Smoothness, Emotion Matching, and Naturalness

Users can choose the default mode at runtime. Therefore, arbitrary folder names
such as `test_a`, `S1_zh`, or `my_model_outputs` are supported.

## Run Evaluation

Example: evaluate all task groups as emotion-transition synthesis.

```bash
python evaluate.py \
  --json_file /path/to/tasks.json \
  --output_dir results \
  --temp_dir temp_combined \
  --dns_primary_model /path/to/sig_bak_ovr.onnx \
  --dns_p808_model /path/to/model_v8.onnx \
  --asr_device cuda:0 \
  --default_eval_mode transition \
  --max_workers 3
```

Use CPU when CUDA is unavailable:

```bash
--asr_device cpu
```

## Use Different Modes for Different Folders

Create a `mode_config.json` file:

```json
{
  "single_emotion_zh": "single",
  "single_emotion_en": "single",
  "transition_zh": "transition",
  "transition_en": "transition"
}
```

Then run:

```bash
python evaluate.py \
  --json_file /path/to/tasks.json \
  --output_dir results \
  --dns_primary_model /path/to/sig_bak_ovr.onnx \
  --dns_p808_model /path/to/model_v8.onnx \
  --default_eval_mode transition \
  --mode_config mode_config.json
```

The priority is:

1. `evaluation_mode` in an individual task record
2. Group name in `mode_config.json`
3. `--default_eval_mode`

## Optional Span-Level Metadata

For emotion-transition evaluation, optional Chinese and English metadata files
can provide `split_results` containing temporally ordered emotion spans.

```bash
python evaluate.py \
  --json_file /path/to/tasks.json \
  --output_dir results \
  --dns_primary_model /path/to/sig_bak_ovr.onnx \
  --dns_p808_model /path/to/model_v8.onnx \
  --zh_split_json /path/to/chinese_data.json \
  --en_split_json /path/to/english_data.json \
  --default_eval_mode transition
```

The evaluator derives directional switch descriptions from these annotations.

## Output Format

One resumable JSONL file is created for each top-level task group:

```text
results/
├── results_single_emotion_zh.jsonl
├── results_transition_zh.jsonl
└── results_my_custom_folder.jsonl
```

Each JSONL record includes:

```json
{
  "group_name": "transition_zh",
  "evaluation_mode": "transition",
  "audio_id": "1_example",
  "model": "Model-A",
  "target_emotion_path": "happiness -> sadness",
  "emotion_switch_positions": "After '...', before '...': happiness -> sadness",
  "ai_scores": {},
  "traditional_metrics": {
    "wer": 0.0,
    "transcription": "",
    "dnsmos_sig": 0.0,
    "dnsmos_bak": 0.0,
    "dnsmos_ovr": 0.0,
    "p808_mos": 0.0
  }
}
```

Completed `(audio_id, model)` pairs are skipped automatically when the same
result file already exists.

