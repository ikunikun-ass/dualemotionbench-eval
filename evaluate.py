#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DualEmotionBench evaluation pipeline.

Metrics:
  - LLM-based Emotion Matching (EM)
  - LLM-based Naturalness (NA)
  - LLM-based Transition Smoothness (TS), for transition mode only
  - SenseVoiceSmall WER
  - DNSMOS P.835: SIG / BAK / OVR
  - DNSMOS P.808 MOS

Evaluation modes:
  - single:     EM + NA
  - transition: TS + EM + NA

The top-level keys in the task JSON are treated only as user-defined
folder/group names. They do not determine the evaluation mode.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from dotenv import load_dotenv
from funasr import AutoModel
from jiwer import wer
from openai import OpenAI
from opencc import OpenCC
from pydub import AudioSegment


# ============================================================================
# Global configuration
# ============================================================================

SAMPLE_RATE = 16000
DNSMOS_INPUT_LENGTH = 9.01
VALID_EVALUATION_MODES = {"single", "transition"}

MODEL_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Prompt templates
# ============================================================================

SINGLE_SENTENCE_PROMPT = """
You are a strict TTS single-emotion synthesis evaluator.

{reference_note}

Score ONLY the synthesized TTS audio.

## Evaluation Context
- Synthesized TTS model: {model_name}
- Text content: "{text}"
- Target emotion: {emotion_path}

## Core Rules
1. The synthesized audio must fully express the target emotion and appropriate
   intensity. If it does not clearly express the target emotion, Expression
   Matching must not exceed 2.
2. Follow the 1--5 criteria strictly. Do not give high scores without clear
   evidence from the audio.
3. Your reasoning must address both dimensions using audible evidence.

## Scoring Dimensions
### 1. Expression Matching
Does the synthesized audio match the target emotion type and intensity?
- 1: Emotion is absent, contradictory, or severely confused.
- 2: Emotion is only weakly expressed or intensity is strongly mismatched.
- 3: Emotion type is present, but intensity has an obvious deviation.
- 4: Emotion and intensity highly match the target with minor deviations.
- 5: Emotion is vivid, precise, and fully matches the target.

### 2. Naturalness
Does the synthesized audio sound natural and human-like?
- 1: Extremely unnatural, mechanical, or linguistically disordered.
- 2: Rigid prosody, monotony, obvious stuttering, or splicing artifacts.
- 3: Generally acceptable, but obvious synthesis artifacts remain.
- 4: Natural rhythm and prosody with only minor artifacts.
- 5: Indistinguishable from natural human speech.

Return JSON only:
{{
  "scores": {{
    "expression_matching": 1,
    "naturalness": 1
  }},
  "reasoning": "Brief evidence for both scores."
}}
"""


TRANSITION_PROMPT = """
You are a strict evaluator for intra-utterance emotion-transition TTS.

{reference_note}

Score ONLY the synthesized TTS audio.

## Evaluation Context
- Synthesized TTS model: {model_name}
- Text content: "{text}"
- Ordered target emotion transition: {emotion_path}
- Annotated emotion-switch location(s): {emotion_switch_positions}

A transition is directional and sequential, for example
"happiness -> sadness". It is not a mixed or co-occurring emotion state.

## Core Rules
1. The synthesized audio must express all target emotions in the specified
   order and perform the transition near the annotated location.
2. If one target emotion is missing, the emotional order is incorrect, or no
   transition occurs, Transition Smoothness and Expression Matching must not
   exceed 2.
3. Follow the 1--5 criteria strictly. Do not give high scores without clear
   evidence from the audio.
4. Your reasoning must discuss all three dimensions.

## Scoring Dimensions
### 1. Transition Smoothness
How naturally does emotion change at the annotated switch location?
- 1: No transition, or severe discontinuity, pause, pitch, or energy break.
- 2: Transition is attempted but very abrupt or jarring.
- 3: Transition is identifiable but noticeably stiff.
- 4: Natural and gradual transition with minor roughness.
- 5: Seamless human-like emotional progression.

### 2. Expression Matching
Does the audio match all target emotions, their order, intensity, and
approximate switch location?
- 1: Target emotions are absent, contradictory, or confused.
- 2: Only part of the emotion path is expressed, or intensity is very weak.
- 3: All emotions are present, but intensity or switch location is inaccurate.
- 4: The target path is highly matched with minor deviations.
- 5: All emotion types, order, intensity, and transition location are precise.

### 3. Naturalness
Does the synthesized audio sound natural and human-like?
- 1: Extremely unnatural, mechanical, or linguistically disordered.
- 2: Rigid prosody, monotony, obvious stuttering, or splicing artifacts.
- 3: Generally acceptable, but obvious synthesis artifacts remain.
- 4: Natural rhythm and prosody with only minor artifacts.
- 5: Indistinguishable from natural human speech.

Return JSON only:
{{
  "scores": {{
    "transition_smoothness": 1,
    "expression_matching": 1,
    "naturalness": 1
  }},
  "reasoning": "Brief evidence for all three scores."
}}
"""


# ============================================================================
# Utility functions
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Make NumPy values serializable in JSONL output."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_filename(name: str) -> str:
    """Convert an arbitrary user folder/group name to a valid result filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def normalize_text(text: str, converter: OpenCC) -> str:
    """Normalize Chinese and English text before WER calculation."""
    text = converter.convert(str(text)).lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    return " ".join(tokens)


def parse_json_response(text: str | None) -> dict[str, Any] | None:
    """Parse direct JSON or JSON enclosed in a Markdown code block."""
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    patterns = (
        r"```json\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    )

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def load_mode_config(config_path: str) -> dict[str, str]:
    """
    Load user-defined folder/group-to-mode mapping.

    Example:
    {
      "single_emotion_zh": "single",
      "single_emotion_en": "single",
      "emotion_transition_zh": "transition",
      "emotion_transition_en": "transition"
    }
    """
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mode config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        mode_map = json.load(f)

    if not isinstance(mode_map, dict):
        raise ValueError("Mode config must be a JSON object.")

    for group_name, mode in mode_map.items():
        if mode not in VALID_EVALUATION_MODES:
            raise ValueError(
                f"Invalid mode '{mode}' for group '{group_name}'. "
                "Only 'single' and 'transition' are supported."
            )

    return mode_map


def resolve_evaluation_mode(
    group_name: str,
    item: dict[str, Any],
    mode_map: dict[str, str],
    default_mode: str,
) -> str:
    """
    Resolve evaluation mode in the following priority order:

    1. item["evaluation_mode"]
    2. mode_config[group_name]
    3. --default_eval_mode
    """
    mode = item.get("evaluation_mode")
    if mode is None:
        mode = mode_map.get(group_name, default_mode)

    if mode not in VALID_EVALUATION_MODES:
        raise ValueError(
            f"Invalid evaluation mode '{mode}' for group '{group_name}'. "
            "Use 'single' or 'transition'."
        )

    return mode


def infer_is_chinese(group_name: str, item: dict[str, Any]) -> bool:
    """
    Prefer the optional task field `language`; otherwise preserve the old
    convention that a group name containing 'zh' is Chinese.
    """
    language = str(item.get("language", "")).lower()

    if language in {"zh", "zh-cn", "chinese"}:
        return True
    if language in {"en", "en-us", "english"}:
        return False

    return "zh" in group_name.lower()


def get_numeric_audio_id(audio_id: str) -> int | str:
    """Use numeric prefix before '_' for matching legacy span metadata."""
    prefix = str(audio_id).split("_", maxsplit=1)[0]

    try:
        return int(prefix)
    except ValueError:
        return str(audio_id)


# ============================================================================
# Emotion span metadata
# ============================================================================

def load_emotion_split_json(
    zh_path: str,
    en_path: str,
) -> tuple[dict[Any, dict[str, Any]], dict[Any, dict[str, Any]]]:
    """Load optional Chinese and English span-level annotation JSON files."""

    def load_one(path_str: str, language_name: str) -> dict[Any, dict[str, Any]]:
        if not path_str:
            return {}

        path = Path(path_str)
        if not path.is_file():
            logger.warning("%s span metadata not found: %s", language_name, path)
            return {}

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            result = {item["id"]: item for item in data}
            logger.info(
                "Loaded %d %s span annotations from %s",
                len(result),
                language_name,
                path,
            )
            return result

        except Exception as exc:
            logger.error("Failed to load %s span metadata: %s", language_name, exc)
            return {}

    return load_one(zh_path, "Chinese"), load_one(en_path, "English")


def find_split_data(
    split_map: dict[Any, dict[str, Any]],
    audio_id: str,
) -> dict[str, Any] | None:
    """Try both numeric and string IDs for robust metadata matching."""
    numeric_id = get_numeric_audio_id(audio_id)

    if numeric_id in split_map:
        return split_map[numeric_id]

    if str(numeric_id) in split_map:
        return split_map[str(numeric_id)]

    return None


def extract_emotion_switch_positions(
    split_data: dict[str, Any] | None,
    is_chinese: bool,
) -> str:
    """Convert ordered span metadata to a human-readable switch description."""
    if not split_data or "split_results" not in split_data:
        return "No span-level transition annotation is available."

    split_results = split_data["split_results"]

    if not isinstance(split_results, list) or len(split_results) < 2:
        return "No emotion transition (one annotated segment)."

    positions = []
    previous_emotion = None
    previous_text = ""

    for segment in split_results:
        current_emotion = str(segment.get("emotion", "unknown"))
        text_key = "text" if is_chinese else "text_en"
        current_text = str(
            segment.get(text_key) or segment.get("text") or ""
        ).strip()

        if previous_emotion is not None and current_emotion != previous_emotion:
            positions.append(
                f"After '{previous_text}', before '{current_text}': "
                f"{previous_emotion} -> {current_emotion}"
            )

        previous_emotion = current_emotion
        previous_text = current_text

    if not positions:
        return "No emotion transition (same emotion across annotated spans)."

    return "; ".join(positions)


# ============================================================================
# Objective metrics: SenseVoice WER + DNSMOS
# ============================================================================

def get_audio_melspec(
    audio: np.ndarray,
    n_mels: int = 120,
    frame_size: int = 320,
    hop_length: int = 160,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=frame_size + 1,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def get_polyfit_val(
    sig: float,
    bak: float,
    ovr: float,
) -> tuple[float, float, float]:
    """Official DNSMOS non-personalized polynomial calibration."""
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))


class MetricsCalculator:
    def __init__(
        self,
        asr_model: Any,
        dns_primary_sess: ort.InferenceSession,
        dns_p808_sess: ort.InferenceSession,
    ) -> None:
        self.asr_model = asr_model
        self.dns_primary_sess = dns_primary_sess
        self.dns_p808_sess = dns_p808_sess
        self.converter = OpenCC("t2s")

    def calculate(
        self,
        audio_path: str,
        reference_text: str,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "wer": None,
            "transcription": "",
            "dnsmos_sig": None,
            "dnsmos_bak": None,
            "dnsmos_ovr": None,
            "p808_mos": None,
        }

        if not audio_path or not os.path.isfile(audio_path):
            logger.error("Synthesized audio does not exist: %s", audio_path)
            return results

        # --------------------------------------------------------------------
        # A. SenseVoice transcription and WER
        # --------------------------------------------------------------------
        try:
            with MODEL_LOCK:
                asr_res = self.asr_model.generate(
                    input=audio_path,
                    cache={},
                    language="auto",
                    use_itn=True,
                )

            raw_hypothesis = asr_res[0].get("text", "") if asr_res else ""
            hypothesis = re.sub(r"<\|.*?\|>", "", raw_hypothesis)
            results["transcription"] = hypothesis

            reference_norm = normalize_text(reference_text, self.converter)
            hypothesis_norm = normalize_text(hypothesis, self.converter)

            if reference_norm or hypothesis_norm:
                results["wer"] = float(wer(reference_norm, hypothesis_norm))
            else:
                results["wer"] = 0.0

        except Exception as exc:
            logger.exception("ASR/WER failed for %s: %s", audio_path, exc)

        # --------------------------------------------------------------------
        # B. DNSMOS
        # --------------------------------------------------------------------
        try:
            audio, sample_rate = sf.read(audio_path)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            audio = np.asarray(audio, dtype=np.float32)

            if len(audio) == 0:
                raise ValueError("Audio file contains no samples.")

            if sample_rate != SAMPLE_RATE:
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=SAMPLE_RATE,
                )

            peak = float(np.max(np.abs(audio)))

            if peak > 0:
                audio = audio / (peak + 1e-7)

            required_samples = int(DNSMOS_INPUT_LENGTH * SAMPLE_RATE)

            while len(audio) < required_samples:
                audio = np.concatenate([audio, audio])

            num_hops = max(
                1,
                1 + (len(audio) - required_samples) // SAMPLE_RATE,
            )

            scores = {
                "sig": [],
                "bak": [],
                "ovr": [],
                "p808": [],
            }

            for index in range(num_hops):
                start = index * SAMPLE_RATE
                segment = audio[start:start + required_samples]

                if len(segment) < required_samples:
                    break

                primary_input = segment.astype(np.float32)[np.newaxis, :]
                p808_input = get_audio_melspec(
                    segment[:-160]
                ).astype(np.float32)[np.newaxis, :, :]

                with MODEL_LOCK:
                    mos_raw = self.dns_primary_sess.run(
                        None,
                        {"input_1": primary_input},
                    )[0][0]

                    p808_raw = self.dns_p808_sess.run(
                        None,
                        {"input_1": p808_input},
                    )[0][0][0]

                sig, bak, ovr = get_polyfit_val(
                    mos_raw[0],
                    mos_raw[1],
                    mos_raw[2],
                )

                scores["sig"].append(sig)
                scores["bak"].append(bak)
                scores["ovr"].append(ovr)
                scores["p808"].append(float(p808_raw))

            if scores["sig"]:
                results["dnsmos_sig"] = float(np.mean(scores["sig"]))
                results["dnsmos_bak"] = float(np.mean(scores["bak"]))
                results["dnsmos_ovr"] = float(np.mean(scores["ovr"]))
                results["p808_mos"] = float(np.mean(scores["p808"]))

        except Exception as exc:
            logger.exception("DNSMOS failed for %s: %s", audio_path, exc)

        return results


# ============================================================================
# Qwen audio judge
# ============================================================================

class TTSJudge:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "qwen3-omni-flash",
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name

    @staticmethod
    def encode_audio_to_base64(audio_path: str) -> str:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    def evaluate(
        self,
        audio_path: str,
        prompt: str,
        max_retries: int = 3,
    ) -> dict[str, Any] | None:
        base64_audio = self.encode_audio_to_base64(audio_path)

        content = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:audio/wav;base64,{base64_audio}",
                    "format": "wav",
                },
            },
        ]

        for retry in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    temperature=0.1,
                    stream=False,
                )

                response_text = completion.choices[0].message.content
                parsed = parse_json_response(response_text)

                if parsed is not None:
                    return parsed

                logger.warning(
                    "Judge returned invalid JSON (%d/%d).",
                    retry + 1,
                    max_retries,
                )

            except Exception as exc:
                logger.warning(
                    "Judge API request failed (%d/%d): %s",
                    retry + 1,
                    max_retries,
                    exc,
                )

            time.sleep(2 * (retry + 1))

        return None


# ============================================================================
# Audio preparation and task processing
# ============================================================================

def concatenate_tts_audio(
    gt_path: str | None,
    model_path: str,
    temp_dir: str,
    task_id: str,
) -> tuple[str, bool]:
    """
    Create a 16-kHz mono WAV for LLM judging.

    If GT exists:
        GT reference + 1 second silence + TTS audio

    Otherwise:
        TTS audio only
    """
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Synthesized audio not found: {model_path}")

    model_audio = AudioSegment.from_file(model_path)

    has_reference = bool(gt_path and os.path.isfile(gt_path))

    if has_reference:
        gt_audio = AudioSegment.from_file(gt_path)
        silence = AudioSegment.silent(duration=1000)
        combined_audio = gt_audio + silence + model_audio
    else:
        combined_audio = model_audio

    combined_audio = (
        combined_audio
        .set_frame_rate(SAMPLE_RATE)
        .set_channels(1)
    )

    os.makedirs(temp_dir, exist_ok=True)

    safe_task_id = safe_filename(task_id)

    with tempfile.NamedTemporaryFile(
        prefix=f"{safe_task_id}_",
        suffix=".wav",
        dir=temp_dir,
        delete=False,
    ) as temp_file:
        combined_path = temp_file.name

    combined_audio.export(combined_path, format="wav")

    return combined_path, has_reference


def build_prompt(
    task: dict[str, Any],
    has_reference: bool,
) -> str:
    if has_reference:
        reference_note = (
            "The input audio contains two segments: a human ground-truth "
            "reference first, then one second of silence, then the synthesized "
            "TTS audio."
        )
    else:
        reference_note = (
            "The input audio contains only synthesized TTS audio; no ground-truth "
            "reference audio is provided."
        )

    common_kwargs = {
        "reference_note": reference_note,
        "model_name": task["model_name"],
        "text": task["text"],
        "emotion_path": task["emotions"],
    }

    if task["evaluation_mode"] == "single":
        return SINGLE_SENTENCE_PROMPT.format(**common_kwargs)

    return TRANSITION_PROMPT.format(
        **common_kwargs,
        emotion_switch_positions=task["emotion_switch_positions"],
    )


def load_processed_keys(output_file: str) -> set[str]:
    """Load completed (audio_id, model) pairs for resumable evaluation."""
    processed = set()

    if not os.path.isfile(output_file):
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                processed.add(f"{item['audio_id']}::{item['model']}")
            except (json.JSONDecodeError, KeyError):
                logger.warning(
                    "Ignored malformed line in existing result file: %s",
                    output_file,
                )

    return processed


def append_result(
    output_file: str,
    result: dict[str, Any],
) -> None:
    """Thread-safe JSONL output."""
    with WRITE_LOCK:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    cls=NumpyEncoder,
                )
                + "\n"
            )


def process_single_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    judge: TTSJudge,
    metrics_calculator: MetricsCalculator,
) -> None:
    task_id = (
        f"{task['group_name']}_"
        f"{task['audio_id']}_"
        f"{task['model_name']}"
    )

    combined_audio_path = None
    ai_scores = None

    try:
        combined_audio_path, has_reference = concatenate_tts_audio(
            gt_path=task.get("groundtruth_audio"),
            model_path=task["model_audio"],
            temp_dir=args.temp_dir,
            task_id=task_id,
        )

        prompt = build_prompt(task, has_reference)
        ai_scores = judge.evaluate(combined_audio_path, prompt)

    except Exception as exc:
        logger.exception(
            "LLM evaluation failed for %s: %s",
            task_id,
            exc,
        )

    finally:
        if combined_audio_path and os.path.isfile(combined_audio_path):
            try:
                os.remove(combined_audio_path)
            except OSError as exc:
                logger.warning(
                    "Could not remove temporary audio %s: %s",
                    combined_audio_path,
                    exc,
                )

    traditional_metrics = metrics_calculator.calculate(
        task["model_audio"],
        task["text"],
    )

    result = {
        "group_name": task["group_name"],
        "evaluation_mode": task["evaluation_mode"],
        "audio_id": task["audio_id"],
        "model": task["model_name"],
        "text": task["text"],
        "target_emotion_path": task["emotions"],
        "emotion_switch_positions": task["emotion_switch_positions"],
        "ai_scores": ai_scores,
        "traditional_metrics": traditional_metrics,
    }

    append_result(task["output_file"], result)

    logger.info(
        "Completed | group=%s | mode=%s | audio=%s | model=%s",
        task["group_name"],
        task["evaluation_mode"],
        task["audio_id"],
        task["model_name"],
    )


# ============================================================================
# Task creation
# ============================================================================

def iter_task_items(groups: list[Any]) -> list[dict[str, Any]]:
    """
    Support both common task JSON layouts:

    Layout A:
      "group_name": [[{sample_1}], [{sample_2}]]

    Layout B:
      "group_name": [{sample_1}, {sample_2}]
    """
    items = []

    for group in groups:
        if isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    items.append(item)
        elif isinstance(group, dict):
            items.append(group)

    return items


def build_tasks(
    all_data: dict[str, Any],
    args: argparse.Namespace,
    mode_map: dict[str, str],
    zh_split_map: dict[Any, dict[str, Any]],
    en_split_map: dict[Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build pending tasks from arbitrary user-defined task groups."""
    tasks = []

    for group_name, groups in all_data.items():
        safe_group_name = safe_filename(group_name)
        output_file = os.path.join(
            args.output_dir,
            f"results_{safe_group_name}.jsonl",
        )

        processed_keys = load_processed_keys(output_file)
        group_items = iter_task_items(groups)

        logger.info(
            "Preparing group '%s': %d task records",
            group_name,
            len(group_items),
        )

        for item in group_items:
            required_fields = {"audio_id", "text", "emotions", "dialogue"}
            missing_fields = required_fields - set(item.keys())

            if missing_fields:
                logger.warning(
                    "Skipping invalid item in group '%s'; missing fields: %s",
                    group_name,
                    sorted(missing_fields),
                )
                continue

            evaluation_mode = resolve_evaluation_mode(
                group_name=group_name,
                item=item,
                mode_map=mode_map,
                default_mode=args.default_eval_mode,
            )

            is_chinese = infer_is_chinese(group_name, item)
            split_map = zh_split_map if is_chinese else en_split_map

            audio_id = str(item["audio_id"])
            split_data = find_split_data(split_map, audio_id)

            switch_positions = extract_emotion_switch_positions(
                split_data,
                is_chinese,
            )

            dialogue = item["dialogue"]

            if not isinstance(dialogue, dict):
                logger.warning(
                    "Skipping audio '%s': 'dialogue' must be a model-to-path dictionary.",
                    audio_id,
                )
                continue

            for model_name, model_audio_path in dialogue.items():
                task_key = f"{audio_id}::{model_name}"

                if task_key in processed_keys:
                    continue

                tasks.append(
                    {
                        "group_name": group_name,
                        "evaluation_mode": evaluation_mode,
                        "audio_id": audio_id,
                        "text": item["text"],
                        "emotions": item["emotions"],
                        "groundtruth_audio": item.get("question"),
                        "model_name": str(model_name),
                        "model_audio": str(model_audio_path),
                        "output_file": output_file,
                        "emotion_switch_positions": switch_positions,
                    }
                )

    return tasks


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DualEmotionBench TTS evaluation pipeline"
    )

    # Input/output
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the task JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for resumable JSONL result files.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="./temp_combined",
        help="Directory for temporary concatenated WAV files.",
    )

    # Optional span metadata
    parser.add_argument(
        "--zh_split_json",
        type=str,
        default="",
        help="Optional Chinese emotion-span metadata JSON.",
    )
    parser.add_argument(
        "--en_split_json",
        type=str,
        default="",
        help="Optional English emotion-span metadata JSON.",
    )

    # DNSMOS and ASR
    parser.add_argument(
        "--dns_primary_model",
        type=str,
        required=True,
        help="Path to DNSMOS sig_bak_ovr.onnx.",
    )
    parser.add_argument(
        "--dns_p808_model",
        type=str,
        required=True,
        help="Path to DNSMOS model_v8.onnx.",
    )
    parser.add_argument(
        "--asr_device",
        type=str,
        default="cuda:0",
        help="SenseVoice device, for example cuda:0 or cpu.",
    )

    # LLM judge
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible endpoint for the audio judge.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="qwen3-omni-flash",
        help="LLM audio judge model.",
    )

    # Evaluation-mode configuration
    parser.add_argument(
        "--default_eval_mode",
        type=str,
        choices=["single", "transition"],
        required=True,
        help=(
            "Fallback mode for groups not listed in --mode_config. "
            "'single': EM + NA; 'transition': TS + EM + NA."
        ),
    )
    parser.add_argument(
        "--mode_config",
        type=str,
        default="",
        help=(
            "Optional JSON mapping from group/folder name to evaluation mode. "
            "A task-level evaluation_mode field has higher priority."
        ),
    )

    # Runtime
    parser.add_argument(
        "--max_workers",
        type=int,
        default=3,
        help="Number of concurrent evaluation workers.",
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. "
            "Create a .env file and set DASHSCOPE_API_KEY."
        )

    if not os.path.isfile(args.dns_primary_model):
        raise FileNotFoundError(
            f"DNSMOS primary model not found: {args.dns_primary_model}"
        )

    if not os.path.isfile(args.dns_p808_model):
        raise FileNotFoundError(
            f"DNSMOS P808 model not found: {args.dns_p808_model}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    logger.info("Loading SenseVoiceSmall on %s...", args.asr_device)
    asr_model = AutoModel(
        model="iic/SenseVoiceSmall",
        device=args.asr_device,
        disable_update=True,
    )

    logger.info("Loading DNSMOS ONNX models...")
    dns_primary_sess = ort.InferenceSession(
        args.dns_primary_model,
        providers=["CPUExecutionProvider"],
    )
    dns_p808_sess = ort.InferenceSession(
        args.dns_p808_model,
        providers=["CPUExecutionProvider"],
    )

    judge = TTSJudge(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.judge_model,
    )

    metrics_calculator = MetricsCalculator(
        asr_model=asr_model,
        dns_primary_sess=dns_primary_sess,
        dns_p808_sess=dns_p808_sess,
    )

    zh_split_map, en_split_map = load_emotion_split_json(
        args.zh_split_json,
        args.en_split_json,
    )

    mode_map = load_mode_config(args.mode_config)

    with open(args.json_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if not isinstance(all_data, dict):
        raise ValueError(
            "Task JSON must be an object whose top-level keys are "
            "user-defined folder/group names."
        )

    tasks = build_tasks(
        all_data=all_data,
        args=args,
        mode_map=mode_map,
        zh_split_map=zh_split_map,
        en_split_map=en_split_map,
    )

    logger.info("Total pending tasks: %d", len(tasks))

    if not tasks:
        logger.info("No pending tasks. Evaluation finished.")
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(
            executor.map(
                lambda task: process_single_task(
                    task,
                    args,
                    judge,
                    metrics_calculator,
                ),
                tasks,
            )
        )

    logger.info("All evaluation tasks completed.")


if __name__ == "__main__":
    main()
