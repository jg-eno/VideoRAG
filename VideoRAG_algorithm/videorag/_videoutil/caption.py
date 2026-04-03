import json
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

# Public-ish checkpoint (full MiniCPM-V-2_6 is gated on Hugging Face).
_DEFAULT_HUB_CAPTION = "openbmb/MiniCPM-V-2_6-int4"


def _videorag_repo_root() -> Path:
    """``VideoRAG/`` (parent of ``VideoRAG_algorithm``)."""
    return Path(__file__).resolve().parents[3]


def _resolve_caption_model_id() -> str:
    """
    Prefer a local directory (avoids Hub / gated repos), then Hugging Face id.

    Override with env ``VIDEO_RAG_CAPTION_MODEL`` (absolute path or model id).

    CPU-only override: ``VIDEO_RAG_CAPTION_MODEL_CPU`` (defaults to same resolution
    if unset — no longer forces gated ``openbmb/MiniCPM-V-2_6``).
    """
    env = os.environ.get("VIDEO_RAG_CAPTION_MODEL", "").strip()
    if env:
        if os.path.isdir(os.path.abspath(env)):
            return os.path.abspath(env)
        return env

    root = _videorag_repo_root()
    for candidate in (
        root / "MiniCPM-V-2_6-int4",
        Path.cwd() / "MiniCPM-V-2_6-int4",
        Path("./MiniCPM-V-2_6-int4").resolve(),
    ):
        try:
            if candidate.is_dir():
                return str(candidate.resolve())
        except OSError:
            continue

    return os.environ.get("VIDEO_RAG_CAPTION_MODEL_HUB", _DEFAULT_HUB_CAPTION)


def _caption_checkpoint_needs_cuda_gpu(model_id: str) -> bool:
    """INT4 / bitsandbytes checkpoints cannot run on CPU-only PyTorch."""
    mid = model_id.lower()
    if "int4" in mid or "4bit" in mid or "4-bit" in mid:
        return True
    if os.path.isdir(model_id):
        cfg = Path(model_id) / "config.json"
        if cfg.is_file():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                if data.get("quantization_config"):
                    return True
            except (OSError, json.JSONDecodeError, TypeError):
                pass
    return False


def _cpu_caption_dtype():
    name = os.environ.get("VIDEO_RAG_CAPTION_DTYPE", "float16").lower()
    if name in ("float16", "half"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float16


def load_caption_model_and_tokenizer():
    """
    Loads MiniCPM-V (typically int4) for captioning.

    Resolution order:

    1. ``VIDEO_RAG_CAPTION_MODEL`` if set (local path or Hub id).
    2. Local ``MiniCPM-V-2_6-int4`` next to the repo root (or cwd).
    3. Hub ``openbmb/MiniCPM-V-2_6-int4`` (override with ``VIDEO_RAG_CAPTION_MODEL_HUB``).

    The **full** ``openbmb/MiniCPM-V-2_6`` model is gated; it is no longer the default
    CPU fallback. For gated Hub models, run ``huggingface-cli login`` or set ``HF_TOKEN``.

    CPU-only explicit override: ``VIDEO_RAG_CAPTION_MODEL_CPU`` (path or model id).

    **INT4 checkpoints** (``MiniCPM-V-2_6-int4``) use bitsandbytes and **require a CUDA
    GPU**. They cannot be loaded with ``device_map="cpu"``.
    """
    if torch.cuda.is_available():
        model_id = _resolve_caption_model_id()
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer

    cpu_id = os.environ.get("VIDEO_RAG_CAPTION_MODEL_CPU", "").strip()
    model_id = cpu_id or _resolve_caption_model_id()
    if _caption_checkpoint_needs_cuda_gpu(model_id):
        raise RuntimeError(
            f"Caption checkpoint {model_id!r} is 4-bit quantized (bitsandbytes). "
            "It needs an NVIDIA GPU and PyTorch with CUDA. "
            "This process has torch.cuda.is_available() == False, so loading on CPU is not supported.\n\n"
            "What to do:\n"
            "  • Run on a machine with a GPU; check `nvidia-smi` and install the CUDA build of PyTorch "
            "(https://pytorch.org/get-started/locally/).\n"
            "  • Verify: `python -c \"import torch; print(torch.cuda.is_available())\"` prints True.\n"
            "  • Do not use the int4 folder on CPU; there is no supported CPU path for this checkpoint."
        )
    dtype = _cpu_caption_dtype()
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames
    
def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:
        model, tokenizer = load_caption_model_and_tokenizer()
        model.eval()
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                msgs = [{'role': 'user', 'content': video_frames + [query]}]
                params = {}
                params["use_image_id"] = False
                params["max_slice_nums"] = 2
                segment_caption = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    **params
                )
                caption_result[index] = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        
def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # model.eval()
    
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        # query = f"The transcript of the current video:\n{segment_transcript}.\nGiven a question: {query}, you have to extract relevant information from the video and transcript for answering the question."
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return caption_result
