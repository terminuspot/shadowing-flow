from datetime import datetime
import os
from urllib.parse import urlparse

import feedparser
import logging
import requests

import torch
import whisperx
import whisper
from typing import cast, List, Dict, Any
from pathlib import Path

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
origin_episode_folder = PROJECT_ROOT / "episodes"

origin_episode_folder.mkdir(exist_ok=True)  # 确保保存音频的文件夹存在


def fetch_latest_podcast(rss_url):
    """从 RSS 获取最新一期播客并下载 MP3"""

    # 解析 RSS feed
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        raise Exception("无法获取 RSS 内容")

    # 获取最新一期播客信息
    latest = feed.entries[0]
    episode_info = {
        "title": latest.title,
        "published": latest.published,
        "audio_url": None,
        "episode_id": None,
    }

    # 提取音频 URL
    if hasattr(latest, "enclosures") and latest.enclosures:
        episode_info["audio_url"] = latest.enclosures[0].href
    # 尝试从 link 提取 episode ID
    if hasattr(latest, "id"):
        episode_info["episode_id"] = latest.id

    logging.info(f"找到最新播客: {episode_info}")
    return episode_info


def fetch_audio(audio_url):
    """下载音频文件"""
    if not os.path.exists(origin_episode_folder):
        os.makedirs(origin_episode_folder)
    # 从 URL 中提取文件名
    path = urlparse(audio_url).path
    file_name = os.path.basename(path)
    if not file_name:
        file_name = "downloaded_audio.mp3"

    today_str = datetime.now().strftime("%Y%m%d")
    file_name = f"{today_str}_{file_name}"

    # 生成完整的保存路径
    save_path = origin_episode_folder / file_name

    # --- 新增：检查文件是否存在 ---
    if save_path.exists():
        logging.info(f"文件已存在，直接跳过下载: {save_path}")
        return save_path

    # 下载音频文件
    logging.info(f"开始下载音频: {audio_url}")
    response = requests.get(audio_url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()

    if response.status_code == 200:
        logging.info(f"音频下载中: {save_path}")
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"下载失败，状态码: {response.status_code}")

    logging.info(f"音频下载成功，保存路径: {save_path}")
    return save_path


def transcribe_audio_with_whisper(audio_path):
    """使用本地 Whisper 模型生成带时间戳的文稿"""
    logging.info("正在启动本地 Whisper 模型进行语音识别...")
    # 这里使用 'base' 或 'small' 模型即可，速度快且准确率足够
    model = whisper.load_model("base")

    logging.info("正在进行语音转文字并提取时间戳...")
    result = model.transcribe(
        str(audio_path),
        fp16=False,
        beam_size=5,
        no_speech_threshold=0.4,
        word_timestamps=False,
    )  # 禁用 fp16 以避免某些 GPU 上的兼容性问题
    transcript_text = ""

    # 在循环开始前，把整个 segments 列表强制转化为“字符串键的字典列表”
    segments = cast(List[Dict[str, Any]], result.get("segments", []))

    # 将结果格式化为带有开始和结束时间的字符串
    for idx, segment in enumerate(segments):
        start = round(float(segment.get("start", 0)), 2)
        end = round(float(segment.get("end", 0)), 2)
        text = segment.get("text", "")
        # transcript_text += f"[{start} - {end}] {text}\n"
        # 输出格式变为：[Line 5] [12.5 - 15.0] The company reported...
        transcript_text += f"[Line {idx + 1}] [{start} - {end}] {text}\n"

    logging.info("文稿解析完成！")
    return transcript_text, segments


def transcribe_audio_with_whisperx(audio_path):
    # ----- 1. 设备自适应 + 合理的 batch_size -----
    # Mac 上 torch.cuda.is_available() 必为 False，自动走 CPU 分支
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU 用 int8 省内存，Apple Silicon 上也稳定
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    # 建议 1~4，默认 16 对 CPU 负担太大
    batch_size = 4 if device == "cpu" else 16

    logging.info(
        f"启动 WhisperX (设备: {device}, 计算类型: {compute_type}, batch_size: {batch_size})"
    )

    # 2. 加载模型（首次会自动下载 base 模型）
    model = whisperx.load_model("base", device, compute_type=compute_type)

    logging.info("正在进行初步转录...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)

    language_code = result["language"]

    # 3. 强制对齐（这一步在 CPU 上会慢一些，但精度极高）
    logging.info(f"加载对齐模型 (语言: {language_code})...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )

    logging.info("正在执行强制对齐以修正时间轴...")
    result_aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    segments = cast(List[Dict[str, Any]], result_aligned.get("segments", []))

    # 4. 格式化输出（保留毫秒级精度）
    transcript_text = ""
    for idx, segment in enumerate(segments):
        start = round(float(segment.get("start", 0)), 3)
        end = round(float(segment.get("end", 0)), 3)
        text = segment.get("text", "").strip()
        transcript_text += f"[Line {idx + 1}] [{start} - {end}] {text}\n"

    return transcript_text, segments
