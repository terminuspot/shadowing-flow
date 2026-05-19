from datetime import datetime
import os
import re
import subprocess
from urllib.parse import urlparse

import feedparser
import logging
import mlx_whisper
import requests

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

    convert_to_cbr(save_path)  # 转换为 CBR 格式，确保后续处理稳定

    logging.info(f"音频下载成功，保存路径: {save_path}")
    return save_path


def convert_to_cbr(input_path):
    """将下载的 VBR MP3 强制转换为 CBR (128k)，消除 Pydub 寻址 Bug"""
    output_path = input_path.replace(".mp3", "_cbr.mp3")

    # 调用系统 ffmpeg 进行标准重编码
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",  # 强行固定 128kbps 码率
        output_path,
    ]

    # 隐藏控制台输出运行
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 覆盖原文件
    os.remove(input_path)
    os.rename(output_path, input_path)
    logging.info("音频已成功重构为标准 CBR 格式")


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


def transcribe_audio_with_whisper_mlx(audio_path):
    """使用 MLX Whisper 模型生成带时间戳的文稿 (Mac 专属硬件加速)"""

    # MLX Whisper 推荐直接使用 Hugging Face 上的 mlx 优化格式模型
    # 这里对应你原本的 "base" 模型
    model_id = "mlx-community/whisper-base-mlx"

    logging.info(f"正在启动 MLX Whisper 模型 ({model_id}) 进行语音识别...")

    # mlx-whisper.transcribe 会自动处理模型的加载、缓存和 Metal GPU 调度
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_id,
        word_timestamps=True,
        # 注意：MLX 会自动且完美地使用 Mac GPU 的混合精度 (fp16/bf16)
        # 所以我们不需要像原版那样写 fp16=False 来规避兼容性问题
    )

    # --- 新增的按句子合并逻辑 ---
    transcript_text = ""

    # 这里的逻辑与原版完全保持一致，因为 MLX 返回的字典结构和原版 100% 相同
    segments = cast(List[Dict[str, Any]], result.get("segments", []))

    # 1. 先合并成短句
    sentences = merge_into_short_sentences(segments)

    # 2. 按时长打包（每个包 ≤30秒）
    packages = pack_by_duration(sentences, max_duration=30.0)

    # --- 重新格式化输出 ---
    for idx, (start, end, text) in enumerate(packages):
        start_rounded = round(start, 2)
        end_rounded = round(end, 2)
        transcript_text += (
            f"[Line {idx + 1}] [{start_rounded} - {end_rounded}] {text}\n"
        )

    logging.info("文稿解析完成！")
    return transcript_text, packages


def merge_into_short_sentences(segments, max_sentence_duration=40.0):
    """
    将 whisper 片段按标点合并成短句（不含时长限制）
    返回: [(start, end, text), ...]
    """
    COMMON_ABBREVIATIONS = {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "vs",
        "e.g",
        "i.e",
        "Inc",
        "Corp",
        "Ltd",
    }

    def is_sentence_end(t):
        return bool(re.search(r'[.!?。？！][\'"\)\]\s]*$', t.strip()))

    def is_abbrev(t):
        m = re.match(r"^([A-Za-z\.]+)\.$", t.strip())
        return m and m.group(1) in COMMON_ABBREVIATIONS

    sentences = []
    cur_start = None
    cur_text = ""
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        if cur_start is None:
            cur_start = seg["start"]

        # 拼接文本
        cur_text += " " + text if cur_text else text

        # 当前累计时长（从 cur_start 到本片段结束）
        current_duration = seg["end"] - cur_start

        # 判断是否为正常句子结束（非缩写）或超时熔断
        is_normal_end = is_sentence_end(text) and not is_abbrev(text)
        is_timeout = current_duration >= max_sentence_duration

        if is_normal_end or is_timeout:
            sentences.append((cur_start, seg["end"], cur_text.strip()))
            cur_start = None
            cur_text = ""

    if cur_text and cur_start is not None:
        sentences.append((cur_start, segments[-1]["end"], cur_text.strip()))
    return sentences


def pack_by_duration(sentences, max_duration=30.0):
    """
    将短句列表按累计时长打包，每个包时长不超过 max_duration 秒
    返回: [(start, end, combined_text), ...]
    """
    if not sentences:
        return []
    packages = []
    cur_start = sentences[0][0]
    cur_end = sentences[0][1]
    cur_text = sentences[0][2]

    for s in sentences[1:]:
        s_start, s_end, s_text = s
        # 如果加入当前句后总时长 <= max_duration，则合并
        new_duration = s_end - cur_start
        if new_duration <= max_duration:
            cur_end = s_end
            cur_text += " " + s_text
        else:
            # 超出限制，保存当前包，开始新包
            packages.append((cur_start, cur_end, cur_text.strip()))
            cur_start = s_start
            cur_end = s_end
            cur_text = s_text

    # 最后一个包
    if cur_text:
        packages.append((cur_start, cur_end, cur_text.strip()))
    return packages
