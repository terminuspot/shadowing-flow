import re
import mlx_whisper
import logging
from typing import List


def extract_clean_words_flow_mlx(result: dict) -> List[dict]:
    """
    【核心去重算法】从 MLX-Whisper 结果中提取绝对单调递增、毫无重叠的纯净单词流
    """
    raw_segments = result.get("segments", [])
    all_words = []

    for seg in raw_segments:
        # 提取 MLX 单词级时间戳列表
        words = seg.get("words", [])
        if not words:
            continue

        for w in words:
            all_words.append(
                {
                    "word": w["word"].strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
            )

    if not all_words:
        return []

    # 1. 按物理开始时间排序，建立绝对一维参考系
    all_words.sort(key=lambda x: x["start"])

    # 2. 斩断一切因为 Mac 滑窗边界带来的“交织重叠”脏数据
    clean_words = [all_words[0]]
    for next_w in all_words[1:]:
        last_w = clean_words[-1]

        # 如果当前词的开始时间，居然比上一个词的结束时间还要早（发生了滑窗重叠）
        if next_w["start"] < last_w["end"]:
            # 情况 A：如果两个单词一模一样，说明是 Whisper 重复输出的幻觉词，直接无视丢弃
            if next_w["word"].lower() == last_w["word"].lower():
                continue
            # 情况 B：连读导致的时间轴轻微交叉，强行将当前词的 start 修正为上一句的 end，使其单调递增
            next_w["start"] = last_w["end"]

        # 确保脏数据不会混入
        if next_w["end"] > next_w["start"]:
            clean_words.append(next_w)

    overlap_count = len(all_words) - len(clean_words)
    logging.info(
        {
            "event": "mlx_word_cleanup_complete",
            "total_raw_words": len(all_words),
            "clean_words_kept": len(clean_words),
            "dropped_overlap_words": overlap_count,
            "audio_duration_sec": (
                round(all_words[-1]["end"] - all_words[0]["start"], 2)
                if all_words
                else 0
            ),
        }
    )

    return clean_words


def merge_words_into_sentences(clean_words: List[dict], max_sentence_duration=40.0):
    """
    第一阶段（语义级）：利用洗干净的绝对线性单词流，重新拼装出结构完整的句子
    """
    if not clean_words:
        return []

    def is_sentence_end(text):
        return bool(re.search(r'[.!?。？！][\'"\)\]\s]*$', text))

    sentences = []
    cur_start = None
    cur_text = []

    for w in clean_words:
        if cur_start is None:
            cur_start = w["start"]

        cur_text.append(w["word"])
        current_duration = w["end"] - cur_start

        # 结合标点断句与防大难句超时熔断
        is_normal_end = is_sentence_end(w["word"])
        is_timeout = current_duration >= max_sentence_duration

        if is_normal_end or is_timeout:
            combined_text = " ".join(cur_text)
            sentences.append((cur_start, w["end"], combined_text))
            cur_start = None
            cur_text = []

    # 兜底
    if cur_text and cur_start is not None:
        sentences.append((cur_start, clean_words[-1]["end"], " ".join(cur_text)))

    return sentences


def transcribe_audio_with_whisper_mlx(audio_path):
    """使用 MLX Whisper 模型生成带时间戳的文稿 (彻底消灭时间轴交织重叠版)"""
    model_id = "mlx-community/whisper-base-mlx"
    logging.info(f"正在启动 MLX Whisper 模型 ({model_id}) 进行高精度语音识别...")

    # 核心修改点 1：必须显式传入 word_timestamps=True 开启单词扫描
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_id,
        word_timestamps=True,  # ⭐ 开启高精度单词时间轴
    )

    transcript_text = ""

    # 核心修改点 2：提纯单词流，抹平所有重叠和边界倒退问题
    clean_words = extract_clean_words_flow_mlx(result)

    # 核心修改点 3：第一阶段清洗（语义切分）
    sentences = merge_words_into_sentences(clean_words, max_sentence_duration=40.0)

    # 核心修改点 4：第二阶段装箱（基于净说话时长累加，每个包 ≤25秒）
    # 这里的 pack_by_duration 请沿用我们之前改好的【净时长版】
    packages = pack_by_duration(sentences, max_duration=25.0)

    print("=" * 50)
    print(f"第一阶段共切出 {len(sentences)} 个标准单句")
    print(f"第二阶段经过 25s 限制后，共打包出 {len(packages)} 个素材包")
    # --- 重新格式化输出给 DeepSeek 消费的文本 ---
    for idx, (start, end, text) in enumerate(packages):
        print(
            f" -> 包 {idx + 1}: 时长 = {round(end - start, 2)}秒, 字数 = {len(text.split())}"
        )

        start_rounded = round(float(start), 2)
        end_rounded = round(float(end), 2)
        # 格式化为标准 Line ID 格式，彻底防止 DeepSeek 时间轴幻觉
        transcript_text += (
            f"[Line {idx + 1}] [{start_rounded} - {end_rounded}] {text}\n"
        )
    print("=" * 50)
    logging.info("文稿高精度解析与装箱完成！")

    # 保持原有接口返回格式不变
    return transcript_text, packages


def pack_by_duration(sentences, max_duration=25.0):
    """
    将短句列表按【净说话时长】打包，每个包内的文字实际音频跨度更紧凑
    返回: [(start, end, combined_text), ...]
    """
    if not sentences:
        return []

    packages = []

    # 初始化第一个包裹
    cur_start, cur_end, cur_text = sentences[0]
    # 【核心改变】：记录这个包里所有句子的“净说话时间”总和
    cur_net_duration = cur_end - cur_start

    for s in sentences[1:]:
        s_start, s_end, s_text = s
        s_duration = s_end - s_start  # 当前这句话的净时长

        # 策略升级：如果加上这一句的【净时长】依然 <= max_duration，就允许合并
        # 并且，为了防止两句之间静音期实在太长（比如超过 12 秒的超级大断层），加一个辅助防御
        is_time_ok = (cur_net_duration + s_duration) <= max_duration
        is_gap_acceptable = (s_start - cur_end) < 12.0

        if is_time_ok and is_gap_acceptable:
            cur_end = s_end
            cur_text += " " + s_text
            cur_net_duration += s_duration  # 累加净时长
        else:
            # 超过限制，封包
            packages.append((cur_start, cur_end, cur_text.strip()))
            # 开新包
            cur_start = s_start
            cur_end = s_end
            cur_text = s_text
            cur_net_duration = s_duration

    # 收尾最后一个包
    if cur_text:
        packages.append((cur_start, cur_end, cur_text.strip()))

    return packages
