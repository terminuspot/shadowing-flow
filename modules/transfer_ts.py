import re
import logging
import whisper_timestamped as whisper


def transcribe_audio_with_whisper_timestamped(audio_path, max_duration=25.0):
    """
    使用 whisper-timestamped 高精度单词流模型进行识别
    天然消除滑窗重叠与时差漂移，完美对接下游两阶段装箱逻辑
    """
    logging.info("正在启动本地 Whisper-Timestamped 高精度模型...")
    # 可以直接加载 base 或 small，模型会天然被高精度追踪算法强化
    model = whisper.load_model(
        "base", device="cpu"
    )  # 在 Mac M系列芯片上可以指定 cpu 或 mps

    logging.info("正在进行高精度语音转文字，通过动态规划（DTW）锁死字词时间轴...")

    # 执行转写：whisper-timestamped 的核心接口
    result = whisper.transcribe(
        model,
        str(audio_path),
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad=True,  # ⭐ 开启内置的轻量 VAD 防幻觉机制
    )

    # 它的原生结果里包含非常干净的 segments，且 segments 内部带有每个单词的精确卡点
    raw_segments = result.get("segments", [])

    # --- 第一阶段：降维提取单调递增、毫无重叠的纯净单词流 ---
    all_words = []
    for seg in raw_segments:
        words = seg.get("words", [])
        for w in words:
            all_words.append(
                {
                    "word": w["text"].strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
            )

    if not all_words:
        raise ValueError("Whisper-Timestamped 未能成功从音频中解析出任何有效单词")

    # 物理层去重与单调性修正（双保险，防止极端特殊连读）
    all_words.sort(key=lambda x: x["start"])
    clean_words = [all_words[0]]
    for next_w in all_words[1:]:
        last_w = clean_words[-1]
        if next_w["start"] < last_w["end"]:
            if next_w["word"].lower() == last_w["word"].lower():
                continue  # 剔除幻觉重复词
            next_w["start"] = last_w["end"]  # 强行校准时间轴向前
        if next_w["end"] > next_w["start"]:
            clean_words.append(next_w)

    logging.info(f"成功提纯 {len(clean_words)} 个高精度单词。开始执行语义切句...")

    # --- 第二阶段：利用洗干净的绝对线性单词流，组装短句（一阶段语义断句） ---
    sentences = []

    def is_sentence_end(text):
        return bool(re.search(r'[.!?。？！][\'"\)\]\s]*$', text))

    cur_start = None
    cur_text = []

    for w in clean_words:
        if cur_start is None:
            cur_start = w["start"]
        cur_text.append(w["word"])

        # 结合标点断句与超时（40秒）防大难句熔断
        if is_sentence_end(w["word"]) or (w["end"] - cur_start >= 40.0):
            sentences.append((cur_start, w["end"], " ".join(cur_text)))
            cur_start = None
            cur_text = []

    if cur_text and cur_start is not None:
        sentences.append((cur_start, clean_words[-1]["end"], " ".join(cur_text)))

    # --- 第三阶段：按【净说话时长】装箱（二阶段控制素材跨度 ≤25秒） ---
    packages = pack_by_duration_timestamped_version(
        sentences, max_duration=max_duration
    )

    # --- 第四阶段：格式化输出给 DeepSeek 消费的规范文本 ---
    transcript_text = ""
    for idx, (start, end, text) in enumerate(packages):
        start_rounded = round(start, 2)
        end_rounded = round(end, 2)
        transcript_text += (
            f"[Line {idx + 1}] [{start_rounded} - {end_rounded}] {text}\n"
        )

    logging.info(f"高精度文稿解析与装箱完成！共打包出 {len(packages)} 个黄金素材包。")
    return transcript_text, packages


def pack_by_duration_timestamped_version(sentences: list, max_duration=25.0) -> list:
    """
    针对短句列表的【净说话时长】装箱算法
    """
    if not sentences:
        return []
    packages = []
    cur_start, cur_end, cur_text = sentences[0]
    cur_net_duration = cur_end - cur_start

    for s in sentences[1:]:
        s_start, s_end, s_text = s
        s_duration = s_end - s_start

        is_time_ok = (cur_net_duration + s_duration) <= max_duration
        is_gap_acceptable = (s_start - cur_end) < 3.0  # 静音断层超过3秒强制切分

        if is_time_ok and is_gap_acceptable:
            cur_end = s_end
            cur_text += " " + s_text
            cur_net_duration += s_duration
        else:
            packages.append((cur_start, cur_end, cur_text.strip()))
            cur_start = s_start
            cur_end = s_end
            cur_text = s_text
            cur_net_duration = s_duration

    if cur_text:
        packages.append((cur_start, cur_end, cur_text.strip()))
    return packages
