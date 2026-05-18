import logging
import json
import os
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from pydub import AudioSegment
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/"
DEEPSEEK_MODEL = "deepseek-v4-flash"
AI_TEMPERATURE = 0.2

# 1. Path(__file__).resolve() 获取 cliper.py 的绝对路径
# 2. .parent 得到 module 目录
# 3. .parent.parent 得到 项目根目录
storage_path = os.getenv("STORAGE_DIR", "data")
OBSIDIAN_VAULT = Path(storage_path)

# 4. 拼接目标路径
MD_OUTPUT_FOLDER = OBSIDIAN_VAULT / "WSJ-2026"
CLIP_OUTPUT_FOLDER = OBSIDIAN_VAULT / "WSJ-Audio"

MD_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CLIP_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 系统 Prompt
SYSTEM_PROMPT = """你是一名英语语音训练专家。请从以下带时间戳的播客转录文本中，筛选 3-4 段最适合 Shadowing 的片段，每段时长控制在 20-40 秒。

【核心目标】
训练重点切忌追求完美的美式发音或复杂的连读，而是必须聚焦于【意群停顿（Chunking）】与【逻辑重音（Stress）】，以英语表达的清晰度与专业感。

【执行步骤】
1. **内容归纳**：
    - 设定核心主题 `episode_theme`，必须严格执行格式：`[英文单词: 中文概括]`
        * 要求：抽象生动、具备行业深度，如 [Silicon Gambit: 巨头间的策略博弈]，严禁输出“新闻汇总”类直白词汇。
    - 英文控制在 2-4 词，中文控制在 9 字以内。
    - 提取 2-4 个与片段相关的子话题标签。
    - 禁止翻译公司名称、专有名词等，必须保留原文（如 "Apple"、"ChatGPT"、"Google"、"Meta"、"Intel"、"TSMC"、"Nintendo"、"AI regulation" 等）。

2. **片段筛选与时间轴锚定**：
    - 时长 20~40 秒（自然语速）
    - 发音清晰、句式完整、包含高频商务/科技/新闻表达
    - 连读/弱读/语调特征明显，适合模仿
    - 避开纯数据罗列、广告口播、多人快速插话或背景杂音段

3. **文本二次加工逻辑（关键）**：
    - **智能纠错**：原始转录文本中可能包含听觉识别错误（如将 "those spots" 错转为 "love spots"，"forward-facing" 错转为 "form of facing"），请务必修正这些错误，确保输出文本的准确性。
    - **双版本文本输出**：
        - `raw_text`：修复错词后的纯净原文，不带斜杠。
        - `chunked_text`：在 raw_text 基础上加入 '/' 划分意群。
    - **意群划分规则（针对 chunked_text）**：
        - 每个意群 3-7 词。
        - 【禁止断句点】：严禁拆散紧密相关的语法结构（如名词短语、介词短语）。
        - 示例：✅ "you learned from your father /"  ❌ "you learned from / your father"

【输出要求】
- 仅输出 JSON，禁止代码块外的文字。格式如下：

{
    "episode_theme": "[英文词组: 中文概括]",
    "snippets": [
        {
            "topic_tag": "子话题标签",
            "title": "6字内标题",
            "start_line_id": 12,   // 必须为整数，如选中的起始行是 [Line 12]
            "end_line_id": 18,     // 必须为整数，如选中的结束行是 [Line 18]
            "raw_text": "修复错词后的连续原文段落（作为时间轴校验基准）",
            "chunked_text": "带 '/' 的意群划分文本。每个意群 3-7 词，逻辑连贯。",
            "key_phrases": ["短语1", "短语2"],
            "practice_focus": "指出核心重音词，说明如何通过停顿和重音增强说服力。"
        }
    ]
}
"""

# Obsidian Markdown 模板
MD_TEMPLATE = """
## 🎙️ {topic_tag} {title}
**⏱️ 时间轴：** `{start_sec}s` → `{end_sec}s`  
**🔊 音频：**  ![[{audio_filename}]]

**🧩 原文与意群断句：**
> {chunked_text}

**🎯 重点语句：** {key_phrases_str}  
**🗣️ 练习点（发音/意群/重音）：** {practice_focus}  

---
"""

# 初始化 DeepSeek 客户端 (DeepSeek 兼容 OpenAI SDK)
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)


def generate_shadowing_script(transcript: str):
    """调用 DeepSeek 提取适合 Shadowing 的片段，并强制返回 JSON 格式"""
    logging.info("正在调用 DeepSeek API 分析文稿...")

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            response_format={"type": "json_object"},  # 强制返回 JSON
            temperature=AI_TEMPERATURE,  # 降低温度以获得更稳定的输出
        )

        # 1. 安全地获取内容
        content = response.choices[0].message.content

        if not content:
            logging.error("错误：模型返回内容为空")
            return {}

        # 2. 解析 JSON
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解析失败: {e}")
        return {}
    except Exception as e:
        logging.error(f"请求发生异常: {e}")
        return {}


def validate_snippets(data: Dict, whisper_segments: List[Dict]) -> List[Dict]:
    """基础结构校验与时间戳合法性检查"""
    snippets = data.get("snippets", [])
    if not isinstance(snippets, list) or len(snippets) == 0:
        raise ValueError("响应中未找到有效的 'snippets' 列表")

    valid_clips = []
    for clip in snippets:
        start_line = clip["start_line_id"]
        end_line = clip["end_line_id"]

        # 通过行号，直接从 whisper 原始数据中获取最精准的时间
        try:
            # 起始行的 start 时间
            actual_start_sec = float(whisper_segments[start_line - 1].get("start", 0))
            # 结束行的 end 时间
            actual_end_sec = float(whisper_segments[end_line - 1].get("end", 0))
        except IndexError:
            logging.error(f"AI 幻觉行号: {start_line} - {end_line}")
            continue

        if actual_start_sec >= actual_end_sec:
            logging.warning(
                f"⚠️ 时间戳无效 [{actual_start_sec}s -> {actual_end_sec}s]，已跳过"
            )
            continue
        clip["start_sec"] = round(actual_start_sec, 2)
        clip["end_sec"] = round(actual_end_sec, 2)
        valid_clips.append(clip)
    return valid_clips


def process_audio_and_markdown(
    text_title: str, audio_path: str, snippets_info: dict, whisper_segments: List[Dict]
):
    """根据 DeepSeek 返回的片段信息，剪裁音频并生成 Obsidian 笔记"""
    logging.info("正在处理音频并生成 Obsidian 笔记...")

    valid_clips = validate_snippets(snippets_info, whisper_segments)
    if not valid_clips:
        logging.warning("没有有效的片段可供剪裁和生成笔记")
        return

    audio_name = os.path.basename(audio_path).split(".")[0]  # 去掉扩展名
    # 这里可以使用 pydub 或 ffmpeg-python 来剪裁音频
    # 同时根据 snippets_info 生成对应的 Markdown 内容
    audio = AudioSegment.from_mp3(audio_path)

    # 生成文件名或标题
    today_chinese = datetime.now().strftime("%Y年%-m月%-d日")
    md_content = [f"# {text_title}"]
    md_content.append("\n---")
    md_content.append(f">核心主题：{snippets_info.get('episode_theme', '未知')}")
    md_content.append(
        ">新闻来源：WSJ Tech News Briefing | 仅供英语听说与 Shadowing 练习分享"
    )
    md_content.append(f">整理日期：{today_chinese}")
    md_content.append("---\n")

    for idx, clip in enumerate(valid_clips):
        start_ms = int(clip["start_sec"] * 1000)
        end_ms = int(clip["end_sec"] * 1000)

        # 剪裁并保存音频片段
        audio_clip = audio[start_ms:end_ms]
        clip_filename = f"{audio_name}_clip{idx+1}.mp3"
        clip_path = CLIP_OUTPUT_FOLDER / clip_filename
        audio_clip.export(clip_path, format="mp3")

        # 生成 Markdown 内容
        key_phrases_str = ", ".join(clip.get("key_phrases", []))
        md_content.append(
            MD_TEMPLATE.format(
                topic_tag=clip.get("topic_tag", "通用"),
                title=clip.get("title", f"片段 {idx+1}"),
                start_sec=clip["start_sec"],
                end_sec=clip["end_sec"],
                audio_filename=clip_filename,
                chunked_text=clip.get("chunked_text", ""),
                key_phrases_str=key_phrases_str,
                practice_focus=clip.get("practice_focus", ""),
            )
        )

    # 保存 Markdown 文件
    md_filename = f"{audio_name}.md"
    md_path = MD_OUTPUT_FOLDER / md_filename
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    logging.info(f"Markdown 笔记已生成: {md_filename}")
