# 🎙️ shadowing-flow
**shadowing-flow** 是一个半自动化工具链，用于从 *The Wall Street Journal* 的 **Tech News Briefing (TNB)** 播客中自动提取适合影子跟读（shadowing）的片段，生成结构化的 Markdown 学习笔记，并可选地裁剪对应音频片段，最终保存至 Obsidian 知识库。

> 项目名称：`shadowing-flow` – 让影子跟读材料的准备像流水线一样顺畅。

## ✨ 主要功能

- 📡 **自动获取最新节目**：通过 RSS 获取每期 TNB 播客的音频 URL 与元数据。
- 🧠 **AI 智能筛选**：调用 DeepSeek / OpenAI API 识别适合影子跟读的句子，标注重点短语和练习点。
- ✂️ **音频自动剪辑**：基于 pydub，自动裁剪对应的音频片段。
- 📝 **生成 Obsidian 笔记**：输出规范的 Markdown 文件，包含原文、练习提示和音频嵌入占位符。

## 🛠 技术栈
| 模块               | 技术选型                                         |
| ------------------ | ------------------------------------------------ |
| 数据获取           | `feedparser` + `requests`                       |
| 语音识别          | `openai-whisper` (本地)                          |
| AI 分析与结构化    | DeepSeek API                                     |
| 音频处理           | `pydub` + `ffmpeg`                |
| 笔记存储           | Obsidian `.md` 文件 |

📂 Output Example
## 🎙️ Apple & Intel Chip Deal
**⏱️ Time:** 30.8s → 46.08s  
**🔊 Audio:** ![[clip1.mp3]]
**Chunked Text:** We exclusively report / that Apple and Intel / have reached a preliminary agreement...
**Practice Focus:** Stress on 'exclusively', chunked pauses for clarity.

🚀 Quick Start
`
git clone https://github.com/yourusername/shadowing-flow.git
cd shadowing-flow
uv sync
cp .env.example .env   # add your DEEPSEEK_API_KEY
python main.py
`

🤝 License
MIT