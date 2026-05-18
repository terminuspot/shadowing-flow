import logging
from pathlib import Path
from modules import fetcher
from modules import cliper

# WSJ Tech News Briefing 的官方 RSS 地址
WSJ_RSS_URL = "https://video-api.wsj.com/podcast/rss/wsj/tech-news-briefing"
WSJ_RSS_MIN_URL = "https://video-api.wsj.com/podcast/rss/wsj/the-journal"

# 1. 获取当前脚本所在的绝对路径目录
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)  # 如果 logs 文件夹不存在则自动创建
LOG_FILE = LOG_DIR / "shadowing_flow.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    logging.info("Start to fetch the latest podcast from WSJ Tech News Briefing...")
    # 步骤 1：从 RSS 拉取今天最新的 WSJ 播客音频
    episode_info = fetcher.fetch_latest_podcast(WSJ_RSS_MIN_URL)
    # 步骤 2：下载音频文件，用本地 Whisper 听写音频，生成带时间戳的精准文稿
    audio_path = fetcher.fetch_audio(episode_info["audio_url"])
    transcript, segments = fetcher.transcribe_audio_with_whisper_mlx(audio_path)

    logging.info(f"生成的文稿:\n{transcript}")
    # 步骤 3：把文稿发给 DeepSeek 提取 Shadowing 学习片段
    snippets_info = cliper.generate_shadowing_script(transcript)
    logging.info(f"Shadowing 片段: {snippets_info}")
    # 步骤 4：根据 AI 的时间戳自动剪裁音频，并生成 Obsidian 笔记
    cliper.process_audio_and_markdown(
        episode_info["title"], str(audio_path), snippets_info, segments
    )
    logging.info("流程执行完成！")
    print("✅ 脚本执行完毕")


if __name__ == "__main__":
    main()
