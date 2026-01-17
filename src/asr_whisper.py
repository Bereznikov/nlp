"""
ASR для аудио сообщений Telegram с использованием faster-whisper.

Input: Путь до аудио (ogg/mp3/wav)
Output: Распознанный текст

Использование:
python -m src.asr_whisper --audio path/to/voice.ogg --model small
"""

import argparse
from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model = WhisperModel(args.model, device=args.device)
    segments, info = model.transcribe(args.audio, language="ru")
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    print(text)


if __name__ == "__main__":
    main()
