# TG Audio Sentiment (Women’s Clothing Reviews)

Цель: классифицировать голосовые сообщения Telegram по тональности **позитив/негатив**.

Датасет: *Women’s Clothing E-Commerce Reviews*.

## ВАЖНО
model.safetensors не влезла в GitHub.
Скачайте ее из архива или по [ссылке](https://disk.yandex.ru/d/9_d1uBcoJ2ErJw) и добавьте в /models/bert/, чтобы получить сразу рабочий проект.

## Схема
Telegram voice -> сервер -> ASR (Whisper) -> текст -> классификатор (лучший: RuBERT) -> метка -> ответ в Telegram.

## Быстрый старт

### 1) Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Препроцессинг датасета (2 колонки)
```bash
python -m src.preprocess_womens_dataset \
  --input "data/raw/Womens Clothing E-Commerce Reviews.csv" \
  --output "data/processed/dataset_en.csv"
```

### 3) Перевод EN->RU
```bash
python -m src.translate_dataset \
  --input "data/processed/dataset_en.csv" \
  --output "data/processed/dataset_ru.csv" \
  --batch-size 16
```

### 4) Обучение моделей (ноутбук)
Откройте `notebooks/01_train_text_models_ru_womens.ipynb` и выполните все ячейки.

### 5) Сервер
После обучения сохраните модель RuBERT в `models/bert/` (в ноутбуке есть ячейка сохранения).

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 6) Бот
Скопируйте `.env.example` в `.env`, вставьте токен.

```bash
python -m bot.bot
```
