# Видеоархив музея-заповедника «Царское Село»

Цифровой медиаархив с AI-powered семантическим поиском и интеллектуальным ассистентом на базе GigaChat.

## О проекте

Система для поиска и анализа видеоархива музея Царского Села (1995-2010 гг.). Автоматическая индексация метаданных, векторный поиск и генерация ответов через AI.


## Возможности

- **Семантический поиск** по метаданным видеозаписей
- **AI-ассистент GigaChat** для генерации развёрнутых ответов
- **Векторные эмбеддинги E5-large** для понимания контекста
- **Веб-интерфейс** с интуитивным поиском
- **Безопасное хранение** API ключей

## Технологии

- **Backend:** Flask
- **AI:** GigaChat (Сбер), HuggingFace Embeddings
- **Vector Search:** FAISS
- **ML:** sentence-transformers, langchain
- **Data:** Pandas, openpyxl

## Быстрый старт

### Требования

- Python 3.9+
- pip

### Установка

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/kvoytova-cyber/tsarskoye-video-archive.git
cd tsarskoye-video-archive

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Настройте переменные окружения
cp .env.example .env
nano .env  # Добавь свой GIGA_KEY
```

### Создание FAISS индекса

```bash
# Векторный индекс создается один раз:
python "create index e5.py"

# Это создаст папку faiss_index_e5/ (~10 минут)
```

### Запуск приложения

```bash
# Запустите Flask сервер
python app_giga.py

# Откройте в браузере
# http://localhost:5000
```

## Структура проекта

```
tsarskoye-video-archive/
├── .gitignore              
├── .env.example           
├── README.md             
├── requirements.txt       
├── app_giga.py           
├── create index e5.py     
├── video_archive1.xlsx    
└── faiss_index_e5/       #создается локально
```

## Деплой на Render.com

### Подготовка

1. Создайте FAISS индекс локально (см. выше)
2. Зарегистрируйтесь на [render.com](https://render.com)
3. Подключите GitHub репозиторий

### Настройка

1. **New Web Service** → Выберите репозиторий
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `python app_giga.py`
4. **Environment Variables:**
   - `GIGA_KEY` = ваш ключ GigaChat
   - `FLASK_ENV` = production

### Создание индекса на сервере

После деплоя:
1. Render Dashboard → Shell
2. Выполните: `python "create index e5.py"`
3. Перезапустите сервис

## Переменные окружения

Создай файл `.env` на основе `.env.example`:

```bash
# GigaChat API ключ
GIGA_KEY=your_gigachat_api_key_here

# Flask настройки (опционально)
FLASK_ENV=development
PORT=5000
```

**Где получить ключ GigaChat:** https://developers.sber.ru/studio/

## База данных

- **Источник:** Видеоархив музея-заповедника «Царское Село»
- **Период:** 1995-2010 годы
- **Объем:** 348 видеозаписей
- **Контент:** Новости, репортажи, интервью, документальные фильмы
- **Темы:** Реставрация, Янтарная комната, официальные визиты, культурные события и многое другое


---
