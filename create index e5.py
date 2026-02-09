"""
Создание FAISS-индекса с моделью E5-large
Улучшенное качество поиска для 348 видео
"""

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("="*60)
print("СОЗДАНИЕ FAISS-ИНДЕКСА С МОДЕЛЬЮ E5-LARGE")
print("="*60)

# 1. ЧИТАЕМ EXCEL
print("\n[1/4] Читаю Excel...")
df = pd.read_excel("video_archive1.xlsx")
print(f"✓ Загружено {len(df)} видео")

# Определяем названия колонок (поддержка разных форматов)
def get_column(df, *possible_names):
    """Ищет колонку по нескольким возможным названиям"""
    for name in possible_names:
        if name in df.columns:
            return name
    # Если не нашли, возвращаем первое название
    return possible_names[0]

col_transcript = get_column(df, "Транскрипция", "transcript", "text", "title")
col_filename = get_column(df, "Имя файла", "filename", "file")
col_url = get_column(df, "Ссылка на видео на диске", "url", "path")
col_title = get_column(df, "Название", "title")
col_year = get_column(df, "Год", "year")
col_channel = get_column(df, "Канал", "channel")
col_type = get_column(df, "Тип", "type")

print(f"  Используемые колонки:")
print(f"    Текст: {col_transcript}")
print(f"    Файл: {col_filename}")
print(f"    Название: {col_title}")

# 2. РЕЖЕМ НА ЧАНКИ
print("\n[2/4] Режу на чанки...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

chunks = []
for idx, row in df.iterrows():
    # Используем найденную колонку с текстом
    text = str(row.get(col_transcript, row.get(col_title, "")))
    
    # Режем на чанки
    texts = splitter.split_text(text)
    
    for i, chunk_text in enumerate(texts):
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "filename": str(row.get(col_filename, f"video_{idx}")),
                "url": str(row.get(col_url, "")),
                "title": str(row.get(col_title, "")),
                "year": str(row.get(col_year, "")),
                "channel": str(row.get(col_channel, "")),
                "type": str(row.get(col_type, "")),
                "chunk_id": f"{idx}_{i}"
            }
        ))
    
    # Прогресс
    if (idx + 1) % 50 == 0:
        print(f"  Обработано {idx + 1}/{len(df)} видео...")

print(f"✓ Получилось {len(chunks)} чанков")

# 3. СОЗДАЁМ FAISS ИНДЕКС С E5-LARGE
print("\n[3/4] Создаю векторы с моделью E5-large...")
print("  ⚠️  Это займёт 5-10 минут (модель тяжелее, но качественнее)")
print("  Загружаю модель intfloat/multilingual-e5-large...")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},  # Используем CPU (для GPU поставь 'cuda')
    encode_kwargs={'normalize_embeddings': True}  # Нормализация для лучших результатов
)

print("  Создаю векторный индекс...")
vector_store = FAISS.from_documents(chunks, embeddings)

# 4. СОХРАНЯЕМ
print("\n[4/4] Сохраняю индекс...")
vector_store.save_local("faiss_index_e5")

print("\n" + "="*60)
print("✓ ГОТОВО!")
print("="*60)
print(f"Индекс сохранён в папку: faiss_index_e5/")
print(f"Модель: intfloat/multilingual-e5-large")
print(f"Обработано видео: {len(df)}")
print(f"Всего чанков: {len(chunks)}")
print(f"\n💡 E5-large даёт лучшее качество поиска, чем MiniLM!")
print(f"\nТеперь запусти Flask-приложение:")
print(f"  python search_app_e5.py")
print("="*60)