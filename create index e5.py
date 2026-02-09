import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. ЧИТАЕМ EXCEL
df = pd.read_excel("video_archive1.xlsx")

# Определяем названия колонок
def get_column(df, *possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return possible_names[0]

col_transcript = get_column(df, "Транскрипция", "transcript", "text", "title")
col_filename = get_column(df, "Имя файла", "filename", "file")
col_url = get_column(df, "Ссылка на видео на диске", "url", "path")
col_title = get_column(df, "Название", "title")
col_year = get_column(df, "Год", "year")
col_channel = get_column(df, "Канал", "channel")
col_type = get_column(df, "Тип", "type")

# 2. РЕЖЕМ НА ЧАНКИ
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

chunks = []
for idx, row in df.iterrows():
    text = str(row.get(col_transcript, row.get(col_title, "")))
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

# 3. СОЗДАЁМ FAISS ИНДЕКС
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = FAISS.from_documents(chunks, embeddings)

# 4. СОХРАНЯЕМ
vector_store.save_local("faiss_index_e5")

print(f"✓ Индекс создан: {len(df)} видео, {len(chunks)} чанков")
