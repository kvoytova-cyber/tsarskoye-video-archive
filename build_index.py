import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. ЧИТАЕМ EXCEL
print("Читаю Excel...")
df = pd.read_excel("video_archive.xlsx")
print(f"Загружено {len(df)} видео")

# 2. РЕЖЕМ НА ЧАНКИ
print("Режу на чанки...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

chunks = []
for _, row in df.iterrows():
    texts = splitter.split_text(str(row["Транскрипция"]))
    for i, text in enumerate(texts):
        chunks.append(Document(
            page_content=text,
            metadata={
                "filename": row["Имя файла"],
                "url": row["Ссылка на видео на диске"]
            }
        ))

print(f"Получилось {len(chunks)} чанков")

# 3. СОЗДАЁМ FAISS ИНДЕКС
print("Создаю векторы (подожди пару минут)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)

# 4. СОХРАНЯЕМ
vector_store.save_local("faiss_index")
print("Готово! Индекс сохранён в папку faiss_index/")