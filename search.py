from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Загружаем индекс
print("Загружаю индекс...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("Готово!\n")

# Поиск
print("Поиск по видеоархиву. Напиши 'выход' чтобы выйти.\n")

while True:
    query = input("Твой вопрос: ").strip()
    if query.lower() in ["выход", "exit", "q"]:
        break
    
    results = vector_store.similarity_search_with_score(query, k=3)
    
    print("\n📹 Найдено:\n")
    for doc, score in results:
        print(f"[{doc.metadata['filename']}]")
        print(f"{doc.page_content}")
        print(f"(релевантность: {1-score:.0%})\n")
    print("-" * 40 + "\n")