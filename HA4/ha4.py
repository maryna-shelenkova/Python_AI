import google.generativeai as genai
import time
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def get_embedding(text):
    try:
        response = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")
        return response["embedding"]
    except Exception as e:
        print(f"Ошибка получения эмбеддинга: {e}")
        return None

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

text1 = "Привет, как дела?"
text2 = "Здравствуйте, как у вас настроение?"

embedding1 = get_embedding(text1)
embedding2 = get_embedding(text2)

similarity = cosine_similarity(embedding1, embedding2)
print(f"Сходство текстов: {similarity:.4f}")

def find_most_similar(query, texts):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []
    similarities = [(text, cosine_similarity(query_embedding, get_embedding(text))) for text in texts]
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Пример поиска
texts = [
    "Сегодня солнечный день",
    "Как дела у тебя?",
    "Я люблю программировать",
    "Приветствую, как настроение?"
]

query = "Как поживаешь?"
results = find_most_similar(query, texts)

print("\nТоп похожих текстов:")
for text, score in results:
    print(f"{text} (сходство: {score:.4f})")

