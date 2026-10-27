from minio import Minio
import pymupdf
import chromadb
import requests
import os


# ==========================================
# 설정
# ==========================================

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

BUCKET_NAME = "documents"

PDF_FILE = "workplan.pdf"
OBJECT_NAME = "workplan.pdf"

OLLAMA_URL = "http://localhost:11435"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:12b"


# ==========================================
# 1. MinIO 연결
# ==========================================

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)


# ==========================================
# 2. Bucket 생성
# ==========================================

if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)
    print("Bucket 생성:", BUCKET_NAME)
else:
    print("Bucket 존재:", BUCKET_NAME)


# ==========================================
# 3. PDF → MinIO 업로드
# ==========================================

minio_client.fput_object(
    BUCKET_NAME,
    OBJECT_NAME,
    PDF_FILE
)

print("MinIO 업로드 완료:", OBJECT_NAME)


# ==========================================
# 4. MinIO → PDF 다운로드
# ==========================================

download_file = "downloaded_workplan.pdf"

minio_client.fget_object(
    BUCKET_NAME,
    OBJECT_NAME,
    download_file
)

print("MinIO 다운로드 완료:", download_file)


# ==========================================
# 5. PDF Parsing
# ==========================================

doc = pymupdf.open(download_file)

text = ""

for page in doc:
    text += page.get_text()

doc.close()

print("\n===== PDF TEXT =====")
print(text[:1000])


# ==========================================
# 6. Chunking
# ==========================================

def split_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = split_text(
    text,
    chunk_size=500,
    overlap=100
)

print("\nChunk 개수:", len(chunks))


# ==========================================
# 7. Ollama Embedding 생성
# ==========================================

def create_embedding(text):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"][0]


# ==========================================
# 8. Chroma Vector DB 생성
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# 기존 collection이 있을 경우 제거
try:
    chroma_client.delete_collection(
        name="documents"
    )
except Exception:
    pass


collection = chroma_client.create_collection(
    name="documents"
)


# ==========================================
# 9. Chunk → Embedding → Chroma 저장
# ==========================================

for i, chunk in enumerate(chunks):

    embedding = create_embedding(chunk)

    collection.add(

        ids=[
            f"chunk-{i}"
        ],

        documents=[
            chunk
        ],

        embeddings=[
            embedding
        ],

        metadatas=[
            {
                "source": OBJECT_NAME,
                "chunk": i
            }
        ]
    )

    print(
        f"Chunk {i} 저장 완료"
    )


print("\nVector DB 저장 완료")

# ==========================================
# 10. 반복 질문 / 응답
# ==========================================

print("\n====================================")
print("PDF RAG 질의응답을 시작합니다.")
print("종료하려면 'exit', 'quit', '종료'를 입력하세요.")
print("====================================")


while True:

    # --------------------------------------
    # 1. 사용자 질문
    # --------------------------------------

    question = input("\n질문을 입력하세요: ").strip()


    # --------------------------------------
    # 2. 종료 명령 확인
    # --------------------------------------

    if question.lower() in ["exit", "quit", "종료"]:
        print("\n프로그램을 종료합니다.")
        break


    # 빈 질문 처리
    if not question:
        print("질문을 입력해주세요.")
        continue


    # ======================================
    # 11. 질문 Embedding
    # ======================================

    question_embedding = create_embedding(
        question
    )


    # ======================================
    # 12. Similarity Search
    # ======================================

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=3
    )


    retrieved_chunks = results["documents"][0]


    print("\n===== 검색된 문서 =====")

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"\n--- 검색 결과 {i} ---"
        )

        print(chunk)


    # ======================================
    # 13. Context 생성
    # ======================================

    context = "\n\n".join(
        retrieved_chunks
    )


    # ======================================
    # 14. Prompt 생성
    # ======================================

    prompt = f"""
다음 문서를 참고하여 질문에 답하세요.

[문서]

{context}

[질문]

{question}

[답변 규칙]

1. 문서의 내용을 근거로 답하세요.
2. 문서에 없는 내용은 추측하지 마세요.
3. 문서에서 답을 찾을 수 없다면
   "제공된 문서에서 해당 내용을 찾을 수 없습니다."
   라고 답하세요.
4. 한국어로 명확하고 간결하게 답하세요.
"""


    # ======================================
    # 15. Gemma 호출
    # ======================================

    response = requests.post(

        f"{OLLAMA_URL}/api/generate",

        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )


    response.raise_for_status()

    answer = response.json()["response"]


    # ======================================
    # 16. 최종 답변
    # ======================================

    print("\n===== Answer =====")

    print(answer)