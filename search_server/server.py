from brand_mapping_data import BRAND_MAPPING, get_official_maker_name
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
import chromadb
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
import numpy as np

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
DB_PATH = "../embedder/chroma_db"
COLLECTION_NAME = "rakuten_products"

app = FastAPI(title="Mono-Log AI Server", description="이미지 중심 하이브리드 검색 엔진")

print("⏳ 시스템 초기화 중...")
# (2) CLIP 모델 로드
model = SentenceTransformer('clip-ViT-B-32')

# (3) OCR 엔진 로드 (서버 켤 때 한 번만!)
ocr = PaddleOCR(use_angle_cls=True, lang='japan', show_log=False)

# (4) DB 연결
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print("✅ 서버 준비 완료!")

# ==========================================
# 2. 핵심 로직: 가중치 계산 함수
# ==========================================
def calculate_final_score(item, user_inputs):
    # 1. 기본 점수: 이미지 벡터 유사도 (0.0 ~ 1.0)
    base_score = item['similarity_score']
    
    # 2. 가중치 점수 합산
    bonus_score = 0.0
    
    # [필터 1] 브랜드 (가장 강력한 힌트)
    user_brand = user_inputs.get('brand')
    if user_brand:
        # 입력값을 소문자로 바꾸고, 매핑된 일본어가 있으면 가져옴
        target_maker_keyword = get_official_maker_name(user_brand)
        
        # DB의 제조사(maker) 정보와 비교 (부분 일치)
        # 예: 'nissin' -> '日清' 반환 -> DB의 '日清食品'에 포함되므로 OK!
        if target_maker_keyword in item.get('maker', ''):
            bonus_score += 0.15

    # [필터 2] 가격 (비슷하면 점수)
    user_price = user_inputs.get('price')
    if user_price:
        try:
            target_price = int(user_price)
            item_price = int(item.get('price', 0))
            # 가격 차이가 작을수록 점수 높음 (최대 0.1점)
            diff = abs(target_price - item_price)
            if diff <= 50: # 50엔 차이 이내면 만점
                bonus_score += 0.1
            elif diff <= 200: # 200엔 차이 이내면 부분 점수
                bonus_score += 0.05
        except:
            pass # 가격 입력이 숫자가 아니면 무시

    # [필터 3] 제품명/키워드 (텍스트 포함 여부)
    keywords = [user_inputs.get('name'), user_inputs.get('keyword')]
    for kw in keywords:
        if kw:
            # 상품명이나 OCR 텍스트에 키워드가 있으면 가산점
            full_text = (item.get('name', '') + item.get('ocr_text', '')).lower()
            if kw.lower() in full_text:
                bonus_score += 0.05

    # 최종 점수 반환 (1.0을 넘을 수도 있음)
    return base_score + bonus_score

# ==========================================
# 3. 메인 API: 이미지 검색 (+ 필터)
# ==========================================
@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),      # 필수: 이미지 파일
    name: Optional[str] = Form(None),  # 선택: 제품명
    price: Optional[str] = Form(None), # 선택: 가격
    brand: Optional[str] = Form(None), # 선택: 브랜드
    keyword: Optional[str] = Form(None)# 선택: 기타 키워드
):
    try:
        # 1. 이미지 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 2. [정밀 모드] 업로드된 이미지 OCR 수행 (속도 희생, 정확도 UP)
        ocr_result = ocr.ocr(np.array(image), cls=True)
        detected_texts = []
        if ocr_result and ocr_result[0]:
            detected_texts = [line[1][0] for line in ocr_result[0]]
        full_ocr_text = " ".join(detected_texts)
        
        print(f"📸 OCR 감지된 텍스트: {full_ocr_text}")

        # 3. 이미지 벡터 변환
        query_vector = model.encode(image).tolist()

        # 4. 1차 후보군 검색 (벡터로 상위 50개 가져옴 - 넉넉하게)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=50, 
            include=["metadatas", "distances", "ids"]
        )

        # 5. 2차 재순위화 (Re-ranking)
        candidates = []
        user_inputs = {"name": name, "price": price, "brand": brand, "keyword": keyword}
        
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for i in range(len(ids)):
            item = metadatas[i]
            item['id'] = ids[i]
            item['similarity_score'] = 1 - distances[i] # 기본 벡터 점수
            
            # 여기서 가중치 계산!
            final_score = calculate_final_score(item, user_inputs)
            
            # (옵션) 업로드된 이미지의 OCR 텍스트와 DB 데이터 매칭 보너스
            # 예: 사진에 'BIG'이라 써있고, DB 상품명에도 'BIG'이 있으면 추가 점수
            for text in detected_texts:
                if len(text) > 2 and text in (item.get('name', '') + item.get('ocr_text', '')):
                     final_score += 0.05

            item['final_score'] = final_score
            candidates.append(item)

        # 6. 점수 높은 순으로 정렬 후 상위 20개 자르기
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        top_20 = candidates[:20]

        return {
            "status": "success",
            "detected_text": full_ocr_text, # 디버깅용: OCR이 뭘 읽었는지 알려줌
            "results": top_20
        }

    except Exception as e:
        print(f"❌ 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))