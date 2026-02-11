from brand_mapping_data import BRAND_MAPPING, get_official_maker_name
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
import numpy as np
import uvicorn
import traceback
from pathlib import Path
from scipy.spatial.distance import cosine
from difflib import SequenceMatcher

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
DB_PATH = "../embedder/chroma_db"
COLLECTION_NAME = "rakuten_products"
RESPONSE_JSON_PATH = os.path.join(os.path.dirname(__file__), "response.json")
DEBUG_SCORING = True
DEBUG_SCORING_LIMIT = 5

def similarity(a: str, b: str) -> float:
    """두 문자열의 유사도 계산 (0.0~1.0)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_weights() -> dict:
    _load_env_file()
    weights = {
        "base_score_weight": _get_float("MONOLOG_BASE_SCORE_WEIGHT", 0.3),
        "brand_bonus": _get_float("MONOLOG_BRAND_BONUS", 0.15),
        "name_bonus": _get_float("MONOLOG_NAME_BONUS", 0.15),
        "ocr_threshold_minimum": _get_int("MONOLOG_OCR_THRESHOLD_MINIMUM", 10),
        "ocr_threshold_fair": _get_int("MONOLOG_OCR_THRESHOLD_FAIR", 30),
        "ocr_threshold_good": _get_int("MONOLOG_OCR_THRESHOLD_GOOD", 60),
        "ocr_bonus_poor": _get_float("MONOLOG_OCR_BONUS_POOR", 0.0),
        "ocr_bonus_fair": _get_float("MONOLOG_OCR_BONUS_FAIR", 0.025),
        "ocr_bonus_good": _get_float("MONOLOG_OCR_BONUS_GOOD", 0.05),
        "price_bonus_10pct": _get_float("MONOLOG_PRICE_BONUS_10PCT", 0.10),
        "price_bonus_20pct": _get_float("MONOLOG_PRICE_BONUS_20PCT", 0.05),
        "price_threshold_10pct": _get_int("MONOLOG_PRICE_THRESHOLD_10PCT", 10),
        "price_threshold_20pct": _get_int("MONOLOG_PRICE_THRESHOLD_20PCT", 20),
        "similarity_threshold": _get_float("MONOLOG_SIMILARITY_THRESHOLD", 0.8),
    }
    
    # 최대 가능한 보너스 합계 계산 (모든 조건 만족)
    weights["max_bonus"] = (
        weights["brand_bonus"] +  # 브랜드 일치
        weights["name_bonus"] +  # 제품명 일치
        weights["ocr_bonus_good"] +  # OCR 우수
        weights["price_bonus_10pct"]  # 가격 10% 이내
    )
    
    return weights


WEIGHTS = load_weights()
SIMILARITY_THRESHOLD = WEIGHTS.get("similarity_threshold", 0.8)

app = FastAPI(title="Mono-Log AI Server", description="이미지 중심 하이브리드 검색 엔진")

# 모델은 첫 요청 시 로드 (lazy loading)
model = None
ocr = None
client = None
collection = None

def initialize_models():
    """모델 초기화 (첫 요청 시 한 번만 실행)"""
    global model, ocr, client, collection
    if model is None:
        print("⏳ 시스템 초기화 중...")
        import time
        
        start = time.time()
        model = SentenceTransformer('clip-ViT-B-32')
        print(f"  ✓ CLIP 모델 로드: {time.time()-start:.2f}초")
        
        start = time.time()
        ocr = PaddleOCR(use_textline_orientation=True, lang='japan')
        print(f"  ✓ OCR 엔진 로드: {time.time()-start:.2f}초")
        
        start = time.time()
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"  ✓ DB 연결: {time.time()-start:.2f}초")
        
        print("✅ 서버 준비 완료!")

# ==========================================
# 2. 핵심 로직: 가중치 계산 함수
# ==========================================
def calculate_final_score(item, user_inputs, detected_texts=None):
    # 1. 기본 점수: 이미지 벡터 유사도 (0.0 ~ 1.0)
    base_score = item['similarity_score']
    
    # 2. 가중치 점수 합산
    bonus_score = 0.0
    
    # [필터 1] 브랜드 (가장 강력한 힌트)
    brand_matched = False
    user_brand = user_inputs.get('brand')
    if user_brand:
        # 입력값을 소문자로 바꾸고, 매핑된 일본어가 있으면 가져옴
        target_maker_keyword = get_official_maker_name(user_brand)
        
        # DB의 제조사(maker) 정보와 비교 (부분 일치)
        # 예: 'nissin' -> '日清' 반환 -> DB의 '日清食品'에 포함되므로 OK!
        if target_maker_keyword in item.get('maker', ''):
            bonus_score += WEIGHTS["brand_bonus"]
            brand_matched = True
    
    # [필터 1-2] OCR에서 브랜드명 발견 (user_brand 없어도 작동)
    if not brand_matched and detected_texts:
        detected_full = ' '.join(detected_texts)
        item_maker = item.get('maker', '')
        # 완전 일치 체크
        if item_maker and item_maker in detected_full:
            bonus_score += WEIGHTS["brand_bonus"]
            brand_matched = True
        # 유사도 체크 (OCR 오류 대응: HISSIN vs NISSIN)
        elif item_maker:
            for word in detected_texts:
                if len(word) >= 3 and similarity(word, item_maker) >= SIMILARITY_THRESHOLD:
                    bonus_score += WEIGHTS["brand_bonus"]
                    brand_matched = True
                    break

    # [필터 2] 제품명 (name 파라미터 또는 OCR 자동 감지)
    name_matched = False
    user_name = user_inputs.get('name')
    if user_name:
        # API name 입력: DB name에 포함되는지 확인
        if user_name.lower() in item.get('name', '').lower():
            bonus_score += WEIGHTS["name_bonus"]
            name_matched = True
    
    # OCR에서 제품명 자동 감지
    if not name_matched and detected_texts:
        detected_full = ' '.join(detected_texts)
        item_name = item.get('name', '')
        # 완전 일치 체크
        if item_name and (item_name in detected_full or 
                          any(word in item_name for word in detected_texts if len(word) >= 2)):
            bonus_score += WEIGHTS["name_bonus"]
            name_matched = True
        # 유사도 체크 (OCR 오류 대응)
        elif item_name:
            for word in detected_texts:
                if len(word) >= 3 and similarity(word, item_name) >= SIMILARITY_THRESHOLD:
                    bonus_score += WEIGHTS["name_bonus"]
                    name_matched = True
                    break

    # [필터 3] 가격 (비슷하면 점수)
    user_price = user_inputs.get('price')
    if user_price:
        try:
            target_price = int(user_price)
            item_price = int(item.get('price', 0))
            diff = abs(target_price - item_price)
            price_ratio = (diff / target_price * 100) if target_price > 0 else 100
            if price_ratio <= WEIGHTS["price_threshold_10pct"]:
                bonus_score += WEIGHTS["price_bonus_10pct"]
            elif price_ratio <= WEIGHTS["price_threshold_20pct"]:
                bonus_score += WEIGHTS["price_bonus_20pct"]
        except:
            pass

    # [필터 4] OCR 일치율 (업로드 이미지 OCR과 DB ocr_lines 비교)
    if detected_texts:
        _, ocr_bonus = _calculate_ocr_match_score(detected_texts, item, debug_ocr=False)
        bonus_score += ocr_bonus

    # 정규화: 0~1 범위로 변환
    # 최대 가능 점수 = base(1.0) * BASE_WEIGHT + max_bonus * BONUS_WEIGHT
    bonus_weight = 1.0 - WEIGHTS["base_score_weight"]
    max_possible = 1.0 * WEIGHTS["base_score_weight"] + WEIGHTS["max_bonus"] * bonus_weight
    
    final_score = base_score * WEIGHTS["base_score_weight"] + bonus_score * bonus_weight
    normalized_score = final_score / max_possible if max_possible > 0 else 0.0
    
    return min(normalized_score, 1.0)  # 1.0을 넘지 않도록


def calculate_score_with_debug(item, user_inputs, detected_texts=None, debug_ocr=False):
    base_score = item['similarity_score']
    bonus_score = 0.0
    reasons = []
    breakdown = {
        "brand": 0.0,
        "name": 0.0,
        "price": 0.0,
        "ocr": 0.0,
        "ocr_ratio": 0.0
    }

    brand_matched = False
    user_brand = user_inputs.get('brand')
    if user_brand:
        target_maker_keyword = get_official_maker_name(user_brand)
        if target_maker_keyword in item.get('maker', ''):
            bonus_score += WEIGHTS["brand_bonus"]
            breakdown["brand"] = WEIGHTS["brand_bonus"]
            reasons.append(f"brand:+{WEIGHTS['brand_bonus']:.2f}({target_maker_keyword})")
            brand_matched = True
    
    # OCR에서 브랜드명 발견
    if not brand_matched and detected_texts:
        detected_full = ' '.join(detected_texts)
        item_maker = item.get('maker', '')
        matched_word = None
        match_type = None
        # 완전 일치
        if item_maker and item_maker in detected_full:
            bonus_score += WEIGHTS["brand_bonus"]
            breakdown["brand"] = WEIGHTS["brand_bonus"]
            matched_word = item_maker
            match_type = "exact"
            brand_matched = True
        # 유사도 체크
        elif item_maker:
            for word in detected_texts:
                if len(word) >= 3:
                    sim = similarity(word, item_maker)
                    if sim >= SIMILARITY_THRESHOLD:
                        bonus_score += WEIGHTS["brand_bonus"]
                        breakdown["brand"] = WEIGHTS["brand_bonus"]
                        matched_word = f"{word}≈{item_maker}"
                        match_type = f"{sim:.0%}"
                        brand_matched = True
                        break
        if matched_word:
            reasons.append(f"brand:+{WEIGHTS['brand_bonus']:.2f}(OCR:{matched_word})")

    # 제품명 (name 파라미터 또는 OCR 자동 감지)
    name_matched = False
    user_name = user_inputs.get('name')
    if user_name:
        if user_name.lower() in item.get('name', '').lower():
            bonus_score += WEIGHTS["name_bonus"]
            breakdown["name"] = WEIGHTS["name_bonus"]
            reasons.append(f"name:+{WEIGHTS['name_bonus']:.2f}(API:{user_name})")
            name_matched = True
    
    # OCR에서 제품명 자동 감지
    if not name_matched and detected_texts:
        detected_full = ' '.join(detected_texts)
        item_name = item.get('name', '')
        matched_word = None
        # 완전 일치
        if item_name and (item_name in detected_full or 
                          any(word in item_name for word in detected_texts if len(word) >= 2)):
            bonus_score += WEIGHTS["name_bonus"]
            breakdown["name"] = WEIGHTS["name_bonus"]
            matched_word = next((w for w in detected_texts if len(w) >= 2 and w in item_name), item_name[:10])
            name_matched = True
        # 유사도 체크
        elif item_name:
            for word in detected_texts:
                if len(word) >= 3:
                    sim = similarity(word, item_name)
                    if sim >= SIMILARITY_THRESHOLD:
                        bonus_score += WEIGHTS["name_bonus"]
                        breakdown["name"] = WEIGHTS["name_bonus"]
                        matched_word = f"{word}≈{item_name[:10]}"
                        name_matched = True
                        break
        if matched_word:
            reasons.append(f"name:+{WEIGHTS['name_bonus']:.2f}(OCR:{matched_word})")

    # 가격
    user_price = user_inputs.get('price')
    if user_price:
        try:
            target_price = int(user_price)
            item_price = int(item.get('price', 0))
            diff = abs(target_price - item_price)
            price_ratio = (diff / target_price * 100) if target_price > 0 else 100
            if price_ratio <= WEIGHTS["price_threshold_10pct"]:
                bonus_score += WEIGHTS["price_bonus_10pct"]
                breakdown["price"] = WEIGHTS["price_bonus_10pct"]
                reasons.append(f"price:+{WEIGHTS['price_bonus_10pct']:.2f}(<={WEIGHTS['price_threshold_10pct']:.0f}%)")
            elif price_ratio <= WEIGHTS["price_threshold_20pct"]:
                bonus_score += WEIGHTS["price_bonus_20pct"]
                breakdown["price"] = WEIGHTS["price_bonus_20pct"]
                reasons.append(f"price:+{WEIGHTS['price_bonus_20pct']:.2f}(<={WEIGHTS['price_threshold_20pct']:.0f}%)")
        except Exception:
            pass

    # OCR 일치율 (업로드 이미지 vs DB 메타데이터)
    if detected_texts:
        match_ratio, ocr_bonus = _calculate_ocr_match_score(
            detected_texts,
            item,
            debug_ocr=debug_ocr,
        )
        breakdown["ocr_ratio"] = match_ratio
        if ocr_bonus > 0:  # 최소치 이상일 때만
            bonus_score += ocr_bonus
            breakdown["ocr"] = ocr_bonus
            if match_ratio >= WEIGHTS["ocr_threshold_good"]:
                level = "우수"
            elif match_ratio >= WEIGHTS["ocr_threshold_fair"]:
                level = "보통"
            else:
                level = "미흡"
            reasons.append(f"ocr:+{ocr_bonus:.2f}({match_ratio:.0f}%-{level})")
        elif match_ratio > 0:
            # 최소치 미만: 보너스 없지만 일치율 기록
            reasons.append(f"ocr:+0.00({match_ratio:.0f}%-미흡,최소치미만)")

    # 정규화: 0~1 범위로 변환
    bonus_weight = 1.0 - WEIGHTS["base_score_weight"]
    max_possible = 1.0 * WEIGHTS["base_score_weight"] + WEIGHTS["max_bonus"] * bonus_weight
    
    final_score = base_score * WEIGHTS["base_score_weight"] + bonus_score * bonus_weight
    normalized_score = final_score / max_possible if max_possible > 0 else 0.0
    normalized_score = min(normalized_score, 1.0)  # 1.0을 넘지 않도록
    
    return normalized_score, reasons, breakdown


def _extract_texts(res):
    if isinstance(res, dict):
        texts = res.get("rec_texts") or res.get("texts")
        if texts:
            return list(texts)
        if "text" in res:
            return [res.get("text")]
        return []

    for attr in ("to_json", "json"):
        if hasattr(res, attr):
            try:
                data = getattr(res, attr)()
                return _extract_texts(data)
            except Exception:
                pass

    if isinstance(res, list):
        texts = []
        for line in res:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                payload = line[1]
                if isinstance(payload, (list, tuple)) and len(payload) >= 1:
                    text = str(payload[0]).strip()
                    if text:
                        texts.append(text)
        return texts

    return []


def _calculate_ocr_match_score(detected_texts, item, debug_ocr=False):
    """
    업로드 이미지의 OCR 텍스트와 DB 상품 정보(name, maker, ocr_lines)의 일치율 계산
    반환: (일치율%, 보너스 점수)
    """
    # DB에서 텍스트 추출
    db_texts = []
    
    # name과 maker 추가
    if item.get('name'):
        db_texts.extend(item['name'].split())
    if item.get('maker'):
        db_texts.extend(item['maker'].split())
    
    # ocr_lines 파싱 (JSON 문자열)
    ocr_lines_str = item.get('ocr_lines', '[]')
    try:
        ocr_lines = json.loads(ocr_lines_str)
        for line in ocr_lines:
            if isinstance(line, dict) and 'text' in line:
                db_texts.extend(line['text'].split())
    except:
        pass
    
    # 겹치는 단어 계산 (완전 일치 + 유사도)
    detected_set = set(w.lower() for w in detected_texts if w)
    db_set = set(w.lower() for w in db_texts if w)
    
    # 완전 일치
    exact_overlap = detected_set & db_set
    overlap_count = len(exact_overlap)
    
    # 유사도 매칭 (완전 일치 못한 것들끼리)
    remaining_detected = detected_set - exact_overlap
    remaining_db = db_set - exact_overlap
    fuzzy_matches = []
    
    for det_word in remaining_detected:
        if len(det_word) < 3:  # 너무 짧은 단어는 skip
            continue
        for db_word in remaining_db:
            if len(db_word) < 3:
                continue
            sim = similarity(det_word, db_word)
            if sim >= SIMILARITY_THRESHOLD:
                fuzzy_matches.append((det_word, db_word, sim))
                overlap_count += 1
                remaining_db.discard(db_word)  # 중복 매칭 방지
                break
    
    # 🔍 DEBUG: OCR 매칭 과정 출력
    if debug_ocr:
        print(f"    🔍 OCR DEBUG for {item.get('name', 'Unknown')[:30]}")
        print(f"       Detected: {list(detected_set)[:5]}... (total: {len(detected_set)})")
        print(f"       DB: {list(db_set)[:5]}... (total: {len(db_set)})")
        print(f"       Exact match: {exact_overlap}")
        if fuzzy_matches:
            print(f"       Fuzzy match: {[(d, b, f'{s:.0%}') for d, b, s in fuzzy_matches[:3]]}")
    
    if not detected_set or not db_set:
        return 0.0, 0.0
    
    overlap = overlap_count
    total = max(len(detected_set), len(db_set))
    
    match_ratio = (overlap / total * 100) if total > 0 else 0.0
    
    # 3단계 구간 (임계값은 .env에서 로드)
    # 최소치 미만이면 보너스 미지급
    if match_ratio >= WEIGHTS["ocr_threshold_good"]:
        return match_ratio, WEIGHTS["ocr_bonus_good"]  # 우수
    elif match_ratio >= WEIGHTS["ocr_threshold_fair"]:
        return match_ratio, WEIGHTS["ocr_bonus_fair"]  # 보통
    elif match_ratio >= WEIGHTS["ocr_threshold_minimum"]:
        return match_ratio, WEIGHTS["ocr_bonus_poor"]  # 미흡
    else:
        return match_ratio, 0.0  # 최소치 미만: 보너스 없음

# ==========================================
# 3. 메인 API: 이미지 검색 (+ 필터)
# ==========================================
@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),      # 필수: 이미지 파일
    name: Optional[str] = Form(None),  # 선택: 제품명
    price: Optional[str] = Form(None), # 선택: 가격
    brand: Optional[str] = Form(None)  # 선택: 브랜드
):
    # 모델 초기화 (첫 요청 시 한 번만)
    initialize_models()
    
    try:
        # 1. 이미지 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 2. [정밀 모드] 업로드된 이미지 OCR 수행 (속도 희생, 정확도 UP)
        ocr_result = ocr.predict(input=np.array(image))
        detected_texts = []
        for res in ocr_result:
            detected_texts.extend(_extract_texts(res))
        full_ocr_text = " ".join(detected_texts)
        
        print(f"📸 OCR 감지된 텍스트: {full_ocr_text}")

        # 3. 이미지 벡터 변환
        query_vector = model.encode(image).tolist()

        # 4. 1차 후보군 검색 (벡터로 상위 50개 가져옴 - 넉넉하게)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=50,
            include=["metadatas", "distances", "embeddings"]
        )

        if DEBUG_SCORING:
            print(
                "DEBUG query results:",
                {
                    "ids": len(results.get("ids", [])),
                    "metadatas": len(results.get("metadatas", [])),
                    "distances": len(results.get("distances", [])),
                },
            )

        # 5. 2차 재순위화 (Re-ranking)
        candidates = []
        user_inputs = {"name": name, "price": price, "brand": brand}
        
        ids_list = results.get('ids', [])
        metas_list = results.get('metadatas', [])
        dists_list = results.get('distances', [])
        embeddings_list = results.get('embeddings', [])

        if not ids_list or not ids_list[0]:
            return {
                "status": "success",
                "detected_text": full_ocr_text,
                "results": []
            }

        ids = ids_list[0]
        metadatas = metas_list[0] if metas_list else []
        distances = dists_list[0] if dists_list else []
        embeddings = embeddings_list[0] if embeddings_list else []

        if DEBUG_SCORING:
            print(
                "DEBUG first batch sizes:",
                {
                    "ids": len(ids),
                    "metadatas": len(metadatas),
                    "distances": len(distances),
                    "embeddings": len(embeddings),
                },
            )

        debug_scored = 0
        for item_id, meta, dist, embedding in zip(ids, metadatas, distances, embeddings):
            item = meta
            item['id'] = item_id
            # Cosine similarity (0~1 범위)
            cosine_dist = cosine(query_vector, embedding)
            item['similarity_score'] = 1 - cosine_dist
            
            # 여기서 가중치 계산! (detected_texts 포함)
            if DEBUG_SCORING:
                final_score, reasons, breakdown = calculate_score_with_debug(
                    item,
                    user_inputs,
                    detected_texts,
                    debug_ocr=debug_scored < DEBUG_SCORING_LIMIT,
                )
            else:
                final_score = calculate_final_score(item, user_inputs, detected_texts)
                reasons = []
                breakdown = {}

            if DEBUG_SCORING and debug_scored < DEBUG_SCORING_LIMIT:
                print("=" * 80)
                print(f"🔍 DEBUG [{debug_scored + 1}/{DEBUG_SCORING_LIMIT}] - {item.get('name', 'Unknown')}")
                print(f"📦 ID: {item_id}")
                print(f"🏢 Maker: {item.get('maker', 'N/A')}")
                print(f"💰 Price: {item.get('price', 'N/A')}")
                print("-" * 80)
                print(f"📊 Base Score (벡터 유사도): {item['similarity_score']:.4f}")
                print(f"🎁 Bonus Breakdown:")
                print(f"   • Brand:   +{breakdown.get('brand', 0.0):.3f}")
                print(f"   • Name:    +{breakdown.get('name', 0.0):.3f}")
                print(f"   • Price:   +{breakdown.get('price', 0.0):.3f}")
                print(f"   • OCR:     +{breakdown.get('ocr', 0.0):.3f} (일치율: {breakdown.get('ocr_ratio', 0.0):.1f}%)")
                print(f"   💡 Total Bonus: {sum([breakdown.get('brand', 0), breakdown.get('name', 0), breakdown.get('price', 0), breakdown.get('ocr', 0)]):.3f}")
                print("-" * 80)
                print(f"⭐ Final Score (정규화): {final_score:.4f}")
                if reasons:
                    print(f"📝 Reasons: {' | '.join(reasons)}")
                print("=" * 80)
                print()
                debug_scored += 1

            item['final_score'] = final_score
            candidates.append(item)

        # 6. 점수 높은 순으로 정렬 후 상위 20개 자르기
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        top_20 = candidates[:20]

        response_payload = {
            "status": "success",
            "detected_text": full_ocr_text, # 디버깅용: OCR이 뭘 읽었는지 알려줌
            "results": top_20
        }

        with open(RESPONSE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(response_payload, f, ensure_ascii=False, indent=2)

        return response_payload

    except Exception as e:
        print("❌ 에러 발생:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import sys
    # reload 모드에서는 메인 프로세스에서 불필요한 모델 로드 방지
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        reload_excludes=["*.pyc", "__pycache__"]
    )