import os
import json
import hashlib
from datetime import datetime, timezone
from PIL import Image
from sentence_transformers import SentenceTransformer

# Work around oneDNN/PIR issues on some Windows CPU installs.
os.environ.setdefault("FLAGS_use_onednn", "false")

from paddleocr import PaddleOCR
import chromadb

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
JSON_FILE = "../crawl/example.json"  # 최신 데이터 파일 경로
IMAGE_DIR = "../crawl/images"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rakuten_products"
DEBUG_OCR = True

def run_embedding():
    print("🚀 스마트 임베딩 시스템 가동 (중복 방지 & 이미지 검증 포함)...")

    # 1. DB 연결
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 2. 모델 로드
    print("📥 CLIP 모델 로딩 중...")
    model = SentenceTransformer('clip-ViT-B-32')

    print("📥 PaddleOCR 로딩 중...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='japan')

    # 3. JSON 데이터 로드
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except FileNotFoundError:
        print(f"❌ JSON 파일을 찾을 수 없습니다: {JSON_FILE}")
        return

    print(f"📦 처리 대상 상품: {len(products)}개")

    # 이미 DB에 저장된 ID 목록을 필요한 만큼만 조회합니다. (중복 방지용)
    # 전체 데이터를 통째로 가져오지 않고, 현재 처리 대상 ID만 배치 조회합니다.
    existing_ids = set()
    existing_meta_by_id = {}
    candidate_ids = [str(item['id']) for item in products if 'id' in item]
    chunk_size = 1000
    for i in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[i:i + chunk_size]
        try:
            found = collection.get(ids=chunk, include=["ids", "metadatas"])
            ids = found.get('ids', [])
            existing_ids.update(ids)
            for idx, meta in enumerate(found.get('metadatas', [])):
                if idx < len(ids):
                    existing_meta_by_id[ids[idx]] = meta or {}
        except Exception:
            # 일부 버전에서는 include 인자가 제한될 수 있어 안전하게 재시도
            found = collection.get(ids=chunk)
            ids = found.get('ids', [])
            existing_ids.update(ids)
            for idx, meta in enumerate(found.get('metadatas', [])):
                if idx < len(ids):
                    existing_meta_by_id[ids[idx]] = meta or {}

    print(f"💾 현재 DB 저장된 상품 수(대상 기준): {len(existing_ids)}개")

    # 배치 처리를 위한 임시 저장소
    batch_ids = []
    batch_embeddings = []
    batch_metadatas = []
    
    new_count = 0
    update_count = 0
    skip_count = 0
    error_count = 0

    # 4. 이미지 인덱스 생성 (O(이미지 수) 한 번만)
    image_index = {}
    for f_name in os.listdir(IMAGE_DIR):
        if f_name.endswith('.jpg') or f_name.endswith('.png'):
            product_id = f_name.split('_', 1)[0].split('.', 1)[0]
            if product_id not in image_index:
                image_index[product_id] = f_name

    # 5. 데이터 순회
    debug_printed = 0
    for idx, item in enumerate(products):
        product_id = str(item['id'])
        
        # [체크 2] 이미지가 실제로 존재하는가?
        # JSON에는 파일명이 없으므로 ID로 시작하는 이미지 파일을 찾습니다.
        image_filename = image_index.get(product_id)
        
        if not image_filename:
            # print(f"   ⚠️ 이미지 파일 없음 (Skip): {item['name']}")
            error_count += 1
            continue

        image_path = os.path.join(IMAGE_DIR, image_filename)
        image_hash = _hash_file(image_path)

        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            existing_meta = existing_meta_by_id.get(product_id)
            created_at = (existing_meta or {}).get("created_at") or now_iso
            metadata_base = {
                "name": item['name'],
                "price": item['price'],
                "maker": item['maker'],
                "category": item['category'],
                "image_path": image_path, # 나중에 웹에서 보여줄 때 필요
                "product_url": item['product_url'],
                "image_hash": image_hash,
                "created_at": created_at,
                "updated_at": now_iso,
            }

            if product_id in existing_ids:
                changed = False
                if existing_meta:
                    compare_keys = [
                        "name",
                        "price",
                        "maker",
                        "category",
                        "product_url",
                        "image_hash",
                    ]
                    for key in compare_keys:
                        if existing_meta.get(key) != metadata_base.get(key):
                            changed = True
                            break
                else:
                    changed = True

                if not changed:
                    skip_count += 1
                    continue

                # [체크 3] 이미지가 깨지지 않고 열리는가? (Validation)
                with Image.open(image_path) as img:
                    # 이미지를 모델이 이해할 수 있게 벡터로 변환
                    vector = model.encode(img).tolist()

                ocr_lines = _run_ocr(ocr, image_path)
                if DEBUG_OCR:
                    print(f"   🧪 OCR 디버그: {product_id} lines={len(ocr_lines)}")
                    for line in ocr_lines[:3]:
                        print(f"      - {line}")
                    debug_printed += 1
                metadata = dict(metadata_base)
                metadata["ocr_lines"] = _serialize_ocr_lines(ocr_lines)

                batch_ids.append(product_id)
                batch_embeddings.append(vector)
                batch_metadatas.append(metadata)
                update_count += 1
                print(f"   🔁 [갱신] {item['name'][:15]}... 업데이트됨")
            else:
                # [체크 3] 이미지가 깨지지 않고 열리는가? (Validation)
                with Image.open(image_path) as img:
                    # 이미지를 모델이 이해할 수 있게 벡터로 변환
                    vector = model.encode(img).tolist()

                ocr_lines = _run_ocr(ocr, image_path)
                if DEBUG_OCR:
                    print(f"   🧪 OCR 디버그: {product_id} lines={len(ocr_lines)}")
                    for line in ocr_lines[:3]:
                        print(f"      - {line}")
                    debug_printed += 1
                metadata = dict(metadata_base)
                metadata["ocr_lines"] = _serialize_ocr_lines(ocr_lines)

                batch_ids.append(product_id)
                batch_embeddings.append(vector)
                batch_metadatas.append(metadata)
                new_count += 1
                print(f"   ✅ [신규] {item['name'][:15]}... 추가됨")

        except Exception as e:
            print(f"   ❌ 이미지 손상 또는 에러 ({item['name']}): {e}")
            error_count += 1
            continue

    # 5. DB에 저장 (신규/갱신 데이터가 있을 때만)
    if batch_ids:
        print(f"\n📥 신규/갱신 데이터 {len(batch_ids)}개를 DB에 저장합니다...")
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
        print("🎉 저장 완료!")
    else:
        print("\n✨ 추가할 신규 데이터가 없습니다.")

    # 6. 최종 리포트
    print("\n" + "="*30)
    print(f"📊 처리 결과 요약")
    print(f"   - 총 데이터: {len(products)}")
    print(f"   - 이미 존재함 (Skip): {skip_count}")
    print(f"   - 이미지 오류/없음: {error_count}")
    print(f"   - 새로 추가됨: {new_count}")
    print(f"   - 갱신됨: {update_count}")
    print("="*30)

def _run_ocr(ocr: PaddleOCR, image_path: str) -> list:
    lines = []

    try:
        result = ocr.predict(input=image_path)
        for res in result:
            lines.extend(_extract_ocr_lines(res))
    except Exception:
        return []

    return lines


def _extract_ocr_lines(result) -> list:
    if isinstance(result, dict):
        texts = result.get("rec_texts") or result.get("texts")
        scores = result.get("rec_scores") or result.get("scores")
        if texts:
            lines = []
            for text, score in zip(texts, scores or [None] * len(texts)):
                clean_text = str(text).strip()
                clean_score = float(score) if score is not None else None
                if not clean_text:
                    continue
                if clean_score is not None and clean_score <= 0.0:
                    continue
                lines.append({"text": clean_text, "score": clean_score})
            return lines
        if "text" in result:
            return [{"text": result.get("text"), "score": result.get("score")}]
        return []

    for attr in ("to_json", "json"):
        if hasattr(result, attr):
            try:
                data = getattr(result, attr)()
                return _extract_ocr_lines(data)
            except Exception:
                pass

    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        lines = []
        for line in result:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                payload = line[1]
                if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                    text = str(payload[0]).strip()
                    score = float(payload[1]) if payload[1] is not None else None
                    if not text:
                        continue
                    if score is not None and score <= 0.0:
                        continue
                    lines.append({"text": text, "score": score})
        return lines

    return []


def _serialize_ocr_lines(lines: list) -> str:
    if not lines:
        return ""
    try:
        return json.dumps(lines, ensure_ascii=False)
    except Exception:
        return ""


def _hash_file(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    run_embedding()