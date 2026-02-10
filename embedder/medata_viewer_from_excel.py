import sqlite3
import pandas as pd
import os
import json

# ==========================================
# 1. 설정 (내 환경에 맞게 수정)
# ==========================================
# DB 파일 경로 (폴더명/chroma.sqlite3)
DB_PATH = "./chroma_db/chroma.sqlite3" 
# 결과로 나올 파일 이름
OUTPUT_FILE = "metadata_view.csv"

# ==========================================
# 2. SQL 쿼리 (파일 안에 내장!)
# ==========================================
# 작성자님이 만드신 그 완벽한 쿼리를 여기에 넣었습니다.
QUERY = """
SELECT 
    id,
    MAX(CASE WHEN key = 'name' THEN string_value END) AS 상품명,
    MAX(CASE WHEN key = 'price' THEN int_value END) AS 가격,
    MAX(CASE WHEN key = 'maker' THEN string_value END) AS 제조사,
    MAX(CASE WHEN key = 'category' THEN string_value END) AS 카테고리,
    MAX(CASE WHEN key = 'image_path' THEN string_value END) AS 이미지경로,
    MAX(CASE WHEN key = 'product_url' THEN string_value END) AS 상품URL,
    MAX(CASE WHEN key = 'image_hash' THEN string_value END) AS 이미지해시,
    MAX(CASE WHEN key = 'created_at' THEN string_value END) AS 생성일,
    MAX(CASE WHEN key = 'updated_at' THEN string_value END) AS 수정일,
    MAX(CASE WHEN key = 'ocr_lines' THEN string_value END) AS OCR내용
FROM embedding_metadata
GROUP BY id;
"""


def _normalize_ocr_value(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)

    text = value.strip()
    if not text:
        return ""

    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception:
            return value

        if isinstance(parsed, list):
            parts = []
            for item in parsed:
                if isinstance(item, dict):
                    line_text = item.get("text")
                    if line_text:
                        parts.append(str(line_text))
                elif isinstance(item, str):
                    parts.append(item)
            return " | ".join(parts)

        if isinstance(parsed, dict):
            line_text = parsed.get("text")
            return str(line_text) if line_text else ""

    return value

def run_export():
    print(f"📂 DB 읽는 중... ({DB_PATH})")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ 에러: DB 파일을 찾을 수 없습니다. 경로를 확인해주세요: {DB_PATH}")
        return

    try:
        # 1. DB 연결 (sqlite3로 직접 연결)
        conn = sqlite3.connect(DB_PATH)
        
        # 2. 쿼리 실행 및 데이터프레임 변환
        df = pd.read_sql_query(QUERY, conn)
        conn.close()

        # 3. 데이터가 비어있는지 확인
        if df.empty:
            print("⚠️ 저장된 데이터가 없습니다.")
            return

        if "OCR내용" in df.columns:
            df["OCR내용"] = df["OCR내용"].apply(_normalize_ocr_value)

        # 4. CSV 파일로 저장 (엑셀에서 한글 안 깨지게 utf-8-sig 사용)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*40)
        print(f"🎉 성공! 데이터가 '{OUTPUT_FILE}'로 추출되었습니다.")
        print(f"📊 총 상품 수: {len(df)}개")
        print("="*40)
        
        # (선택) 상위 5개 미리보기 출력
        print("\n[미리보기]")
        print(df[['상품명', '가격', '제조사']].head().to_string())

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        input("엔터를 누르면 종료합니다...") # 에러 메시지 읽을 수 있게 대기

if __name__ == "__main__":
    run_export()
    # 윈도우에서 더블클릭 실행 시 창이 바로 꺼지는 것 방지
    os.system("pause")