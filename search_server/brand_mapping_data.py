# brand_data.py
# 공식적인 제조사 명칭 및 브랜드 매핑 데이터
# Key: 사용자가 입력할 수 있는 다양한 키워드 (소문자 영어, 한국어, 브랜드명 등)
# Value: DB에 실제로 저장된 'maker' 필드의 핵심 키워드 (일본어)

BRAND_MAPPING = {
    # 1. 닛신 (Nissin) - DB값: "日清食品", "日清食品株式会社"
    "nissin": "日清",
    "cup noodle": "日清",      # 대표 상품
    "ufo": "日清",             # 대표 상품
    "donbei": "日清",          # 대표 상품
    "どん兵衛": "日清",
    "닛신": "日清",
    "닛신식품": "日清",

    # 2. 토요수이산 (Toyo Suisan) - DB값: "東洋水産"
    # 해외에서는 'Maruchan(마루짱)' 브랜드로 더 유명함
    "toyo suisan": "東洋水産",
    "maruchan": "東洋水産",
    "red fox": "東洋水産",     # 赤いきつね (Red Fox)
    "green tanuki": "東洋水産", # 緑のたぬき (Green Tanuki)
    "마루짱": "東洋水産",
    "토요수이산": "東洋水産",
    "동양수산": "東洋水産",

    # 3. 산요식품 (Sanyo Foods) - DB값: "サンヨー食品"
    # 대표 브랜드: 삿포로 이치반 (Sapporo Ichiban)
    "sanyo foods": "サンヨー食品",
    "sapporo ichiban": "サンヨー食品",
    "cup star": "サンヨー食品",
    "삿포로이치반": "サンヨー食品",
    "산요식품": "サンヨー食品",

    # 4. 묘조식품 (Myojo Foods) - DB값: "明星食品"
    # 대표 브랜드: 잇페이짱 (Ippei-chan), 차루메라
    "myojo": "明星食品",
    "ippei": "明星食品",
    "charumera": "明星食品",
    "ippei-chan": "明星食品",
    "묘조": "明星食品",
    "명성식품": "明星食品",
    "잇페이짱": "明星食品",

    # 5. 에이스쿡 (Acecook) - DB값: "エースコック"
    # 대표 브랜드: 슈퍼컵 (Super Cup)
    "acecook": "エースコック",
    "super cup": "エースコック",
    "에이스쿡": "エースコック",
    "슈퍼컵": "エースコック",

    # 6. 마루카식품 (Maruka Foods) - DB값: "まるか食品"
    # 사명보다 '페양그(Peyoung)'라는 브랜드가 압도적으로 유명함
    "maruka": "まるか食品",
    "peyoung": "まるか食品",
    "maruka foods": "まるか食品",
    "페양그": "まるか食品",
    "마루카": "まるか食品",

    # 7. 야마다이 (Yamadai) - DB값: "ヤマダイ", "ヤマダイ株式会社"
    # 대표 브랜드: 뉴터치 (New Touch), 스고멘 (Sugomen)
    "yamadai": "ヤマダイ",
    "new touch": "ヤマダイ",
    "sugomen": "ヤマダイ",
    "야마다이": "ヤマダイ",
    "뉴터치": "ヤマダイ",

    # 8. 농심 (Nongshim) - DB값: "農心ジャパン"
    "nongshim": "農心",
    "shin ramyun": "農心",
    "neoguri": "農心",
    "농심": "農心",
    "신라면": "農心",
    "너구리": "農心",

    # 9. 삼양 (Samyang) - DB값: "三養ジャパン"
    # 대표 브랜드: 불닭볶음면 (Buldak)
    "samyang": "三養",
    "buldak": "三養",
    "삼양": "三養",
    "불닭": "三養",
    "불닭볶음면": "三養",

    # 10. 아지노모토 (Ajinomoto) - DB값: "アジノモト"
    "ajinomoto": "アジノモト",
    "yumyum": "アジノモト", # 얌얌(태국라면 브랜드 소유)
    "아지노모토": "アジノモト",

    # 11. 이치란 (Ichiran) - DB값: "一蘭"
    "ichiran": "一蘭",
    "이치란": "一蘭",

    # 12. 마루타이 (Marutai) - DB값: "マルタイ"
    # 후쿠오카의 봉지라면으로 유명
    "marutai": "マルタイ",
    "마루타이": "マルタイ",

    # 13. 히가시마루 (Higashimaru) - DB값: "ヒガシマル"
    "higashimaru": "ヒガシマル",
    "히가시마루": "ヒガシマル",

    # 14. 코쿠부 그룹 (Kokubu) - DB값: "国分グループ本社"
    # 브랜드: Tabete (타베테)
    "kokubu": "国分",
    "tabete": "国分",
    "코쿠부": "国分",
    "타베테": "国分",
    
    # 15. 도쿠시마 제분 (Tokushima Seifun) - DB값: "徳島製粉"
    # 브랜드: 킨짱 누들 (Kin-chan)
    "tokushima": "徳島製粉",
    "kin-chan": "徳島製粉",
    "kinchan": "徳島製粉",
    "도쿠시마": "徳島製粉",
    "킨짱": "徳島製粉",

    # 16. 나가타니엔 (Nagatanien) - DB값: "永谷園"
    "nagatanien": "永谷園",
    "나가타니엔": "永谷園",
    
    # 17. 마스모토 (Masumoto) - DB값: "桝元"
    # 미야자키 카라멘(매운면)으로 유명
    "masumoto": "桝元",
    "karamen": "桝元",
    "마스모토": "桝元",
}

def get_official_maker_name(query: str):
    """
    사용자 입력(영어, 한국어, 브랜드명)을 받아 
    DB 검색용 일본어 제조사명(일부분)을 반환합니다.
    """
    if not query:
        return None
    return BRAND_MAPPING.get(query.lower(), query)