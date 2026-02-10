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
    MAX(CASE WHEN key = 'updated_at' THEN string_value END) AS 수정일
FROM embedding_metadata
GROUP BY id;