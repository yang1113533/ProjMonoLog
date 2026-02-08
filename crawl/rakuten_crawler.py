import os
import json
import time
import re
import requests
import sys
from datetime import datetime
from playwright.sync_api import sync_playwright

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
DEFAULT_URL = "https://sm.rakuten.co.jp/search/200029"

# [수정 3] 수집할 페이지 구간 설정 (예: 2페이지부터 5페이지까지)
START_PAGE = 1
MAX_PAGES = 99

IMAGE_DIR = "images"
NAV_TIMEOUT_MS = 30000
LIST_TIMEOUT_MS = 15000
NAV_RETRIES = 2

# [수정 2] 파일명에 시/분/초 추가 (예: 20240208_153022_products.json)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_FILE = f"{current_time}_products.json"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# ==========================================
# 2. 헬퍼 함수
# ==========================================
def clean_text(text):
    """텍스트 정리"""
    if not text: return "Unknown"
    return text.strip().replace("\n", "").replace("\t", "")

def get_high_res_url(img_url):
    """고해상도 이미지 URL 변환 및 https 프로토콜 추가"""
    if not img_url: return ""
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    return img_url.split("?")[0]

def download_image(url, filename):
    """이미지 다운로드"""
    if not url: return False
    if url.startswith("//"):
        url = "https:" + url
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(IMAGE_DIR, filename), 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

def navigate_and_wait(page, url):
    """페이지 이동 후 상품 리스트가 보일 때까지 대기 (재시도 포함)."""
    last_error = None
    for attempt in range(1, NAV_RETRIES + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            page.wait_for_selector("#item-list .product-item", timeout=LIST_TIMEOUT_MS)
            return True
        except Exception as e:
            last_error = e
            print(f"   ⚠️ 로딩 지연으로 재시도 {attempt}/{NAV_RETRIES}: {e}")
            time.sleep(2)
    print(f"   ❌ 페이지 로딩 실패: {last_error}")
    return False

# ==========================================
# 3. 메인 실행 함수
# ==========================================
def run():
    # URL 인자 처리
    target_urls = sys.argv[1:]
    if not target_urls:
        print("ℹ️ 입력된 URL이 없어 기본 라면 카테고리를 수집합니다.")
        target_urls = [DEFAULT_URL]
    
    print(f"🚀 Rakuten Seiyu 크롤러 v7.0 시작... (총 {len(target_urls)}개 URL)")
    print(f"📄 수집 구간: {START_PAGE}페이지 ~ {MAX_PAGES}페이지")
    print(f"📁 저장 파일명: {DATA_FILE}")

    with sync_playwright() as p:
        # Stealth 모드: 봇 감지 우회를 위한 설정
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
        )
        
        # navigator.webdriver를 false로 설정 (봇 감지 우회 핵심)
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['ja-JP', 'ja', 'en-US', 'en']
            });
            window.chrome = { runtime: {} };
        """)
        
        page = context.new_page()

        all_products_total = []

        # --- 입력된 URL 순회 ---
        for url_idx, target_url in enumerate(target_urls):
            print(f"\n==========================================")
            print(f"🌍 [{url_idx+1}/{len(target_urls)}] 타겟 URL 처리 중...")
            print(f"🔗 {target_url}")
            print(f"==========================================")
            
            # 시작 페이지로 이동 (START_PAGE가 1이면 원본 URL, 아니면 ?page=N 추가)
            if START_PAGE == 1:
                start_url = target_url
            else:
                separator = "&" if "?" in target_url else "?"
                start_url = f"{target_url}{separator}page={START_PAGE}"
            
            print(f"\n📄 {START_PAGE}페이지로 이동 중...")
            if not navigate_and_wait(page, start_url):
                print("   ❌ 페이지 로딩 실패, 다음 URL로 넘어갑니다.")
                continue

            # 실제 수집 시작
            for current_page in range(START_PAGE, MAX_PAGES + 1):
                print(f"\n📄 {current_page}페이지 수집 중...")

                # START_PAGE 이후 페이지는 다음 버튼 클릭으로 이동
                if current_page > START_PAGE:
                    next_link = page.locator(".paging .paging-next-page a")
                    if next_link.count() == 0:
                        print("   ⚠️ 다음 페이지 버튼이 없습니다. (마지막 페이지)")
                        break
                    
                    current_first_id = ""
                    first_item = page.locator("#item-list .product-item").first
                    if first_item.count() > 0:
                        current_first_id = first_item.get_attribute("data-ratid") or ""
                    
                    next_link.first.click()
                    
                    page_changed = False
                    for wait_sec in range(15):
                        time.sleep(1)
                        new_first_item = page.locator("#item-list .product-item").first
                        if new_first_item.count() > 0:
                            new_first_id = new_first_item.get_attribute("data-ratid") or ""
                            if new_first_id and new_first_id != current_first_id:
                                page_changed = True
                                break
                        print(f"   ⏳ 페이지 로딩 대기... {wait_sec+1}초")
                    
                    if not page_changed:
                        print("   ❌ 페이지 전환 실패 (타임아웃)")
                        break

                time.sleep(1) # 렌더링 안정화 대기

                # 스크롤 (이미지 로딩 트리거)
                page.mouse.wheel(0, 4000)
                time.sleep(1)

                # --- 카테고리 추출 ---
                category = "Unknown"
                try:
                    cat_el = page.locator('xpath=//*[@id="container"]/div[1]/div/div/div[3]/h1')
                    if cat_el.count() > 0:
                        raw_cat = clean_text(cat_el.first.inner_text())
                        category = re.sub(r'\s*\d+～\d+件.*', '', raw_cat).strip()
                except Exception:
                    pass

                # 상품 리스트 찾기
                products = page.locator("#item-list .product-item:not(.product-item-next)")
                count = products.count()
                print(f"   -> {count}개 상품 발견 (카테고리: {category})")

                if count == 0:
                    print("   ⚠️ 상품이 없습니다. (페이지 끝 도달 가능성)")
                    break

                # --- 상품 순회 ---
                for i in range(count):
                    try:
                        item = products.nth(i)
                        
                        # 1. 기본 정보
                        maker_el = item.locator(".product-item-info-maker")
                        maker = clean_text(maker_el.inner_text()) if maker_el.count() > 0 else "Unknown"

                        name_el = item.locator(".product-item-info-name")
                        name = clean_text(name_el.inner_text()) if name_el.count() > 0 else "Unknown"

                        price_el = item.locator(".product-item-info-price")
                        price_text = price_el.inner_text() if price_el.count() > 0 else "0"
                        price = re.sub(r'[^0-9]', '', price_text)

                        # 2. ID 추출
                        item_id = item.get_attribute("data-ratid")
                        if not item_id:
                            link_el = item.locator("a").first
                            href = link_el.get_attribute("href")
                            match = re.search(r'/item/(\d+)', href)
                            item_id = match.group(1) if match else f"unknown_{i}"

                        # 3. 제품 URL
                        product_url = ""
                        link_el = item.locator("a").first
                        if link_el.count() > 0:
                            href = link_el.get_attribute("href")
                            if href:
                                if href.startswith("/"):
                                    product_url = f"https://sm.rakuten.co.jp{href}"
                                else:
                                    product_url = href

                        # 4. 이미지 URL
                        img_el = item.locator("img.img-base-size")
                        raw_img_url = ""
                        if img_el.count() > 0:
                            raw_img_url = img_el.get_attribute("data-src") or img_el.get_attribute("src")
                        
                        final_img_url = get_high_res_url(raw_img_url)

                        # 데이터 저장
                        product_data = {
                            "id": item_id,
                            "category": category,
                            "maker": maker,
                            "name": name,
                            "price": int(price) if price else 0,
                            "product_url": product_url,
                            "image_url": final_img_url,
                            "page": current_page,
                            "source_url": target_url
                        }
                        all_products_total.append(product_data)
                        
                        print(f"   [{i+1}/{count}] {item_id} | {name[:10]}... | {category}")

                        # 이미지 다운로드
                        if final_img_url:
                            safe_name = re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")
                            safe_maker = re.sub(r'[\\/*?:"<>|]', "", maker).replace(" ", "_")
                            ext = ".png" if ".png" in final_img_url else ".jpg"
                            filename = f"{item_id}_{safe_maker}_{safe_name[:20]}{ext}"
                            download_image(final_img_url, filename)

                    except Exception as e:
                        print(f"   ❌ {i}번 에러: {e}")
                        continue
        
        # 파일 저장
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(all_products_total, f, ensure_ascii=False, indent=4)
        
        print(f"\n🎉 모든 작업 완료!")
        print(f"   ✅ 총 데이터: {len(all_products_total)}개")
        print(f"   ✅ 파일명: {DATA_FILE}")
        
        browser.close()

if __name__ == "__main__":
    run()