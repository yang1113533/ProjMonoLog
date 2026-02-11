import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st


JSON_PATH = "response.json"
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent


if __name__ == "__main__" and os.environ.get("STREAMLIT_RUN_FROM_PY") != "1":
    os.environ["STREAMLIT_RUN_FROM_PY"] = "1"
    raise SystemExit(subprocess.call(["streamlit", "run", __file__] + sys.argv[1:]))


def _load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_image_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    normalized = raw_path.replace("\\", "/")
    if os.path.isabs(normalized):
        return normalized
    candidates = [
        (BASE_DIR / normalized),
        (WORKSPACE_ROOT / normalized),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return str((BASE_DIR / normalized).resolve())


def _parse_ocr_lines(raw_value: str) -> list:
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        try:
            return json.loads(raw_value)
        except Exception:
            return []
    return []


st.set_page_config(page_title="Search Results Viewer", layout="wide")
st.title("Mono-Log Search Results Viewer")

json_path = st.text_input("Result JSON path", JSON_PATH)
if not os.path.exists(json_path):
    st.warning("JSON file not found. Save the API response as response.json and try again.")
    st.stop()

payload = _load_results(json_path)
results = payload.get("results", [])

st.caption(f"Detected text: {payload.get('detected_text', '')}")
st.caption(f"Total results: {len(results)}")

for item in results:
    cols = st.columns([1, 3])
    with cols[0]:
        image_path = _normalize_image_path(item.get("image_path", ""))
        if image_path and os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.write("(no image)")

    with cols[1]:
        st.subheader(item.get("name", "(no name)"))
        st.caption(
            f"ID: {item.get('id', '')} | maker: {item.get('maker', '')} | price: {item.get('price', '')}"
        )
        st.caption(
            f"score: {item.get('final_score', '')} | similarity: {item.get('similarity_score', '')}"
        )

        ocr_lines = _parse_ocr_lines(item.get("ocr_lines", ""))
        if ocr_lines:
            top_lines = [line.get("text", "") for line in ocr_lines[:10] if line.get("text")]
            if top_lines:
                st.text("OCR (top 10): " + " | ".join(top_lines))

        with st.expander("Raw metadata"):
            st.json(item)

    st.divider()
