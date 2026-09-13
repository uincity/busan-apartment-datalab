"""School Map Marker Visual Fix용 비운영 시각 점검 화면."""
import pandas as pd
import streamlit as st

from src.visualization import combined_pydeck_map

st.set_page_config(page_title="학교 도형 크기 점검", layout="wide")
st.title("학교 도형 8–28px 점검")
st.caption("개발용 가상 점수이며 운영 snapshot에는 저장되지 않습니다.")

scores = [0, 25, 50, 75, 100]
rows = []
for level, latitude in (("elementary", 35.18), ("middle", 35.16)):
    for index, score in enumerate(scores):
        rows.append({
            "school_id": f"preview-{level}-{score}", "school_level": level,
            "school_name": f"{'초등학교' if level == 'elementary' else '중학교'} {score}점", "sigungu": "개발 미리보기",
            "score": float(score), "score_type": "미리보기 점수",
            "latitude": latitude, "longitude": 129.02 + index * 0.025,
        })

st.pydeck_chart(combined_pydeck_map(pd.DataFrame(), pd.DataFrame(rows)), height=650, width="stretch")
st.caption("왼쪽부터 0·25·50·75·100점 = 8·13·18·23·28px · 초등학교는 삼각형, 중학교는 사각형")
