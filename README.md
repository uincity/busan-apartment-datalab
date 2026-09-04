# 부산 아파트 데이터 분석

부산광역시 16개 구·군의 국토교통부 아파트 매매·전월세 실거래와 K-apt 단지정보를 결합해 부산 → 구·군 → 법정동 → 단지 → 면적그룹 → 거래로 탐색하는 재사용 가능한 분석 프로젝트입니다.

분석 결과는 투자 추천이나 매수 신호가 아닙니다. K-apt 미등록 단지도 실거래 원본에서 삭제하지 않고 내부 ID를 부여해 보존합니다.

## 1. 환경 설치

Python 3.11 이상이 필요합니다.

```powershell
cd busan_apartment_analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux에서는 활성화 명령만 `source .venv/bin/activate`로 바꿉니다.

## 2. API Key 설정

`.env.example`을 `.env`로 복사하고 공공데이터포털에서 발급받은 키를 입력합니다.

```dotenv
PUBLIC_DATA_API_KEY=인증키
KAPT_API_KEY=인증키
KAKAO_API_KEY=카카오_REST_API_키
```

- `PUBLIC_DATA_API_KEY`: [국토교통부 아파트 매매 실거래가 상세 자료](https://www.data.go.kr/data/15126468/openapi.do) 활용신청 키
- 같은 `PUBLIC_DATA_API_KEY`로 [국토교통부 아파트 전월세 실거래가 자료](https://www.data.go.kr/data/15126474/openapi.do)도 별도로 활용신청해야 합니다.
- `KAPT_API_KEY`: [공동주택 단지 목록](https://www.data.go.kr/data/15057332/openapi.do) 및 [공동주택 기본정보](https://www.data.go.kr/data/15058453/openapi.do) 활용신청 키
- 같은 공공데이터포털 키에 두 K-apt 서비스 권한이 있으면 `KAPT_API_KEY` 하나를 공용으로 사용할 수 있습니다.
- `KAKAO_API_KEY`: 카카오 디벨로퍼스 애플리케이션의 REST API 키입니다. K-apt 주소를 지도용 위·경도로 변환할 때 사용합니다.

인증키는 코드·로그·Git에 기록되지 않습니다. API URL과 동작명은 `config/settings.yaml`에서 교체할 수 있습니다. 2026-08-07 개정 명세에 맞춰 공동주택 목록은 `AptListService4`, 기본·상세정보는 `AptBasisInfoServiceV5`를 사용합니다.

## 3. 데이터 수집

```powershell
python main.py collect-trade --start 202001 --end 202608
python main.py collect-rent --start 202001 --end 202608
python main.py collect-trade --start 202301 --end 202606 --lawd-cd 26350 --force
python main.py collect-rent --start 202301 --end 202606 --lawd-cd 26350 --force
python main.py collect-kapt
python main.py geocode-kapt
python main.py export-kapt-excel
```

매매·전월세 실거래 수집은 각각 부산 16개 지역 × 월을 순회합니다. 저장된 지역/월 파일은 건너뛰며 다시 받을 때만 `--force`를 사용합니다. API 호출은 페이지네이션, timeout, retry, exponential backoff를 적용하고 한 지역의 빈 응답·실패가 전체 수집을 중단하지 않습니다.

K-apt 수집은 유효한 실데이터 원본이 있으면 재사용하고, 데모·불완전 원본은 자동으로 감지해 다시 수집합니다. 최신값을 강제로 다시 받으려면 `python main.py collect-kapt --force`를 사용합니다.

원본 위치:

- `data/raw/trade/YYYYMM/{lawd_cd}.parquet`
- `data/raw/rent/YYYYMM/{lawd_cd}.parquet`
- `data/raw/kapt/busan_complexes.parquet`
- `data/processed/busan_kapt_apartment_analysis.xlsx`

`geocode-kapt`는 위·경도가 없는 K-apt 단지만 카카오 주소검색 API로 변환합니다. 성공·실패 결과는
`data/interim/kapt_coordinates.parquet`에 주기적으로 저장하므로 중단 후 다시 실행해도 완료된 단지는 건너뜁니다.
검색 실패 주소를 다시 요청하려면 `python main.py geocode-kapt --retry-failed`를 사용합니다. 좌표 캐시는 이후
`build` 실행 시 자동으로 병합되며, 기존 월 패널과 요약 파일도 지오코딩 완료 직후 갱신됩니다.
K-apt 주소가 오래되었거나 카카오에서 검색되지 않는 단지는 `config/geocode_address_overrides.csv`에
`kapt_code,address`를 등록한 뒤 `--retry-failed`로 다시 변환할 수 있습니다.

## 4. 정제·매칭·분석 데이터 생성

```powershell
python main.py build
python main.py build --incremental
python main.py report
```

월별 실거래 데이터만 추가·갱신한 경우에는 `build --incremental`을 사용합니다. 기존 빌드의 정제·단지 매칭 결과 중 잠정 구간 이전 자료를 재사용하고, 최근 잠정 구간과 새로 추가된 월만 다시 정제·매칭합니다. K-apt 원본 또는 잠정 구간보다 오래된 실거래 원본이 기존 빌드 이후 변경되었다면 정확성을 위해 자동으로 전체 빌드로 전환합니다. 월 패널의 롤링·누적 지표는 전체 이력에 의존하므로 다시 계산하며, 장시간 작업은 단계 및 그룹 진행률을 로그로 표시합니다.

`build`는 다음 순서로 실행됩니다.

1. 실거래 금액·면적·층·계약일·건축연도 정제
2. 해제 거래를 원본/중간자료에 보존하고 분석 자료에서 제외
3. 단지명 정규화 후 도로명+단지명, 법정동+지번+단지명, 같은 법정동 fuzzy 순서로 K-apt 매칭
4. K-apt 미매칭 단지에 안정적인 `internal_complex_id` 부여
5. 단지×월×면적그룹 패널과 거래회전율·중앙가격 추세 생성
6. 구·군, 법정동, 단지, 대표단지, 시장회복 관찰대상, 노후단지 표 생성
7. 매칭률과 데이터 품질 보고서 생성

전월세 원본이 있으면 `build`가 전세·월세를 정제하고 K-apt 단지에 매칭한 뒤
`busan_apartment_rent_monthly.parquet`도 생성합니다. 상세화면의 전세가율은 같은 단지·면적그룹에서
최근 12개월 전세 보증금 중앙값을 최근 12개월 매매가 중앙값으로 나눈 값입니다. 월세 계약은
보증금과 월세 금액을 별도로 표시하며 전세가율 계산에는 포함하지 않습니다.

전월세 API의 도로명에 건물번호가 함께 들어오는 경우도 분리해 매칭합니다. 또한 법정동·지번·정규화
단지명이 같거나 정규화 단지명·도로명주소가 같은 매매와 전월세의 내부 단지 ID가 다르면,
양쪽에서 단일 ID로 확인되는 경우에만 매매 ID로 교차교정합니다.
교정 내역은 `apartment_rent_sale_crosswalk.csv`, 교정 후 남은 불일치는
`apartment_rent_cross_market_issues.csv`에 기록하며, 불일치가 남으면 `build`가 실패합니다.

84㎡ 가격은 전용면적 80㎡ 이상 90㎡ 미만의 실제 거래만 사용하며 다른 면적 가격을 환산하지 않습니다. 세대수 누락 시 거래회전율은 0이 아니라 결측값입니다.

## 5. API 키 없는 데모

```powershell
## python main.py demo
## streamlit run app.py
```

`demo`는 재현 가능한 표본 원본을 생성한 뒤 실제 정제·매칭·집계 파이프라인을 그대로 실행합니다. 기존 데모 대상 경로의 파일은 동일 이름으로 갱신될 수 있으므로 실데이터와 데모는 별도 작업 복사본에서 운용하는 것을 권장합니다.

실데이터 보호를 위해 `data/raw`에 기존 원본이 있으면 `demo` 실행을 중단합니다. 데모는 반드시 별도의 빈 작업 복사본에서 실행하세요. 특정 구·군 원본만 재수집하려면 `--lawd-cd` 뒤에 하나 이상의 법정동 코드를 지정합니다.

## 6. 대시보드

```powershell
streamlit run app.py
```

메뉴는 부산 Overview, 구군 비교, 동 비교, 아파트 상세, 아파트 비교(최대 5개), 시장회복 Watch, 노후단지 Watch로 구성됩니다. 아파트 상세에서는 매매·전세 가격 추이, 전세가율, 전세·월세 계약량과 최근 개별 임대차 계약을 함께 확인할 수 있습니다. 기간·구군·법정동·단지·면적그룹·연식·세대수 필터를 제공하며 최신 잠정 월이 포함되면 화면 하단에 표시합니다.

## 7. 테스트

```powershell
python -m pytest -q
```

금액 변환, 면적그룹 경계, 해제 제거, 단지명 정규화, fuzzy 매칭, ㎡당 가격, 세대당 주차, 거래회전율과 rolling 거래량을 검증합니다.

## 생성 파일과 역할

```text
busan_apartment_analysis/
├─ config/                 API·분석 설정, 부산 16개 법정 시군구 코드
├─ data/raw/               API 원본 캐시
├─ data/interim/           정제 거래, 정제 K-apt, 매칭 거래
├─ data/processed/         월 패널과 최종 분석 CSV
├─ notebooks/              모듈 재사용형 점검/EDA 진입점
├─ reports/                품질·매칭률·그림·표
├─ src/
│  ├─ collect_trade.py     실거래 페이지 수집
│  ├─ collect_rent.py      전월세 실거래 페이지 수집
│  ├─ collect_kapt.py      K-apt 목록/기본정보 수집
│  ├─ clean_trade.py       실거래 파싱·정제·면적그룹
│  ├─ clean_rent.py        전월세 파싱·정제·전세/월세 분류
│  ├─ clean_kapt.py        단지 정제·연식/주차 파생변수
│  ├─ match_complex.py     주소 우선 exact/fuzzy 단지 매칭
│  ├─ feature_engineering.py 거래량·회전율·가격추세
│  ├─ aggregate.py         단지×월×면적 패널
│  ├─ rent_analysis.py     전월세 월 집계·전세가율 계산
│  ├─ analysis.py          요약·대표단지·Watch·품질검증
│  ├─ visualization.py     Plotly 공용 차트
│  └─ pipeline.py          build/report/demo 오케스트레이션
├─ tests/                  pytest 테스트
├─ app.py                  Streamlit 대시보드
└─ main.py                 CLI 진입점
```

핵심 출력은 `busan_apartment_monthly.parquet`, `busan_apartment_rent_monthly.parquet`, `rent_matched.parquet`, `busan_district_summary.csv`, `busan_dong_summary.csv`, `busan_complex_summary.csv`, `representative_complexes.csv`, `recovery_watchlist.csv`, `old_apartment_watchlist.csv`, `apartment_match_log.csv`, `apartment_rent_match_log.csv`, `reports/data_quality_report.csv`입니다.

## 데이터 품질 주의사항

- 최신 2개월은 `provisional=True`로 표시되며 신고 진행에 따라 값이 달라질 수 있습니다.
- 거래량이 적은 단지·면적그룹은 `low_sample_flag=True`이며 단일 거래로 추세를 판단하지 않습니다.
- K-apt는 의무관리대상 중심이므로 소규모 단지 누락이 정상적으로 존재할 수 있습니다.
- fuzzy 매칭 점수 90 미만 및 미매칭은 `manual_review=True`이며 `apartment_match_log.csv`에서 검토해야 합니다.
- 주소·필드명은 공급 API 개편의 영향을 받을 수 있습니다. 파서 별칭과 `settings.yaml` 엔드포인트를 먼저 갱신하세요.
- 품질 보고서의 `REVIEW`는 자동 삭제 지시가 아니라 원본 확인 대상입니다.

## 현재 범위와 다음 우선순위

현재 구현은 매매·전월세 실거래/K-apt 수집, 정제, 3단계 매칭, 월 패널, 전세가율·회전율·가격 추세, 3종 요약, 대표단지 점수, 두 Watch 목록, 품질검증, Plotly/Streamlit 탐색까지입니다.

아직 구현하지 않은 확장 기능은 지하철·학교·학군·상권·해안·공원·고도, 입주물량, 인구, 한국부동산원 지수와 가격예측 모델입니다. 다음 개발 우선순위는 ① 실제 API 응답 샘플로 K-apt 필드 별칭 회귀검증, ② 수동 매칭 보정 테이블, ③ 공간정보 플러그인 계층, ④ 시계열 기반 모델 검증 순입니다.
