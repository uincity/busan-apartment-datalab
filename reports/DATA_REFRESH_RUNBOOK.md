# 웹서비스 데이터 갱신 운영 매뉴얼

이 문서는 부산 아파트 분석 웹서비스에 다음 변경을 안전하게 반영하는 표준 절차를 설명합니다.

- 국토교통부 아파트 매매 실거래 갱신
- 국토교통부 아파트 전월세 실거래 갱신
- `area_master`의 평형별 세대수·KB 가격·혼합단지 검수 결과 갱신
- 일반 분석 화면과 KB/실거래 기반 시가총액 갱신
- 검증, 배포, 서비스 확인 및 장애 대응

명령은 이 프로젝트 루트에서 PowerShell로 실행하는 것을 기준으로 합니다.

---

## 1. 전체 작업 흐름

```text
API 키·대상 월 확인
  → 최근 매매 원본 수집
  → 최근 전월세 원본 수집
  → 정제·매칭·월 패널 생성(build)
  → 필요 시 area_master 기반 KB 시가총액 배치
  → 실거래 기반 시가총액 배치
  → 테스트와 로컬 화면 확인
  → 웹서비스용 산출물 커밋·배포
  → 운영 화면과 데이터 기준일 확인
```

웹서비스는 접속 시 원본 API를 호출하거나 시가총액을 계산하지 않습니다. 모든 수집과 배치는 로컬 또는 별도 배치 환경에서 완료하고, 검증된 산출물을 웹서비스에 배포해야 합니다.

## 2. 갱신 유형별 필수 작업

| 변경 내용 | `collect-*` | `build` | `market_cap_kb` | `market_cap_batch` |
|---|---:|---:|---:|---:|
| 최근 매매·전월세 갱신 | 필요 | 필요 | 불필요 | 필요 |
| 매매만 갱신 | 필요 | 필요 | 불필요 | 필요 |
| 전월세만 갱신 | 필요 | 필요 | 불필요 | 불필요 |
| KB 가격만 갱신 | 불필요 | 불필요 | 필요 | 불필요 |
| 평형별 세대수 마스터 갱신 | 불필요 | 불필요 | 필요 | 필요 |
| 혼합단지 분양·임대 구분 갱신 | 불필요 | 불필요 | 필요 | 필요할 수 있음 |
| K-apt 단지 원본 갱신 | 불필요 | 전체 빌드 필요 | 필요 | 필요 |
| 시가총액 보정 정책 갱신 | 불필요 | 불필요 | 필요 | 불필요 |

`market_cap_batch`는 매매 원본과 매매 단지 매칭 결과를 사용합니다. 따라서 매매 자료를 갱신한 경우 반드시 `build`가 성공한 뒤 실행합니다. 전월세 자료는 실거래 기반 시가총액 계산에 사용하지 않습니다.

매매·전월세와 `area_master`를 한 번에 갱신하는 정기 작업의 명령 순서는 다음과 같습니다. 날짜와 배포본은 실제 작업 대상에 맞게 변경합니다.

```powershell
.venv\Scripts\python.exe main.py collect-trade --start 202607 --end 202609 --force
.venv\Scripts\python.exe main.py collect-rent --start 202607 --end 202609 --force
.venv\Scripts\python.exe main.py build --incremental
.venv\Scripts\python.exe ..\area_master\scripts\run_market_cap_kb_batch.py
.venv\Scripts\python.exe -m src.sync_area_master_market_cap
.venv\Scripts\python.exe -m src.market_cap_batch --reason "월간 실거래 및 area_master 갱신"
.venv\Scripts\python.exe -m pytest -q
```

`area_master`가 바뀌지 않았다면 KB 배치와 동기화 단계는 생략합니다. 평형 마스터와 매매를 모두 갱신할 때는 동기화가 로컬 실거래용 마스터를 먼저 교체해야 하므로 반드시 `market_cap_batch`보다 앞에 실행합니다.

---

## 3. 작업 전 준비

### 3.1 실행 환경 확인

```powershell
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -c "import pandas, pyarrow, streamlit; print('환경 확인 완료')"
git status --short
```

작업 트리에 기존 수정사항이 있다면 이번 갱신 파일과 구분할 수 있도록 현재 상태를 기록합니다. 기존 변경사항을 임의로 되돌리거나 삭제하지 않습니다.

### 3.2 API 키 확인

프로젝트 루트의 `.env`에 다음 값이 있어야 합니다.

```dotenv
PUBLIC_DATA_API_KEY=공공데이터포털_인증키
KAPT_API_KEY=K-apt_인증키
KAKAO_API_KEY=카카오_REST_API_키
```

매매와 전월세 API는 같은 키를 사용하더라도 각각 활용신청이 완료되어 있어야 합니다. `.env`와 인증키는 Git에 추가하지 않습니다.

### 3.3 수집 기간 결정

실거래는 신고 지연, 취소, 정정이 발생하므로 정기 갱신 시 **현재 월과 직전 2개월**, 총 3개월을 다시 받는 것을 기본으로 합니다.

예를 들어 2026년 9월에 작업하면 `202607`부터 `202609`까지 수집합니다. 시가총액 월별 산정은 진행 중인 현재 월을 제외하고 최근 종료 월까지만 생성합니다.

### 3.4 원본 보존

`--force`는 지정한 `data/raw/trade` 또는 `data/raw/rent` 파일을 새 API 응답으로 교체합니다. 원본은 Git 추적 대상이 아니므로, 중요한 운영 갱신 전에는 `data/raw`를 별도 저장소나 백업 위치에 보존합니다.

`demo` 명령은 운영 데이터 갱신에 사용하지 않습니다.

---

## 4. 매매·전월세 정기 갱신

아래 예시는 2026년 7월부터 9월까지 다시 수집하는 경우입니다. 실제 작업 월에 맞게 `--start`와 `--end`를 변경합니다.

### 4.1 매매 원본 수집

```powershell
.venv\Scripts\python.exe main.py collect-trade --start 202607 --end 202609 --force
```

### 4.2 전월세 원본 수집

```powershell
.venv\Scripts\python.exe main.py collect-rent --start 202607 --end 202609 --force
```

두 명령은 부산 16개 구·군을 순회합니다. 결과 JSON에서 다음을 확인합니다.

- 실패 건수가 0인지
- 요청한 월과 16개 구·군이 모두 처리되었는지
- `data/metadata/collection_status.json`에 성공 상태와 수집 시각이 기록되었는지
- `data/raw/trade/YYYYMM/`와 `data/raw/rent/YYYYMM/`에 구·군별 Parquet가 있는지

일부 구·군만 실패했다면 성공한 전체 범위를 다시 받을 필요 없이 해당 코드만 재실행합니다.

```powershell
.venv\Scripts\python.exe main.py collect-trade --start 202609 --end 202609 --lawd-cd 26350 --force
.venv\Scripts\python.exe main.py collect-rent --start 202609 --end 202609 --lawd-cd 26350 --force
```

빈 응답은 실제 거래가 없는 경우일 수 있지만, 인증·서비스 오류와 구분하여 로그를 확인해야 합니다.

### 4.3 정제·매칭·분석 산출물 생성

최근 월만 추가하거나 다시 수집했다면 증분 빌드를 실행합니다.

```powershell
.venv\Scripts\python.exe main.py build --incremental
```

증분 빌드는 기본적으로 최근 잠정 구간을 다시 정제·매칭하고 월 패널 및 롤링 지표는 전체 이력을 기준으로 다시 계산합니다. 다음 경우에는 프로그램이 자동으로 전체 빌드로 전환합니다.

- 필요한 기존 정제·매칭 산출물이 없는 경우
- K-apt 원본이 기존 매칭 결과보다 새로운 경우
- 잠정 구간보다 오래된 매매 원본이 기존 빌드 이후 변경된 경우

K-apt 구조나 오래된 이력을 의도적으로 변경했다면 처음부터 전체 빌드를 실행합니다.

```powershell
.venv\Scripts\python.exe main.py build
```

빌드가 성공하면 매매·전월세 패널, 단지 매칭 결과, 요약표와 `data/processed/data_update_status.json`이 갱신됩니다. 이 상태 파일은 모든 최종 패널이 만들어진 후 마지막에 교체됩니다.

### 4.4 일반 분석 산출물 확인

다음 파일의 수정 시각과 크기가 갱신되었는지 확인합니다.

```powershell
Get-Item data\processed\busan_apartment_monthly.parquet
Get-Item data\processed\busan_apartment_rent_monthly.parquet
Get-Item data\interim\trade_matched.parquet
Get-Item data\interim\rent_matched.parquet
Get-Content data\processed\data_update_status.json -Encoding UTF8
```

`data_update_status.json`에서 다음 항목을 확인합니다.

- `trade.latest_month`, `rent.latest_month`
- `trade.collection_status`, `rent.collection_status`가 `complete`인지
- `successful_regions`가 `target_regions`와 같은지
- `dashboard_applied_at`이 이번 작업 시각인지
- `data_version`이 이전 값에서 변경되었는지

다음 검수 파일도 확인합니다.

- `reports/data_quality_report.csv`
- `data/processed/apartment_match_manual_review.csv`
- `data/processed/apartment_rent_match_manual_review.csv`
- `data/processed/apartment_rent_cross_market_issues.csv`

매매·전월세 단지 ID 교차검증 오류가 있으면 빌드는 실패합니다. 이때 오류 파일을 검토하기 전에는 기존 운영 산출물을 교체하지 않습니다.

---

## 5. 실거래 기반 시가총액 갱신

매매 수집과 `build`가 성공한 다음 실행합니다.

### 5.1 최근 종료 월 갱신

정기 월간 작업에서는 다음 명령으로 최근 종료 월을 산정합니다.

```powershell
.venv\Scripts\python.exe -m src.market_cap_batch --reason "월간 매매 실거래 갱신"
```

기존 월에 새 실행 결과를 추가할 때 `--reason`은 필수입니다. 동일 입력·규칙·마스터로 다시 실행하면 새 revision을 만들지 않습니다.

### 5.2 특정 과거 월 정정

지연 신고, 취소 또는 정정이 특정 종료 월의 시가총액에도 반영되어야 하면 월을 명시합니다.

```powershell
.venv\Scripts\python.exe -m src.market_cap_batch --month 2026-07 --reason "지연 신고·취소 반영"
```

가격 산정은 최근 3·6·12개월 거래 창을 사용합니다. 과거 원본 변경이 이후 평가월의 가격 창에도 포함된다면, 변경 월부터 현재 최근 종료 월까지 영향을 받는 월을 각각 재산정합니다.

### 5.3 전체 과거 이력 재구성

다음 경우에만 전체 재구성을 사용합니다.

- 평형별 세대수 마스터를 크게 보완한 경우
- 계산 규칙을 변경한 경우
- 과거 원본을 광범위하게 정정한 경우

```powershell
.venv\Scripts\python.exe -m src.market_cap_batch --backfill --reason "평형 마스터 또는 과거 원본 전면 갱신"
```

`--backfill`은 현재 확보한 마스터와 원본으로 과거를 재구성하며, 기존 최초 결과를 덮어쓰지 않고 새 revision을 추가합니다. 현재 KB 가격을 과거 가격으로 사용하는 작업은 아닙니다.

### 5.4 결과 확인

확인 대상은 다음과 같습니다.

- `data/processed/market_cap/manifest.json`
- `data/processed/market_cap/audit.json`
- `data/processed/market_cap/supplement_needed.csv`
- `data/processed/market_cap/sensitivity.csv`
- `data/processed/market_cap/top10.csv`
- `data/processed/market_cap/YYYY-MM/<run_id>/`

`audit.json`의 `latest_month`, `new_snapshots`, 등급별 단지 수와 확인 세대수를 직전 실행과 비교합니다.

---

## 6. `area_master` 및 KB 시가총액 갱신

### 6.1 현재 프로젝트가 읽는 외부 파일

기본 외부 경로는 형제 프로젝트 `../area_master`입니다.

| 파일 | 역할 |
|---|---|
| `data/releases/area_master_*/market_cap_area_master.csv` | 평형별 전용·공급면적, 세대수, 적용 기간과 검증 상태 |
| `data/raw/kb/kb_area_types.csv` | KB 타입별 일반매매가와 수집 시각 |
| `data/releases/area_master_*/area_master_complex_status.csv` | 단지별 최종 검수 상태 |
| `data/qa/phase4_mixed_complex_audit.csv` | 혼합단지 분양·임대 세대 구분 |
| `data/releases/area_master_*/RELEASE_INFO.json` | 배포 식별과 입력 해시 이력 |
| `config/market_cap_kb_adjustments.json` | 승인된 단지·평형별 KB 누락가격 보정 정책과 출처 |
| `data/processed/market_cap/kb/` | `area_master`가 생성한 최종 KB 시가총액 불변 snapshot |

보정 정책과 최종 KB 시가총액 산출 책임은 `area_master`에 있습니다. 웹서비스 프로젝트는 보정 정책을 별도로 유지하지 않고 검증된 최종 snapshot을 동기화해 표시합니다.

### 6.2 배포본 선택

`--release`를 생략하면 디렉터리 이름순으로 가장 최신인 `area_master_*`가 선택됩니다. 운영 갱신에서는 잘못된 배포본 선택을 막기 위해 경로를 명시하는 것을 권장합니다.

```powershell
.venv\Scripts\python.exe ..\area_master\scripts\run_market_cap_kb_batch.py
.venv\Scripts\python.exe -m src.sync_area_master_market_cap
```

첫 번째 명령은 `area_master`에서 다음을 수행합니다.

1. 배포 마스터 형식과 K-apt 단지 식별자를 검증합니다.
2. `VERIFIED_MASTER` 단지와 마스터 포함 단지의 일치를 검사합니다.
3. 평형 면적·타입명·세대수가 모두 같은 KB 타입을 연결합니다.
4. 혼합단지의 검증된 분양 세대 범위를 적용합니다.
5. `area_master/config/market_cap_kb_adjustments.json`에 승인된 단지만 보정합니다.
6. `area_master/data/processed/market_cap/kb/`에 새 불변 snapshot과 `summary.csv`를 생성합니다.
7. `area_master/data/processed/market_cap/market_cap_area_master.csv`를 재생성합니다.

두 번째 명령은 산출물 SHA-256, 필수 컬럼, 단지·평형 식별자 중복, 단지별 보정 시가총액과 평형별 합계, 메타데이터 건수를 검증합니다. 모두 통과한 경우에만 snapshot과 실거래용 평형 마스터를 웹서비스 폴더로 복사하고 마지막에 `latest.json`을 원자적으로 교체합니다.

입력 파일들이 서로 다른 시점의 자료이면 검증 실패 또는 잘못된 부분 산정이 발생할 수 있습니다. 마스터, 단지 상태, 혼합단지 감사 파일과 `RELEASE_INFO.json`은 하나의 일관된 배포본으로 관리합니다.

### 6.3 승인 보정 정책 변경

KB가 일부 또는 전체 평형의 일반매매가를 제공하지 않지만 운영상 검증된 보정가격을 반영해야 한다면 **`../area_master/config/market_cap_kb_adjustments.json`**을 수정합니다. 웹서비스 프로젝트에 별도 보정 규칙을 추가하지 않습니다. 보정은 반드시 단지의 K-apt 코드 단위로 등록하고 다음 정보를 남깁니다.

- `households`: 보정 대상 단지의 검증된 전체 세대수
- `scope`: 분양·임대 구분을 포함한 산정 범위
- `reason`: KB 가격이 누락된 이유와 보정이 필요한 이유
- `source`: 외부 가격의 출처, 검증 조건, 거래 기간
- `price_method`: 화면에 표시할 보정 방식
- `type_price_overrides_krw`: 평형 ID별 승인 가격(원). 외부 평형가격을 직접 적용할 때만 사용

`type_price_overrides_krw`의 가격은 **KB 일반매매가가 없는 평형에만** 적용됩니다. 같은 평형에 KB 가격이 있으면 KB 가격을 우선합니다. 등록한 평형 ID가 최신 `market_cap_area_master.csv`에 없으면 배치를 중단하고 마스터와 설정을 다시 대조해야 합니다.

가까운 KB 평형의 면적단가를 이용하는 기존 보정은 `type_price_overrides_krw` 없이 사용할 수 있습니다. 외부 실거래 기반 가격을 직접 등록하는 경우에는 취소 거래 제외 여부, 단지 매칭 검증, 산정 기간과 평형별 대표가격 계산법을 `source`에 기록합니다. 이 값은 실제 KB 시세가 아니므로 화면에서도 `실거래 기반 보정 추정` 등 별도 방식으로 표시해야 합니다.

설정을 변경한 뒤 다음 순서로 재산정합니다.

```powershell
.venv\Scripts\python.exe ..\area_master\scripts\run_market_cap_kb_batch.py
.venv\Scripts\python.exe -m src.sync_area_master_market_cap
.venv\Scripts\python.exe -m src.market_cap_batch --reason "KB 시세·평형 마스터 및 대상 단지 보정 반영"
.venv\Scripts\python.exe -m pytest -q
```

세 번째 명령은 동기화된 `config/market_cap_area_master.csv`의 해시나 내용이 기존 값과 달라졌을 때 실행합니다. KB 보정 설정만 바뀌고 실거래용 평형 마스터가 동일하다면 `market_cap_batch`는 생략할 수 있습니다.

#### 2026-09-21 적용 사례

이번 적용은 `area_master_20260918`, 보정 정책 `approved-price-adjustments-v2`를 사용했습니다. `area_master`가 생성하고 서비스가 검증·동기화한 KB snapshot run ID는 `285aaf0c2b23697c9d41f95b`이며, KB 시세만으로 완전 산정된 단지는 385개, 보정 추정까지 포함하면 421개입니다.

| 단지 | K-apt 코드 | 적용 방식 | 보정 후 시가총액 | 확인 사항 |
|---|---|---|---:|---|
| 해운대두산위브더제니스 | `A61202007` | KB가 15개 평형 모두 일반매매가를 제공하지 않아, 검증된 2024-01~2026-08 매매 실거래로 평형별 대표가격을 등록 | 34,731.75억원 | 15개 평형·1,788세대가 모두 `실거래 기반 보정 추정`으로 표시되어야 함 |
| 남천자이아파트 | `A10023420` | KB 가격이 있는 907세대에 미고시 펜트하우스 6세대의 인접 49평 면적단가 보정을 합산 | 13,863.30억원 | 25개 평형·913세대가 산정되고 보정 세대수가 6세대로 표시되어야 함 |

이 적용에서 해운대두산위브더제니스 금액은 KB 시세가 아니라 별도 승인된 실거래 기반 보정 추정치입니다. 남천자이는 KB 부분합계 13,700.15억원과 누락 세대 보정액을 합한 값입니다. 운영 화면의 기본 `보정 추정 포함` 모드와 부산 Overview 지도에서 두 단지가 모두 조회되어야 합니다.

### 6.4 KB 결과 확인

```powershell
Get-Content ..\area_master\data\processed\market_cap\kb\latest.json -Encoding UTF8
Get-Content data\processed\market_cap\kb\latest.json -Encoding UTF8
Get-Item data\processed\market_cap\kb\summary.csv
Get-Item config\market_cap_area_master.csv
```

`latest.json`에서 다음을 확인합니다.

- `release`가 의도한 배포본인지
- `collection_start`, `collection_end`가 최신 KB 수집 구간인지
- `master_types`, `master_complexes`, `targets`
- `complete`, `adjusted_complete`
- `price_issues`
- `source_files`의 경로와 SHA-256 해시
- `artifact_files`의 `complexes.parquet`, `areas.parquet` 콘텐츠 SHA-256
- `adjustment_policy.version`이 이번에 수정한 보정 정책 버전인지

보정 대상 단지는 `data/processed/market_cap/kb/<run_id>/complexes.parquet`에서 다음 항목도 대조합니다.

- `adjusted_status`가 완전 산정 상태인지
- `adjusted_market_cap_krw`, `estimated_households`가 승인 내용과 일치하는지
- `adjustment_source`에 실제 가격 출처와 방식이 표시되는지
- KB 원가격이 없는 값을 실제 KB 가격처럼 표시하지 않는지

새 snapshot은 `data/processed/market_cap/kb/<run_id>/`에 저장됩니다. 이전 snapshot은 롤백과 감사에 필요하므로 임의로 삭제하지 않습니다.

### 6.5 평형 마스터도 변경된 경우

KB 가격이나 보정 정책만 바뀌어도 `area_master` 배치와 서비스 동기화를 모두 실행합니다. 동기화된 `market_cap_area_master.csv`의 평형별 세대수나 범위까지 변경되었다면 실거래 기반 시가총액도 다시 산정합니다.

```powershell
.venv\Scripts\python.exe -m src.market_cap_batch --reason "area_master 평형 마스터 갱신"
```

과거 월 전체를 새 평형 구성으로 재구성해야 한다는 운영 판단이 있을 때만 `--backfill`을 사용합니다.

---

## 7. 테스트와 로컬 화면 검증

### 7.1 자동 테스트

전체 회귀 테스트를 실행합니다.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

시간을 줄여 1차 확인할 때는 관련 테스트를 먼저 실행할 수 있지만, 배포 전에는 전체 테스트 통과를 권장합니다.

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_market_cap.py tests\test_market_cap_kb.py tests\test_overview_map.py tests\test_overview_ui.py
```

### 7.2 로컬 웹서비스 실행

```powershell
.venv\Scripts\streamlit.exe run app.py
```

이미 앱이 실행 중인 상태에서 일반 매매·전월세 Parquet를 교체했다면 프로세스를 재시작합니다. 일반 패널은 `st.cache_resource`로 메모리에 보관되므로 브라우저 새로고침만으로 이전 데이터가 남을 수 있습니다. 필요하면 중지 상태에서 캐시를 비운 뒤 다시 시작합니다.

```powershell
.venv\Scripts\streamlit.exe cache clear
```

### 7.3 화면 점검표

- 상단 데이터 갱신 현황의 반영 시각과 최신 월
- 부산 Overview의 거래금액 및 시가총액 마커
- 구·군/법정동/단지 필터와 최신 월 거래량
- 아파트 상세의 매매가격, 전세가격, 전세가율
- 최근 매매 및 전월세 계약 목록
- 아파트 시가총액의 KB 기준일·배포본·완전 산정 건수
- `보정 추정 포함` 모드에서 해운대두산위브더제니스 34,731.75억원, 남천자이 13,863.30억원 표시
- 두산위브더제니스의 가격 산정 방법이 `실거래 기반 보정 추정`으로 표시되고 실제 KB 시세로 표기되지 않는지
- `KB 시세만`과 `보정 추정 포함` 전환 시 대상 단지의 포함 여부와 설명이 올바른지
- 가격 기준을 `실거래 월별 추정`으로 바꿨을 때 최근 종료 월
- 빈 화면, 파일 읽기 오류, 캐시로 인한 이전 값 노출 여부

---

## 8. 웹서비스 배포

현재 저장소에는 특정 호스팅 사업자용 배포 설정이 없습니다. 따라서 아래 공통 절차 뒤 실제 운영 환경의 Git 연동 재배포 또는 파일 배포 절차를 수행합니다.

### 8.1 변경 파일 검토

```powershell
git status --short
git diff --stat
```

매매·전월세 갱신 시 웹서비스에 필요한 주요 산출물은 다음과 같습니다.

- `data/processed/busan_apartment_monthly.parquet`
- `data/processed/busan_apartment_rent_monthly.parquet`
- `data/processed/busan_complex_summary.csv`
- `data/processed/recovery_watchlist.csv`
- `data/processed/old_apartment_watchlist.csv`
- `data/processed/data_update_status.json`
- `data/interim/trade_matched.parquet`
- `data/interim/rent_matched.parquet`

시가총액 갱신 시에는 다음도 포함합니다.

- `config/market_cap_area_master.csv`
- `data/processed/market_cap/manifest.json`
- 새 `data/processed/market_cap/YYYY-MM/<run_id>/`
- `data/processed/market_cap/audit.json`과 관련 CSV
- 새 `data/processed/market_cap/kb/<run_id>/`
- `data/processed/market_cap/kb/latest.json`
- `data/processed/market_cap/kb/summary.csv`

원본 `data/raw`와 수집 메타데이터는 로컬 배치용이며 기본 Git 배포 대상이 아닙니다. `.env`도 절대 배포 커밋에 포함하지 않습니다.

보정 정책 원본과 생성 측 snapshot은 `area_master` 저장소의 `config/market_cap_kb_adjustments.json`, `data/processed/market_cap/`에서 별도로 관리합니다. 웹서비스 배포에는 동기화된 사본만 포함합니다.

### 8.2 커밋 전 확인

생성 산출물만 선택적으로 stage한 뒤 포함 파일을 다시 확인합니다.

```powershell
git diff --cached --stat
git diff --cached --name-only
```

다음을 확인한 뒤 커밋하고 운영 브랜치로 push합니다.

- 의도하지 않은 원본·임시 파일이 없는지
- `latest.json`이 가리키는 KB snapshot 디렉터리가 함께 포함되었는지
- `manifest.json`이 참조하는 실거래 시가총액 snapshot이 함께 포함되었는지
- `.env`, API 키, 개인 경로가 포함되지 않았는지
- 대용량 파일 제한을 초과하지 않는지

Git 연동 Streamlit 배포라면 push 후 새 배포가 정상 완료됐는지 확인합니다. 서버에 파일을 직접 복사하는 환경이라면 산출물을 모두 복사한 후 Streamlit 프로세스를 재시작합니다. 새 프로세스가 시작되면 메모리 캐시도 초기화됩니다.

---

## 9. 배포 후 확인과 롤백

### 9.1 배포 후 확인

1. 웹서비스가 오류 없이 열리는지 확인합니다.
2. 데이터 갱신 현황의 `최신 월`과 `반영 시각`을 확인합니다.
3. 최신 월의 매매·전세·월세 건수를 로컬 `data_update_status.json`과 대조합니다.
4. 대표 단지 2~3개의 최근 거래와 가격 추이를 확인합니다.
5. KB 시가총액 화면의 배포본과 산정 완료 건수를 `kb/latest.json`과 대조합니다.
6. 실거래 시가총액의 최신 종료 월과 등급별 건수를 `market_cap/audit.json`과 대조합니다.

### 9.2 실패 시 원칙

- 수집 일부 실패: 실패 구·군/월만 다시 수집한 뒤 `build`부터 재실행합니다.
- `build` 실패: 오류 원인을 수정하기 전에는 새 중간 산출물을 배포하지 않습니다.
- KB 검증 실패: `area_master`의 마스터·상태·혼합단지 파일 조합을 확인합니다.
- 운영 화면이 이전 데이터 표시: 배포 파일과 프로세스 재시작 여부를 확인합니다.
- 운영 오류: 직전 정상 Git 커밋 또는 직전 정상 배포 산출물로 되돌리고 서비스를 재시작합니다.

시가총액 snapshot은 불변 이력으로 저장됩니다. KB만 롤백해야 한다면 직전 정상 snapshot의 `metadata.json` 내용을 기준으로 `kb/latest.json`을 복구할 수 있지만, 운영에서는 관련 산출물이 모두 일치하도록 직전 정상 Git 커밋 단위로 롤백하는 것이 안전합니다.

---

## 10. 월간 운영 체크리스트

- [ ] 현재 월과 직전 2개월의 매매 수집 완료
- [ ] 현재 월과 직전 2개월의 전월세 수집 완료
- [ ] 16개 구·군 수집 상태 확인
- [ ] `build --incremental` 성공
- [ ] `data_update_status.json` 최신 월·완료 상태 확인
- [ ] 매매 갱신 시 최근 종료 월 시가총액 배치 성공
- [ ] `area_master` 변경 시 KB 시가총액 배치 성공
- [ ] 보정 정책 변경 시 평형 ID·세대수·출처·방식 검증 및 `adjustment_policy.version` 갱신
- [ ] 평형 마스터 변경 시 실거래 시가총액 재산정 여부 결정
- [ ] 보정 대상 대표 단지의 금액·보정 세대수·출처 표시 확인
- [ ] 품질·수동검토·보완 목록 확인
- [ ] 전체 테스트 통과
- [ ] 로컬 화면 확인
- [ ] 배포 대상 파일과 snapshot 참조 무결성 확인
- [ ] 운영 배포 완료
- [ ] 운영 화면 최신 월·반영 시각·대표 단지 확인
