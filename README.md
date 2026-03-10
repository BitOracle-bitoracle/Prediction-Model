# BitOracle - 가상자산 데이터 분석 및 정보 공유 플랫폼

> 숭실대학교 컴퓨터학부 전공종합설계 1, 2 캡스톤 디자인 프로젝트

BitOracle은 GRU(Gated Recurrent Unit) 딥러닝 모델을 기반으로 비트코인 가격 방향(상승/하락)을 예측하고, 예측 결과를 실시간으로 시각화하는 가상자산 분석 플랫폼입니다.

---

## 주요 기능

- **BTC/USDT 가격 방향 예측** — GRU 모델을 통해 다음 1시간 봉 기준 상승 확률 제공
- **실시간 가격 차트** — Binance API 연동으로 최신 BTC 가격 데이터 제공
- **기술적 보조지표 계산** — RSI, MACD, ATR, Stochastic(%K/%D), 이동평균선 등 자동 계산
- **REST API 서버** — FastAPI 기반 백엔드, React 프론트엔드 연동 지원

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 모델 | GRU (Keras / TensorFlow 2.x) |
| 백엔드 | FastAPI, Uvicorn |
| 데이터 수집 | CCXT (Binance), yfinance |
| 데이터 처리 | Pandas, NumPy, scikit-learn |
| 시각화 | Matplotlib |

---

## 모델 버전 히스토리

| 버전 | 설명 |
|------|------|
| v12 | 초기 GRU 분류 모델 (LSTM → GRU 전환) |
| v14 | 하이퍼파라미터 튜닝, 외부 시장 데이터(S&P500, DXY 등) 실험 |
| v15 | 학습 안정화 (외부 데이터 제거, BTC 내부 지표 14개 확정) |
| v16 | 스마트 피처 엔지니어링 적용 |
| v17 | CNN + GRU 하이브리드 아키텍처 실험 |
| v18 | 수익 최대화 목표 손실 함수 실험 |

> **최종 사용 모델: v14** — 48시간 윈도우, 14개 피처(OHLCV + 기술적 지표)

---

## 모델 입력 피처 (14개)

**가격 피처 (7개):** `Open`, `High`, `Low`, `Close`, `Volume`, `MA5`, `MA20`

**기술적 지표 (7개):** `RSI`, `MACD`, `Signal_Line`, `Log_Return`, `ATR`, `%K`, `%D`

- 전처리: 가격 피처는 `pct_change`, 지표 피처는 `diff` 적용 후 `StandardScaler` 정규화
- 시퀀스 윈도우: 과거 **48시간** 데이터를 입력으로 다음 1시간 방향 예측

---

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/predict/chart` | GRU 모델 예측 결과 (상승 확률 0.0~1.0) 반환 |
| GET | `/api/price/chart` | 실제 BTC/USDT 가격 히스토리 반환 |

---

## 프로젝트 구조

```
LSTMServer/
├── v14/
│   ├── GRUServer.py              # FastAPI 서버 (최종 배포용)
│   ├── GRU_v14_Tunned.py         # v14 모델 학습 코드
│   └── v14_Backtest(StopLoss).py # 백테스트 (스탑로스 전략)
├── v15&v16/
│   ├── GRU_v15_Fixed.py          # v15 학습 코드
│   ├── GRU_v16_Smart.py          # v16 학습 코드
│   ├── GRU_v17_Hybrid_CNN.py     # v17 CNN+GRU 실험
│   ├── GRU_v18_Profit_Maximizer.py # v18 수익 최적화 실험
│   └── Feature_Importance_Check.py # 피처 중요도 분석
├── Cryptocurrency-Prediction-Model/ # 초기 LSTM 실험 코드
├── BitOracle_PredictModelV12.py  # v12 학습 코드
└── README.md
```

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install fastapi uvicorn tensorflow ccxt yfinance pandas numpy scikit-learn
```

### 2. 서버 실행

```bash
cd v14
uvicorn GRUServer:app --host 0.0.0.0 --port 8000 --reload
```

### 3. API 확인

```
http://localhost:8000/docs
```

---

## 주의사항

- 이 프로젝트의 예측 결과는 **학술 연구 목적**이며, 실제 투자 판단의 근거로 사용하지 마십시오.
- 모델 `.keras` 파일은 용량 문제로 저장소에 포함되지 않습니다. 학습 코드를 직접 실행하여 생성하세요.

---

## 개발자

숭실대학교 컴퓨터학부 | 전공종합설계 1, 2 (2024)
