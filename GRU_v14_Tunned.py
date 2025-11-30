import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



def get_binance_data(symbol='BTC/USDT', since='2014-10-12', limit=1000):
    exchange = ccxt.binance()
    since_ms = exchange.parse8601(since + 'T00:00:00Z')
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', since=since_ms, limit=limit)
        if not ohlcv: break
        all_ohlcv += ohlcv
        since_ms = ohlcv[-1][0] + 60 * 60 * 1000
        if len(ohlcv) < limit: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    return df.drop(columns=['timestamp'])

# === 1. 데이터 다운로드 (1시간 봉) ===
btc_data = get_binance_data("BTC/USDT" , since='2018-01-01')

# === 2. 외부 시장 데이터 (1시간 봉) ===
tickers = {'S&P500': '^GSPC', '10YR_Yield': '^TNX', 'DXY': 'DX-Y.NYB', 'Gold': 'GLD'}
start_date = btc_data.index.min().strftime('%Y-%m-%d')
end_date = (btc_data.index.max() + timedelta(days=1)).strftime('%Y-%m-%d')
yf_data = yf.download(list(tickers.values()), start=start_date, end=end_date, interval='1h')

# === 3. Feature 리스트 정의 ===
price_features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20',
    'Open_S&P500', 'Open_10YR_Yield', 'Open_DXY', 'Open_Gold'
]
indicator_features = [
    'RSI', 'MACD', 'Signal_Line', 'Log_Return', 'ATR', '%K', '%D'
]
features = price_features + indicator_features
external_features = ['Open_S&P500', 'Open_10YR_Yield', 'Open_DXY', 'Open_Gold']

# === 4. 데이터 병합 및 결측치 처리 (Robust) ===
if yf_data.empty or yf_data.get('Open') is None or yf_data['Open'].isnull().all().all():
    print("="*50)
    print("!!! 경고: YFinance 1h 데이터 다운로드 실패. !!!")
    print("!!! 외부 Feature 없이 비트코인 데이터만으로 학습을 진행합니다. !!!")
    print("="*50)
    combined_data = btc_data
    features = [f for f in features if f not in external_features]
    price_features = [f for f in price_features if f not in external_features]
else:
    print("YFinance 1h 데이터 다운로드 성공. 데이터 병합을 진행합니다.")
    yf_close_data = yf_data['Close'] 
    yf_close_data.columns = [f"Open_{key}" for key in tickers.keys()]
    combined_data = pd.merge(btc_data, yf_close_data, left_index=True, right_index=True, how='left')
    combined_data.ffill(inplace=True)
    combined_data.bfill(inplace=True)

# === 5. 보조지표 계산 ===
def calculate_technical_indicators(df):
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0)).rolling(window=14).mean() / (-df['Close'].diff().where(df['Close'].diff() < 0, 0)).rolling(window=14).mean()))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    df['%K'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
    df['%D'] = df['%K'].rolling(3).mean()
    return df

btc_data_with_indicators = calculate_technical_indicators(combined_data.copy())

# === 6. Feature 변환 (Stationarizing) ===
for col in price_features:
    if col in btc_data_with_indicators.columns:
        btc_data_with_indicators[col] = btc_data_with_indicators[col].pct_change(1)
for col in indicator_features:
     if col in btc_data_with_indicators.columns:
        btc_data_with_indicators[col] = btc_data_with_indicators[col].diff(1)

# === 7. inf 값 처리 ===
btc_data_with_indicators.replace([np.inf, -np.inf], np.nan, inplace=True)

# === 8. 최종 데이터 전처리 ===
btc_data_processed = btc_data_with_indicators.dropna()

# === 9. Feature 정규화 (StandardScaler) ===
feature_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(btc_data_processed[features])

# === 10. Y(Target) 데이터 생성 (1시간 뒤 분류) ===
window_size = 48  # === 1. 윈도우 90 -> 48 (2일)로 단축 ===
future_target_day = 1
x, y = [], []
close_prices = combined_data.loc[btc_data_processed.index, 'Close'].values

for i in range(len(scaled_features) - window_size - (future_target_day - 1)):
    x.append(scaled_features[i:i + window_size, :])
    current_price = close_prices[i + window_size - 1]
    future_price = close_prices[i + window_size - 1 + future_target_day]
    y.append(1 if future_price > current_price else 0)

x, y = np.array(x), np.array(y)

# 학습 및 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
print(f"Total Samples: {len(x)}, Train Samples: {len(x_train)}, Test Samples: {len(x_test)}")


# === 11. 모델 구성 (GRU 용량 증가) (NEW) ===
model = Sequential([
    # GRU 유닛 32 -> 64로 증가 (패턴 학습 능력 향상)
    GRU(64, activation='tanh', input_shape=(window_size, x_train.shape[2]), 
        return_sequences=False
    ),
    BatchNormalization(),
    Dropout(0.4), # 규제 유지
    
    # Dense 유닛 16 -> 32로 증가
    Dense(32, activation='relu'),
    Dropout(0.3), # 규제 유지
    
    Dense(1, activation='sigmoid')
])
model.summary()

# === 12. 컴파일 설정 ===
model.compile(
    optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# === 13. 콜백 함수 ===
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('best_model_GRU_tuned_v14.keras', monitor='val_accuracy', save_best_only=True, mode='max')
]

# === 14. 모델 학습 ===
history = model.fit(
    x_train, y_train,
    epochs=150,
    batch_size=256,
    validation_split=0.2,
    callbacks=callbacks,
    shuffle=False
)

# === 15. 모델 평가 ===
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")

predictions_prob = model.predict(x_test)
predicted_classes = (predictions_prob > 0.5).astype(int)

print("\n--- Classification Report (GRU v14 - Tuned) ---")
print(classification_report(y_test, predicted_classes, target_names=['Down (0)', 'Up (1)'], zero_division=0))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, predicted_classes))


# === 16. 시각화 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy (GRU v14 - Tuned)')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid()
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss (GRU v14 - Tuned)')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid()
plt.tight_layout()
plt.show()