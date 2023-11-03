import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import streamlit as st

st.markdown(
    f"""
    <style>
        .centered-text {{
            text-align: center;
        }}
    </style>
    """
    , unsafe_allow_html=True
)

st.markdown("<h1 class='centered-text'>창원 중앙고 프로그래밍 대회</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>LSTM model (predicting stock price)</h2>", unsafe_allow_html=True)
# st.markdown("<h3 class='centered-text'>주가 예측</h3>", unsafe_allow_html=True)

# 화면 나누기
left_column, right_column = st.columns(2)

## 선택 박스
# 회사
input_company = st.sidebar.text_input("주식 회사(심볼) ex)AAPL")

# 주가 종류
type_list = ['Open', 'High', 'Low', 'Close', 'Adj Close']
# 기본값
default_index_type = type_list.index('Adj Close') if 'Adj Close' in type_list else 0
selected_type = st.sidebar.selectbox("주가", type_list, index=default_index_type)

# 가져올 데이터   
# 1~15
year_list = list(range(1, 16))
# 기본값
default_index_year = year_list.index(10) if 10 in year_list else 0
selected_number_year = st.sidebar.selectbox("가져올 주가 기간(년)", year_list, index=default_index_year)

## 모델 학습 관련
    
# 입력값 형태
# 14~30
lookback_list = list(range(14, 31))
# 기본값
default_index_lookback = lookback_list.index(21) if 21 in lookback_list else 0
selected_number_lookback = st.sidebar.selectbox("모델 학습 입력값의 시퀀스 길이(일)", lookback_list, index=default_index_lookback)
    
# 데이터 분할
# 0.7~0,9
split_list = [0.7, 0.75, 0.8, 0.85, 0.9]
# 기본값
default_index_split = split_list.index(0.8) if 0.8 in split_list else 0
selected_number_split = st.sidebar.selectbox("학습용 데이터 비율(0.8추천)", split_list, index=default_index_split)
    
# 경사하강
# 50~100
epochs_list = list(range(50,101))
# 기본값
default_index_epochs = epochs_list.index(50) if 50 in epochs_list else 0
selected_number_epochs = st.sidebar.selectbox("경사하강 회수(epochs)", epochs_list, index=default_index_epochs) 


year = int(selected_number_year)
        
import datetime
# 현재 날짜
today = datetime.date.today()
days=365*year
# 전의 날짜 계산
date_ago = today - datetime.timedelta(days=days)
# 주식 데이터 다운로드
company = input_company
data = yf.download(company, start= date_ago, end=today) #AAPL, GOOG, MSFT, TSLA, AMZN
price = data[selected_type].values

scaler = MinMaxScaler()
price = price.reshape(-1,1)
price_scaled = scaler.fit_transform(price)

# 데이터셋 생성
X, y = [], []
look_back = int(selected_number_lookback)  # 몇일치 데이터를 활용할 것인지 설정
for i in range(len(price_scaled) - look_back):
    X.append(price_scaled[i:i + look_back])
    y.append(price_scaled[i + look_back])
X, y = np.array(X), np.array(y)

# 데이터셋 분할
split_ratio = selected_number_split
split = int(len(X) * split_ratio)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape = (look_back, 1)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs=selected_number_epochs, batch_size=64)

predicted = model.predict(X_test)

with left_column:
    
    from sklearn.metrics import r2_score
    # R2 결정 계수
    r2 = r2_score(y_test, predicted)
    
    # 역변환
    predicted = scaler.inverse_transform(predicted)
    y_test = scaler.inverse_transform(y_test)
    
    st.write("## 전체 데이터 그래프")
    # 전체 데이터 시각화
    predicted_all = model.predict(X)
    predicted_all = scaler.inverse_transform(predicted_all)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, price, label='Real stock price', alpha=0.7)
    plt.plot(data.index[-len(predicted_all):-len(y_test)], predicted_all[:-len(y_test)], label='Predicted stock price(train)', linestyle='--')
    plt.plot(data.index[-len(y_test):], predicted, label='Predicted stock price(test)', linestyle='--', c='r')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title("Whole Data")
    plt.legend()
    
    st.pyplot(plt)
    
    
    st.write("## 테스트용 데이터 그래프")
    # 그래프 생성(테스트용 데이터)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], y_test, label='Real stock price')
    plt.plot(data.index[-len(y_test):], predicted, label='Predicted stock price', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price (test data)')
    plt.legend()
    
    st.pyplot(plt)
    

with right_column:
    st.write("## 최근 그래프" )
    # 확대 그래프
    back = 31
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-back:], y_test[-back:], label='Real stock price')
    plt.plot(data.index[-back:], predicted[-back:], label='Predicted stock price', linestyle = '--')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price(recent data)')
    plt.legend()
        
    st.pyplot(plt)
    st.subheader(f"\nR2결정계수 : {r2}")
    st.write("R2(결정계수)는 모델이 데이터에 얼마나 적합한지 평가하는 통계적 척도이다. R2는 주로 0~1사이의 값을 가지며, 1에 가까울수록 모델이 데이터를 잘 설명함을 의미하고 0에 가까울수록 설명하지 못함을 의미한다. 주로 R2가 0.7이상이면 좋은 모델이라 평가한다.")

acc = 0
adj_acc = 0
for i in range(0, len(y_test)-1):
    if (predicted[i+1][0]-predicted[i][0])*(y_test[i+1]-y_test[i]) > 0:
        acc += 1
for i in range(0, len(y_test)-2):
    if (predicted[i+1][0]-predicted[i][0])*(y_test[i+2]-y_test[i+1]) > 0:
        adj_acc += 1
acc_perc = acc/(len(y_test)-1)*100
adj_acc_perc = adj_acc/(len(y_test)-2)*100

st.write("## ⓘ")
st.write("**3일 이상 뒤의 예측부터는 예측력이 매우 떨어진다. 1,2일 뒤 예측 주가만 참고하는게 바람직하다.**")
st.write(f'**이 모델은 <span style="color: red;">{acc_perc}%</span>의 확률로 주가의 상승, 하락을 올바르게 예측한다.**', unsafe_allow_html=True)
st.write(f'**이 모델은 <span style="color: red;">{adj_acc_perc}%</span>의 확률로 주가의 상승, 하락을 올바르게 예측한다.(조정된 예측)**', unsafe_allow_html=True)




st.write("## 미래 주가 예측 그래프") 
after = 0
after_days = []
price = data[selected_type].values
price = price.reshape(-1,1)
last_days = price[-look_back:]
future_price = []
future_price = np.array(future_price)
future_price = future_price.reshape(-1, 1)

for i in range(1,5):
    after += 1
    after_days.append(after)
    # 내일 주가 예측
    last_days = last_days[-look_back:]  # 마지막 데이터
    scaled_last_days = scaler.transform(last_days.reshape(-1, 1))

    # 주가 예측
    predicted_price = model.predict(scaled_last_days.reshape(1, look_back, 1))
    predicted_price = scaler.inverse_transform(predicted_price)
    last_days = np.append(last_days, predicted_price, axis=0)
    future_price = np.append(future_price, predicted_price, axis=0)
    # sake = predicted_price[0][0] - price[-1][0]

last_days = list(last_days)
after_days = [0] + after_days
plt.figure(figsize=(12, 6))
plt.plot(after_days, last_days[-5:], label='Predicted stock price', c = 'orange', marker='o')
plt.xlabel('Days after')
plt.ylabel('Stock Price')
plt.title('Stock Price(futere)')
for i, price in enumerate(last_days[-5:]):
    plt.text(after_days[i], price, f'{round(float(price), 2)}', ha='left', va='bottom')
plt.legend()
st.pyplot(plt)

st.write(f"오늘 주가 : {data[selected_type].values[-1]}")
for i in range(0,4):
    d = after_days[i+1]
    p = future_price[i][0]
    st.write(f"{d}일 뒤 예상 주가 : {p}")


st.subheader("도큐먼트")

code = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import streamlit as st

st.markdown(
    f"""
    <style>
        .centered-text {{
            text-align: center;
        }}
    </style>
    """
    , unsafe_allow_html=True
)

st.markdown("<h1 class='centered-text'>창원 중앙고 프로그래밍 대회</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>LSTM model (predicting stock price)</h2>", unsafe_allow_html=True)
# st.markdown("<h3 class='centered-text'>주가 예측</h3>", unsafe_allow_html=True)

# 화면 나누기
left_column, right_column = st.columns(2)

## 선택 박스
# 회사
input_company = st.sidebar.text_input("주식 회사(심볼) ex)AAPL")

# 주가 종류
type_list = ['Open', 'High', 'Low', 'Close', 'Adj Close']
# 기본값
default_index_type = type_list.index('Adj Close') if 'Adj Close' in type_list else 0
selected_type = st.sidebar.selectbox("주가", type_list, index=default_index_type)

# 가져올 데이터   
# 1~15
year_list = list(range(1, 16))
# 기본값
default_index_year = year_list.index(10) if 10 in year_list else 0
selected_number_year = st.sidebar.selectbox("가져올 주가 기간(년)", year_list, index=default_index_year)

## 모델 학습 관련
    
# 입력값 형태
# 14~30
lookback_list = list(range(14, 31))
# 기본값
default_index_lookback = lookback_list.index(21) if 21 in lookback_list else 0
selected_number_lookback = st.sidebar.selectbox("모델 학습 입력값의 시퀀스 길이(일)", lookback_list, index=default_index_lookback)
    
# 데이터 분할
# 0.7~0,9
split_list = [0.7, 0.75, 0.8, 0.85, 0.9]
# 기본값
default_index_split = split_list.index(0.8) if 0.8 in split_list else 0
selected_number_split = st.sidebar.selectbox("학습용 데이터 비율(0.8추천)", split_list, index=default_index_split)
    
# 경사하강
# 50~100
epochs_list = list(range(50,101))
# 기본값
default_index_epochs = epochs_list.index(50) if 50 in epochs_list else 0
selected_number_epochs = st.sidebar.selectbox("경사하강 회수(epochs)", epochs_list, index=default_index_epochs) 


year = int(selected_number_year)
        
import datetime
# 현재 날짜
today = datetime.date.today()
days=365*year
# 전의 날짜 계산
date_ago = today - datetime.timedelta(days=days)
# 주식 데이터 다운로드
company = input_company
data = yf.download(company, start= date_ago, end=today) #AAPL, GOOG, MSFT, TSLA, AMZN
price = data[selected_type].values

scaler = MinMaxScaler()
price = price.reshape(-1,1)
price_scaled = scaler.fit_transform(price)

# 데이터셋 생성
X, y = [], []
look_back = int(selected_number_lookback)  # 몇일치 데이터를 활용할 것인지 설정
for i in range(len(price_scaled) - look_back):
    X.append(price_scaled[i:i + look_back])
    y.append(price_scaled[i + look_back])
X, y = np.array(X), np.array(y)

# 데이터셋 분할
split_ratio = selected_number_split
split = int(len(X) * split_ratio)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape = (look_back, 1)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs=selected_number_epochs, batch_size=64)

predicted = model.predict(X_test)

with left_column:
    
    from sklearn.metrics import r2_score
    # R2 결정 계수
    r2 = r2_score(y_test, predicted)
    
    # 역변환
    predicted = scaler.inverse_transform(predicted)
    y_test = scaler.inverse_transform(y_test)
    
    st.write("## 전체 데이터 그래프")
    # 전체 데이터 시각화
    predicted_all = model.predict(X)
    predicted_all = scaler.inverse_transform(predicted_all)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, price, label='Real stock price', alpha=0.7)
    plt.plot(data.index[-len(predicted_all):-len(y_test)], predicted_all[:-len(y_test)], label='Predicted stock price(train)', linestyle='--')
    plt.plot(data.index[-len(y_test):], predicted, label='Predicted stock price(test)', linestyle='--', c='r')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title("Whole Data")
    plt.legend()
    
    st.pyplot(plt)
    
    
    st.write("## 테스트용 데이터 그래프")
    # 그래프 생성(테스트용 데이터)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], y_test, label='Real stock price')
    plt.plot(data.index[-len(y_test):], predicted, label='Predicted stock price', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price (test data)')
    plt.legend()
    
    st.pyplot(plt)
    

with right_column:
    st.write("## 최근 그래프" )
    # 확대 그래프
    back = 31
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-back:], y_test[-back:], label='Real stock price')
    plt.plot(data.index[-back:], predicted[-back:], label='Predicted stock price', linestyle = '--')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price(recent data)')
    plt.legend()
        
    st.pyplot(plt)
    st.subheader(f"\nR2결정계수 : {r2}")
    st.write("R2(결정계수)는 모델이 데이터에 얼마나 적합한지 평가하는 통계적 척도이다. R2는 주로 0~1사이의 값을 가지며, 1에 가까울수록 모델이 데이터를 잘 설명함을 의미하고 0에 가까울수록 설명하지 못함을 의미한다. 주로 R2가 0.7이상이면 좋은 모델이라 평가한다.")

acc = 0
adj_acc = 0
for i in range(0, len(y_test)-1):
    if (predicted[i+1][0]-predicted[i][0])*(y_test[i+1]-y_test[i]) > 0:
        acc += 1
for i in range(0, len(y_test)-2):
    if (predicted[i+1][0]-predicted[i][0])*(y_test[i+2]-y_test[i+1]) > 0:
        adj_acc += 1
acc_perc = acc/(len(y_test)-1)*100
adj_acc_perc = adj_acc/(len(y_test)-2)*100

st.write("## ⓘ")
st.write("**3일 이상 뒤의 예측부터는 예측력이 매우 떨어진다. 1,2일 뒤 예측 주가만 참고하는게 바람직하다.**")
st.write(f'**이 모델은 <span style="color: red;">{acc_perc}%</span>의 확률로 주가의 상승, 하락을 올바르게 예측한다.**', unsafe_allow_html=True)
st.write(f'**이 모델은 <span style="color: red;">{adj_acc_perc}%</span>의 확률로 주가의 상승, 하락을 올바르게 예측한다.(조정된 예측)**', unsafe_allow_html=True)




st.write("## 미래 주가 예측 그래프") 
after = 0
after_days = []
price = data[selected_type].values
price = price.reshape(-1,1)
last_days = price[-look_back:]
future_price = []
future_price = np.array(future_price)
future_price = future_price.reshape(-1, 1)

for i in range(1,5):
    after += 1
    after_days.append(after)
    # 내일 주가 예측
    last_days = last_days[-look_back:]  # 마지막 데이터
    scaled_last_days = scaler.transform(last_days.reshape(-1, 1))

    # 주가 예측
    predicted_price = model.predict(scaled_last_days.reshape(1, look_back, 1))
    predicted_price = scaler.inverse_transform(predicted_price)
    last_days = np.append(last_days, predicted_price, axis=0)
    future_price = np.append(future_price, predicted_price, axis=0)
    # sake = predicted_price[0][0] - price[-1][0]

last_days = list(last_days)
after_days = [0] + after_days
plt.figure(figsize=(12, 6))
plt.plot(after_days, last_days[-5:], label='Predicted stock price', c = 'orange', marker='o')
plt.xlabel('Days after')
plt.ylabel('Stock Price')
plt.title('Stock Price(futere)')
for i, price in enumerate(last_days[-5:]):
    plt.text(after_days[i], price, f'{round(float(price), 2)}', ha='left', va='bottom')
plt.legend()
st.pyplot(plt)

st.write(f"오늘 주가 : {data[selected_type].values[-1]}")
for i in range(0,4):
    d = after_days[i+1]
    p = future_price[i][0]
    st.write(f"{d}일 뒤 예상 주가 : {p}")


st.subheader("도큐먼트")
 
    '''
st.code(code, language='python')
  
