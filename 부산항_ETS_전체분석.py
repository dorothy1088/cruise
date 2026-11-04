# -*- coding: utf-8 -*-
"""부산항 ETS 감쇠 추세 계절모형 전체 분석.
원본 CSV 두 개를 작업 폴더에 놓고 실행하세요.
실행: python 부산항_ETS_전체분석.py
필요 패키지와 데이터 경로는 아래를 확인하세요.
"""

# # 부산항 크루즈 월별 승객 수: ETS 감쇠 추세 계절모형
# 원본 CSV 불러오기부터 월별 집계, 결측 보완, 모형 학습, 평가, 95% 근사 예측구간까지 순서대로 실행합니다.
# 별도 Python 모듈이나 미리 집계한 학습 CSV는 필요하지 않습니다.
# 
# **준비:** 이 노트북과 같은 폴더에 `i.csv`, `2027년 크루즈 부산항 입항 예정 스케줄 - sheet1.csv`를 놓으세요.
# 입력 파일명이 다르면 아래 경로 두 줄만 바꾸세요. VS Code에서 Python 커널을 선택하고 모두 실행합니다.
# 
# ETS(A, Ad, A), 가법 오차·가법 감쇠 추세·가법 계절성, 12개월 주기입니다. 로그 변환을 사용하지 않습니다.
# 2026년 평가: 2023-01~2025-12 학습 → 2026-01~09.
# 최종 전망: 2023-01~2026-09 재학습 → 2026-10~2027-12.
# 2026년 결과는 이미 이전 모형 비교에 사용했으므로 독립 최종 시험으로 해석하지 않습니다.

# 처음 환경을 구성할 때만 아래 주석을 해제해 실행하고 커널을 다시 시작하세요.
# %pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 statsmodels==0.15.0 plotly==7.1.0 streamlit==1.66.1 nbformat ipykernel
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
def display(value):
    """일반 Python 실행에서는 표를 터미널에 출력합니다."""
    if isinstance(value, pd.DataFrame):
        print(value.to_string())
    else:
        print(value)
    print()
DATA_PATH = Path('./data/i.csv')
SCHEDULE_PATH = Path('2027년 크루즈 부산항 입항 예정 스케줄 - sheet1.csv')
SEED = 20261030
SIMULATIONS = 10000
pd.set_option('display.max_columns', 20)


# ## 1. 원본 데이터 불러오기와 입항 기록 추출
# 팀이 선종·입항목적·청코드 조건으로 수집한 CSV를 사용합니다. 입출 구분이 입항인 기록만 합산하고 출항 행을 중복 합산하지 않습니다. 수집 조건 자체를 원자료의 모든 열에서 재검증한 것은 아닙니다.

def read_csv_auto(path):
    for encoding in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'CSV 인코딩을 확인하세요: {path}')
df = read_csv_auto(DATA_PATH)
df.columns = df.columns.str.strip()
required = {'입출', '입항일시', '승객', '호출부호', '선명'}
if not required.issubset(df.columns):
    raise ValueError(f'필수 열 누락: {required-set(df.columns)}')
inbound = df.loc[df['입출'].astype(str).str.strip().eq('입항')].copy()
inbound['입항일시'] = pd.to_datetime(inbound['입항일시'], errors='coerce')
if inbound['입항일시'].isna().any():
    raise ValueError('입항일시가 해석되지 않는 행을 먼저 확인하세요.')
inbound['승객'] = pd.to_numeric(inbound['승객'].astype('string').str.replace(',', '', regex=False).str.strip(), errors='coerce')
if (inbound['승객'].dropna()<0).any():
    raise ValueError('음수 승객 기록을 확인하세요.')
inbound['호출부호'] = inbound['호출부호'].astype('string').str.strip()
display(pd.DataFrame({'항목':['원본 행 수','입항 행 수','입항 승객 결측 행 수'], '값':[len(df),len(inbound),inbound['승객'].isna().sum()]}))
display(inbound[['입항일시','호출부호','선명','승객']].head())


# ## 2. 월별 집계와 학습용 결측 보완
# 2023년 1월~2026년 9월을 사용하고 2026년 10월 부분 기록은 제외합니다. 9월까지 완료월이라는 기존 가정을 유지합니다.
# 팀 확인 무입항 월은 2023년 1·2·12월, 2024년 1월로 0명 처리합니다. 이외에 새로운 무기록 월이 있으면 자동으로 0을 넣지 않고 중단합니다. 2020~2022년은 학습 범위 밖입니다.
# 승객 결측 입항이 하나라도 있는 월은 **실제 월합계를 결측**으로 둡니다.
# 학습용 값만 해당 입항보다 과거의 동일 호출부호 관측 중앙값으로 보완합니다. 미래 기록과 보완된 기록을 참조하지 않습니다.

passenger_rows = inbound.loc[inbound['입항일시'].between('2023-01-01', '2026-09-30 23:59:59'), ['입항일시','호출부호','선명','승객']].copy().sort_values('입항일시')
passenger_rows['월'] = passenger_rows['입항일시'].dt.to_period('M')
month_index = pd.period_range('2023-01','2026-09',freq='M')
passenger_monthly = passenger_rows.groupby('월').agg(입항건수=('승객','size'), 유효기록수=('승객','count'), 확인승객합계=('승객',lambda x:x.sum(min_count=1))).reindex(month_index)
no_calls = passenger_monthly['입항건수'].isna()
confirmed_zero = {'2023-01','2023-02','2023-12','2024-01'}
unknown = set(month_index[no_calls].astype(str)) - confirmed_zero
if unknown: raise ValueError(f'미확인 무기록 월: {unknown}')
passenger_monthly.loc[no_calls, ['입항건수','유효기록수','확인승객합계']] = 0
passenger_monthly['결측입항건수'] = passenger_monthly['입항건수']-passenger_monthly['유효기록수']
passenger_monthly['실제승객합계'] = passenger_monthly['확인승객합계'].where(passenger_monthly['결측입항건수'].eq(0))
passenger_monthly['자료상태'] = np.where(no_calls,'팀 확인 무입항 0명',np.where(passenger_monthly['결측입항건수'].gt(0),'승객 기록 불완전','승객 기록 완전'))
passenger_monthly.index = passenger_monthly.index.to_timestamp()
passenger_monthly.index.name = '연월'
actual_passengers = passenger_monthly['실제승객합계'].asfreq('MS')

def training_passengers(cutoff, quantile=0.5):
    # Only this training window and earlier observed records are eligible.
    end = pd.Timestamp(cutoff)+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)
    rows = passenger_rows.loc[passenger_rows['입항일시']<end].copy()
    rows['학습승객'] = rows['승객']
    log = []
    for idx,row in rows.loc[rows['승객'].isna()].iterrows():
        prior = rows.loc[(rows['입항일시']<row['입항일시']) & rows['승객'].notna()]
        same = prior.loc[prior['호출부호'].eq(row['호출부호'])]
        if same.empty:raise ValueError('동일 선박 과거 승객 기록이 없어 별도 결측 처리 검토 필요')
        value = float(same['승객'].quantile(quantile))
        rows.loc[idx,'학습승객'] = value
        log.append({'입항일시':str(row['입항일시']),'선명':row['선명'],'이전기록수':len(same),'보완값':value,'참조최종일':str(same['입항일시'].max())})
    index = pd.period_range('2023-01',pd.Timestamp(cutoff).to_period('M'),freq='M')
    y = rows.groupby('월')['학습승객'].sum().reindex(index,fill_value=0).astype(float)
    y.index = y.index.to_timestamp();y.index.freq='MS'
    assert y.notna().all() and (y>=0).all()
    return y,pd.DataFrame(log)


training_y, imputation_log = training_passengers('2026-09-01')
passenger_monthly['학습용승객합계'] = training_y
display(passenger_monthly)
display(imputation_log)


# ## 3. ETS 모형과 95% 근사 예측구간 함수
# 천 명 단위로 적합한 뒤 명으로 복원합니다. heuristic 초기화와 최대 2,000회 반복을 사용하며 수렴을 확인합니다. 고정 모수 조건의 시뮬레이션 10,000경로에서 월별 2.5·97.5백분위수를 구합니다. 음수 예측과 경로는 0으로 제한합니다. 모수 추정·결측 보완·일정 변경의 불확실성을 충분히 반영하지 못하며 실제 포함률 95%를 보장하지 않습니다.

def forecast(train, months, seed=SEED):
    # 수치 안정화를 위한 천 명 단위 환산. 로그 변환은 하지 않습니다.
    y = np.asarray(train, dtype=float) / 1000
    fit = ETSModel(y, error='add', trend='add', damped_trend=True,
                   seasonal='add', seasonal_periods=12,
                   initialization_method='heuristic').fit(maxiter=2000, disp=False)
    if not fit.mle_retvals.get('converged', False):
        raise RuntimeError('ETS 최적화가 수렴하지 않았습니다.')
    h = len(months)
    point = np.maximum(np.asarray(fit.forecast(h)), 0) * 1000
    paths = np.asarray(fit.simulate(h, anchor='end', repetitions=SIMULATIONS,
                       rng=np.random.default_rng(seed))).reshape(h, SIMULATIONS)
    lower, upper = np.quantile(np.maximum(paths, 0) * 1000, [.025, .975], axis=1)
    return pd.DataFrame({'연월': months, '예측승객수': point,
                         '95% 하한': lower, '95% 상한': upper})

def percentage_error(actual, predicted):
    # 실제값 0 또는 결측인 월의 APE는 정의되지 않아 NaN으로 유지합니다.
    a = pd.Series(actual, dtype=float)
    p = pd.Series(predicted, dtype=float)
    return (a-p).abs().div(a.where(a > 0)) * 100

def metrics(frame):
    valid = frame[['실제승객합계', '예측승객수']].dropna()
    actual, pred = valid['실제승객합계'], valid['예측승객수']
    error = actual - pred
    return {'MAPE(%)': float(percentage_error(actual, pred).mean()),
            'MAE(명)': float(error.abs().mean()),
            'RMSE(명)': float(np.sqrt((error**2).mean())),
            'WAPE(%)': float(error.abs().sum()/actual.sum()*100) if actual.sum()>0 else np.nan,
            'MAPE 평가월수': int((actual > 0).sum())}



# ## 4. 2026년 1~9월 실제값과 예측값 비교
# 2025년 12월까지만 학습합니다. 오차=실제−예측, APE=절대오차/실제×100, MAPE=월별 APE의 단순평균입니다. 실제값이 0 또는 결측인 달은 APE와 MAPE에서 제외합니다. MAE·RMSE·WAPE는 유효한 실제값으로 계산합니다.

train_2026, log_2026 = training_passengers('2025-12-01')
months_2026 = pd.period_range('2026-01','2026-09',freq='M').astype(str).tolist()
validation = forecast(train_2026, months_2026)
validation['실제승객합계'] = actual_passengers.loc['2026-01':'2026-09'].to_numpy()
validation['오차(실제−예측)'] = validation['실제승객합계']-validation['예측승객수']
validation['절대오차'] = validation['오차(실제−예측)'].abs()
validation['월별 APE(%)'] = percentage_error(validation['실제승객합계'], validation['예측승객수'])
score = metrics(validation)
display(validation.round(2))
display(pd.DataFrame([score]).round(2))


# ## 5. 월별 실제·예측 및 오차비율 시각화
# 노트북에서는 Plotly 그래프를 바로 표시합니다. 마지막 선택 실행 셀에서 동일 코드를 Streamlit 화면으로도 실행할 수 있습니다.

def plot_forecast(frame, title, actual_col=None, interval=True):
    fig=go.Figure()
    if interval:
        fig.add_trace(go.Scatter(x=frame['연월'],y=frame['95% 상한'],mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=frame['연월'],y=frame['95% 하한'],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(35,117,190,.18)',name='95% 근사 예측구간'))
    if actual_col:
        fig.add_trace(go.Scatter(x=frame['연월'],y=frame[actual_col],mode='lines+markers',name=actual_col))
    fig.add_trace(go.Scatter(x=frame['연월'],y=frame['예측승객수'],mode='lines+markers',name='ETS 예측'))
    fig.update_layout(title=title,xaxis_title='연월',yaxis_title='승객 수 (명)',hovermode='x unified')
    fig.show()

def plot_rate(frame, column, title):
    mean=frame[column].mean()
    fig=go.Figure(go.Bar(x=frame['연월'],y=frame[column],text=frame[column].round(2),textposition='outside'))
    fig.add_hline(y=mean,line_dash='dash',annotation_text=f'월별 비율 평균 {mean:.2f}%')
    fig.update_layout(title=title,xaxis_title='연월',yaxis_title='비율 (%)')
    fig.show()

plot_forecast(validation,'2026년 1~9월 실제 승객 수와 예측','실제승객합계')
plot_rate(validation,'월별 APE(%)','2026년 월별 오차비율과 평균 MAPE')


# ## 6. 최종 재학습 및 15개월 전망
# 2023년 1월~2026년 9월로 재학습합니다. 2026년 10~12월과 2027년 12개월을 예측합니다. 월별 구간의 경계를 합산해서 연간 구간으로 해석하지 않습니다.

train_final, log_final = training_passengers('2026-09-01')
future = forecast(train_final, pd.period_range('2026-10','2027-12',freq='M').astype(str).tolist())
display(future.round(2))
display(pd.DataFrame([{'2026년 10~12월 예측 합계':future.iloc[:3]['예측승객수'].sum(), '2027년 예측 합계':future.iloc[3:]['예측승객수'].sum()}]).round(2))
plot_forecast(future,'2026년 10월~2027년 12월 전망과 95% 근사 예측구간')


# ## 7. 2027년 원본 예정 스케줄 불러오기와 정원 비교
# 같은 선박의 반복 입항은 회차별로 합산합니다. 정원은 학습에 사용하지 않습니다. 정원 대비 절대차이율은 실제 승객 수에 대한 MAPE나 탑승률이 아닙니다.

schedule = read_csv_auto(SCHEDULE_PATH)
schedule['입항예정일'] = pd.to_datetime(schedule['입항예정일'], errors='raise')
schedule['선명'] = schedule['선명'].str.strip()
schedule['정원'] = pd.to_numeric(schedule['승객정원수'].astype(str).str.replace(',','',regex=False).str.strip(),errors='raise')
if not schedule['입항예정일'].dt.year.eq(2027).all() or not schedule['정원'].gt(0).all():
    raise ValueError('스케줄의 연도 또는 정원을 확인하세요.')
if schedule.duplicated(['선명','입항예정일']).any():
    raise ValueError('동일 선명·입항예정일 중복 기록을 확인하세요.')
capacity = schedule.groupby(schedule['입항예정일'].dt.strftime('%Y-%m')).agg(예정입항건수=('정원','size'),승객정원합계=('정원','sum')).rename_axis('연월').reset_index()
comparison = future.loc[future['연월'].str.startswith('2027')].merge(capacity,on='연월',how='left',validate='one_to_one')
if comparison['승객정원합계'].isna().any():
    raise ValueError('스케줄에 기록 없는 달이 있습니다. 0 여부를 먼저 확인하세요.')
comparison['차이(정원−예측)'] = comparison['승객정원합계']-comparison['예측승객수']
comparison['정원 대비 절대차이율(%)'] = percentage_error(comparison['승객정원합계'], comparison['예측승객수'])
cap_total=comparison['승객정원합계'].sum()
pred_total=comparison['예측승객수'].sum()
display(comparison.round(2))
display(pd.DataFrame([{'정원 합계':cap_total,'예측 합계':pred_total,'월별 절대차이율 평균(%)':comparison['정원 대비 절대차이율(%)'].mean(),'연간 합계 기준 차이율(%)':abs(cap_total-pred_total)/cap_total*100}]).round(2))
plot_forecast(comparison,'2027년 예정 정원과 예측','승객정원합계',interval=False)
plot_rate(comparison,'정원 대비 절대차이율(%)','2027년 월별 정원 대비 절대차이율')


# ## 8. Streamlit으로 보기 — 선택 실행
# 분석은 앞의 셀까지로 완료됩니다. 아래 셀은 이 노트북 안의 동일한 계산 코드를 Streamlit 실행용 파일로 자동 생성합니다.
# 별도 코드를 다운로드할 필요가 없습니다. 생성 후 터미널에서 `python -m streamlit run ets_streamlit_app.py`를 실행하세요.
# Streamlit은 노트북 출력 영역에서 직접 실행되는 방식이 아니므로 실행용 파일이 필요합니다. 원본 CSV 두 개는 같은 작업 폴더에 둡니다.

# 선택 실행: False를 True로 바꾸면 Streamlit 실행용 파일을 생성합니다.
MAKE_STREAMLIT_APP = False
if MAKE_STREAMLIT_APP:
    app_source = "import streamlit as st\nst.set_page_config(page_title='부산항 ETS 승객 전망',layout='wide')\nst.title('부산항 ETS 감쇠 추세 계절모형')\n# 처음 환경을 구성할 때만 아래 주석을 해제해 실행하고 커널을 다시 시작하세요.\n# %pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 statsmodels==0.15.0 plotly==7.1.0 streamlit==1.66.1 nbformat ipykernel\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport plotly.graph_objects as go\nfrom statsmodels.tsa.exponential_smoothing.ets import ETSModel\ndisplay = st.dataframe\nDATA_PATH = Path('i.csv')\nSCHEDULE_PATH = Path('2027년 크루즈 부산항 입항 예정 스케줄 - sheet1.csv')\nSEED = 20261030\nSIMULATIONS = 10000\npd.set_option('display.max_columns', 20)\n\nst.markdown('## 1. 원본 데이터 불러오기와 입항 기록 추출\\n팀이 선종·입항목적·청코드 조건으로 수집한 CSV를 사용합니다. 입출 구분이 입항인 기록만 합산하고 출항 행을 중복 합산하지 않습니다. 수집 조건 자체를 원자료의 모든 열에서 재검증한 것은 아닙니다.')\ndef read_csv_auto(path):\n    for encoding in ['utf-8-sig', 'cp949', 'euc-kr']:\n        try:\n            return pd.read_csv(path, encoding=encoding)\n        except UnicodeDecodeError:\n            continue\n    raise ValueError(f'CSV 인코딩을 확인하세요: {path}')\ndf = read_csv_auto(DATA_PATH)\ndf.columns = df.columns.str.strip()\nrequired = {'입출', '입항일시', '승객', '호출부호', '선명'}\nif not required.issubset(df.columns):\n    raise ValueError(f'필수 열 누락: {required-set(df.columns)}')\ninbound = df.loc[df['입출'].astype(str).str.strip().eq('입항')].copy()\ninbound['입항일시'] = pd.to_datetime(inbound['입항일시'], errors='coerce')\nif inbound['입항일시'].isna().any():\n    raise ValueError('입항일시가 해석되지 않는 행을 먼저 확인하세요.')\ninbound['승객'] = pd.to_numeric(inbound['승객'].astype('string').str.replace(',', '', regex=False).str.strip(), errors='coerce')\nif (inbound['승객'].dropna()<0).any():\n    raise ValueError('음수 승객 기록을 확인하세요.')\ninbound['호출부호'] = inbound['호출부호'].astype('string').str.strip()\ndisplay(pd.DataFrame({'항목':['원본 행 수','입항 행 수','입항 승객 결측 행 수'], '값':[len(df),len(inbound),inbound['승객'].isna().sum()]}))\ndisplay(inbound[['입항일시','호출부호','선명','승객']].head())\n\nst.markdown('## 2. 월별 집계와 학습용 결측 보완\\n2023년 1월~2026년 9월을 사용하고 2026년 10월 부분 기록은 제외합니다. 9월까지 완료월이라는 기존 가정을 유지합니다.\\n팀 확인 무입항 월은 2023년 1·2·12월, 2024년 1월로 0명 처리합니다. 이외에 새로운 무기록 월이 있으면 자동으로 0을 넣지 않고 중단합니다. 2020~2022년은 학습 범위 밖입니다.\\n승객 결측 입항이 하나라도 있는 월은 **실제 월합계를 결측**으로 둡니다.\\n학습용 값만 해당 입항보다 과거의 동일 호출부호 관측 중앙값으로 보완합니다. 미래 기록과 보완된 기록을 참조하지 않습니다.\\n')\npassenger_rows = inbound.loc[inbound['입항일시'].between('2023-01-01', '2026-09-30 23:59:59'), ['입항일시','호출부호','선명','승객']].copy().sort_values('입항일시')\npassenger_rows['월'] = passenger_rows['입항일시'].dt.to_period('M')\nmonth_index = pd.period_range('2023-01','2026-09',freq='M')\npassenger_monthly = passenger_rows.groupby('월').agg(입항건수=('승객','size'), 유효기록수=('승객','count'), 확인승객합계=('승객',lambda x:x.sum(min_count=1))).reindex(month_index)\nno_calls = passenger_monthly['입항건수'].isna()\nconfirmed_zero = {'2023-01','2023-02','2023-12','2024-01'}\nunknown = set(month_index[no_calls].astype(str)) - confirmed_zero\nif unknown: raise ValueError(f'미확인 무기록 월: {unknown}')\npassenger_monthly.loc[no_calls, ['입항건수','유효기록수','확인승객합계']] = 0\npassenger_monthly['결측입항건수'] = passenger_monthly['입항건수']-passenger_monthly['유효기록수']\npassenger_monthly['실제승객합계'] = passenger_monthly['확인승객합계'].where(passenger_monthly['결측입항건수'].eq(0))\npassenger_monthly['자료상태'] = np.where(no_calls,'팀 확인 무입항 0명',np.where(passenger_monthly['결측입항건수'].gt(0),'승객 기록 불완전','승객 기록 완전'))\npassenger_monthly.index = passenger_monthly.index.to_timestamp()\npassenger_monthly.index.name = '연월'\nactual_passengers = passenger_monthly['실제승객합계'].asfreq('MS')\n\ndef training_passengers(cutoff, quantile=0.5):\n    # Only this training window and earlier observed records are eligible.\n    end = pd.Timestamp(cutoff)+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)\n    rows = passenger_rows.loc[passenger_rows['입항일시']<end].copy()\n    rows['학습승객'] = rows['승객']\n    log = []\n    for idx,row in rows.loc[rows['승객'].isna()].iterrows():\n        prior = rows.loc[(rows['입항일시']<row['입항일시']) & rows['승객'].notna()]\n        same = prior.loc[prior['호출부호'].eq(row['호출부호'])]\n        if same.empty:raise ValueError('동일 선박 과거 승객 기록이 없어 별도 결측 처리 검토 필요')\n        value = float(same['승객'].quantile(quantile))\n        rows.loc[idx,'학습승객'] = value\n        log.append({'입항일시':str(row['입항일시']),'선명':row['선명'],'이전기록수':len(same),'보완값':value,'참조최종일':str(same['입항일시'].max())})\n    index = pd.period_range('2023-01',pd.Timestamp(cutoff).to_period('M'),freq='M')\n    y = rows.groupby('월')['학습승객'].sum().reindex(index,fill_value=0).astype(float)\n    y.index = y.index.to_timestamp();y.index.freq='MS'\n    assert y.notna().all() and (y>=0).all()\n    return y,pd.DataFrame(log)\n\n\ntraining_y, imputation_log = training_passengers('2026-09-01')\npassenger_monthly['학습용승객합계'] = training_y\ndisplay(passenger_monthly)\ndisplay(imputation_log)\n\nst.markdown('## 3. ETS 모형과 95% 근사 예측구간 함수\\n천 명 단위로 적합한 뒤 명으로 복원합니다. heuristic 초기화와 최대 2,000회 반복을 사용하며 수렴을 확인합니다. 고정 모수 조건의 시뮬레이션 10,000경로에서 월별 2.5·97.5백분위수를 구합니다. 음수 예측과 경로는 0으로 제한합니다. 모수 추정·결측 보완·일정 변경의 불확실성을 충분히 반영하지 못하며 실제 포함률 95%를 보장하지 않습니다.')\ndef forecast(train, months, seed=SEED):\n    # 수치 안정화를 위한 천 명 단위 환산. 로그 변환은 하지 않습니다.\n    y = np.asarray(train, dtype=float) / 1000\n    fit = ETSModel(y, error='add', trend='add', damped_trend=True,\n                   seasonal='add', seasonal_periods=12,\n                   initialization_method='heuristic').fit(maxiter=2000, disp=False)\n    if not fit.mle_retvals.get('converged', False):\n        raise RuntimeError('ETS 최적화가 수렴하지 않았습니다.')\n    h = len(months)\n    point = np.maximum(np.asarray(fit.forecast(h)), 0) * 1000\n    paths = np.asarray(fit.simulate(h, anchor='end', repetitions=SIMULATIONS,\n                       rng=np.random.default_rng(seed))).reshape(h, SIMULATIONS)\n    lower, upper = np.quantile(np.maximum(paths, 0) * 1000, [.025, .975], axis=1)\n    return pd.DataFrame({'연월': months, '예측승객수': point,\n                         '95% 하한': lower, '95% 상한': upper})\n\ndef percentage_error(actual, predicted):\n    # 실제값 0 또는 결측인 월의 APE는 정의되지 않아 NaN으로 유지합니다.\n    a = pd.Series(actual, dtype=float)\n    p = pd.Series(predicted, dtype=float)\n    return (a-p).abs().div(a.where(a > 0)) * 100\n\ndef metrics(frame):\n    valid = frame[['실제승객합계', '예측승객수']].dropna()\n    actual, pred = valid['실제승객합계'], valid['예측승객수']\n    error = actual - pred\n    return {'MAPE(%)': float(percentage_error(actual, pred).mean()),\n            'MAE(명)': float(error.abs().mean()),\n            'RMSE(명)': float(np.sqrt((error**2).mean())),\n            'WAPE(%)': float(error.abs().sum()/actual.sum()*100) if actual.sum()>0 else np.nan,\n            'MAPE 평가월수': int((actual > 0).sum())}\n\n\nst.markdown('## 4. 2026년 1~9월 실제값과 예측값 비교\\n2025년 12월까지만 학습합니다. 오차=실제−예측, APE=절대오차/실제×100, MAPE=월별 APE의 단순평균입니다. 실제값이 0 또는 결측인 달은 APE와 MAPE에서 제외합니다. MAE·RMSE·WAPE는 유효한 실제값으로 계산합니다.')\ntrain_2026, log_2026 = training_passengers('2025-12-01')\nmonths_2026 = pd.period_range('2026-01','2026-09',freq='M').astype(str).tolist()\nvalidation = forecast(train_2026, months_2026)\nvalidation['실제승객합계'] = actual_passengers.loc['2026-01':'2026-09'].to_numpy()\nvalidation['오차(실제−예측)'] = validation['실제승객합계']-validation['예측승객수']\nvalidation['절대오차'] = validation['오차(실제−예측)'].abs()\nvalidation['월별 APE(%)'] = percentage_error(validation['실제승객합계'], validation['예측승객수'])\nscore = metrics(validation)\ndisplay(validation.round(2))\ndisplay(pd.DataFrame([score]).round(2))\n\nst.markdown('## 5. 월별 실제·예측 및 오차비율 시각화\\n노트북에서는 Plotly 그래프를 바로 표시합니다. 마지막 선택 실행 셀에서 동일 코드를 Streamlit 화면으로도 실행할 수 있습니다.')\ndef plot_forecast(frame, title, actual_col=None, interval=True):\n    fig=go.Figure()\n    if interval:\n        fig.add_trace(go.Scatter(x=frame['연월'],y=frame['95% 상한'],mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))\n        fig.add_trace(go.Scatter(x=frame['연월'],y=frame['95% 하한'],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(35,117,190,.18)',name='95% 근사 예측구간'))\n    if actual_col:\n        fig.add_trace(go.Scatter(x=frame['연월'],y=frame[actual_col],mode='lines+markers',name=actual_col))\n    fig.add_trace(go.Scatter(x=frame['연월'],y=frame['예측승객수'],mode='lines+markers',name='ETS 예측'))\n    fig.update_layout(title=title,xaxis_title='연월',yaxis_title='승객 수 (명)',hovermode='x unified')\n    st.plotly_chart(fig, width='stretch')\n\ndef plot_rate(frame, column, title):\n    mean=frame[column].mean()\n    fig=go.Figure(go.Bar(x=frame['연월'],y=frame[column],text=frame[column].round(2),textposition='outside'))\n    fig.add_hline(y=mean,line_dash='dash',annotation_text=f'월별 비율 평균 {mean:.2f}%')\n    fig.update_layout(title=title,xaxis_title='연월',yaxis_title='비율 (%)')\n    st.plotly_chart(fig, width='stretch')\n\nplot_forecast(validation,'2026년 1~9월 실제 승객 수와 예측','실제승객합계')\nplot_rate(validation,'월별 APE(%)','2026년 월별 오차비율과 평균 MAPE')\n\nst.markdown('## 6. 최종 재학습 및 15개월 전망\\n2023년 1월~2026년 9월로 재학습합니다. 2026년 10~12월과 2027년 12개월을 예측합니다. 월별 구간의 경계를 합산해서 연간 구간으로 해석하지 않습니다.')\ntrain_final, log_final = training_passengers('2026-09-01')\nfuture = forecast(train_final, pd.period_range('2026-10','2027-12',freq='M').astype(str).tolist())\ndisplay(future.round(2))\ndisplay(pd.DataFrame([{'2026년 10~12월 예측 합계':future.iloc[:3]['예측승객수'].sum(), '2027년 예측 합계':future.iloc[3:]['예측승객수'].sum()}]).round(2))\nplot_forecast(future,'2026년 10월~2027년 12월 전망과 95% 근사 예측구간')\n\nst.markdown('## 7. 2027년 원본 예정 스케줄 불러오기와 정원 비교\\n같은 선박의 반복 입항은 회차별로 합산합니다. 정원은 학습에 사용하지 않습니다. 정원 대비 절대차이율은 실제 승객 수에 대한 MAPE나 탑승률이 아닙니다.')\nschedule = read_csv_auto(SCHEDULE_PATH)\nschedule['입항예정일'] = pd.to_datetime(schedule['입항예정일'], errors='raise')\nschedule['선명'] = schedule['선명'].str.strip()\nschedule['정원'] = pd.to_numeric(schedule['승객정원수'].astype(str).str.replace(',','',regex=False).str.strip(),errors='raise')\nif not schedule['입항예정일'].dt.year.eq(2027).all() or not schedule['정원'].gt(0).all():\n    raise ValueError('스케줄의 연도 또는 정원을 확인하세요.')\nif schedule.duplicated(['선명','입항예정일']).any():\n    raise ValueError('동일 선명·입항예정일 중복 기록을 확인하세요.')\ncapacity = schedule.groupby(schedule['입항예정일'].dt.strftime('%Y-%m')).agg(예정입항건수=('정원','size'),승객정원합계=('정원','sum')).rename_axis('연월').reset_index()\ncomparison = future.loc[future['연월'].str.startswith('2027')].merge(capacity,on='연월',how='left',validate='one_to_one')\nif comparison['승객정원합계'].isna().any():\n    raise ValueError('스케줄에 기록 없는 달이 있습니다. 0 여부를 먼저 확인하세요.')\ncomparison['차이(정원−예측)'] = comparison['승객정원합계']-comparison['예측승객수']\ncomparison['정원 대비 절대차이율(%)'] = percentage_error(comparison['승객정원합계'], comparison['예측승객수'])\ncap_total=comparison['승객정원합계'].sum()\npred_total=comparison['예측승객수'].sum()\ndisplay(comparison.round(2))\ndisplay(pd.DataFrame([{'정원 합계':cap_total,'예측 합계':pred_total,'월별 절대차이율 평균(%)':comparison['정원 대비 절대차이율(%)'].mean(),'연간 합계 기준 차이율(%)':abs(cap_total-pred_total)/cap_total*100}]).round(2))\nplot_forecast(comparison,'2027년 예정 정원과 예측','승객정원합계',interval=False)\nplot_rate(comparison,'정원 대비 절대차이율(%)','2027년 월별 정원 대비 절대차이율')\n\n"
    Path('ets_streamlit_app.py').write_text(app_source,encoding='utf-8')
    print('터미널에서 실행: python -m streamlit run ets_streamlit_app.py')


# 모형 참고: https://www.statsmodels.org/stable/examples/notebooks/generated/ets.html
# 
# 승객 수는 실제 하선 인원이나 필요 인력 수를 직접 뜻하지 않습니다. 터미널 준비는 실제 일정·예약 정보를 함께 확인하세요.
