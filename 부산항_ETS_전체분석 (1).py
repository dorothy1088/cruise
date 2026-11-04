# -*- coding: utf-8 -*-
"""실행: python -m streamlit run 부산항_ETS_전체분석.py
원본 CSV를 이 파일과 같은 폴더에 두거나 화면에서 업로드하세요.
필요 패키지: streamlit plotly numpy pandas scipy statsmodels
기존 재현 환경: statsmodels 0.15.0, pandas 2.2.3, numpy 2.3.5
"""
from pathlib import Path
from io import BytesIO
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:
    print('다음 명령으로 실행하세요:')
    print(f'"{sys.executable}" -m streamlit run "{Path(__file__).resolve()}"')
    raise SystemExit(0)

BASE_DIR = Path(__file__).resolve().parent
SEED = 20261030
SIMULATIONS = 10000

def read_csv_auto(content):
    for encoding in ['utf-8-sig','cp949','euc-kr']:
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError('CSV 인코딩을 확인하세요.')

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


@st.cache_data(show_spinner='원본 집계와 ETS 예측을 계산하고 있습니다.')
def calculate(data_bytes, schedule_bytes):
    df = read_csv_auto(data_bytes)
    df.columns = df.columns.str.strip()
    required = {'입출', '입항일시', '승객', '호출부호', '선명'}
    if not required.issubset(df.columns):
        raise ValueError(f'필수 열 누락: {required - set(df.columns)}')
    inbound = df.loc[df['입출'].astype(str).str.strip().eq('입항')].copy()
    inbound['입항일시'] = pd.to_datetime(inbound['입항일시'], errors='coerce')
    if inbound['입항일시'].isna().any():
        raise ValueError('입항일시가 해석되지 않는 행을 먼저 확인하세요.')
    inbound['승객'] = pd.to_numeric(inbound['승객'].astype('string').str.replace(',', '', regex=False).str.strip(), errors='coerce')
    if (inbound['승객'].dropna() < 0).any():
        raise ValueError('음수 승객 기록을 확인하세요.')
    inbound['호출부호'] = inbound['호출부호'].astype('string').str.strip()
    passenger_rows = inbound.loc[inbound['입항일시'].between('2023-01-01', '2026-09-30 23:59:59'), ['입항일시', '호출부호', '선명', '승객']].copy().sort_values('입항일시')
    passenger_rows['월'] = passenger_rows['입항일시'].dt.to_period('M')
    month_index = pd.period_range('2023-01', '2026-09', freq='M')
    passenger_monthly = passenger_rows.groupby('월').agg(입항건수=('승객', 'size'), 유효기록수=('승객', 'count'), 확인승객합계=('승객', lambda x: x.sum(min_count=1))).reindex(month_index)
    no_calls = passenger_monthly['입항건수'].isna()
    confirmed_zero = {'2023-01', '2023-02', '2023-12', '2024-01'}
    unknown = set(month_index[no_calls].astype(str)) - confirmed_zero
    if unknown:
        raise ValueError(f'미확인 무기록 월: {unknown}')
    passenger_monthly.loc[no_calls, ['입항건수', '유효기록수', '확인승객합계']] = 0
    passenger_monthly['결측입항건수'] = passenger_monthly['입항건수'] - passenger_monthly['유효기록수']
    passenger_monthly['실제승객합계'] = passenger_monthly['확인승객합계'].where(passenger_monthly['결측입항건수'].eq(0))
    passenger_monthly['자료상태'] = np.where(no_calls, '팀 확인 무입항 0명', np.where(passenger_monthly['결측입항건수'].gt(0), '승객 기록 불완전', '승객 기록 완전'))
    passenger_monthly.index = passenger_monthly.index.to_timestamp()
    passenger_monthly.index.name = '연월'
    actual_passengers = passenger_monthly['실제승객합계'].asfreq('MS')
    def training_passengers(cutoff, quantile=0.5):
        end = pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        rows = passenger_rows.loc[passenger_rows['입항일시'] < end].copy()
        rows['학습승객'] = rows['승객']
        log = []
        for idx, row in rows.loc[rows['승객'].isna()].iterrows():
            prior = rows.loc[(rows['입항일시'] < row['입항일시']) & rows['승객'].notna()]
            same = prior.loc[prior['호출부호'].eq(row['호출부호'])]
            if same.empty:
                raise ValueError('동일 선박 과거 승객 기록이 없어 별도 결측 처리 검토 필요')
            value = float(same['승객'].quantile(quantile))
            rows.loc[idx, '학습승객'] = value
            log.append({'입항일시': str(row['입항일시']), '선명': row['선명'], '이전기록수': len(same), '보완값': value, '참조최종일': str(same['입항일시'].max())})
        index = pd.period_range('2023-01', pd.Timestamp(cutoff).to_period('M'), freq='M')
        y = rows.groupby('월')['학습승객'].sum().reindex(index, fill_value=0).astype(float)
        y.index = y.index.to_timestamp()
        y.index.freq = 'MS'
        assert y.notna().all() and (y >= 0).all()
        return (y, pd.DataFrame(log))
    training_y, imputation_log = training_passengers('2026-09-01')
    passenger_monthly['학습용승객합계'] = training_y
    train_2026, log_2026 = training_passengers('2025-12-01')
    months_2026 = pd.period_range('2026-01', '2026-09', freq='M').astype(str).tolist()
    validation = forecast(train_2026, months_2026)
    validation['실제승객합계'] = actual_passengers.loc['2026-01':'2026-09'].to_numpy()
    validation['오차(실제−예측)'] = validation['실제승객합계'] - validation['예측승객수']
    validation['절대오차'] = validation['오차(실제−예측)'].abs()
    validation['월별 APE(%)'] = percentage_error(validation['실제승객합계'], validation['예측승객수'])
    score = metrics(validation)
    train_final, log_final = training_passengers('2026-09-01')
    future = forecast(train_final, pd.period_range('2026-10', '2027-12', freq='M').astype(str).tolist())
    schedule = read_csv_auto(schedule_bytes)
    schedule['입항예정일'] = pd.to_datetime(schedule['입항예정일'], errors='raise')
    schedule['선명'] = schedule['선명'].str.strip()
    schedule['정원'] = pd.to_numeric(schedule['승객정원수'].astype(str).str.replace(',', '', regex=False).str.strip(), errors='raise')
    if not schedule['입항예정일'].dt.year.eq(2027).all() or not schedule['정원'].gt(0).all():
        raise ValueError('스케줄의 연도 또는 정원을 확인하세요.')
    if schedule.duplicated(['선명', '입항예정일']).any():
        raise ValueError('동일 선명·입항예정일 중복 기록을 확인하세요.')
    capacity = schedule.groupby(schedule['입항예정일'].dt.strftime('%Y-%m')).agg(예정입항건수=('정원', 'size'), 승객정원합계=('정원', 'sum')).rename_axis('연월').reset_index()
    comparison = future.loc[future['연월'].str.startswith('2027')].merge(capacity, on='연월', how='left', validate='one_to_one')
    if comparison['승객정원합계'].isna().any():
        raise ValueError('스케줄에 기록 없는 달이 있습니다. 0 여부를 먼저 확인하세요.')
    comparison['차이(정원−예측)'] = comparison['승객정원합계'] - comparison['예측승객수']
    comparison['정원 대비 절대차이율(%)'] = percentage_error(comparison['승객정원합계'], comparison['예측승객수'])
    cap_total = comparison['승객정원합계'].sum()
    pred_total = comparison['예측승객수'].sum()
    monthly = passenger_monthly.reset_index()
    monthly["연월"] = monthly["연월"].dt.strftime("%Y-%m")
    return monthly, validation, future, comparison, score, imputation_log

st.set_page_config(page_title='부산항 ETS 승객 전망',page_icon='⚓',layout='wide')
st.title('부산항 크루즈 월별 승객 전망')
st.caption('ETS 감쇠 추세 계절모형 · 12개월 계절 주기 · 95% 근사 예측구간')
with st.sidebar:
    st.header('원본 데이터')
    upload_data=st.file_uploader('입출항 원자료 CSV',type=['csv'],key='raw')
    upload_schedule=st.file_uploader('2027년 예정 스케줄 CSV',type=['csv'],key='schedule')
    st.caption('업로드하지 않으면 이 파이썬 파일과 같은 폴더의 원본을 읽습니다.')
    st.caption('입항 기록의 승객 수 기준이며 실제 하선 인원이 아닙니다.')

def input_bytes(upload, filename):
    if upload is not None:
        return upload.getvalue()
    path=BASE_DIR/filename
    if not path.is_file():
        st.info(f'{filename} 파일을 왼쪽에서 업로드하거나 파이썬 파일과 같은 폴더에 놓아주세요.')
        return None
    return path.read_bytes()

data_bytes=input_bytes(upload_data,'i.csv')
schedule_bytes=input_bytes(upload_schedule,'2027년 크루즈 부산항 입항 예정 스케줄 - sheet1.csv')
if data_bytes is None or schedule_bytes is None:
    st.stop()
try:
    monthly,validation,future,comparison,score,imputation_log=calculate(data_bytes,schedule_bytes)
except Exception as exc:
    st.error(f'계산을 완료하지 못했습니다: {exc}')
    st.stop()
tabs=st.tabs(['2026년 실제값 검증','15개월 승객 전망','2027년 정원 비교','자료와 계산 기준'])
def table_download(df, filename):
    st.dataframe(df.round(2), hide_index=True, width='stretch')
    st.download_button('표 CSV 내려받기', df.to_csv(index=False).encode('utf-8-sig'), filename, 'text/csv')

def line_chart(df, actual_col=None, interval=False):
    fig = go.Figure()
    if interval:
        fig.add_trace(go.Scatter(x=df['연월'], y=df['95% 상한'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=df['연월'], y=df['95% 하한'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(35,117,190,0.18)', name='95% 근사 예측구간'))
    if actual_col:
        fig.add_trace(go.Scatter(x=df['연월'], y=df[actual_col], mode='lines+markers', name=actual_col, line=dict(color='#182f48')))
    fig.add_trace(go.Scatter(x=df['연월'], y=df['예측승객수'], mode='lines+markers', name='ETS 예측', line=dict(color='#2375be')))
    fig.update_layout(yaxis_title='승객 수 (명)', xaxis_title='연월', hovermode='x unified', legend=dict(orientation='h', y=1.15), height=440)
    st.plotly_chart(fig, width='stretch')

def error_chart(df, column, mean):
    fig = go.Figure(go.Bar(x=df['연월'], y=df[column], name=column, marker_color='#2375be', text=df[column].round(2), textposition='outside'))
    fig.add_hline(y=mean, line_dash='dash', line_color='#d46a35', annotation_text=f'월별 비율 평균 {mean:.2f}%')
    fig.update_layout(yaxis_title='비율 (%)', xaxis_title='연월', height=360)
    st.plotly_chart(fig, width='stretch')

with tabs[0]:
    st.subheader('2026년 1~9월 실제값과 예측값')
    st.write('학습: 2023년 1월~2025년 12월 · 평가: 2026년 1~9월')
    cols = st.columns(4)
    for col, key in zip(cols, ['MAPE(%)','MAE(명)','RMSE(명)','WAPE(%)']):
        col.metric(key, f'{score[key]:,.2f}')
    line_chart(validation, '실제승객합계', interval=True)
    st.subheader('월별 오차비율과 평균 MAPE')
    error_chart(validation, '월별 APE(%)', score['MAPE(%)'])
    st.caption('오차 = 실제−예측. APE = |실제−예측| / 실제 × 100. MAPE는 월별 APE의 단순평균입니다. 실제가 0 또는 결측이면 APE와 MAPE 평가에서 제외합니다.')
    table_download(validation,'2026_validation.csv')
    st.info('이 기간은 이전 모형 비교에서 이미 확인한 자료입니다. 독립적인 최종 시험 성적으로 해석하지 않습니다.')
with tabs[1]:
    st.subheader('2026년 10월~2027년 12월 전망')
    st.write('학습: 2023년 1월~2026년 9월 · 2026년 9월 말 기준 전망')
    c1,c2=st.columns(2)
    c1.metric('2026년 10~12월 예측 합계', f"{future.iloc[:3]['예측승객수'].sum():,.0f}명")
    c2.metric('2027년 예측 합계', f"{future.iloc[3:]['예측승객수'].sum():,.0f}명")
    line_chart(future, interval=True)
    st.caption('고정된 모형 모수로 생성한 10,000개 경로의 월별 2.5~97.5백분위수입니다. 실제 포함률 95%를 보장하지 않으며 15개월 전체의 동시 구간도 아닙니다. 월별 경계를 합산해 연간 구간으로 해석하지 않습니다.')
    table_download(future,'forecast_95.csv')
with tabs[2]:
    st.subheader('예정 스케줄 승객 정원과 예측 비교')
    st.write('예정 정원은 학습에 사용하지 않았습니다. 실제 승객 수가 아니므로 예측 정확도 또는 탑승률로 해석하지 않습니다.')
    avg=comparison['정원 대비 절대차이율(%)'].mean()
    cap=comparison['승객정원합계'].sum(); pred=comparison['예측승객수'].sum()
    c1,c2,c3=st.columns(3)
    c1.metric('예정 정원 합계',f'{cap:,.0f}명')
    c2.metric('월별 절대차이율 평균',f'{avg:.2f}%')
    c3.metric('연간 합계 기준 차이율',f'{abs(cap-pred)/cap*100:.2f}%')
    line_chart(comparison,'승객정원합계')
    error_chart(comparison,'정원 대비 절대차이율(%)',avg)
    st.caption('정원 대비 절대차이율 = |정원−예측| / 정원 × 100. 월별 차이율 평균과 연간 합계 차이율은 서로 다른 지표입니다.')
    table_download(comparison,'2027_capacity_comparison.csv')
with tabs[3]:
    st.subheader('자료 범위와 전처리')
    st.write('원본 i.csv의 입항 기록을 직접 집계하여 45개월 자료를 만듭니다.')
    st.write('팀 확인 무입항 월: 2023년 1·2·12월, 2024년 1월 → 0명. 2020~2022년은 학습 범위 밖입니다.')
    st.write('2025년 3·12월은 승객 결측 입항이 각각 1건입니다. 해당 입항 이전 동일 호출부호의 중앙값 316.5명·129명을 학습용으로 보완했습니다. 실제 월합계는 결측으로 유지합니다.')
    st.write('2026년 9월까지 완료월이라는 기존 가정을 유지합니다. 10월 부분 기록은 사용하지 않습니다.')
    st.subheader('모형과 구간 계산')
    st.code("ETSModel(y / 1000, error='add', trend='add', damped_trend=True,\n         seasonal='add', seasonal_periods=12,\n         initialization_method='heuristic')")
    st.write('로그 변환 없이 천 명 단위로 적합한 뒤 명 단위로 복원합니다. 최대 반복 2,000회, 수렴 확인, 음수 예측 및 경로는 0으로 제한합니다. 난수 시드 20261030을 고정했습니다.')
    st.write('95% 근사 예측구간은 모수 추정·결측 보완·일정 변경의 불확실성을 충분히 반영하지 못합니다. 운영 준비는 실제 입항 일정 및 예약 자료와 함께 검토해야 합니다.')
    table_download(monthly,'monthly_passengers.csv')

    st.subheader("학습용 결측 보완 내역")
    st.dataframe(imputation_log, hide_index=True, width="stretch")
