# -*- coding: utf-8 -*-
"""실행: IDE의 Python 실행 버튼 또는 python 부산항_ETS_전체분석.py
Streamlit 서버와 브라우저가 자동으로 시작됩니다.
실제 입출항 CSV를 업로드하고 완료월을 선택하세요. 예정 스케줄은 선택 사항입니다.
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

def launch_streamlit():
    """일반 Python 실행 시 서버를 한 번 시작하고 준비되면 브라우저를 엽니다."""
    import socket
    import subprocess
    import threading
    import time
    import urllib.request
    import webbrowser

    script = Path(__file__).resolve()
    # 실행 중인 다른 프로그램과 충돌하지 않는 로컬 포트를 선택합니다.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [sys.executable, '-m', 'streamlit', 'run', str(script),
               '--server.address=127.0.0.1', f'--server.port={port}',
               '--server.headless=true', '--global.developmentMode=false',
               '--browser.gatherUsageStats=false']
    # 인수를 리스트로 전달하므로 파일명에 공백이나 (1)이 있어도 실행됩니다.
    process = subprocess.Popen(command, cwd=str(script.parent))
    stopped = threading.Event()

    def open_when_ready():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 90
        while not stopped.is_set() and process.poll() is None and time.monotonic() < deadline:
            try:
                with opener.open(url + '/_stcore/health', timeout=1) as response:
                    ready = response.status == 200
                if ready:
                    print(f'브라우저 주소: {url}', flush=True)
                    if not webbrowser.open(url, new=2):
                        print('브라우저 자동 열기가 차단되면 위 주소를 직접 열어주세요.', flush=True)
                    return
            except (OSError, TimeoutError):
                pass
            stopped.wait(0.25)
        if not stopped.is_set() and process.poll() is None:
            print(f'서버 시작이 지연되고 있습니다. 터미널 로그를 확인하세요: {url}', flush=True)

    thread = threading.Thread(target=open_when_ready, daemon=True)
    thread.start()
    print('Streamlit을 시작합니다. 종료하려면 실행 중지 또는 Ctrl+C를 누르세요.', flush=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        stopped.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        thread.join(timeout=2)


# Streamlit 내부 재실행에서는 서버를 추가로 시작하지 않습니다.
if get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit(launch_streamlit())

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

def fit_ets(train):
    # 수치 안정화를 위한 천 명 단위 환산. 로그 변환은 하지 않습니다.
    y = np.asarray(train, dtype=float) / 1000
    if len(y)<24 or not np.isfinite(y).all() or (y<0).any():
        raise ValueError("24개월 이상의 유효한 비음수 학습자료가 필요합니다.")
    fit = ETSModel(y, error='add', trend='add', damped_trend=True,
                   seasonal='add', seasonal_periods=12,
                   initialization_method='heuristic').fit(maxiter=2000, disp=False)
    if not fit.mle_retvals.get('converged', False):
        raise RuntimeError('ETS 최적화가 수렴하지 않았습니다.')
    return fit

def simulate_interval(fit, months, seed, repetitions=SIMULATIONS):
    h = len(months)
    point = np.maximum(np.asarray(fit.forecast(h)), 0) * 1000
    paths = np.asarray(fit.simulate(h, anchor='end', repetitions=repetitions,
                       rng=np.random.default_rng(seed))).reshape(h, repetitions)
    lower, upper = np.quantile(np.maximum(paths, 0) * 1000, [.025, .975], axis=1)
    return pd.DataFrame({'연월': months, '예측승객수': point,
                         '95% 하한': lower, '95% 상한': upper})

def forecast(train, months, seed=SEED):
    return simulate_interval(fit_ets(train), months, seed)

@st.cache_data(show_spinner='동일한 ETS 모형으로 여러 시드의 95% 구간을 비교합니다.')
def seed_stability(train, months, seeds):
    # 모형은 한 번만 적합합니다. 바뀌는 것은 미래 오차 생성용 시드뿐입니다.
    fit = fit_ets(train)
    frames = []
    for seed in seeds:
        frame = simulate_interval(fit, list(months), seed, repetitions=10000)
        frame['시드'] = seed
        frame['경로 수'] = 10000
        frame['구간 폭'] = frame['95% 상한'] - frame['95% 하한']
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    group = detail.groupby('연월', sort=True)
    summary = group.agg(
        예측승객수=('예측승객수','first'),
        하한_최솟값=('95% 하한','min'), 하한_최댓값=('95% 하한','max'),
        하한_평균=('95% 하한','mean'), 하한_표준편차=('95% 하한','std'),
        상한_최솟값=('95% 상한','min'), 상한_최댓값=('95% 상한','max'),
        상한_평균=('95% 상한','mean'), 상한_표준편차=('95% 상한','std'),
        평균_구간폭=('구간 폭','mean'),
        하한0_시드수=('95% 하한',lambda x: int(x.eq(0).sum()))
    ).reset_index()
    summary['하한 변동폭(명)'] = summary['하한_최댓값']-summary['하한_최솟값']
    summary['상한 변동폭(명)'] = summary['상한_최댓값']-summary['상한_최솟값']
    denominator = summary['평균_구간폭'].replace(0, np.nan)
    summary['하한 변동폭/평균 구간폭(%)'] = summary['하한 변동폭(명)']/denominator*100
    summary['상한 변동폭/평균 구간폭(%)'] = summary['상한 변동폭(명)']/denominator*100
    return detail, summary


import hashlib
import json
import uuid
from datetime import datetime, timezone

KNOWN_ZERO = {'2023-01','2023-02','2023-12','2024-01'}
MODEL_LABEL = 'ETS(A,Ad,A), heuristic, period=12, scale=1000'

def clean_actual(content, remove_duplicates=False):
    df=read_csv_auto(content)
    df.columns=df.columns.str.strip()
    required={'입출','입항일시','승객','호출부호','선명'}
    if not required.issubset(df.columns):
        raise ValueError(f'필수 열 누락: {required-set(df.columns)}')
    duplicates=int(df.duplicated().sum())
    if duplicates and not remove_duplicates:
        raise ValueError(f'완전히 같은 원본 행이 {duplicates}개 있습니다. 중복 여부를 확인한 뒤 왼쪽의 중복 제거 항목을 선택하세요.')
    if remove_duplicates: df=df.drop_duplicates()
    rows=df.loc[df['입출'].astype(str).str.strip().eq('입항')].copy()
    rows['입항일시']=pd.to_datetime(rows['입항일시'],errors='coerce')
    if rows.empty or rows['입항일시'].isna().any():
        raise ValueError('입항 기록이 없거나 입항일시를 해석할 수 없습니다.')
    raw=rows['승객'].astype('string').str.replace(',','',regex=False).str.strip()
    parsed=pd.to_numeric(raw,errors='coerce')
    invalid=raw.notna() & ~raw.isin(['','-','nan','None']) & parsed.isna()
    if invalid.any(): raise ValueError('숫자로 해석할 수 없는 승객 기록을 확인하세요.')
    if (parsed.dropna()<0).any() or not np.isfinite(parsed.dropna()).all():
        raise ValueError('승객 수의 음수 또는 무한값을 확인하세요.')
    rows['승객']=parsed.astype(float)
    rows['호출부호']=rows['호출부호'].astype('string').str.strip()
    rows['월']=rows['입항일시'].dt.to_period('M').astype(str)
    rows=rows.sort_values('입항일시').reset_index(drop=True)
    return rows,duplicates

def aggregate(rows,start,end,confirmed_zero):
    rows=rows.loc[rows['월'].between(start,end)].copy()
    periods=pd.period_range(start,end,freq='M').astype(str)
    out=rows.groupby('월').agg(입항건수=('승객','size'),유효기록수=('승객','count'),확인승객합계=('승객',lambda x:x.sum(min_count=1))).reindex(periods)
    absent=out['입항건수'].isna()
    zero=absent & out.index.isin(confirmed_zero)
    out.loc[absent,['입항건수','유효기록수']]=0
    out.loc[zero,'확인승객합계']=0
    out['결측입항건수']=out['입항건수']-out['유효기록수']
    out['실제승객합계']=out['확인승객합계'].where(out['결측입항건수'].eq(0))
    out['자료상태']=np.where(absent,np.where(zero,'확인된 무입항 0명','미확인 무기록 월'),np.where(out['결측입항건수'].gt(0),'승객 결측 포함','승객 기록 완전'))
    rows['학습승객']=rows['승객']
    logs=[]
    for idx,row in rows.loc[rows['승객'].isna()].iterrows():
        prior=rows.loc[(rows['입항일시']<row['입항일시']) & rows['승객'].notna() & rows['호출부호'].eq(row['호출부호'])]
        value=float(prior['승객'].median()) if len(prior) else np.nan
        rows.loc[idx,'학습승객']=value
        logs.append({'입항일시':str(row['입항일시']),'선명':row['선명'],'이전기록수':len(prior),'보완값':value,'참조최종일':str(prior['입항일시'].max()) if len(prior) else '없음'})
    train=rows.groupby('월')['학습승객'].agg(lambda x:x.sum() if x.notna().all() else np.nan).reindex(periods)
    train.loc[zero]=0
    out['학습용승객합계']=train
    out.index.name='연월'
    return out.reset_index(),pd.DataFrame(logs)

def require_train(frame):
    if len(frame)<24:raise ValueError('12개월 계절모형은 최소 24개월의 연속된 학습자료가 필요합니다.')
    bad=frame.loc[frame['학습용승객합계'].isna(),'연월'].tolist()
    if bad:raise ValueError('무기록 여부 또는 승객 결측을 확인해야 하는 학습월: '+', '.join(bad))
    return frame['학습용승객합계'].to_numpy(dtype=float)

def evaluate(prediction, monthly):
    out=prediction.merge(monthly[['연월','실제승객합계','자료상태']],on='연월',how='left',validate='one_to_one')
    out['오차(실제−예측)']=out['실제승객합계']-out['예측승객수']
    out['절대오차']=out['오차(실제−예측)'].abs()
    out['월별 APE(%)']=out['절대오차']/out['실제승객합계'].where(out['실제승객합계']>0)*100
    valid=out['실제승객합계'].notna()
    out['95% 구간 포함']=pd.Series(pd.NA,index=out.index,dtype='boolean')
    out.loc[valid,'95% 구간 포함']=out.loc[valid,'실제승객합계'].between(out.loc[valid,'95% 하한'],out.loc[valid,'95% 상한'])
    return out

def metrics(frame):
    v=frame.dropna(subset=['실제승객합계','예측승객수'])
    if v.empty:return None
    a=v['실제승객합계'];err=(a-v['예측승객수']).abs()
    return {'MAPE(%)':float((err/a.where(a>0)*100).mean()),'MAE(명)':float(err.mean()),'RMSE(명)':float(np.sqrt((err**2).mean())),
            'WAPE(%)':float(err.sum()/a.sum()*100) if a.sum()>0 else np.nan,
            '95% 실제 포함률(%)':float(v['95% 구간 포함'].mean()*100),'실제 평가월수':len(v),'MAPE 평가월수':int((a>0).sum())}

@st.cache_data(show_spinner='ETS 모형을 재학습하고 95% 구간을 계산합니다.')
def run_forecast(train,months):
    return forecast(train,list(months))

def schedule_compare(content,future,remove_duplicates=False):
    s=read_csv_auto(content);s.columns=s.columns.str.strip()
    if not {'입항예정일','선명','승객정원수'}.issubset(s.columns):raise ValueError('스케줄에 입항예정일·선명·승객정원수 열이 필요합니다.')
    n=int(s.duplicated().sum())
    if n and not remove_duplicates:raise ValueError(f'스케줄에 완전히 같은 행 {n}개가 있습니다. 원자료를 확인하세요.')
    if remove_duplicates:s=s.drop_duplicates()
    s['입항예정일']=pd.to_datetime(s['입항예정일'],errors='raise')
    s['정원']=pd.to_numeric(s['승객정원수'].astype(str).str.replace(',','',regex=False),errors='raise')
    if s['입항예정일'].isna().any() or s['정원'].isna().any() or (s['정원']<0).any() or not np.isfinite(s['정원']).all():raise ValueError('입항예정일과 정원을 확인하세요.')
    if s.duplicated(['선명','입항예정일']).any():raise ValueError('동일 선명·입항예정일이 중복됩니다. 중복인지 다른 회차인지 원자료에서 확인하세요.')
    s['연월']=s['입항예정일'].dt.strftime('%Y-%m')
    cap=s.groupby('연월').agg(예정입항건수=('정원','size'),승객정원합계=('정원','sum')).reset_index()
    out=future.merge(cap,on='연월',how='left',validate='one_to_one')
    out['차이(정원−예측)']=out['승객정원합계']-out['예측승객수']
    out['정원 대비 절대차이율(%)']=out['차이(정원−예측)'].abs()/out['승객정원합계'].where(out['승객정원합계']>0)*100
    return out,cap,n

def make_snapshot(future,start,cutoff,confirmed_zero,fingerprint):
    import importlib.metadata as md
    return {'schema_version':1,'run_id':uuid.uuid4().hex,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'model':MODEL_LABEL,'train_start':start,'train_end':cutoff,'forecast_end':future['연월'].iloc[-1],
            'seed':SEED,'simulations':SIMULATIONS,'confirmed_zero_months':sorted(confirmed_zero),'input_sha256':fingerprint,
            'versions':{p:md.version(p) for p in ['numpy','pandas','statsmodels']},'predictions':future.to_dict(orient='records')}

def parse_snapshot(content):
    o=json.loads(content)
    if o.get('schema_version')!=1:raise ValueError('지원하지 않는 예측 기록 형식입니다.')
    for key in ['train_start','train_end','forecast_end','created_at_utc','run_id','predictions']:
        if key not in o:raise ValueError(f'예측 기록 필수 항목 누락: {key}')
    start=pd.Period(o['train_start'],freq='M');end=pd.Period(o['train_end'],freq='M')
    if start>end:raise ValueError('학습기간이 잘못됐습니다.')
    f=pd.DataFrame(o['predictions'])
    if not {'연월','예측승객수','95% 하한','95% 상한'}.issubset(f.columns) or f.empty:raise ValueError('예측 기록 표 형식을 확인하세요.')
    expected=pd.period_range(end+1,pd.Period(o['forecast_end'],freq='M'),freq='M').astype(str).tolist()
    if f['연월'].tolist()!=expected:raise ValueError('예측월의 중복·누락 또는 학습기간과의 중첩이 있습니다.')
    numbers=f[['예측승객수','95% 하한','95% 상한']].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(numbers).all().all() or (numbers<0).any().any() or (numbers['95% 하한']>numbers['95% 상한']).any():raise ValueError('예측 기록의 수치가 잘못됐습니다.')
    f[numbers.columns]=numbers
    return o,f

def table(df,name):
    st.dataframe(df.round(2),hide_index=True,width='stretch')
    st.download_button('표 CSV 내려받기',df.to_csv(index=False).encode('utf-8-sig'),name,'text/csv',key=name)

def chart(df,actual=None,title='월별 승객 전망'):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['연월'],y=df['95% 상한'],line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['연월'],y=df['95% 하한'],line=dict(width=0),fill='tonexty',fillcolor='rgba(35,117,190,.18)',name='95% 근사 예측구간'))
    fig.add_trace(go.Scatter(x=df['연월'],y=df['예측승객수'],mode='lines+markers',name='ETS 예측'))
    if actual:fig.add_trace(go.Scatter(x=df['연월'],y=df[actual],mode='lines+markers',name=actual))
    fig.update_layout(title=title,xaxis_title='연월',yaxis_title='승객 수 (명)',hovermode='x unified')
    st.plotly_chart(fig,width='stretch')

def score_view(df,key):
    score=metrics(df)
    if score is None:st.info('평가할 수 있는 실제 완료월 자료가 없습니다.');return
    cols=st.columns(4)
    for col,k in zip(cols,['MAPE(%)','MAE(명)','RMSE(명)','WAPE(%)']):
        col.metric(k,f'{score[k]:,.2f}' if np.isfinite(score[k]) else '계산 불가')
    st.caption(f"실제 평가 {score['실제 평가월수']}개월 / MAPE 평가 {score['MAPE 평가월수']}개월 / 95% 구간 실제 포함률 {score['95% 실제 포함률(%)']:.2f}%")
    fig=go.Figure(go.Bar(x=df['연월'],y=df['월별 APE(%)'],name='월별 APE'))
    if np.isfinite(score['MAPE(%)']):fig.add_hline(y=score['MAPE(%)'],line_dash='dash',annotation_text=f"MAPE {score['MAPE(%)']:.2f}%")
    fig.update_layout(yaxis_title='APE (%)',xaxis_title='연월')
    st.plotly_chart(fig,width='stretch',key=key)

st.set_page_config(page_title='부산항 승객 전망 갱신',page_icon='⚓',layout='wide')
st.title('부산항 크루즈 승객 전망 갱신')
st.caption('실제 자료 갱신 → 이전 예측 평가 → ETS 재학습 → 미래 전망 → 예정 정원 비교')
with st.sidebar:
    st.header('자료 업로드')
    raw=st.file_uploader('누적 입출항 원자료 CSV (필수)',type=['csv'],key='raw')
    schedule=st.file_uploader('새 입항 예정 스케줄 CSV (선택)',type=['csv'],key='schedule')
    remove_duplicates=st.checkbox('완전히 같은 원본 행은 중복으로 확인하여 제거',value=False)
    st.caption('자동 파일 읽기는 사용하지 않습니다. 새 실적을 추가한 누적 CSV를 올려주세요.')
if raw is None:st.info('왼쪽에서 입출항 원자료를 업로드하세요. 스케줄 없이도 예측할 수 있습니다.');st.stop()
data=raw.getvalue()
try:rows,duplicates=clean_actual(data,remove_duplicates)
except Exception as e:st.error(str(e));st.stop()
first=pd.Period(rows['월'].min(),freq='M');last=pd.Period(rows['월'].max(),freq='M')
with st.sidebar:
    st.header('학습 및 예측 기간')
    starts=pd.period_range(first,last,freq='M').astype(str).tolist()
    start=st.selectbox('학습 시작월',starts,index=starts.index('2023-01') if '2023-01' in starts else 0)
    ends=pd.period_range(pd.Period(start,freq='M'),last+12,freq='M').astype(str).tolist()
    default=max(pd.Period(start,freq='M'),last-1).strftime('%Y-%m')
    cutoff=st.selectbox('실제 자료가 확정된 마지막 완료월',ends,index=ends.index(default))
    st.caption('초깃값은 마지막 기록월의 전월입니다. 완결성을 자동 판정한 값이 아니므로 직접 확인하세요.')
    confirmed=st.checkbox('선택한 완료월까지 실제 자료 수집이 완료됐음을 확인',value=False, key='confirmed_'+hashlib.sha256(data).hexdigest()[:12]+'_'+start+'_'+cutoff)
    future_options=pd.period_range(pd.Period(cutoff,freq='M')+1,pd.Period(cutoff,freq='M')+120,freq='M').astype(str).tolist()
    forecast_end=st.selectbox('예측 종료월',future_options,index=14)
    month_set=set(rows['월'])
    absent=[m for m in pd.period_range(start,cutoff,freq='M').astype(str) if m not in month_set]
    zero=st.multiselect('무입항으로 확인된 무기록 월만 선택',absent,default=[m for m in absent if m in KNOWN_ZERO])
    st.caption('2023년 1·2·12월, 2024년 1월은 기존 팀 확인을 기본 반영합니다. 새 무기록 월은 직접 확인해야 합니다.')
if not confirmed:st.info('학습 종료월을 확인한 뒤 왼쪽의 자료 수집 완료 확인을 선택하세요.');st.stop()
monthly,impute_log=aggregate(rows,start,cutoff,set(zero))
tabs=st.tabs(['미래 전망·저장','이전 예측 평가','과거 시점 재검증','예정 스케줄 비교','자료·계산 기준','시드 안정성'])
try:
    train=require_train(monthly)
    months=tuple(pd.period_range(pd.Period(cutoff,freq='M')+1,forecast_end,freq='M').astype(str))
    future=run_forecast(tuple(train),months)
except Exception as e:
    future=None
    st.warning(f'새 전망은 아직 계산할 수 없습니다: {e}')
with tabs[0]:
    if future is not None:
        st.text(f'학습: {start} ~ {cutoff} / 예측: {months[0]} ~ {forecast_end}')
        if len(months)>24:st.warning('2년을 넘는 장기 전망입니다. 실제 일정·정책·선박 구성 변화가 반영되지 않아 운영 확정값으로 사용하기 어렵습니다.')
        chart(future)
        table(future,'future_forecast.csv')
        annual=future.assign(연도=future['연월'].str[:4]).groupby('연도').agg(예측개월수=('연월','size'),예측승객합계=('예측승객수','sum')).reset_index()
        st.caption('아래 합계는 표에 있는 예측월만 합산합니다. 12개월 미만이면 해당 연도 전체 전망이 아닙니다. 월별 구간 경계는 연간 구간으로 합산하지 않습니다.')
        st.dataframe(annual.round(2),hide_index=True,width='stretch')
        snapshot_key=hashlib.sha256(data+json.dumps([start,cutoff,forecast_end,sorted(zero),remove_duplicates]).encode()).hexdigest()
        if st.session_state.get('snapshot_key')!=snapshot_key:
            st.session_state['snapshot_key']=snapshot_key
            st.session_state['snapshot']=make_snapshot(future,start,cutoff,zero,hashlib.sha256(data).hexdigest())
        snap=st.session_state['snapshot'];encoded=json.dumps(snap,ensure_ascii=False,indent=2).encode('utf-8')
        st.subheader('현재 예측 기록 보관')
        st.caption('당시 학습 종료월·예측값·구간·생성시각을 함께 저장합니다. 재학습해도 기존 기록은 덮어쓰지 않습니다.')
        st.download_button('예측 기록 JSON 내려받기',encoded,f"forecast_{cutoff}_{snap['run_id'][:8]}.json",'application/json')
        if st.button('이 컴퓨터에도 예측 기록 저장'):
            try:
                folder=BASE_DIR/'forecast_history';folder.mkdir(exist_ok=True)
                path=folder/f"forecast_{cutoff}_{snap['run_id']}.json"
                if not path.exists():
                    with path.open('x',encoding='utf-8') as f:f.write(encoded.decode('utf-8'))
                st.success(f'저장 완료: {path.name} (파이썬 파일 옆 forecast_history 폴더)')
            except OSError as e:st.error(f'로컬 저장 실패: {e}. JSON 내려받기를 이용하세요.')
    else:st.info('학습자료의 미확인 무기록 또는 승객 결측 문제를 먼저 확인하세요.')
with tabs[1]:
    st.subheader('저장해 둔 이전 예측과 새 실제값 비교')
    uploaded=st.file_uploader('이전에 저장한 예측 기록 JSON',type=['json'],key='archive')
    files=sorted((BASE_DIR/'forecast_history').glob('*.json')) if (BASE_DIR/'forecast_history').exists() else []
    chosen=st.selectbox('이 컴퓨터의 저장 기록', ['선택 안 함']+[f.name for f in files])
    archive=uploaded.getvalue() if uploaded else ((BASE_DIR/'forecast_history'/chosen).read_bytes() if chosen!='선택 안 함' else None)
    if archive:
        try:
            old,pred=parse_snapshot(archive)
            if pd.Period(cutoff,freq='M')<pd.Period(old['train_end'],freq='M'):
                st.info('현재 완료월이 저장 기록의 학습 종료월보다 이전입니다.')
            else:
                actual,_=aggregate(rows,old['train_start'],cutoff,set(zero)|set(old.get('confirmed_zero_months',[])))
                evaluated=evaluate(pred,actual)
                st.text(f"예측 생성시각(UTC): {old['created_at_utc']} / 당시 학습 종료월: {old['train_end']}")
                st.caption('저장된 예측을 그대로 평가하며 다시 학습하지 않습니다. 완료되지 않은 월과 실제 합계가 불완전한 월은 평가에서 제외합니다. 저장 시각만으로 사전 예측임이 증명되지는 않으며 원자료 수정도 비교에 영향을 줄 수 있습니다.')
                chart(evaluated,'실제승객합계','이전 예측과 갱신된 실제값')
                score_view(evaluated,'archive_error_chart');table(evaluated,'previous_forecast_evaluation.csv')
        except Exception as e:st.error(f'기록 평가 실패: {e}')
    else:st.info('첫 사용에서는 미래 전망 탭에서 예측 기록을 저장하세요. 새 실적을 확보한 뒤 이 탭에서 비교할 수 있습니다.')
with tabs[2]:
    st.subheader('과거 시점에서 예측했다면 어땠을까?')
    options=monthly['연월'].tolist()[23:-1]
    if options:
        back_end=st.selectbox('검증용 학습 종료월',options,index=options.index('2025-12') if '2025-12' in options else len(options)-1)
        test_options=pd.period_range(pd.Period(back_end,freq='M')+1,cutoff,freq='M').astype(str).tolist()
        test_end=st.selectbox('평가 종료월',test_options,index=len(test_options)-1)
        try:
            bmonthly,_=aggregate(rows,start,back_end,set(zero))
            btrain=require_train(bmonthly)
            bm=tuple(pd.period_range(pd.Period(back_end,freq='M')+1,test_end,freq='M').astype(str))
            pred=run_forecast(tuple(btrain),bm)
            checked=evaluate(pred,monthly)
            st.text(f'학습: {start} ~ {back_end} / 평가: {bm[0]} ~ {test_end}')
            st.caption('현재 확보한 자료로 다시 계산한 과거 시점 검증입니다. 당시 저장한 예측 기록과 구분하며, 독립적인 최종 시험을 뜻하지 않습니다.')
            chart(checked,'실제승객합계','과거 시점 검증');score_view(checked,'backtest_error_chart');table(checked,'historical_validation.csv')
        except Exception as e:st.error(str(e))
    else:st.info('24개월 학습자료와 최소 1개월의 추가 평가기간이 필요합니다.')
with tabs[3]:
    st.subheader('새 예정 스케줄과 같은 월의 예측 비교')
    st.caption('예정 정원은 학습 자료에 섞지 않습니다. 스케줄 무기록 월은 0으로 단정하지 않고 미확인으로 표시합니다.')
    if schedule is None:st.info('예정 스케줄 CSV를 올리면 해당 연도·월을 비교합니다. 스케줄 없이도 미래 전망은 실행됩니다.')
    elif future is not None:
        try:
            compare,capacity,dups=schedule_compare(schedule.getvalue(),future,remove_duplicates)
            outside=capacity.loc[~capacity['연월'].isin(future['연월']),'연월'].tolist()
            if outside:st.info('현재 예측기간 밖의 스케줄 월: '+', '.join(outside)+'. 필요하면 예측 종료월을 늘리세요.')
            matched=compare.dropna(subset=['승객정원합계'])
            if matched.empty:st.info('스케줄과 예측의 연월이 겹치지 않습니다.')
            else:
                year=st.selectbox('비교 연도',sorted(matched['연월'].str[:4].unique()))
                selected=compare.loc[compare['연월'].str.startswith(year)]
                valid=selected.dropna(subset=['승객정원합계'])
                total=valid['승객정원합계'].sum()
                st.caption(f'정원이 확인된 {len(valid)}개월만 합계·차이율에 포함합니다. 전체 연도 합계와 다를 수 있습니다.')
                a,b,c=st.columns(3);a.metric('확인된 월의 정원 합계',f'{total:,.0f}명')
                b.metric('월별 절대차이율 평균',f"{valid['정원 대비 절대차이율(%)'].mean():.2f}%" if total>0 else '계산 불가')
                c.metric('확인된 월 합계 차이율',f"{abs(total-valid['예측승객수'].sum())/total*100:.2f}%" if total>0 else '계산 불가')
                chart(selected,'승객정원합계','예정 정원과 예측');table(selected,'schedule_comparison.csv')
                st.info('정원은 실제 승객 수가 아닙니다. 정원 차이율을 MAPE·예측 정확도·실제 탑승률로 해석하지 않습니다.')
        except Exception as e:st.error(f'스케줄 비교 실패: {e}')
with tabs[4]:
    st.write(f'중복 제거 확인 건수: {duplicates}행. 선택한 완료월 이후의 실제 기록은 학습과 평가에서 제외합니다.')
    st.write('무기록 월은 확인한 경우만 0명으로 둡니다. 승객 결측이 하나라도 있는 월은 실제 월합계를 결측으로 유지합니다.')
    st.write('학습용 결측 입항만 해당 시점 이전의 동일 호출부호 관측 중앙값으로 보완합니다. 미래 관측과 이미 보완한 값은 참조하지 않습니다. 참조 기록이 없으면 새 전망을 중단합니다.')
    table(monthly,'monthly_data.csv')
    st.subheader('학습용 결측 보완 내역');st.dataframe(impute_log,hide_index=True,width='stretch')
    st.write('ETS(A, Ad, A), 계절 주기 12개월, heuristic 초기화, 천 명 단위 학습 후 명 단위 복원, 최대 반복 2,000회. 기본 미래 경로 10,000개, 시드 20261030, 95% 구간만 표시합니다.')
    st.write('오차=실제−예측. MAPE=월별 절대백분율오차 평균(실제 0 또는 결측 제외). WAPE=절대오차 합계/실제 합계×100. 포함률=실제값이 구간 안에 든 유효월 비율입니다.')
    st.warning('구간은 고정 모수 조건의 근사 구간이며 실제 포함률 95%를 보장하지 않습니다. 음수 점예측·모의값은 0으로 제한합니다. 자료 보완·모수 추정·일정 변경의 불확실성을 충분히 반영하지 못합니다.')
with tabs[5]:
    st.subheader('현재 미래 전망의 시드 안정성')
    st.caption('시드만 바꿔 구간 계산의 흔들림을 점검합니다. 실제 예측 정확도를 평가하는 지표는 아닙니다.')
    nseeds=st.selectbox('시드 수',[5,10,20],index=1)
    if future is not None and st.checkbox('시드 안정성 비교 실행'):
        detail,summary=seed_stability(tuple(train),months,tuple(SEED+i for i in range(nseeds)))
        table(summary,'seed_summary.csv');table(detail,'seed_detail.csv')
        st.caption('각 시드의 경로 수 10,000개. 변동폭은 경계 최댓값−최솟값이며 보편적 합격 기준은 없습니다. 하한을 0으로 제한한 영향도 고려하세요.')
