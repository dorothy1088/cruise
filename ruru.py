from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path("./data/i.csv")

# 한글 그래프 글꼴: Windows 기준
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

df = pd.read_csv(DATA_PATH, encoding="euc-kr")

# 입항 기록만 추출
inbound = df.loc[df["입출"].astype(str).str.strip().eq("입항")].copy()


#데이터 불러오기 및 전처리
# 날짜/시간 및 숫자형 변수 정리
inbound["입항일시"] = pd.to_datetime(inbound["입항일시"], errors="coerce")
inbound["출항일시"] = pd.to_datetime(inbound["출항일시"], errors="coerce")
inbound["승객"] = pd.to_numeric(inbound["승객"], errors="coerce")

inbound["날짜"] = inbound["입항일시"].dt.date
inbound["연도"] = inbound["입항일시"].dt.year
inbound["월"] = inbound["입항일시"].dt.month
inbound["요일번호"] = inbound["입항일시"].dt.dayofweek
inbound["요일"] = inbound["입항일시"].dt.day_name()
inbound["시각"] = inbound["입항일시"].dt.hour

요일_한글 = {0:"월", 1:"화", 2:"수", 3:"목", 4:"금", 5:"토", 6:"일"}
inbound["요일"] = inbound["요일번호"].map(요일_한글)

# 실제 체류시간(선박의 입항~출항 간격): 보조지표로만 사용
inbound["체류시간_시간"] = (
    inbound["출항일시"] - inbound["입항일시"]
).dt.total_seconds() / 3600

print("원본 데이터 크기:", df.shape)
print("입항 기록 수:", len(inbound))
print("분석 기간:", inbound["입항일시"].min(), "~", inbound["입항일시"].max())
print("승객 수 결측:", inbound["승객"].isna().sum())
inbound.head()

#데이터 품질 점검
#결측 승객 수를 0으로 바꾸지 않기. 결측이 있는 날짜는 유형 분류 때 별도로 표시
quality = pd.Series({
    "입항 기록 수": len(inbound),
    "입항일시 결측": inbound["입항일시"].isna().sum(),
    "출항일시 결측": inbound["출항일시"].isna().sum(),
    "승객 수 결측": inbound["승객"].isna().sum(),
    "완전 중복 행 수": inbound.duplicated().sum(),
    "입항일시·선명 중복 행 수": inbound.duplicated(["입항일시", "선명"]).sum()
})
display(quality.to_frame("건수"))

# 승객 결측이 발생한 항차를 직접 확인
display(inbound.loc[inbound["승객"].isna(),
                    ["입항일시", "출항일시", "선명", "계선장소.2", "승객"]])

#연도별·월별 입항 패턴
#2026년은 10월 16일까지의 부분자료이므로 완전연도와 구분해 해석하기. 2020~2022년은 자료에 포함 안됨.
annual = inbound.groupby("연도").size().rename("입항건수").reset_index()
display(annual)

annual.set_index("연도")["입항건수"].plot(kind="bar", figsize=(10,4), title="연도별 크루즈 입항 건수")
plt.ylabel("입항 건수")
plt.tight_layout()
plt.show()

# 월별 패턴 비교는 2026년 제외
monthly_complete = inbound.loc[inbound["연도"] != 2026].groupby("월").size().reindex(range(1,13), fill_value=0)
display(monthly_complete.rename("입항건수").to_frame())

monthly_complete.plot(kind="bar", figsize=(10,4), title="월별 크루즈 입항 건수 (2026년 제외)")
plt.xlabel("월")
plt.ylabel("입항 건수")
plt.tight_layout()
plt.show()

# 시간대·요일 패턴
hour_counts = inbound.groupby("시각").size().reindex(range(24), fill_value=0)
display(hour_counts.rename("입항건수").to_frame())

hour_counts.plot(kind="bar", figsize=(12,4), title="시간대별 크루즈 입항 건수")
plt.xlabel("입항 시각(시)")
plt.ylabel("입항 건수")
plt.tight_layout()
plt.show()

# 요일 비교도 부분연도 영향을 피하기 위해 2026년 제외
weekday_order = ["월","화","수","목","금","토","일"]
weekday_counts = (inbound.loc[inbound["연도"] != 2026]
                  .groupby("요일").size().reindex(weekday_order, fill_value=0))
display(weekday_counts.rename("입항건수").to_frame())

# 월 × 요일 히트맵: 2026년 제외
heat_data = inbound.loc[inbound["연도"] != 2026].pivot_table(
    index="월", columns="요일", values="입항일시", aggfunc="count", fill_value=0
).reindex(index=range(1,13), columns=weekday_order, fill_value=0)
plt.figure(figsize=(9,5))
sns.heatmap(heat_data, annot=True, fmt=".0f", cmap="Blues")
plt.title("월·요일별 입항 건수 (2026년 제외)")
plt.tight_layout()
plt.show()

# 날짜별 입항 척수와 승객 규모
#일별 승객 합계는 기록된 승객 수의 합
daily_ops = inbound.groupby("날짜").agg(
    입항건수=("입항일시", "size"),
    승객합계=("승객", "sum"),
    승객기록항차수=("승객", "count")
).reset_index()

daily_ops["승객기록완전"] = daily_ops["승객기록항차수"].eq(daily_ops["입항건수"])
# 결측 승객 기록이 있는 날은 승객 합계가 완전하지 않으므로 유형 구분에 쓰지 않습니다.
daily_ops.loc[~daily_ops["승객기록완전"], "승객합계_분류용"] = np.nan
daily_ops.loc[daily_ops["승객기록완전"], "승객합계_분류용"] = daily_ops.loc[daily_ops["승객기록완전"], "승객합계"]

display(daily_ops["입항건수"].describe().to_frame("일별 입항 척수"))
display(daily_ops["승객합계_분류용"].describe().to_frame("완전 기록일의 일별 승객 합계"))
display(daily_ops["입항건수"].value_counts().sort_index().rename("날짜수").to_frame())

# 같은 날짜·같은 시각의 중복 입항
# 입항 시각을 시(hour) 단위로 묶었을 때 2척 이상 기록된 경우를 표시. 실제 하선 중첩이나 시설 혼잡을 입증은 아님
inbound["시간대키"] = inbound["입항일시"].dt.floor("h")
hourly_group = inbound.groupby(["날짜", "시간대키"]).agg(
    입항건수=("입항일시", "size"),
    승객합계=("승객", "sum"),
    승객기록항차수=("승객", "count")
).reset_index()
hourly_group["중복시간대"] = hourly_group["입항건수"] >= 2

overlap_summary = pd.Series({
    "날짜·시간대 그룹 수": len(hourly_group),
    "2척 이상 그룹 수": int(hourly_group["중복시간대"].sum()),
    "중복 시간대에 속한 항차 수": int(hourly_group.loc[hourly_group["중복시간대"], "입항건수"].sum()),
    "전체 입항 항차 중 중복 시간대 비율(%)": round(
        inbound["입항일시"].notna().sum() and
        hourly_group.loc[hourly_group["중복시간대"], "입항건수"].sum() / len(inbound) * 100, 2
    )
})
display(overlap_summary.to_frame("값"))
display(hourly_group.loc[hourly_group["중복시간대"]].sort_values(["날짜","시간대키"]))

# A~D 운영유형 분류
# 분류 기준은 분석을 위한 탐색적 기준
# 중복 시간대를 60분 이내의 같은 날짜·시각(hour)으로 보고, 승객 규모는 승객 기록이 완전한 날짜들의 중앙값을 기준으로 나눔.
# 일별 중복 여부: 해당 날짜에 같은 시각(hour)으로 2척 이상 묶인 그룹이 하나라도 있는지
overlap_dates = set(hourly_group.loc[hourly_group["중복시간대"], "날짜"])
daily_ops["집중입항"] = daily_ops["날짜"].isin(overlap_dates)

passenger_threshold = daily_ops.loc[daily_ops["승객기록완전"], "승객합계"].median()
daily_ops["승객규모"] = np.where(
    daily_ops["승객기록완전"],
    np.where(daily_ops["승객합계"] >= passenger_threshold, "고승객", "저승객"),
    "자료확인필요"
)

def classify_type(row):
    if not row["승객기록완전"]:
        return "자료확인필요"
    if row["집중입항"] and row["승객규모"] == "고승객":
        return "A_중첩·고승객수요형"
    if (not row["집중입항"]) and row["승객규모"] == "고승객":
        return "B_대규모비집중입항형"
    if row["집중입항"] and row["승객규모"] == "저승객":
        return "C_중첩형"
    return "D_일반형"

daily_ops["운영유형"] = daily_ops.apply(classify_type, axis=1)
display(daily_ops.groupby("운영유형", dropna=False).agg(
    날짜수=("날짜","size"),
    평균입항척수=("입항건수","mean"),
    평균승객합계=("승객합계","mean")
).round(2))

# 운영유형을 항차별 데이터에 붙여 시간대 분포를 살펴봅니다.
inbound = inbound.merge(daily_ops[["날짜","운영유형"]], on="날짜", how="left")
type_hour = pd.crosstab(inbound["운영유형"], inbound["시각"])
display(type_hour)

# 분류 기준 민감도 분석
# 중복 기준을 30/60/90분으로 바꾸고, 승객 기준을 중앙값(P50) 또는 75백분위수(P75)로 바꿨을 때 유형별 날짜 수가 얼마나 달라지는지 확인
# 결과가 기준값에 얼마나 민감한지 확인

def sensitivity_table(data, gap_minutes_list=(30,60,90)):
    base = data.drop(columns=["운영유형"], errors="ignore").copy()
    rows = []
    complete_days = base.groupby("날짜").agg(
        입항건수=("입항일시","size"),
        승객합계=("승객","sum"),
        승객기록항차수=("승객","count")
    ).reset_index()
    complete_days["완전"] = complete_days["입항건수"].eq(complete_days["승객기록항차수"])
    for gap in gap_minutes_list:
        temp = base.copy()
        temp["시간대그룹"] = temp["입항일시"].dt.floor(f"{gap}min")
        groups = temp.groupby(["날짜","시간대그룹"]).size()
        overlap_keys = set(groups[groups >= 2].index.get_level_values(0))
        d = complete_days.copy()
        d["중첩"] = d["날짜"].isin(overlap_keys)
        for quantile, label in [(0.50, "P50"), (0.75, "P75")]:
            threshold = d.loc[d["완전"], "승객합계"].quantile(quantile)
            d["승객구간"] = np.where(d["완전"], np.where(d["승객합계"] >= threshold, "고승객", "저승객"), "자료확인필요")
            d["유형"] = np.select(
                [~d["완전"], d["중첩"] & d["승객구간"].eq("고승객"),
                 ~d["중첩"] & d["승객구간"].eq("고승객"), d["중첩"] & d["승객구간"].eq("저승객")],
                ["자료확인필요","A","B","C"], default="D"
            )
            counts = d["유형"].value_counts()
            rows.append({"중복기준(분)":gap,"승객기준":label,
                         **{k:int(counts.get(k,0)) for k in ["A","B","C","D","자료확인필요"]}})
    return pd.DataFrame(rows)

display(sensitivity_table(inbound))

# 월별 입항 예측 검증 (선택 분석)
# 이 부분은 월별 예측을 보고서의 핵심 결과로 제시할 때만 사용
# 과거 관측된 같은 달의 평균을 예측값으로 사용하는 단순 기준모형을 평가

monthly_counts = (inbound.groupby(["연도","월"]).size()
                  .rename("실제입항건수").reset_index())
years = sorted(monthly_counts["연도"].dropna().unique())
evaluation_rows = []
for target_year in years:
    if target_year < 2016 or target_year == 2026:
        continue
    for month in range(1,13):
        actual_row = monthly_counts[(monthly_counts["연도"] == target_year) & (monthly_counts["월"] == month)]
        history = monthly_counts[(monthly_counts["연도"] < target_year) & (monthly_counts["월"] == month)]
        if actual_row.empty or history.empty:
            continue
        actual = float(actual_row["실제입항건수"].iloc[0])
        pred = float(history["실제입항건수"].mean())
        evaluation_rows.append({"연도":target_year,"월":month,"실제":actual,"예측_과거동월평균":pred,
                                "절대오차":abs(actual-pred),"오차_실제-예측":actual-pred})
forecast_eval = pd.DataFrame(evaluation_rows)
display(forecast_eval.head())
if not forecast_eval.empty:
    print("MAE:", round(forecast_eval["절대오차"].mean(), 2))
    print("평균 오차(실제-예측):", round(forecast_eval["오차_실제-예측"].mean(), 2))

# 선석별 체류시간 분석 (보조 분석)
# 체류시간은 선박의 입항~출항 간격이며, 관광시설 방문시간이나 필요한 직원 수와 동일 안함
berth_col = "계선장소.2"
if berth_col in inbound.columns:
    berth_summary = inbound.groupby(berth_col).agg(
        입항건수=("입항일시","size"),
        평균체류시간=("체류시간_시간","mean"),
        중앙값체류시간=("체류시간_시간","median"),
        최대체류시간=("체류시간_시간","max")
    ).sort_values("입항건수", ascending=False)
    display(berth_summary.round(2).head(20))
