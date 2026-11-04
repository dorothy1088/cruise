# 세팅
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
df = pd.read_csv('./data/i.csv', encoding='euc-kr')


# 1. 계선장소별 크루즈선 입항 횟수 구하기
df["입출"] = df["입출"].astype(str).str.strip()
arrival = df[df["입출"] == "입항"].copy()

berth_col = "계선장소.2"
berth_counts = (
     arrival[berth_col]
     .fillna("미상")
     .value_counts()
     .rename_axis("계선장소")
     .reset_index(name="입항횟수")
 )

total_arrivals = berth_counts["입항횟수"].sum()
berth_counts["입항비중(%)"] = (
     berth_counts["입항횟수"] / total_arrivals * 100
 ).round(2)


print("전체 입항 횟수:", total_arrivals)
print("\n계선장소별 입항 횟수 및 비중")
print(berth_counts.to_string(index=False))


plot_df = berth_counts.sort_values("입항횟수")
plt.figure(figsize=(12, 8))
bars = plt.barh(
     plot_df["계선장소"],
     plot_df["입항횟수"]
 )

plt.bar_label(bars, fmt="%.0f", padding=3)
plt.title("계선장소 별 입항 횟수")
plt.xlabel("입항 횟수")
plt.ylabel("계선장소")
plt.tight_layout()
plt.show()




# 2. 계선장소별 승객수 현황

arrival["승객"] = pd.to_numeric(
    arrival["승객"], errors="coerce"
)

passenger_stats = (
    arrival.groupby("계선장소.2", dropna=False)
    .agg(
        총승객수=("승객", "sum"),
        평균승객수=("승객", "mean"),
        승객수_기록건수=("승객", "count"),
        입항횟수=("승객", "size")
    )
    .reset_index()
    .rename(columns={"계선장소.2": "계선장소"})
)

total_passengers = passenger_stats["총승객수"].sum()

passenger_stats["승객비중(%)"] = (
    passenger_stats["총승객수"] / total_passengers * 100
).round(2)


passenger_stats = passenger_stats.sort_values(
    "총승객수", ascending=False
)

print("전체 승객수:", f"{total_passengers:,.0f}")
print(passenger_stats.to_string(index=False))

plot_df = passenger_stats.sort_values("총승객수")
plt.figure(figsize=(12, 8))
bars = plt.barh(
    plot_df["계선장소"].fillna("미상"),
    plot_df["총승객수"]
)
plt.bar_label(bars, fmt="%.0f", padding=3)
plt.title("계선장소 별 총 승객수")
plt.xlabel("총 승객수")
plt.ylabel("계선장소")
plt.tight_layout()
plt.show()


# 3. 연도별 입항 횟수 및 입항 집중 경향

arrival["입항일시"] = pd.to_datetime(
    arrival["입항일시"], errors="coerce"
)

arrival["연도"] = arrival["입항일시"].dt.year

top5_berths = (
    arrival["계선장소.2"]
    .value_counts()
    .head(5)
    .index
    .tolist()
)

print("분석 대상 계선장소:")
for berth in top5_berths:
    print("-", berth)

yearly_counts = (
    arrival[arrival["계선장소.2"].isin(top5_berths)]
    .groupby(["연도", "계선장소.2"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

print("\n연도별 입항 횟수:")
print(yearly_counts)

ax = yearly_counts.plot(
    kind="line",
    marker="o",
    figsize=(12, 6)
)

# 선 그래프
ax.set_title("계선 장소에 따른 년도별 입항 횟수")
ax.set_xlabel("년도")
ax.set_ylabel("입항 횟수")
ax.set_xticks(yearly_counts.index)
ax.grid(True, alpha=0.3)

plt.legend(
    title="계선 장소",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()

# 누적 막대그래프
yearly_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("계선 장소에 따른 년도별 입항 횟수")
plt.xlabel("년도")
plt.ylabel("입항 횟수")
plt.legend(
    title="계선 장소",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()

# 증감률 계산
yearly_change = yearly_counts.pct_change() * 100

print("\n전년 대비 입항 횟수 증감률(%):")
print(yearly_change.round(2))

# 입항 횟수뿐 아니라 전체 입항 중 계선장소별 비중이 연도별로 어떻게 변했는지 확인

arrival["입항일시"] = pd.to_datetime(
    arrival["입항일시"], errors="coerce"
)
arrival["연도"] = arrival["입항일시"].dt.year


yearly_total = arrival.groupby("연도").size()


yearly_counts = (
    arrival.groupby(["연도", "계선장소.2"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

# 연도별 전체 입항 중 계선장소별 비중(%)
yearly_share = (
    yearly_counts
    .div(yearly_total, axis=0)
    .mul(100)
    .round(2)
)

print("연도별 전체 입항 횟수")
print(yearly_total)

print("\n연도별 계선장소 입항 비중(%)")
print(yearly_share.to_string())

# 주요 계선장소 5곳의 비중  선 그래프
top5 = (
    arrival["계선장소.2"]
    .value_counts()
    .head(5)
    .index
)

share_top5 = yearly_share.reindex(columns=top5)

ax = share_top5.plot(
    kind="line",
    marker="o",
    figsize=(12, 7)
)

ax.set_title("연도별 계선장소 입항 비중(%)")
ax.set_xlabel("년도")
ax.set_ylabel("전체 입항 비중 (%)")
ax.set_ylim(0, 100)
ax.set_xticks(yearly_share.index)
ax.grid(True, alpha=0.3)

plt.legend(
    title="Berth Location",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()

# 누적 막대그래프
share_top5.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7)
)

plt.title("Annual Arrival Share of Major Berth Locations")
plt.xlabel("Year")
plt.ylabel("Share of Total Arrivals (%)")
plt.legend(
    title="Berth Location",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()

# 4. 계선장소별 월별·시간대별 입항 패턴 비교

arrival = df.loc[
    df["입출"].astype("string").str.strip().eq("입항")
].copy()

arrival["입항일시"] = pd.to_datetime(
    arrival["입항일시"], errors="coerce"
)

arrival = arrival.dropna(subset=["입항일시", "계선장소.2"])
arrival = arrival.rename(columns={"계선장소.2": "계선장소명"})

# 2026년 제외
arrival = arrival[arrival["입항일시"].dt.year < 2026].copy()

arrival["월"] = arrival["입항일시"].dt.month
arrival["시간"] = arrival["입항일시"].dt.hour

# 3. 장소별 월별·시간대별 입항 건수
monthly_count = pd.crosstab(
    arrival["계선장소명"], arrival["월"]
).reindex(columns=range(1, 13), fill_value=0)

hourly_count = pd.crosstab(
    arrival["계선장소명"], arrival["시간"]
).reindex(columns=range(24), fill_value=0)

# 4. 각 장소의 전체 입항 중 해당 월·시간대가 차지하는 비중
monthly_pct = monthly_count.div(
    monthly_count.sum(axis=1), axis=0
) * 100

hourly_pct = hourly_count.div(
    hourly_count.sum(axis=1), axis=0
) * 100

print("월별 입항 건수")
print(monthly_count)

print("\n시간대별 입항 건수")
print(hourly_count)

# 5. 주요 5개 장소의 패턴 비교
top_places = arrival["계선장소명"].value_counts().head(5).index


fig, axes = plt.subplots(2, 1, figsize=(13, 10))

monthly_pct.loc[top_places].T.plot(
    ax=axes[0], marker="o"
)
axes[0].set(
    title="계선장소별 월별 입항 패턴",
    xlabel="월",
    ylabel="장소별 전체 입항 대비 비중(%)"
)
axes[0].set_xticks(range(1, 13))

hourly_pct.loc[top_places].T.plot(
    ax=axes[1], marker="o"
)
axes[1].set(
    title="계선장소별 시간대별 입항 패턴",
    xlabel="입항 시간대",
    ylabel="장소별 전체 입항 대비 비중(%)"
)
axes[1].set_xticks(range(24))

for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(title="계선장소", bbox_to_anchor=(1.02, 1))

plt.tight_layout()
plt.show()












