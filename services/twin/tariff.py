"""시간대별 전기요금 (한전 일반용전력(을) 고압A 선택Ⅲ, 2026-06-01 시행).

요금표 값·시간대·공휴일은 config/settings.yaml 의 tariff 항목 (출처 PDF: docs/references/).

  계절      여름 6~8월 / 봄·가을 3~5, 9~10월 / 겨울 11~2월
  시간대    경부하(off_peak) · 중간부하(mid) · 최대부하(peak), 계절마다 구간이 다르다
  토요일    최대부하 시간대 → 중간부하 단가
  공휴일    (일요일 포함) 모든 시간 → 경부하 단가
  단가      시간대 전력량요금 + 기후환경요금 + 연료비조정요금 (원/kWh)

시각은 그 시간 구간의 시작 시각이다 (예: 15:00 = 15:00~16:00 사용분).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from services.common.settings import section

PERIODS = ('off_peak', 'mid', 'peak')
PERIOD_LABELS = {'off_peak': '경부하', 'mid': '중간부하', 'peak': '최대부하'}


@dataclass(frozen=True)
class Tariff:
    name: str
    source: str
    effective_date: str
    basic_charge_krw_per_kw: float
    season_of_month: dict[int, str]
    period_hours: dict[str, dict[int, str]]  # 계절 → 시(0~23) → 시간대
    rates: dict[str, dict[str, float]]  # 계절 → 시간대 → 원/kWh
    saturday_peak_as_mid: bool
    sunday_as_holiday: bool
    holidays: frozenset[date]
    adder_krw_per_kwh: float  # 기후환경요금 + 연료비조정요금

    @classmethod
    def from_config(cls, cfg: dict) -> Tariff:
        season_of_month = {m: s for s, months in cfg['seasons'].items() for m in months}
        if sorted(season_of_month) != list(range(1, 13)):
            raise ValueError('tariff.seasons 는 1~12월을 한 번씩 모두 포함해야 합니다')
        period_hours: dict[str, dict[int, str]] = {}
        for season, spans in cfg['periods'].items():
            hours = dict.fromkeys(range(24), 'off_peak')
            for period, ranges in spans.items():
                if period not in PERIODS:
                    raise ValueError(f'알 수 없는 시간대: {period}')
                for start, end in ranges:
                    for h in range(start, end):
                        hours[h] = period
            period_hours[season] = hours
        return cls(
            name=cfg['name'],
            source=cfg['source'],
            effective_date=str(cfg['effective_date']),
            basic_charge_krw_per_kw=float(cfg['basic_charge_krw_per_kw']),
            season_of_month=season_of_month,
            period_hours=period_hours,
            rates={
                s: {p: float(v) for p, v in r.items()} for s, r in cfg['rates_krw_per_kwh'].items()
            },
            saturday_peak_as_mid=bool(cfg['saturday_peak_as_mid']),
            sunday_as_holiday=bool(cfg['sunday_as_holiday']),
            holidays=frozenset(date.fromisoformat(str(d)) for d in cfg.get('holidays') or []),
            adder_krw_per_kwh=float(cfg['climate_env_krw_per_kwh'])
            + float(cfg['fuel_adjust_krw_per_kwh']),
        )

    def season(self, ts: datetime) -> str:
        return self.season_of_month[ts.month]

    def period(self, ts: datetime) -> str:
        """그 시각의 시간대 (주말·공휴일 규칙 적용 후)."""
        if ts.date() in self.holidays or (self.sunday_as_holiday and ts.weekday() == 6):
            return 'off_peak'
        p = self.period_hours[self.season(ts)][ts.hour]
        if p == 'peak' and self.saturday_peak_as_mid and ts.weekday() == 5:
            return 'mid'
        return p

    def price(self, ts: datetime) -> float:
        """그 시각 1kWh 단가 (원)."""
        return self.rates[self.season(ts)][self.period(ts)] + self.adder_krw_per_kwh

    def schedule(self, timestamps: Iterable[datetime]) -> pd.DataFrame:
        """시각 목록 → season, period, price_krw_per_kwh 표."""
        idx = pd.DatetimeIndex(list(timestamps))
        rows = [(self.season(t), self.period(t), self.price(t)) for t in idx]
        return pd.DataFrame(rows, index=idx, columns=['season', 'period', 'price_krw_per_kwh'])


DEFAULT_TARIFF = Tariff.from_config(section('tariff'))
