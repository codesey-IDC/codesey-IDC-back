"""예측 결과 시각화 (plotly → output/forecast/plots/forecast_report.html).

그래프 3개를 한 HTML 로 묶는다.
  1. 24시간 전 예측 vs 실제 (Test 14일, h=24 고정)  + 1주 전 값 기준선
  2. 한 시점에서 본 앞으로 168시간 (fan chart)       + 1주 전 값 기준선
  3. 예측 거리(h)별 MAPE: 모델 vs 1주 전 값 기준선   (168h 모델)

실행 (저장소 루트에서):
  python visualize_forecast.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.forecast.features import TARGET, load_hourly  # noqa: E402

DATA_PATH = 'data/synthetic_idc_90d.parquet'
FORECAST_DIR = 'output/forecast'
PLOT_DIR = os.path.join(FORECAST_DIR, 'plots')

# 색: 실제값은 중립 잉크, 모델은 slot1(blue), 기준선은 slot2(orange). 글자는 잉크색.
C_ACTUAL = '#52514e'
C_MODEL = '#2a78d6'
C_BAND = 'rgba(42, 120, 214, 0.16)'
C_NAIVE = '#eb6834'
C_SURFACE = '#fcfcfb'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'
C_MUTED = '#898781'
C_INK = '#0b0b0b'

LAYOUT = dict(
    paper_bgcolor=C_SURFACE, plot_bgcolor=C_SURFACE,
    font=dict(family='Inter, "Noto Sans KR", Segoe UI, sans-serif', size=13, color=C_INK),
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                font=dict(color=C_INK)),
    margin=dict(l=64, r=24, t=88, b=56),
    height=440,
)
AXIS = dict(gridcolor=C_GRID, gridwidth=1, linecolor=C_AXIS, zeroline=False,
            tickfont=dict(color=C_MUTED), title_font=dict(color=C_MUTED))


def naive_lag168(hourly_y: pd.Series, target_time: pd.Series) -> np.ndarray:
    """1주 전 같은 시각의 실제값 (seasonal naive 기준선)."""
    return hourly_y.reindex(pd.DatetimeIndex(target_time) - pd.Timedelta(hours=168)).to_numpy()


def band_traces(x, lo, hi, name='90% 예측구간') -> list[go.Scatter]:
    return [
        go.Scatter(x=x, y=hi, mode='lines', line=dict(width=0, color=C_MODEL),
                   hoverinfo='skip', showlegend=False),
        go.Scatter(x=x, y=lo, mode='lines', line=dict(width=0, color=C_MODEL),
                   fill='tonexty', fillcolor=C_BAND, name=name,
                   hovertemplate='구간 하한 %{y:,.0f} kW<extra></extra>'),
    ]


def fig_fixed_horizon(tp: pd.DataFrame, hourly_y: pd.Series, h: int, metrics: dict) -> go.Figure:
    d = tp[tp['h'] == h].sort_values('target_time')
    x = d['target_time']
    naive = naive_lag168(hourly_y, d['target_time'])
    fig = go.Figure()
    fig.add_traces(band_traces(x, d['q05'], d['q95']))
    fig.add_trace(go.Scatter(x=x, y=naive, name='기준선: 1주 전 값', mode='lines',
                             line=dict(color=C_NAIVE, width=1.5, dash='dot'),
                             hovertemplate='%{y:,.0f} kW<extra>1주 전 값</extra>'))
    fig.add_trace(go.Scatter(x=x, y=d['q50'], name='모델 예측 (q50)', mode='lines',
                             line=dict(color=C_MODEL, width=2),
                             hovertemplate='%{y:,.0f} kW<extra>모델 예측</extra>'))
    fig.add_trace(go.Scatter(x=x, y=d['y'], name='실제값', mode='lines',
                             line=dict(color=C_ACTUAL, width=2),
                             hovertemplate='%{y:,.0f} kW<extra>실제값</extra>'))
    m = metrics['24h']
    fig.update_layout(
        title=dict(text=f'{h}시간 전에 한 예측 vs 실제 (Test 구간)'
                        f'<br><sup>모델 MAPE {m["mape_model"]:.2f}% · 1주 전 값 기준선 {m["mape_naive_lag168"]:.2f}%'
                        f' · 90% 구간 적중률 {m["coverage_cqr"]:.1f}%</sup>'),
        yaxis_title='IT 부하 (kW)', **LAYOUT)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS, tickformat=',')
    return fig


def fig_fan(tp: pd.DataFrame, hourly_y: pd.Series, origin: pd.Timestamp,
            context_hours: int = 48) -> go.Figure:
    d = tp[tp['origin_time'] == origin].sort_values('h')
    x = d['target_time']
    hist = hourly_y.loc[origin - pd.Timedelta(hours=context_hours):origin]
    naive = naive_lag168(hourly_y, d['target_time'])
    fig = go.Figure()
    fig.add_traces(band_traces(x, d['q05'], d['q95']))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='실제값 (예측 시점 이전)',
                             mode='lines', line=dict(color=C_ACTUAL, width=2),
                             hovertemplate='%{y:,.0f} kW<extra>실제값</extra>'))
    fig.add_trace(go.Scatter(x=x, y=d['y'], name='실제값 (예측 시점 이후)', mode='lines',
                             line=dict(color=C_ACTUAL, width=2, dash='dash'),
                             hovertemplate='%{y:,.0f} kW<extra>실제값</extra>'))
    fig.add_trace(go.Scatter(x=x, y=naive, name='기준선: 1주 전 값', mode='lines',
                             line=dict(color=C_NAIVE, width=1.5, dash='dot'),
                             hovertemplate='%{y:,.0f} kW<extra>1주 전 값</extra>'))
    fig.add_trace(go.Scatter(x=x, y=d['q50'], name='모델 예측 (q50)', mode='lines',
                             line=dict(color=C_MODEL, width=2),
                             hovertemplate='%{y:,.0f} kW<extra>모델 예측</extra>'))
    fig.add_vline(x=origin, line=dict(color=C_MUTED, width=1, dash='dash'))
    fig.add_annotation(x=origin, y=1, yref='paper', text='예측 시점', showarrow=False,
                       xanchor='left', yanchor='top', font=dict(color=C_MUTED, size=12))
    fig.update_layout(
        title=dict(text=f'{origin:%m/%d %H:%M} 에 본 앞으로 {int(d["h"].max())}시간'),
        yaxis_title='IT 부하 (kW)', **LAYOUT)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS, tickformat=',')
    return fig


def fig_mape_by_h(tp: pd.DataFrame, hourly_y: pd.Series) -> go.Figure:
    d = tp.copy()
    d['naive'] = naive_lag168(hourly_y, d['target_time'])
    d['ape_model'] = (d['q50'] - d['y']).abs() / d['y'].abs() * 100
    d['ape_naive'] = (d['naive'] - d['y']).abs() / d['y'].abs() * 100
    d['hit'] = ((d['y'] >= d['q05']) & (d['y'] <= d['q95'])).astype(float) * 100
    g = d.groupby('h')[['ape_model', 'ape_naive', 'hit']].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g['h'], y=g['ape_naive'], name='기준선: 1주 전 값', mode='lines',
                             line=dict(color=C_NAIVE, width=2, dash='dot'),
                             hovertemplate='MAPE %{y:.2f}%<extra>1주 전 값</extra>'))
    fig.add_trace(go.Scatter(x=g['h'], y=g['ape_model'], name='모델 (q50)', mode='lines',
                             line=dict(color=C_MODEL, width=2),
                             hovertemplate='MAPE %{y:.2f}%<extra>모델</extra>'))
    fig.update_layout(
        title=dict(text='예측 거리별 MAPE (168h 모델, Test 구간)'
                        '<br><sup>예측 거리가 길어져도 오차가 거의 그대로: 데이터가 1주 주기 반복이기 때문</sup>'),
        xaxis_title='예측 거리 h (시간 뒤)', yaxis_title='MAPE (%)', **LAYOUT)
    fig.update_xaxes(**AXIS, dtick=24)
    fig.update_yaxes(**AXIS, rangemode='tozero')
    return fig, g


def metrics_table(metrics: dict) -> str:
    rows = [
        ('MAPE (모델 q50)', 'mape_model', '{:.3f}%'),
        ('MAPE (1주 전 값 그대로)', 'mape_naive_lag168', '{:.3f}%'),
        ('90% 적중률 (보정 전)', 'coverage_raw', '{:.1f}%'),
        ('90% 적중률 (CQR 보정 후)', 'coverage_cqr', '{:.1f}%'),
        ('평균 구간 폭 (보정 후)', 'width_cqr_kw', '{:.0f} kW'),
        ('CQR offset', 'cqr_offset_kw', '{:.3f} kW'),
        ('Test 행 수', 'rows', '{:,}'),
    ]
    html = ['<table><thead><tr><th>지표 (Test 5/30 ~ 6/12)</th><th>24h 모델</th><th>168h 모델</th></tr></thead><tbody>']
    for label, key, fmt in rows:
        html.append(f'<tr><td>{label}</td><td>{fmt.format(metrics["24h"][key])}</td>'
                    f'<td>{fmt.format(metrics["168h"][key])}</td></tr>')
    html.append('</tbody></table>')
    return '\n'.join(html)


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(FORECAST_DIR, 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    hourly_y = load_hourly(DATA_PATH)[TARGET]
    tp24 = pd.read_parquet(os.path.join(FORECAST_DIR, 'test_pred_24h.parquet'))
    tp168 = pd.read_parquet(os.path.join(FORECAST_DIR, 'test_pred_168h.parquet'))

    figs = [
        fig_fixed_horizon(tp24, hourly_y, 24, metrics),
        fig_fan(tp168, hourly_y, tp168['origin_time'].min()),
    ]
    f3, by_h = fig_mape_by_h(tp168, hourly_y)
    figs.append(f3)

    parts = [f.to_html(full_html=False, include_plotlyjs='cdn' if i == 0 else False,
                       config={'displaylogo': False, 'responsive': True})
             for i, f in enumerate(figs)]
    css = (f'body{{margin:0;padding:24px;background:#f9f9f7;color:{C_INK};'
           'font-family:Inter,"Noto Sans KR","Segoe UI",sans-serif}'
           'h1{font-size:22px;margin:0 0 4px}p{color:#52514e;margin:0 0 20px}'
           '.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:8px;'
           'padding:8px;margin-bottom:20px}'
           'table{border-collapse:collapse;font-variant-numeric:tabular-nums}'
           'th,td{padding:6px 14px;text-align:right;border-bottom:1px solid #e1e0d9}'
           'th:first-child,td:first-child{text-align:left}th{color:#898781;font-weight:600}')
    html = (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>IT 부하 예측 결과</title><style>{css}</style></head><body>'
        '<h1>IT 부하 예측 결과 (LightGBM 분위수 회귀 + CQR)</h1>'
        '<p>Test 구간 2026-05-30 ~ 06-12 · 냉각 부하 예측값은 IT 부하 예측값과 같다</p>'
        + ''.join(f'<div class="card">{p}</div>' for p in parts)
        + f'<div class="card">{metrics_table(metrics)}</div>'
        '</body></html>'
    )
    out = os.path.join(PLOT_DIR, 'forecast_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    by_h.to_csv(os.path.join(PLOT_DIR, 'mape_by_h_168h.csv'), index=False)
    print(f'저장: {out}  ({os.path.getsize(out) / 1024:.0f} KB)')
    print(f'      {os.path.join(PLOT_DIR, "mape_by_h_168h.csv")}')


if __name__ == '__main__':
    main()
