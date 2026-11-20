"""통합 관제 대시보드 (Streamlit :8501).

세 서비스를 REST 로만 호출한다. 파일(parquet/CSV)을 직접 읽지 않는다.
  twin-service     :8001  계측값(telemetry), 정상 상태 계산(simulate), ESG
  forecast-service :8002  IT 부하 예측
  mpc-service      :8003  설정 온도 계획(plan), 폐루프 시뮬레이션(simulate), 위기 시나리오, 사전 계산 결과

실행 (저장소 루트에서, 세 서비스를 먼저 띄운 뒤):
  streamlit run dashboard/app.py
설정은 config/settings.yaml (다른 파일은 환경변수 IDC_CONFIG).
주소는 환경변수 TWIN_URL, FORECAST_URL, MPC_URL 이 있으면 그 값이 우선한다.
"""

from __future__ import annotations

import os
import time as time_module
from datetime import date, datetime, time
from pathlib import Path

from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import yaml

# 대시보드는 streamlit 으로 따로 실행되므로 services 패키지를 import 하지 않고 설정 파일을 직접 읽는다
CONFIG_PATH = Path(
    os.environ.get('IDC_CONFIG', Path(__file__).resolve().parents[1] / 'config' / 'settings.yaml')
)
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))

TWIN_URL = os.environ.get('TWIN_URL', CONFIG['services']['twin_url'])
FORECAST_URL = os.environ.get('FORECAST_URL', CONFIG['services']['forecast_url'])
MPC_URL = os.environ.get('MPC_URL', CONFIG['services']['mpc_url'])
SERVICES = {'디지털 트윈': TWIN_URL, '예측': FORECAST_URL, 'MPC': MPC_URL}

HISTORY_HOURS: int = CONFIG['simulation']['history_hours']
ROOM_MIN_C: float = CONFIG['datacenter']['room_temp_min_c']
ROOM_MAX_C: float = CONFIG['datacenter']['room_temp_max_c']
CRITICAL_C: float = CONFIG['datacenter']['critical_temp_c']
DEFAULT_START = date(2026, 6, 4)

CONTROLLER_LABELS = {
    'mpc': 'MPC',
    'reactive': '반응형',
    'rule_tes': 'Rule-based + 축열 시간표',
    'rule_pid': 'Rule-based (PID)',
    'fixed_26': '26℃ 고정',
    'fixed_20': '20℃ 고정',
}
MODE_LABELS = {'free_cooling': '외기 냉각', 'hybrid': '혼합', 'chiller': '칠러'}
MPC_MODE_LABELS = {
    'mpc': '정상 (예측 사용)',
    'mpc_naive': '예측 불신 (최근 최대 부하 가정)',
    'fallback': 'Fallback (20℃ 고정)',
}
MPC_MODE_SHORT = {'mpc': '정상', 'mpc_naive': '예측 불신', 'fallback': 'Fallback'}
PLAY_INTERVAL_S = 1.0  # 자동 재생: 1초에 1시간씩 진행
JOB_POLL_S = 0.5  # 시뮬레이션 작업 상태 확인 간격

st.set_page_config(page_title='IDC 통합 관제', page_icon='🌡️', layout='wide')


# ---------------------------------------------------------------- 색 (역할별)
# 화면은 밝은 테마로 고정한다 (.streamlit/config.toml). 격자·보조선은 어느 배경에서도 읽히는 중간 회색.

C = {
    'blue': '#2a78d6',
    'orange': '#eb6834',
    'aqua': '#1baf7a',
    'purple': '#8a5cc2',
    'muted': '#898781',
    'ink': '#52514e',
    'grid': 'rgba(137,135,129,0.25)',
    'critical': '#d03b3b',
    'band': 'rgba(42,120,214,0.18)',
    'ok_band': 'rgba(12,163,12,0.07)',
}
MODE_COLORS = {'free_cooling': C['blue'], 'chiller': C['orange'], 'hybrid': C['aqua']}
# 색은 대상에 고정한다 (순위나 표시 개수에 따라 바뀌지 않게)
CONTROLLER_COLORS = {
    'mpc': C['blue'],
    'reactive': C['orange'],
    'rule_tes': C['purple'],
    'rule_pid': C['ink'],
    'fixed_26': C['aqua'],
    'fixed_20': C['muted'],
}
SCENARIO_COLORS = {
    'normal': C['muted'],
    'server_surge': C['blue'],
    'chiller_failure': C['orange'],
    'heatwave': C['aqua'],
    'heatwave_chiller': C['purple'],
    'surge_heatwave': '#c79a1e',
    'triple_crisis': C['ink'],
    'chiller_double_failure': C['critical'],
}
SITE_COLORS = {'chuncheon': C['blue'], 'seoul': C['orange'], 'jeju': C['aqua']}
# 예측 입력에 일부러 넣어 보는 이상 (IT 부하 예측 탭). 알람이 실제로 뜨는지 확인하는 용도
ANOMALY_LABELS = {
    'none': '없음 (계측값 그대로)',
    'stuck_48h': '센서 고착: 최근 48시간이 한 값으로 멈춤',
    'step_10pct': '계단: 최근 72시간 +10%',
    'drift_12pct': '드리프트: 168시간에 걸쳐 0 → +12%',
    'spike_15pct': '범위 안 스파이크: 마지막 값 +15%',
    'spike_50000': '범위 밖 스파이크: 마지막 값 50,000kW',
    'gap_1h': '결측: 1시간 빠짐',
}
CHECK_MARK = {True: '✅ 정상', False: '🚨 이상', None: '➖ 검사 못 함'}


# ---------------------------------------------------------------- REST 호출


class ApiError(RuntimeError):
    pass


@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(timeout=300.0)


def _call(method: str, url: str, **kwargs: Any) -> Any:
    try:
        r = _client().request(method, url, **kwargs)
    except httpx.HTTPError as e:
        raise ApiError(
            f'{url} 에 연결할 수 없습니다 ({type(e).__name__}). 서비스가 떠 있는지 확인하세요.'
        ) from e
    if not r.is_success:  # 2xx (작업 등록은 202)
        try:
            detail = r.json().get('detail', r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f'{url} → {r.status_code}: {detail}')
    return r.json()


def health(url: str) -> dict | None:
    try:
        return _call('GET', f'{url}/health', timeout=3.0)
    except ApiError:
        return None


@st.cache_data(show_spinner=False)
def telemetry(start: str | None, hours: int) -> dict:
    params = {'hours': hours} | ({'start': start} if start else {})
    return _call('GET', f'{TWIN_URL}/api/v1/telemetry', params=params)


@st.cache_data(show_spinner=False)
def simulate(start: str, hours: int, controller: str, scenario: str) -> dict:
    """폐루프 시뮬레이션. MPC 는 오래 걸려 작업으로 등록하고 끝날 때까지 상태를 확인한다 (API 는 바로 응답)."""
    job = _call(
        'POST',
        f'{MPC_URL}/api/v1/simulate/jobs',
        json={'start': start, 'hours': hours, 'controller': controller, 'scenario': scenario},
    )
    while job['status'] in ('queued', 'running'):
        time_module.sleep(JOB_POLL_S)
        job = _call('GET', f'{MPC_URL}/api/v1/simulate/jobs/{job["job_id"]}')
    if job['status'] == 'failed':
        raise ApiError(f'시뮬레이션 실패: {job["error"]}')
    return job['result']


@st.cache_data(show_spinner=False)
def scenarios() -> list[dict]:
    return _call('GET', f'{MPC_URL}/api/v1/scenarios')


@st.cache_data(show_spinner=False)
def site_results() -> dict:
    """미리 계산해 둔 지역별(서울·춘천·제주) 비교."""
    return _call('GET', f'{MPC_URL}/api/v1/sites')


@st.cache_data(show_spinner=False)
def precomputed(scenario: str) -> dict:
    return _call('GET', f'{MPC_URL}/api/v1/results', params={'scenario': scenario})


def input_history(start: str, anomaly: str = 'none') -> pd.Series:
    """start 직전 HISTORY_HOURS 시간의 IT 부하 (index = 시각). anomaly 를 주면 그 이상을 넣는다."""
    first = (pd.Timestamp(start) - pd.Timedelta(hours=HISTORY_HOURS)).isoformat()
    hist = telemetry(first, HISTORY_HOURS)
    s = pd.Series(hist['it_power_kw'], index=pd.to_datetime(hist['timestamps']), dtype=float)
    if anomaly == 'stuck_48h':
        s.iloc[-48:] = s.iloc[-49]
    elif anomaly == 'step_10pct':
        s.iloc[-72:] *= 1.10
    elif anomaly == 'drift_12pct':
        s.iloc[-168:] *= [1 + 0.12 * k / 167 for k in range(168)]
    elif anomaly == 'spike_15pct':
        s.iloc[-1] *= 1.15
    elif anomaly == 'spike_50000':
        s.iloc[-1] = 50000.0
    elif anomaly == 'gap_1h':
        s = s.drop(s.index[-50])
    return s


@st.cache_data(show_spinner=False)
def forecast(start: str, horizon: int, anomaly: str = 'none') -> dict:
    """start 직전까지의 계측 이력으로 앞으로 horizon 시간을 예측."""
    history = [
        {'timestamp': t.isoformat(), 'it_power_kw': float(v)}
        for t, v in input_history(start, anomaly).items()
    ]
    return _call(
        'POST',
        f'{FORECAST_URL}/api/v1/forecast',
        json={'horizon_hours': horizon, 'history': history},
    )


@st.cache_data(show_spinner=False)
def mpc_plan(start: str, room_temp_c: float) -> dict:
    first = (pd.Timestamp(start) - pd.Timedelta(hours=HISTORY_HOURS)).isoformat()
    hist = telemetry(first, HISTORY_HOURS)
    outdoor = telemetry(start, 24)['outdoor_temp_c']
    outdoor += [outdoor[-1]] * (24 - len(outdoor))
    history = [
        {'timestamp': t, 'it_power_kw': v} for t, v in zip(hist['timestamps'], hist['it_power_kw'])
    ]
    return _call(
        'POST',
        f'{MPC_URL}/api/v1/plan',
        json={'room_temp_c': room_temp_c, 'history': history, 'outdoor_forecast_c': outdoor},
    )


@st.cache_data(show_spinner=False)
def esg(
    it_power_kw: tuple, cooling_power_kw: tuple, cooling_mode: tuple, heat_removed_kw: tuple
) -> dict:
    return _call(
        'POST',
        f'{TWIN_URL}/api/v1/esg',
        json={
            'it_power_kw': list(it_power_kw),
            'cooling_power_kw': list(cooling_power_kw),
            'cooling_mode': list(cooling_mode),
            'heat_removed_kw': list(heat_removed_kw),
        },
    )


@st.cache_data(show_spinner=False)
def setpoint_sweep(it_power_kw: float, outdoor_temp_c: float) -> pd.DataFrame:
    """설정 온도 16~27℃ 를 0.5℃ 간격으로 바꿔가며 정상 상태 계산."""
    setpoints = [16 + 0.5 * i for i in range(23)]
    r = _call(
        'POST',
        f'{TWIN_URL}/api/v1/simulate',
        json={
            'points': [
                {'it_power_kw': it_power_kw, 'outdoor_temp_c': outdoor_temp_c, 'supply_temp_c': s}
                for s in setpoints
            ]
        },
    )
    return pd.DataFrame(r['results']).assign(setpoint_c=setpoints)


def esg_of(sim: dict) -> dict:
    ts = sim['timeseries']
    return esg(
        tuple(ts['it_power_kw']),
        tuple(ts['cooling_power_kw']),
        tuple(ts['cooling_mode']),
        tuple(ts['heat_removed_kw']),
    )


def frame(sim: dict) -> pd.DataFrame:
    df = pd.DataFrame(sim['timeseries'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# ---------------------------------------------------------------- 그래프 공통


def styled(
    fig: go.Figure,
    height: int = 300,
    y_title: str = '',
    x_title: str = '',
    legend: bool = True,
    time_axis: bool = True,
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30 if legend else 10, b=10),
        showlegend=legend,
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0, traceorder='normal'
        ),
        yaxis_title=y_title,
        xaxis_title=x_title,
    )
    fig.update_xaxes(showgrid=False)
    if time_axis:
        fig.update_xaxes(tickformat='%H:%M<br>%m-%d', hoverformat='%m-%d %H:%M')
    fig.update_yaxes(gridcolor=C['grid'], zeroline=False)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width='stretch', key=key, config={'displayModeBar': False})


def room_limits(fig: go.Figure, low: float, high: float) -> None:
    """허용 구간(18~27℃)을 옅은 띠로 칠하고 상한선을 긋는다. 세로축은 [low, high] 로 좁힌다."""
    fig.add_hrect(
        y0=ROOM_MIN_C, y1=ROOM_MAX_C, fillcolor=C['ok_band'], line_width=0, layer='below'
    )
    fig.add_hline(
        y=ROOM_MAX_C,
        line_dash='dash',
        line_color=C['critical'],
        line_width=1,
        annotation_text=f'상한 {ROOM_MAX_C:.0f}℃',
        annotation_position='top left',
    )
    fig.add_hline(
        y=ROOM_MIN_C,
        line_dash='dot',
        line_color=C['muted'],
        line_width=1,
        annotation_text=f'하한 {ROOM_MIN_C:.0f}℃',
        annotation_position='bottom left',
    )
    fig.update_yaxes(range=[low, high])


def room_range(*series: Any) -> tuple[float, float]:
    """서버실 온도 그래프의 세로축 범위: 허용 구간 전체가 보이되 빈 공간은 줄인다."""
    low = min(min(s.min() for s in series), ROOM_MIN_C) - 1.0
    high = max(max(s.max() for s in series), ROOM_MAX_C) + 1.2
    return low, high


def cursor_line(fig: go.Figure, x: Any) -> None:
    """선택한 시각을 세로 점선으로 표시."""
    fig.add_shape(
        type='line',
        x0=x,
        x1=x,
        xref='x',
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color=C['ink'], width=1, dash='dot'),
    )


def mode_bands(fig: go.Figure, df: pd.DataFrame) -> None:
    """시간축 배경에 냉각 모드를 색 띠로 깐다 (같은 모드가 이어지는 구간마다 한 칸)."""
    half = pd.Timedelta(minutes=30)
    run_start = 0
    for k in range(1, len(df) + 1):
        if k == len(df) or df.cooling_mode[k] != df.cooling_mode[run_start]:
            fig.add_vrect(
                x0=df.timestamp[run_start] - half,
                x1=df.timestamp[k - 1] + half,
                fillcolor=MODE_COLORS[df.cooling_mode[run_start]],
                opacity=0.30,
                line_width=0,
                layer='below',
            )
            run_start = k
    for mode, label in MODE_LABELS.items():  # 범례용
        fig.add_scatter(
            x=[df.timestamp[0]],
            y=[None],
            mode='markers',
            name=label,
            hoverinfo='skip',
            marker=dict(size=11, symbol='square', color=MODE_COLORS[mode], opacity=0.55),
        )
    fig.update_xaxes(type='date')


def pue_gauge(value: float, reference: float | None = None) -> go.Figure:
    """현재 PUE. reference(20℃ 고정 운영의 PUE)를 눈금 선으로 함께 표시한다."""
    gauge = {
        'axis': {'range': [1.0, 1.6], 'tickvals': [1.0, 1.2, 1.4, 1.6]},
        'bar': {'color': C['blue'], 'thickness': 0.6},
        'borderwidth': 0,
        'bgcolor': C['grid'],
    }
    if reference is not None:
        gauge['threshold'] = {
            'line': {'color': C['ink'], 'width': 3},
            'thickness': 0.9,
            'value': min(max(reference, 1.0), 1.6),
        }
    fig = go.Figure(
        go.Indicator(
            mode='gauge+number',
            value=value,
            gauge=gauge,
            number={'valueformat': '.3f', 'font': {'size': 34}},
        )
    )
    fig.update_layout(height=150, margin=dict(l=30, r=30, t=18, b=0))
    return fig


def table_view(df: pd.DataFrame, label: str = '표로 보기') -> None:
    with st.expander(label):
        st.dataframe(df, width='stretch', hide_index=True)


def small_stat(col: DeltaGenerator, label: str, value: str) -> None:
    col.markdown(f':gray[{label}]  \n**{value}**')


# ---------------------------------------------------------------- 사이드바

st.title('IDC 통합 관제 대시보드')

with st.sidebar:
    st.header('서비스 상태')
    status = {name: health(url) for name, url in SERVICES.items()}
    for name, h in status.items():
        if h is None:
            st.markdown(f'❌ **{name}** 연결 안 됨  \n`{SERVICES[name]}`')
        elif h.get('model') == 'unavailable':
            st.markdown(f'⚠️ **{name}** 모델 파일 없음')
        elif h.get('status') == 'degraded':
            st.markdown(f'⚠️ **{name}** 저하  \n`{", ".join(h.get("reasons", []))}`')
        else:
            st.markdown(f'✅ **{name}** 정상')

if any(h is None for h in status.values()):
    st.error('연결되지 않은 서비스가 있습니다. 저장소 루트에서 세 서비스를 먼저 띄워 주세요.')
    st.code(
        'uvicorn services.twin.main:app --port 8001\n'
        'uvicorn services.forecast.main:app --port 8002\n'
        'uvicorn services.mpc.main:app --port 8003',
        language='bash',
    )
    st.stop()

try:
    meta = telemetry(None, 1)
    scenario_list = scenarios()
except ApiError as e:
    st.error(str(e))
    st.stop()

data_start, data_end = pd.Timestamp(meta['data_start']), pd.Timestamp(meta['data_end'])
first_day = (data_start + pd.Timedelta(hours=HISTORY_HOURS)).ceil('D').date()
last_day = (data_end - pd.Timedelta(hours=23)).floor('D').date()
SCENARIOS = {s['name']: s for s in scenario_list}


def is_limit_test(name: str) -> bool:
    """한계 시험 시나리오: 과제 요구(27℃ 이하 유지)의 대상이 아니고, 얼마나 버티는지를 본다."""
    return bool(SCENARIOS[name].get('limit_test'))


def scenario_label(name: str) -> str:
    return ('[한계 시험] ' if is_limit_test(name) else '') + SCENARIOS[name]['label']


with st.sidebar:
    st.header('운영 조건')
    start_day = st.date_input(
        '시작 날짜',
        value=min(max(DEFAULT_START, first_day), last_day),
        min_value=first_day,
        max_value=last_day,
        help=f'계측 데이터 {data_start:%Y-%m-%d} ~ {data_end:%Y-%m-%d}. '
        f'예측에 이전 {HISTORY_HOURS}시간 이력이 필요합니다.',
    )
    max_hours = int((data_end - pd.Timestamp(start_day)) / pd.Timedelta(hours=1)) + 1
    hours = st.selectbox(
        '구간',
        [h for h in (24, 48, 72) if h <= max_hours] or [24],
        format_func=lambda h: f'{h}시간',
    )
    controller = st.selectbox(
        '제어기',
        ['mpc', 'rule_pid', 'rule_tes', 'reactive', 'fixed_20'],
        format_func=CONTROLLER_LABELS.get,
        help='MPC 는 1시간당 약 0.3초가 걸립니다.',
    )
    st.header('위기 시나리오 주입')
    scenario = st.radio(
        '관제 화면에 적용할 시나리오',
        list(SCENARIOS),
        format_func=scenario_label,
        label_visibility='collapsed',
    )
    st.caption(
        '시작 시각부터 구간 끝까지 적용됩니다. [한계 시험]은 냉각 용량이 발열보다 작아 27℃ 유지가 '
        '불가능한 조건에서 버티는 시간을 보는 시나리오입니다.'
    )
    if st.button(
        '다시 계산', width='stretch', help='저장된 결과를 지우고 서비스를 다시 호출합니다.'
    ):
        st.cache_data.clear()
        st.rerun()

START = datetime.combine(start_day, time(0)).isoformat()
END = pd.Timestamp(START) + pd.Timedelta(hours=hours - 1)


def system_status() -> tuple[str, list[str]]:
    """실제 시스템 상태 ('정상' | '주의' | '경보')와 이유.

    사이드바에서 고른 시나리오(시뮬레이션 조건)와는 별개로, 지금 서비스와 입력 데이터가 실제로 어떤지 본다:
    서비스 응답, MPC 가 정상 경로로 돌 수 있는지(/health 의 degraded), 예측 입력 이력의 품질·학습 범위.
    """
    level, reasons = 0, []
    mpc_health = status['MPC'] or {}
    if status['예측'] and status['예측'].get('model') == 'unavailable':
        level = 2
        reasons.append('예측 모델 파일 없음')
    if mpc_health.get('status') == 'degraded':
        deps = mpc_health.get('dependencies', {})
        down = [name for name, state in deps.items() if state != 'ok']
        level = max(level, 2 if down else 1)
        if down:
            reasons.append('MPC 가 ' + '·'.join(down) + ' 서비스에 닿지 못함')
        last = mpc_health.get('last_plan_mode')
        if last not in (None, 'mpc'):
            reasons.append(f'MPC 직전 계획이 {MPC_MODE_SHORT.get(last, last)} 모드')
    try:
        fc_now = forecast(START, 24)
    except ApiError as e:
        return '경보', reasons + [f'예측 불가: {e}']
    if not fc_now['in_training_range']:
        level = max(level, 2)
        reasons.append('예측 입력·결과가 학습 범위 밖')
    for check in fc_now.get('input_checks', []):
        if check['ok'] is False:
            level = max(level, 1)
            reasons.append(f'입력 데이터 {check["label"]}')
    return ['정상', '주의', '경보'][level], reasons


# 실제 시스템 상태와, 지금 보고 있는 시뮬레이션 조건을 따로 보여준다 (위기 주입은 조건이지 실제 상태가 아니다)
state, state_reasons = system_status()
state_color, state_icon = {
    '정상': ('green', '✅'),
    '주의': ('orange', '⚠️'),
    '경보': ('red', '🚨'),
}[state]
scenario_badge = (
    ':gray-badge[시뮬레이션 조건: 정상 운영]'
    if scenario == 'normal'
    else f':orange-badge[⚠️ 위기 주입: {SCENARIOS[scenario]["label"]}]'
)
st.markdown(
    f':{state_color}-badge[{state_icon} 시스템 상태: {state}] '
    f':gray-badge[{pd.Timestamp(START):%Y-%m-%d %H:%M} ~ {END:%m-%d %H:%M} · {hours}시간] '
    f':blue-badge[제어기: {CONTROLLER_LABELS[controller]}] {scenario_badge}'
)
if state_reasons:
    (st.error if state == '경보' else st.warning)(
        f'{state_icon} 시스템 상태 {state}: ' + ' · '.join(state_reasons)
    )
else:
    st.caption(
        '시스템 상태: 서비스 3개 응답, MPC 정상 경로, 예측 입력 이력(고착·급변·평균 이동)과 '
        '학습 범위 모두 정상. 위기 주입은 시뮬레이션 조건이라 이 상태에 들어가지 않습니다.'
    )

tab_monitor, tab_forecast, tab_plan, tab_compare, tab_crisis, tab_esg, tab_sites = st.tabs(
    [
        '관제',
        'IT 부하 예측',
        'MPC 계획',
        '제어기 비교',
        '위기 시나리오',
        'ESG · 설정 온도',
        '지역 비교',
    ]
)


# ---------------------------------------------------------------- 관제


def monitor_live(sim: dict, base: dict, playing: bool) -> None:
    """시각 슬라이더에 따라 바뀌는 부분. 자동 재생 중에는 이 부분만 주기적으로 다시 그린다."""
    df = frame(sim)
    # 20℃ 고정이 위험 온도에 닿아 먼저 멈췄으면 그 뒤 시각은 비교값이 없다 (NaN)
    bdf = frame(base).set_index('timestamp').reindex(df.timestamp).reset_index()
    n = len(df)
    key = f'cursor|{START}|{hours}|{controller}|{scenario}'
    if key not in st.session_state:
        st.session_state[key] = n - 1
    elif playing:
        st.session_state[key] = (st.session_state[key] + 1) % n

    c_slider, c_play = st.columns([5, 1], vertical_alignment='bottom')
    i = c_slider.select_slider(
        '시각',
        options=list(range(n)),
        key=key,
        format_func=lambda k: f'{df.timestamp[k]:%m-%d %H:%M}',
    )
    if (
        c_play.toggle(
            '자동 재생', value=playing, key='playing_toggle', help='1초에 1시간씩 진행합니다.'
        )
        != playing
    ):
        st.session_state['playing'] = not playing
        st.rerun()
    now = df.iloc[i]

    compare = controller != 'fixed_20' and pd.notna(bdf.pue[i])
    c_gauge, c_now = st.columns([1, 3], vertical_alignment='center')
    with c_gauge:
        show(pue_gauge(float(now.pue), float(bdf.pue[i]) if compare else None), 'gauge')
        st.caption('현재 PUE' + (f' · 눈금 선: 20℃ 고정 {bdf.pue[i]:.3f}' if compare else ''))
    with c_now:
        m = st.columns(4)
        m[0].metric('냉각 모드', MODE_LABELS[now.cooling_mode], border=True)
        m[1].metric('외기온도', f'{now.outdoor_temp_c:.1f}℃', border=True)
        m[2].metric('설정 온도', f'{now.setpoint_c:.1f}℃', border=True)
        m[3].metric('서버실 온도', f'{now.room_temp_c:.1f}℃', border=True)
        notes = [
            f'전체 전력 {now.facility_power_kw:,.0f} kW'
            + (
                f' (20℃ 고정 대비 {now.facility_power_kw - bdf.facility_power_kw[i]:+,.0f} kW)'
                if compare
                else ''
            )
        ]
        if controller == 'mpc':
            notes.append(f'MPC 동작 모드: {MPC_MODE_LABELS.get(now["mode"], now["mode"])}')
        st.caption(' · '.join(notes))

    left, right = st.columns(2)
    with left:
        st.subheader('전력 소비 추이')
        other = df.facility_power_kw - df.it_power_kw - df.cooling_power_kw
        fig = go.Figure()
        for name, y, color in (
            ('IT', df.it_power_kw, C['blue']),
            ('냉각', df.cooling_power_kw, C['orange']),
            ('기타 손실', other, C['aqua']),
        ):
            fig.add_scatter(
                x=df.timestamp,
                y=y,
                name=name,
                stackgroup='power',
                mode='lines',
                line=dict(width=0.5, color=color),
                hovertemplate='%{y:,.0f} kW',
            )
        cursor_line(fig, now.timestamp)
        show(styled(fig, 290, 'kW'), 'power')
    with right:
        st.subheader('서버실 온도')
        fig = go.Figure()
        fig.add_scatter(
            x=df.timestamp,
            y=df.room_temp_c,
            name='서버실 온도',
            line=dict(color=C['blue'], width=2),
            hovertemplate='%{y:.1f}℃',
        )
        fig.add_scatter(
            x=df.timestamp,
            y=df.setpoint_c,
            name='설정 온도',
            hovertemplate='%{y:.1f}℃',
            line=dict(color=C['orange'], width=2, dash='dot', shape='hv'),
        )
        room_limits(fig, *room_range(df.room_temp_c, df.setpoint_c))
        cursor_line(fig, now.timestamp)
        show(styled(fig, 290, '℃'), 'room')

    st.subheader('냉각 모드와 외기온도')
    fig = go.Figure()
    mode_bands(fig, df)
    fig.add_scatter(
        x=df.timestamp,
        y=df.outdoor_temp_c,
        name='외기온도',
        line=dict(color=C['ink'], width=2),
        customdata=list(zip(df.cooling_mode.map(MODE_LABELS), df.free_cooling_fraction * 100)),
        hovertemplate='%{y:.1f}℃ · %{customdata[0]} (외기 냉각 %{customdata[1]:.0f}%)',
    )
    cursor_line(fig, now.timestamp)
    show(styled(fig, 230, '외기온도 (℃)'), 'modes')
    hours_by_mode = df.cooling_mode.value_counts()
    st.caption(
        '배경색이 냉각 모드입니다. '
        + ' · '.join(f'{MODE_LABELS[m]} {hours_by_mode.get(m, 0)}시간' for m in MODE_LABELS)
    )


with tab_monitor:
    try:
        with st.spinner(f'{CONTROLLER_LABELS[controller]} 폐루프 시뮬레이션 {hours}시간 계산 중…'):
            sim = simulate(START, hours, controller, scenario)
            base = (
                sim if controller == 'fixed_20' else simulate(START, hours, 'fixed_20', scenario)
            )
    except ApiError as e:
        st.error(str(e))
        sim = base = None

    if sim:
        s, df = sim['summary'], frame(sim)
        if s.get('hours_to_critical'):
            st.error(
                f'🚨 경보: 시작 {s["hours_to_critical"]}시간 만에 서버실 온도가 위험 온도 '
                f'{CRITICAL_C:.0f}℃ 에 도달해 시뮬레이션을 멈췄습니다 (최고 {s["max_room_temp_c"]:.1f}℃). '
                '냉각 용량이 발열보다 작아 설정 온도 조정으로는 막을 수 없는 상황입니다.'
                + (
                    ' (한계 시험 시나리오: 27℃ 유지 요구의 대상이 아닙니다.)'
                    if is_limit_test(scenario)
                    else ''
                )
            )
        elif s['diverged'] or not s['temp_ok']:
            st.error(
                f'⚠️ 경고: 서버실 온도가 {ROOM_MAX_C:.0f}℃ 를 넘었습니다 (최고 {s["max_room_temp_c"]:.1f}℃).'
            )
        elif s['violation_hours']:
            st.warning(
                f'⚠️ 주의: 온도는 {ROOM_MAX_C:.0f}℃ 이하지만 {s["violation_hours"]}시간 동안 비상 여유 시간이 '
                f'기준에 못 미쳤습니다 (최소 {s["min_ride_through_min"]:.1f}분).'
            )
        else:
            st.success(
                f'✅ 서버실 온도 정상 (전 구간 {ROOM_MIN_C:.0f}~{ROOM_MAX_C:.0f}℃, 위반 없음)'
            )
        for w in sim['warnings']:
            if not w.startswith('critical_temp_reached'):  # 위 경보와 같은 내용
                st.warning(w)
        degraded_hours = {m: n for m, n in s['mode_hours'].items() if m in MPC_MODE_LABELS}
        degraded_hours.pop('mpc', None)
        if degraded_hours:
            st.warning(
                '⚠️ MPC 가 정상 경로에서 물러나 운전한 시간이 있습니다: '
                + ' · '.join(f'{MPC_MODE_LABELS[m]} {n}시간' for m, n in degraded_hours.items())
            )
        if len(df) < hours or len(frame(base)) < len(df):
            st.caption(
                f'시뮬레이션이 {len(df)}시간에서 멈췄습니다. 아래 지표는 멈출 때까지의 값이고, '
                '20℃ 고정 대비 비교는 두 운전이 모두 진행한 시간만 봅니다.'
            )

        common = min(len(df), len(base['timeseries']['facility_power_kw']))
        saving = (
            1
            - df.facility_power_kw.iloc[:common].sum()
            / sum(base['timeseries']['facility_power_kw'][:common])
        ) * 100
        k = st.columns(5)
        k[0].metric(
            '평균 PUE',
            f'{s["average_pue"]:.3f}',
            border=True,
            height='stretch',
            chart_data=df.pue.tolist(),
            chart_type='line',
            help='전체 전력량 / IT 전력량. 팬 전력이 IT 로 잡히므로 전체 전력과 함께 봐야 합니다.',
        )
        k[1].metric(
            '전체 전력량',
            f'{s["facility_energy_kwh"] / 1000:,.1f} MWh',
            border=True,
            height='stretch',
            chart_data=df.facility_power_kw.tolist(),
            chart_type='line',
            delta=None if controller == 'fixed_20' else f'{-saving:+.1f}% (20℃ 고정 대비)',
            delta_color='inverse',
            help='IT + 냉각 + 기타 손실. 최적화 목표입니다.',
        )
        k[2].metric(
            '최고 서버실 온도',
            f'{s["max_room_temp_c"]:.1f}℃',
            border=True,
            height='stretch',
            chart_data=df.room_temp_c.tolist(),
            chart_type='line',
        )
        k[3].metric(
            '온도 유지율',
            f'{s["temp_maintenance_pct"]:.1f}%',
            border=True,
            height='stretch',
            help=f'서버실 온도가 {ROOM_MIN_C:.0f}~{ROOM_MAX_C:.0f}℃ 인 시간 비율',
        )
        k[4].metric(
            '평균 설정 온도',
            f'{s["average_setpoint_c"]:.1f}℃',
            border=True,
            height='stretch',
            chart_data=df.setpoint_c.tolist(),
            chart_type='line',
        )

        playing = st.session_state.get('playing', False)
        st.fragment(monitor_live, run_every=PLAY_INTERVAL_S if playing else None)(
            sim, base, playing
        )

        if controller == 'mpc':
            st.caption(
                '구간 전체 MPC 동작 모드: '
                + ' · '.join(
                    f'{MPC_MODE_LABELS.get(m, m)} {n}시간' for m, n in s['mode_hours'].items()
                )
            )
        st.subheader('서버 온도 분포')
        compare_dist = controller != 'fixed_20'
        fig = go.Figure()
        bins = dict(
            start=ROOM_MIN_C - 2.0, end=max(ROOM_MAX_C + 2.0, df.room_temp_c.max() + 0.5), size=0.5
        )
        fig.add_histogram(
            x=df.room_temp_c,
            xbins=bins,
            name=CONTROLLER_LABELS[controller],
            marker=dict(color=C['blue']),
            opacity=0.85,
            hovertemplate='%{x}℃: %{y}시간<extra></extra>',
        )
        if compare_dist:
            fig.add_histogram(
                x=frame(base).room_temp_c,
                xbins=bins,
                name=CONTROLLER_LABELS['fixed_20'],
                marker=dict(color=C['muted']),
                opacity=0.55,
                hovertemplate='%{x}℃: %{y}시간<extra></extra>',
            )
        fig.add_vrect(
            x0=ROOM_MIN_C, x1=ROOM_MAX_C, fillcolor=C['ok_band'], line_width=0, layer='below'
        )
        fig.add_vline(
            x=ROOM_MAX_C,
            line_dash='dash',
            line_color=C['critical'],
            line_width=1,
            annotation_text=f'상한 {ROOM_MAX_C:.0f}℃',
            annotation_position='top right',
        )
        fig.update_layout(barmode='overlay', bargap=0.05)
        show(styled(fig, 260, '시간 수', '서버실 온도 (℃)', time_axis=False), 'room_hist')
        q = df.room_temp_c.quantile([0.05, 0.5, 0.95])
        st.caption(
            f'0.5℃ 간격으로 센 시간 수입니다. 중앙값 {q[0.5]:.1f}℃, 5~95% 범위 {q[0.05]:.1f}~{q[0.95]:.1f}℃, '
            f'허용 범위({ROOM_MIN_C:.0f}~{ROOM_MAX_C:.0f}℃) 안 {s["temp_maintenance_pct"]:.1f}%. '
            '서버실은 한 구역(평균 흡기 온도)으로 모델링해 랙별 분포가 아니라 시간에 따른 분포입니다.'
        )
        table_view(df.rename(columns={'timestamp': '시각'}))


# ---------------------------------------------------------------- IT 부하 예측

with tab_forecast:
    c_horizon, c_anomaly = st.columns([1, 2])
    horizon = c_horizon.radio(
        '예측 기간', [24, 168], format_func=lambda h: f'{h}시간', horizontal=True
    )
    anomaly = c_anomaly.selectbox(
        '입력 데이터 이상 주입 (알람 확인용)',
        list(ANOMALY_LABELS),
        format_func=ANOMALY_LABELS.get,
        help='예측에 넣는 최근 이력에 일부러 이상을 넣어, 예측 서비스가 알람을 내는지 확인합니다. '
        '이 탭의 예측에만 적용되고 관제·시스템 상태에는 들어가지 않습니다.',
    )
    try:
        past = input_history(START, anomaly).iloc[-72:]
        fc = forecast(START, horizon, anomaly)
        try:
            actual = telemetry(START, horizon)
        except ApiError:
            actual = None
    except ApiError as e:
        st.error(
            f'🚨 예측 거부: {e}' if anomaly != 'none' else str(e)
        )  # 결측 등은 서비스가 예측을 거부한다 (422)
        fc = None

    if fc:
        t = pd.to_datetime(fc['timestamps'])
        pred = pd.Series(fc['predictions'], index=t)
        st.caption(
            f'예측 시점 {pd.Timestamp(fc["origin_time"]):%Y-%m-%d %H:%M} · 모델 {fc["model"]} · '
            + (
                '정상 조건의 계측 이력 기준 (위기 시나리오 미적용)'
                if anomaly == 'none'
                else f'이상 주입: {ANOMALY_LABELS[anomaly]}'
            )
        )
        checks = fc.get('input_checks', [])
        quality_ok = fc.get('input_quality_ok', True)
        if not fc['in_training_range']:
            st.error(
                '🚨 알람: 입력 또는 예측값이 학습 범위를 벗어났습니다. MPC 는 이 경우 예측을 쓰지 않습니다.'
            )
        elif not quality_ok:
            st.error(
                '🚨 알람: 값은 학습 범위 안이지만 입력 이력에 이상이 있습니다 ('
                + ', '.join(c['label'] for c in checks if c['ok'] is False)
                + '). MPC 는 이 경우 예측을 쓰지 않고 최근 최대 부하를 가정합니다.'
            )
        elif anomaly != 'none':
            st.warning('⚠️ 주입한 이상을 감지하지 못했습니다 (알람 없음).')
        for w in fc['warnings']:
            st.caption(f'경고: {w}')
        if checks:
            with st.expander('입력 데이터 품질 검사', expanded=not quality_ok):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                '검사': c['label'],
                                '판정': CHECK_MARK[c['ok']],
                                '측정값': c['measured'],
                                '기준': c['limit'],
                            }
                            for c in checks
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )
                st.caption(
                    '학습 범위 검사만으로는 잡히지 않는 이상을 봅니다. 기준은 학습 구간의 정상 변동보다 넉넉히 크게 '
                    '잡은 값입니다 (config/settings.yaml 의 forecast.input_quality).'
                )

        cols = st.columns(4)
        cols[0].metric('예측 평균', f'{pred.mean():,.0f} kW', border=True)
        cols[1].metric(
            '90% 구간 평균 폭',
            border=True,
            value=f'{(pd.Series(fc["pi_upper"]) - pd.Series(fc["pi_lower"])).mean():,.0f} kW',
        )
        act = None
        if actual:
            act = pd.Series(actual['it_power_kw'], index=pd.to_datetime(actual['timestamps']))
            both = pd.concat(
                [
                    act,
                    pred,
                    pd.Series(fc['pi_lower'], index=t),
                    pd.Series(fc['pi_upper'], index=t),
                ],
                axis=1,
                keys=['y', 'q50', 'lo', 'hi'],
            ).dropna()
            if len(both):
                cols[2].metric(
                    '오차 (MAPE)',
                    f'{((both.y - both.q50).abs() / both.y).mean() * 100:.2f}%',
                    border=True,
                    help=f'실제값이 있는 {len(both)}시간 기준',
                )
                cols[3].metric(
                    '90% 구간 적중률',
                    border=True,
                    value=f'{((both.y >= both.lo) & (both.y <= both.hi)).mean() * 100:.0f}%',
                )

        fig = go.Figure()
        fig.add_scatter(
            x=list(t) + list(t[::-1]),
            y=fc['pi_upper'] + fc['pi_lower'][::-1],
            fill='toself',
            fillcolor=C['band'],
            line=dict(width=0),
            name='90% 구간',
            hoverinfo='skip',
        )
        fig.add_scatter(
            x=past.index,
            y=past.values,
            name='계측 (과거)' if anomaly == 'none' else '입력 이력 (이상 주입)',
            line=dict(color=C['muted'], width=2),
            hovertemplate='%{y:,.0f} kW',
        )
        if act is not None:
            fig.add_scatter(
                x=act.index,
                y=act.values,
                name='실제',
                line=dict(color=C['orange'], width=2),
                hovertemplate='%{y:,.0f} kW',
            )
        fig.add_scatter(
            x=t,
            y=fc['predictions'],
            name='예측 (중앙값)',
            line=dict(color=C['blue'], width=2),
            hovertemplate='%{y:,.0f} kW',
        )
        cursor_line(fig, pd.Timestamp(fc['origin_time']))
        show(styled(fig, 380, 'IT 부하 (kW)'), 'forecast')
        st.caption('세로 점선이 예측 시점입니다.')
        table_view(
            pd.DataFrame(
                {
                    '시각': t,
                    '예측 (kW)': fc['predictions'],
                    '하한 (kW)': fc['pi_lower'],
                    '상한 (kW)': fc['pi_upper'],
                }
            )
        )


# ---------------------------------------------------------------- MPC 계획

with tab_plan:
    room_now = st.slider('현재 서버실 온도 (℃)', 18.0, 27.0, 20.0, 0.5)
    try:
        with st.spinner('MPC 계획 계산 중…'):
            plan = mpc_plan(START, room_now)
    except ApiError as e:
        st.error(str(e))
        plan = None

    if plan:
        st.caption(
            f'{pd.Timestamp(START):%Y-%m-%d %H:%M} 시점에 세운 24시간 계획 (정상 조건). '
            '실제 운영에서는 첫 1시간만 적용하고 매시간 다시 계산합니다.'
        )
        cols = st.columns(4)
        cols[0].metric('지금 적용할 설정 온도', f'{plan["setpoint_c"]:.1f}℃', border=True)
        cols[1].metric(
            '동작 모드',
            MPC_MODE_SHORT.get(plan['mode'], plan['mode']),
            border=True,
            help=MPC_MODE_LABELS.get(plan['mode'], ''),
        )
        energy_mwh = plan['expected_facility_energy_kwh'] / 1000
        cols[2].metric('24시간 예상 전력량', f'{energy_mwh:,.1f} MWh', border=True)
        saving_pct = plan.get('expected_saving_pct')
        cols[3].metric(
            '예상 절감 (20℃ 고정 대비)',
            '계산 불가' if saving_pct is None else f'{saving_pct:.1f}%',
            border=True,
        )
        if plan['mode'] == 'fallback':
            st.error('⚠️ 트윈을 쓸 수 없어 기준 설정 20℃ 로 돌아갔습니다.')
        elif plan['mode'] == 'mpc_naive':
            st.warning('⚠️ 예측을 믿을 수 없어 최근 3시간 최대 부하가 계속된다고 가정했습니다.')

        if plan['timestamps']:
            t = pd.to_datetime(plan['timestamps'])
            c1, c2 = st.columns(2)
            with c1:
                st.subheader('설정 온도 계획과 예상 서버실 온도')
                fig = go.Figure()
                fig.add_scatter(
                    x=t,
                    y=plan['predicted_room_temp_c'],
                    name='예상 서버실 온도',
                    line=dict(color=C['blue'], width=2),
                    hovertemplate='%{y:.1f}℃',
                )
                fig.add_scatter(
                    x=t,
                    y=plan['plan_c'],
                    name='설정 온도 계획',
                    hovertemplate='%{y:.1f}℃',
                    line=dict(color=C['orange'], width=2, dash='dot', shape='hv'),
                )
                room_limits(
                    fig,
                    *room_range(
                        pd.Series(plan['predicted_room_temp_c']), pd.Series(plan['plan_c'])
                    ),
                )
                show(styled(fig, 320, '℃'), 'plan_temp')
            with c2:
                st.subheader('예상 전력')
                fig = go.Figure()
                fig.add_scatter(
                    x=t,
                    y=plan['predicted_facility_kw'],
                    name='전체 전력',
                    line=dict(color=C['blue'], width=2),
                    hovertemplate='%{y:,.0f} kW',
                )
                fig.add_scatter(
                    x=t,
                    y=plan['forecast_it_kw'],
                    name='IT 부하 예측',
                    line=dict(color=C['orange'], width=2),
                    hovertemplate='%{y:,.0f} kW',
                )
                show(styled(fig, 320, 'kW'), 'plan_power')
        if plan['warnings']:
            with st.expander(f'경고 {len(plan["warnings"])}건'):
                for w in plan['warnings']:
                    st.write(f'- {w}')


# ---------------------------------------------------------------- 제어기 비교 (사전 계산)

with tab_compare:
    try:
        res = precomputed('base')
    except ApiError as e:
        st.error(str(e))
        res = None

    if res:
        t0 = pd.Timestamp(res['start'])
        st.caption(
            f'{t0:%Y-%m-%d} 부터 {res["hours"] // 24}일 (예측 모델 Test 구간). '
            '`scripts/simulate_mpc.py` 로 미리 계산해 둔 결과입니다.'
        )
        order = [c for c in CONTROLLER_LABELS if c in res['controllers']]
        ref = res['controllers']['fixed_20']
        summary = pd.DataFrame(
            [
                {
                    '제어기': CONTROLLER_LABELS[c],
                    '전체 전력량 (MWh)': m['facility_energy_kwh'] / 1000,
                    '전력 절감 (%)': m['saving_vs_fixed_20_pct'],
                    '전기요금 (만원)': m['energy_cost_krw'] / 1e4,
                    '요금 절감 (%)': m['cost_saving_vs_rule_pid_pct'],
                    '냉각 전력량 (MWh)': m['cooling_energy_kwh'] / 1000,
                    '팬 증가분 (MWh)': m['fan_extra_energy_kwh'] / 1000,
                    '평균 PUE': m['average_pue'],
                    '평균 고장률': m['average_xfactor'],
                    '최고 온도 (℃)': m['max_room_temp_c'],
                    '위반 (시간)': m['violation_hours'],
                    '설정 변경 총량 (℃)': m['setpoint_change_c_total'],
                }
                for c in order
                for m in [res['controllers'][c]]
            ]
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader('절감 내역 (20℃ 고정 대비)')
            shown = [c for c in order if c != 'fixed_20']
            labels = [CONTROLLER_LABELS[c] for c in shown]
            cooling_saved = [
                (ref['cooling_energy_kwh'] - res['controllers'][c]['cooling_energy_kwh']) / 1000
                for c in shown
            ]
            fan_added = [
                -(res['controllers'][c]['fan_extra_energy_kwh'] - ref['fan_extra_energy_kwh'])
                / 1000
                for c in shown
            ]
            fig = go.Figure()
            fig.add_bar(
                x=labels,
                y=cooling_saved,
                name='냉각 전력 절감',
                width=0.5,
                marker=dict(color=C['blue'], cornerradius=4),
                text=[f'+{v:.0f}' for v in cooling_saved],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='#ffffff'),
                hovertemplate='%{y:,.1f} MWh',
            )
            fig.add_bar(
                x=labels,
                y=fan_added,
                name='팬 전력 증가',
                width=0.5,
                marker=dict(color=C['orange'], cornerradius=4),
                text=[f'{v:.0f}' for v in fan_added],
                textposition='outside',
                hovertemplate='%{y:,.1f} MWh',
            )
            for label, c, saved in zip(labels, shown, cooling_saved):
                m = res['controllers'][c]
                net = (ref['facility_energy_kwh'] - m['facility_energy_kwh']) / 1000
                fig.add_annotation(
                    x=label,
                    y=saved,
                    yshift=12,
                    showarrow=False,
                    text=f'순 절감 {net:.0f} MWh ({m["saving_vs_fixed_20_pct"]:.1f}%)',
                )
            fig.update_layout(barmode='relative')
            fig.update_yaxes(
                range=[min(fan_added) * 2.4, max(cooling_saved) * 1.18],
                zeroline=True,
                zerolinecolor=C['muted'],
            )
            show(styled(fig, 340, 'MWh', time_axis=False), 'saving')
        with c2:
            st.subheader('일평균 전체 전력')
            fig = go.Figure()
            for c in order:
                ts = pd.DataFrame(res['timeseries'][c])
                daily = (
                    ts.set_index(pd.to_datetime(ts['timestamp']))
                    .facility_power_kw.resample('D')
                    .mean()
                )
                fig.add_scatter(
                    x=daily.index,
                    y=daily.values,
                    name=CONTROLLER_LABELS[c],
                    mode='lines+markers',
                    line=dict(
                        color=CONTROLLER_COLORS[c],
                        width=2,
                        dash='dot' if c == 'fixed_20' else 'solid',
                    ),
                    marker=dict(size=8),
                    hovertemplate='%{y:,.0f} kW',
                )
            fig.update_xaxes(tickformat='%m-%d', hoverformat='%m-%d')
            show(styled(fig, 340, 'kW', time_axis=False), 'daily')

        st.dataframe(
            summary,
            width='stretch',
            hide_index=True,
            column_config={
                '전체 전력량 (MWh)': st.column_config.NumberColumn(format='%.1f'),
                '전력 절감 (%)': st.column_config.NumberColumn(format='%.2f'),
                '전기요금 (만원)': st.column_config.NumberColumn(format='%.0f'),
                '요금 절감 (%)': st.column_config.NumberColumn(
                    format='%.2f', help='Rule-based PID 대비 전기요금 절감률'
                ),
                '냉각 전력량 (MWh)': st.column_config.NumberColumn(format='%.1f'),
                '팬 증가분 (MWh)': st.column_config.NumberColumn(format='%.1f'),
                '평균 PUE': st.column_config.NumberColumn(format='%.3f'),
                '평균 고장률': st.column_config.NumberColumn(format='%.2f'),
                '최고 온도 (℃)': st.column_config.NumberColumn(format='%.1f'),
                '설정 변경 총량 (℃)': st.column_config.NumberColumn(format='%.0f'),
            },
        )
        st.caption(
            '설정 온도를 올리면 냉각 전력이 크게 줄지만 서버 팬 전력이 늘어 일부를 깎아 먹습니다. '
            'MPC 와 반응형의 전력 절감은 거의 같습니다. 트윈의 열용량이 작아 미리 식혀 둘 여지가 적기 때문이며, '
            'MPC 의 차이는 설정 온도를 덜 자주 바꾼다는 점(설정 변경 총량)에서 나타납니다.'
        )


# ---------------------------------------------------------------- 위기 시나리오

with tab_crisis:
    crisis_names = [n for n in SCENARIOS if n != 'normal']
    picked = st.multiselect(
        '시뮬레이션할 위기 시나리오',
        crisis_names,
        default=[n for n in crisis_names if not is_limit_test(n)],
        format_func=scenario_label,
        help='단일 위기 3종과, 위기가 겹치는 복합 시나리오입니다 (config/settings.yaml 의 scenarios). '
        '[한계 시험]은 27℃ 유지 요구의 대상이 아니라 기본 선택에서 빠져 있습니다.',
    )
    c_text, c_button = st.columns([3, 1], vertical_alignment='center')
    c_text.write(
        f'고른 시나리오 {len(picked)}종을 **{CONTROLLER_LABELS[controller]}** 제어로 {hours}시간 '
        '시뮬레이션하고 정상 운영과 한 그래프에 겹쳐 비교합니다.'
    )
    if c_button.button(
        '위기 시나리오 시뮬레이션 실행', type='primary', width='stretch', disabled=not picked
    ):
        st.session_state['crisis'] = (START, hours, controller, tuple(picked))

    if st.session_state.get('crisis') == (START, hours, controller, tuple(picked)):
        # 시나리오 하나가 실패해도 나머지 결과는 남긴다
        runs: dict[str, dict] = {}
        failed: dict[str, str] = {}
        todo = ['normal'] + picked
        progress = st.progress(0.0, text='시뮬레이션 중…')
        for idx, name in enumerate(todo):
            progress.progress(
                idx / len(todo),
                text=f'{SCENARIOS[name]["label"]} 계산 중… ({idx + 1}/{len(todo)})',
            )
            try:
                runs[name] = simulate(START, hours, controller, name)
            except ApiError as e:
                failed[name] = str(e)
        progress.empty()

        crisis = {n: r for n, r in runs.items() if n != 'normal'}
        normal = runs['normal']['summary'] if 'normal' in runs else None
        stopped = [n for n, r in crisis.items() if r['summary'].get('hours_to_critical')]
        over = [
            n
            for n, r in crisis.items()
            if n not in stopped and (r['summary']['diverged'] or not r['summary']['temp_ok'])
        ]
        if failed:
            st.error(
                '🚨 계산하지 못한 시나리오: '
                + ' · '.join(SCENARIOS[n]['label'].split(' (')[0] for n in failed)
            )
        # 한계 시험은 27℃ 유지 요구의 대상이 아니므로 따로 알린다 (경보가 아니라 시험 결과)
        limit_hit = [n for n in stopped + over if is_limit_test(n)]
        stopped_required = [n for n in stopped if not is_limit_test(n)]
        over_required = [n for n in over if not is_limit_test(n)]
        required = [n for n in crisis if not is_limit_test(n)]
        if stopped_required:
            st.error(
                f'🚨 위험 온도 {CRITICAL_C:.0f}℃ 도달: '
                + ' · '.join(
                    f'{SCENARIOS[n]["label"].split(" (")[0]} '
                    f'({crisis[n]["summary"]["hours_to_critical"]}시간 만에)'
                    for n in stopped_required
                )
            )
        if over_required:
            st.error(
                f'⚠️ 서버실 온도 {ROOM_MAX_C:.0f}℃ 초과: '
                + ' · '.join(SCENARIOS[n]['label'].split(' (')[0] for n in over_required)
            )
        if required and not (failed or stopped_required or over_required):
            st.success(
                f'✅ 위기 시나리오 {len(required)}종 모두 서버실 온도 {ROOM_MAX_C:.0f}℃ 이하 유지'
            )
        if limit_hit:
            st.warning(
                '⚠️ 한계 시험 (27℃ 유지 요구의 대상 아님): '
                + ' · '.join(
                    SCENARIOS[n]['label'].split(' (')[0]
                    + (
                        f' — {crisis[n]["summary"]["hours_to_critical"]}시간 만에 '
                        f'위험 온도 {CRITICAL_C:.0f}℃ 도달'
                        if n in stopped
                        else f' — 최고 {crisis[n]["summary"]["max_room_temp_c"]:.1f}℃'
                    )
                    for n in limit_hit
                )
                + '. 냉각 용량이 발열보다 작아 어떤 제어로도 27℃ 를 지킬 수 없는 조건에서 버티는 시간을 봅니다.'
            )

        cards = list(crisis.items()) + [(n, None) for n in failed]
        for row_start in range(0, len(cards), 4):
            row = cards[row_start : row_start + 4]
            for col, (name, r) in zip(st.columns(4), row):
                with col.container(border=True, height='stretch'):
                    st.markdown(f'**{scenario_label(name)}**')
                    if r is None:
                        st.error('계산 못 함')
                        st.caption(failed[name])
                        continue
                    s = r['summary']
                    to_critical = s.get('hours_to_critical')
                    if to_critical:
                        st.metric(
                            '위험 온도 도달까지',
                            f'{to_critical}시간',
                            delta=f'{CRITICAL_C:.0f}℃ 도달 (최고 {s["max_room_temp_c"]:.1f}℃)',
                            delta_arrow='off',
                            delta_color='inverse',
                        )
                        over_hours = int(
                            (pd.Series(r['timeseries']['room_temp_c']) > ROOM_MAX_C).sum()
                        )
                        a, b = st.columns(2)
                        small_stat(
                            a,
                            f'{ROOM_MAX_C:.0f}℃ 이하로 버틴 시간',
                            f'{s["hours"] - over_hours}시간',
                        )
                        small_stat(b, '용량 한계 운전', f'{s["capacity_limited_hours"]}시간')
                    else:
                        st.metric(
                            '최고 서버실 온도',
                            f'{s["max_room_temp_c"]:.1f}℃',
                            delta=f'{ROOM_MAX_C:.0f}℃ ' + ('이하' if s['temp_ok'] else '초과'),
                            delta_arrow='off',
                            delta_color='normal' if s['temp_ok'] else 'inverse',
                        )
                        a, b, c = st.columns(3)
                        small_stat(a, '평균 PUE', f'{s["average_pue"]:.3f}')
                        small_stat(
                            b,
                            '전력량 증가',
                            (
                                '-'
                                if normal is None or normal['hours'] != s['hours']
                                else '{:+.1f}%'.format(
                                    (s['facility_energy_kwh'] / normal['facility_energy_kwh'] - 1)
                                    * 100
                                )
                            ),
                        )
                        small_stat(c, '온도 유지율', f'{s["temp_maintenance_pct"]:.0f}%')
                        if s['violation_hours']:
                            st.warning(
                                f'비상 여유 시간 부족 {s["violation_hours"]}시간 '
                                f'(최소 {s["min_ride_through_min"]:.1f}분)'
                            )
                    mode_note = ''
                    if controller == 'mpc':
                        mode_note = ' · 모드: ' + ', '.join(
                            f'{MPC_MODE_SHORT.get(m, m)} {h}시간'
                            for m, h in s['mode_hours'].items()
                        )
                    st.caption(f'대응: {r["strategy"]}{mode_note}')

        if runs:
            frames = {name: frame(r) for name, r in runs.items()}
            panels = (
                ('서버실 온도', 'room_temp_c', '%{y:.1f}℃', '℃', 'crisis_temp'),
                ('전체 전력', 'facility_power_kw', '%{y:,.0f} kW', 'kW', 'crisis_power'),
            )
            for col, (title, column, hover, unit, key) in zip(st.columns(2), panels):
                with col:
                    st.subheader(title)
                    fig = go.Figure()
                    for name, d in frames.items():
                        short = SCENARIOS[name]['label'].split(' (')[0]
                        fig.add_scatter(
                            x=d.timestamp,
                            y=d[column],
                            name=short,
                            mode='lines',
                            line=dict(
                                color=SCENARIO_COLORS.get(name, C['muted']),
                                width=2,
                                dash='dot' if name == 'normal' else 'solid',
                            ),
                            hovertemplate=hover,
                        )
                        if name in stopped:  # 위험 온도에 닿아 멈춘 지점
                            fig.add_scatter(
                                x=[d.timestamp.iloc[-1]],
                                y=[d[column].iloc[-1]],
                                mode='markers',
                                marker=dict(
                                    symbol='x',
                                    size=10,
                                    color=SCENARIO_COLORS.get(name, C['muted']),
                                ),
                                name=f'{short} 중단',
                                showlegend=False,
                                hovertemplate=f'{short}: 위험 온도 도달로 중단<extra></extra>',
                            )
                    if column == 'room_temp_c':
                        room_limits(fig, *room_range(*[d.room_temp_c for d in frames.values()]))
                        if stopped:
                            fig.add_hline(
                                y=CRITICAL_C,
                                line_dash='dash',
                                line_color=C['critical'],
                                line_width=1,
                                annotation_text=f'위험 {CRITICAL_C:.0f}℃',
                                annotation_position='top left',
                            )
                    show(styled(fig, 340, unit), key)
            st.caption(
                '그래프에 마우스를 올리면 그 시각의 시나리오별 값이 함께 보입니다 (휴대폰에서는 그래프를 누르세요). '
                '× 표시는 서버실 온도가 위험 온도에 닿아 시뮬레이션을 멈춘 지점입니다.'
            )
    else:
        st.info(
            '버튼을 누르면 단일 위기(서버 급증 +30% · 칠러 1대 고장 · 폭염 38℃ 이상)와 '
            '복합 위기(폭염 + 칠러 고장 · 급증 + 폭염 · 세 위기 동시 · 칠러 2대 고장)를 차례로 계산합니다. '
            'MPC 는 시나리오 하나에 수십 초가 걸립니다.'
        )


# ---------------------------------------------------------------- ESG · 설정 온도

with tab_esg:
    if sim:
        try:
            e, e_base = esg_of(sim), esg_of(base)
        except ApiError as err:
            st.error(str(err))
            e = None
        if e:
            st.subheader('ESG 지표')
            st.caption(
                f'관제 탭과 같은 구간 · {CONTROLLER_LABELS[controller]} · {sim["scenario_label"]}. '
                '변화량은 20℃ 고정 대비입니다.'
            )
            same_span = sim['summary']['hours'] == base['summary']['hours']
            compare = controller != 'fixed_20' and same_span
            if not same_span:
                st.caption('두 운전이 멈춘 시각이 달라 20℃ 고정 대비 변화량은 표시하지 않습니다.')
            cols = st.columns(4)
            cols[0].metric(
                '탄소 배출 (tCO2)',
                f'{e["carbon_tco2"]:.1f}',
                delta_color='inverse',
                border=True,
                delta=(f'{e["carbon_tco2"] - e_base["carbon_tco2"]:+.1f}' if compare else None),
                help=f'전체 전력량 × {e["carbon_factor_tco2_per_mwh"]} tCO2/MWh',
            )
            cols[1].metric(
                'CUE (tCO2/MWh)',
                f'{e["cue_tco2_per_mwh"]:.3f}',
                delta_color='inverse',
                border=True,
                delta=(
                    f'{e["cue_tco2_per_mwh"] - e_base["cue_tco2_per_mwh"]:+.3f}'
                    if compare
                    else None
                ),
                help='탄소 배출 / IT 전력량',
            )
            cols[2].metric(
                '물 사용량 (m³)',
                f'{e["water_usage_l"] / 1000:,.1f}',
                delta_color='inverse',
                border=True,
                delta=(
                    f'{(e["water_usage_l"] - e_base["water_usage_l"]) / 1000:+,.1f}'
                    if compare
                    else None
                ),
            )
            cols[3].metric(
                'WUE (L/kWh)',
                f'{e["wue_l_per_kwh"]:.2f}',
                delta_color='inverse',
                border=True,
                delta=(
                    f'{e["wue_l_per_kwh"] - e_base["wue_l_per_kwh"]:+.2f}' if compare else None
                ),
                help='물 사용량 / IT 전력량',
            )

            modes = list(MODE_LABELS)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**냉각 방식별 물 사용량**')
                fig = go.Figure(
                    go.Bar(
                        x=[MODE_LABELS[m] for m in modes],
                        y=[e['water_by_mode_l'][m] / 1000 for m in modes],
                        marker=dict(color=C['blue'], cornerradius=4),
                        width=0.5,
                        text=[f'{e["water_by_mode_l"][m] / 1000:,.1f}' for m in modes],
                        textposition='outside',
                        hovertemplate='%{x}: %{y:,.1f} m³<extra></extra>',
                    )
                )
                show(styled(fig, 260, 'm³', legend=False, time_axis=False), 'water')
            with c2:
                st.markdown('**냉각 방식별 운전 시간**')
                fig = go.Figure(
                    go.Bar(
                        x=[MODE_LABELS[m] for m in modes],
                        y=[e['hours_by_mode'][m] for m in modes],
                        marker=dict(color=C['blue'], cornerradius=4),
                        width=0.5,
                        text=[f'{e["hours_by_mode"][m]:.0f}' for m in modes],
                        textposition='outside',
                        hovertemplate='%{x}: %{y:.0f}시간<extra></extra>',
                    )
                )
                show(styled(fig, 260, '시간', legend=False, time_axis=False), 'mode_hours')
            st.caption(
                '물 원단위(칠러 1.8 · 혼합 1.0 · 외기 냉각 0.3 L/kWh)는 데이터에 없는 가정값입니다.'
            )

    st.divider()
    st.subheader('비용 · 에너지 · 탄소 Trade-off (제어기별)')
    try:
        trade = precomputed('base')
    except ApiError as err:
        st.error(str(err))
        trade = None
    if trade:
        carbon_factor = CONFIG['esg']['carbon_factor_tco2_per_mwh']
        rows = pd.DataFrame(
            [
                {
                    'name': c,
                    '제어기': CONTROLLER_LABELS[c],
                    '전력량 (MWh)': m['facility_energy_kwh'] / 1000,
                    '탄소 (tCO2)': m['facility_energy_kwh'] / 1000 * carbon_factor,
                    '전기요금 (만원)': m['energy_cost_krw'] / 1e4,
                    '평균 단가 (원/kWh)': m['energy_cost_krw'] / m['facility_energy_kwh'],
                    '평균 고장률': m['average_xfactor'],
                }
                for c in CONTROLLER_LABELS
                if c in trade['controllers']
                for m in [trade['controllers'][c]]
            ]
        )
        t0 = pd.Timestamp(trade['start'])
        st.caption(
            f'{t0:%Y-%m-%d} 부터 {trade["hours"] // 24}일, 제어기 6종의 사전 계산 결과입니다. '
            f'탄소는 전력량 × {carbon_factor} tCO2/MWh 라 전력량과 같이 움직이고(위쪽 축), '
            '요금은 언제 쓰느냐(시간대별 단가)에 따라 따로 움직입니다.'
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            for _, r in rows.iterrows():
                fig.add_scatter(
                    x=[r['전력량 (MWh)']],
                    y=[r['전기요금 (만원)']],
                    mode='markers+text',
                    name=r['제어기'],
                    text=[r['제어기']],
                    textposition='top center',
                    marker=dict(
                        size=10 + 40 * (r['평균 고장률'] - 1.0),
                        color=CONTROLLER_COLORS[r['name']],
                        line=dict(width=1, color='#ffffff'),
                    ),
                    customdata=[[r['탄소 (tCO2)'], r['평균 단가 (원/kWh)'], r['평균 고장률']]],
                    hovertemplate='전력량 %{x:,.1f} MWh · 요금 %{y:,.0f}만원<br>'
                    '탄소 %{customdata[0]:,.1f} tCO2 · 평균 단가 %{customdata[1]:.1f}원/kWh · '
                    '고장률 %{customdata[2]:.2f}<extra>%{fullData.name}</extra>',
                )
            lo, hi = rows['전력량 (MWh)'].min(), rows['전력량 (MWh)'].max()
            pad = (hi - lo) * 0.12 or 1.0
            fig.add_scatter(  # 위쪽 축(탄소)을 그리기 위한 보이지 않는 점
                x=[(lo - pad) * carbon_factor, (hi + pad) * carbon_factor],
                y=[rows['전기요금 (만원)'].min()] * 2,
                xaxis='x2',
                mode='markers',
                marker=dict(opacity=0),
                hoverinfo='skip',
                showlegend=False,
            )
            fig.update_layout(
                xaxis=dict(range=[lo - pad, hi + pad]),
                xaxis2=dict(
                    overlaying='x',
                    side='top',
                    title='탄소 배출 (tCO2)',
                    range=[(lo - pad) * carbon_factor, (hi + pad) * carbon_factor],
                    showgrid=False,
                ),
            )
            show(
                styled(
                    fig, 380, '전기요금 (만원)', '전체 전력량 (MWh)', legend=False, time_axis=False
                ),
                'tradeoff',
            )
        with c2:
            st.dataframe(
                rows.drop(columns='name'),
                width='stretch',
                hide_index=True,
                column_config={
                    '전력량 (MWh)': st.column_config.NumberColumn(format='%.1f'),
                    '탄소 (tCO2)': st.column_config.NumberColumn(format='%.1f'),
                    '전기요금 (만원)': st.column_config.NumberColumn(format='%.0f'),
                    '평균 단가 (원/kWh)': st.column_config.NumberColumn(format='%.1f'),
                    '평균 고장률': st.column_config.NumberColumn(format='%.2f'),
                },
            )
            st.caption(
                '왼쪽 아래일수록 요금·에너지·탄소가 모두 적습니다. 점이 클수록 평균 고장률(서버실 온도)이 높습니다. '
                '축열 시간표처럼 전력량(탄소)은 거의 그대로인데 요금만 줄어드는 운전과, '
                '26℃ 고정처럼 전력량은 줄지만 고장률이 오르는 운전이 대비됩니다.'
            )

    st.divider()
    st.subheader('설정 온도를 올리면 얻는 것과 잃는 것')
    default_it = float(frame(sim).it_base_kw.mean()) if sim else 2900.0
    default_out = float(frame(sim).outdoor_temp_c.mean()) if sim else 22.0
    s1, s2 = st.columns(2)
    it_kw = s1.slider(
        'IT 부하 (kW)', 2000, 4500, int(min(max(round(default_it, -1), 2000), 4500)), 10
    )
    out_c = s2.slider(
        '외기온도 (℃)', -5.0, 40.0, min(max(round(default_out * 2) / 2, -5.0), 40.0), 0.5
    )
    try:
        sweep = setpoint_sweep(float(it_kw), float(out_c))
    except ApiError as err:
        st.error(str(err))
        sweep = None

    if sweep is not None:
        best = sweep.loc[sweep.facility_power_kw.idxmin()]
        st.caption(
            f'이 조건에서 전체 전력이 가장 적은 설정 온도는 **{best.setpoint_c:.1f}℃** '
            f'({best.facility_power_kw:,.0f} kW, PUE {best.pue:.3f}) 입니다. '
            'PUE 는 계속 좋아져도 팬 전력 때문에 전체 전력은 다시 늘 수 있습니다.'
        )
        panels = (
            ('전체 전력 (kW)', 'facility_power_kw', '%{y:,.0f} kW'),
            ('상대 고장률 (20℃ = 1.0)', 'xfactor', '%{y:.2f}'),
            ('비상 여유 시간 (분)', 'ride_through_min', '%{y:.1f}분'),
        )
        for col, (title, column, hover) in zip(st.columns(3), panels):
            with col:
                st.markdown(f'**{title}**')
                fig = go.Figure(
                    go.Scatter(
                        x=sweep.setpoint_c,
                        y=sweep[column],
                        name=title,
                        line=dict(color=C['blue'], width=2),
                        hovertemplate=hover,
                    )
                )
                if column == 'facility_power_kw':
                    fig.add_scatter(
                        x=[best.setpoint_c],
                        y=[best.facility_power_kw],
                        mode='markers+text',
                        marker=dict(size=10, color=C['blue']),
                        text=['최소'],
                        textposition='top center',
                        hoverinfo='skip',
                        showlegend=False,
                    )
                if column == 'ride_through_min':
                    fig.add_hline(
                        y=15,
                        line_dash='dash',
                        line_color=C['critical'],
                        line_width=1,
                        annotation_text='기준 15분',
                        annotation_position='bottom left',
                    )
                show(
                    styled(fig, 260, '', '설정 온도 (℃)', legend=False, time_axis=False),
                    f'sweep_{column}',
                )
        table_view(
            sweep[
                [
                    'setpoint_c',
                    'facility_power_kw',
                    'cooling_power_kw',
                    'server_fan_extra_kw',
                    'pue',
                    'cooling_mode',
                    'xfactor',
                    'ride_through_min',
                ]
            ]
        )

# ---------------------------------------------------------------- 지역 비교 (사전 계산)

with tab_sites:
    try:
        site_data = site_results()
    except ApiError as e:
        st.error(str(e))
        site_data = None

    if site_data:
        summary, comparison = site_data['summary'], site_data['comparison']
        names = list(summary)
        first = summary[names[0]]
        st.caption(
            f'같은 가상 데이터센터(같은 IT 부하)를 지역별 외기에 놓았을 때의 비교입니다. '
            f'{first["monthly"][0]["month"]} ~ {first["monthly"][-1]["month"]} '
            f'({first["days"]:.0f}일), '
            '외기는 기상청 ASOS 실측입니다. 지역마다 다른 것은 외기뿐이고, 습도·염해·전력 계통 차이는 '
            '모델에 들어 있지 않습니다.'
        )
        full = comparison['full_period'] if comparison else {}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        '지역': s['label'],
                        '외기 평균 (℃)': s['outdoor_mean_c'],
                        '외기 최고 (℃)': s['outdoor_max_c'],
                        '습도 평균 (%)': s['humidity_mean_pct'],
                        '외기 냉각 (%)': s['free_cooling_pct'],
                        '혼합 (%)': s['hybrid_pct'],
                        '칠러 (%)': s['chiller_pct'],
                        '평균 PUE (20℃ 고정)': s['average_pue'],
                        '냉각 전력량 (MWh)': s['cooling_energy_mwh'],
                        '전기요금 (억원, 20℃ 고정)': (
                            full[n]['controllers']['fixed_20']['energy_cost_krw'] / 1e8
                            if n in full
                            else None
                        ),
                    }
                    for n, s in summary.items()
                ]
            ),
            width='stretch',
            hide_index=True,
            column_config={
                '평균 PUE (20℃ 고정)': st.column_config.NumberColumn(format='%.3f'),
                '전기요금 (억원, 20℃ 고정)': st.column_config.NumberColumn(format='%.2f'),
            },
        )

        panels = (
            ('월별 평균 PUE (20℃ 고정)', 'average_pue', '%{y:.3f}', 'PUE'),
            ('월별 외기 냉각 시간 비율', 'free_cooling_pct', '%{y:.1f}%', '%'),
            ('월별 평균 외기온도', 'outdoor_mean_c', '%{y:.1f}℃', '℃'),
        )
        for col, (title, column, hover, unit) in zip(st.columns(3), panels):
            with col:
                st.markdown(f'**{title}**')
                fig = go.Figure()
                for n in names:
                    monthly = summary[n]['monthly']
                    fig.add_scatter(
                        x=[m['month'][5:] + '월' for m in monthly],
                        y=[m[column] for m in monthly],
                        name=summary[n]['label'],
                        mode='lines+markers',
                        line=dict(color=SITE_COLORS.get(n, C['muted']), width=2),
                        marker=dict(size=7),
                        hovertemplate=hover,
                    )
                show(styled(fig, 280, unit, time_axis=False), f'site_{column}')

        windows = comparison['windows'] if comparison else []
        if windows:
            st.subheader('계절별 MPC 요금 절감률 (Rule-based PID 대비)')
            days = windows[0]['hours'] // 24
            fig = go.Figure()
            for n in names:
                rows = [w for w in windows if w['site'] == n]
                saving = [w['controllers']['mpc']['cost_saving_vs_rule_pid_pct'] for w in rows]
                fig.add_bar(
                    x=[f'{w["start"][5:10]} 부터 {days}일' for w in rows],
                    y=saving,
                    name=summary[n]['label'],
                    marker=dict(color=SITE_COLORS.get(n, C['muted']), cornerradius=4),
                    text=[f'{v:.1f}%' for v in saving],
                    textposition='outside',
                    customdata=[
                        (w['outdoor_mean_c'], w['controllers']['mpc']['max_room_temp_c'])
                        for w in rows
                    ],
                    hovertemplate='절감 %{y:.2f}% · 외기 평균 %{customdata[0]:.1f}℃ · '
                    '최고 서버실 %{customdata[1]:.1f}℃',
                )
            fig.update_layout(barmode='group')
            show(styled(fig, 320, '%', time_axis=False), 'site_mpc_saving')
            worst = max(w['controllers']['mpc']['max_room_temp_c'] for w in windows)
            violations = sum(w['controllers']['mpc']['violation_hours'] for w in windows)
            st.caption(
                f'지역·계절 {len(windows)}개 구간 모두에서 MPC 최고 서버실 온도 {worst:.1f}℃, '
                f'위반 {violations}시간. 예측 모델은 IT 부하만 쓰므로 세 지역에 같은 모델을 썼습니다. '
                '`scripts/compare_sites.py` 로 미리 계산한 결과입니다.'
            )
        else:
            st.info('계절별 MPC 비교는 `python scripts/compare_sites.py` 를 실행하면 표시됩니다.')

st.divider()
st.caption(
    '가정 기반 시뮬레이션입니다. 트윈의 열용량·팬 전력·비상 기준·물 원단위는 데이터에 없는 가정값이고, '
    '외기 예보는 실제값을 그대로 썼습니다(예보 오차 없음). 최적화 목표는 PUE 가 아니라 전체 전력입니다.'
)
