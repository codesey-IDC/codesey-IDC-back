"""통합 관제 대시보드 (Streamlit :8501).

세 서비스를 REST 로만 호출한다. 파일(parquet/CSV)을 직접 읽지 않는다.
  twin-service     :8001  계측값(telemetry), 정상 상태 계산(simulate), ESG
  forecast-service :8002  IT 부하 예측
  mpc-service      :8003  설정 온도 계획(plan), 폐루프 시뮬레이션(simulate), 위기 시나리오, 사전 계산 결과

실행 (저장소 루트에서, 세 서비스를 먼저 띄운 뒤):
  streamlit run dashboard/app.py
주소는 환경변수로 바꿀 수 있다: TWIN_URL, FORECAST_URL, MPC_URL
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# localhost 대신 127.0.0.1: Windows 에서 localhost 는 연결마다 약 2초 지연된다
TWIN_URL = os.environ.get('TWIN_URL', 'http://127.0.0.1:8001')
FORECAST_URL = os.environ.get('FORECAST_URL', 'http://127.0.0.1:8002')
MPC_URL = os.environ.get('MPC_URL', 'http://127.0.0.1:8003')
SERVICES = {'디지털 트윈': TWIN_URL, '예측': FORECAST_URL, 'MPC': MPC_URL}

HISTORY_HOURS = 192
ROOM_MIN_C, ROOM_MAX_C = 18.0, 27.0
DEFAULT_START = date(2026, 6, 4)

CONTROLLER_LABELS = {'mpc': 'MPC', 'reactive': '반응형', 'fixed_26': '26℃ 고정', 'fixed_20': '20℃ 고정'}
MODE_LABELS = {'free_cooling': '외기 냉각', 'hybrid': '혼합', 'chiller': '칠러'}
MPC_MODE_LABELS = {'mpc': '정상 (예측 사용)', 'mpc_naive': '예측 불신 (최근 최대 부하 가정)',
                   'fallback': 'Fallback (20℃ 고정)'}
MPC_MODE_SHORT = {'mpc': '정상', 'mpc_naive': '예측 불신', 'fallback': 'Fallback'}
PLAY_INTERVAL_S = 1.0                   # 자동 재생: 1초에 1시간씩 진행

st.set_page_config(page_title='IDC 통합 관제', page_icon='🌡️', layout='wide')


# ---------------------------------------------------------------- 색 (역할별)
# 화면은 밝은 테마로 고정한다 (.streamlit/config.toml). 격자·보조선은 어느 배경에서도 읽히는 중간 회색.

C = {
    'blue': '#2a78d6',
    'orange': '#eb6834',
    'aqua': '#1baf7a',
    'muted': '#898781',
    'ink': '#52514e',
    'grid': 'rgba(137,135,129,0.25)',
    'critical': '#d03b3b',
    'band': 'rgba(42,120,214,0.18)',
    'ok_band': 'rgba(12,163,12,0.07)',
}
MODE_COLORS = {'free_cooling': C['blue'], 'chiller': C['orange'], 'hybrid': C['aqua']}
# 색은 대상에 고정한다 (순위나 표시 개수에 따라 바뀌지 않게)
CONTROLLER_COLORS = {'mpc': C['blue'], 'reactive': C['orange'], 'fixed_26': C['aqua'], 'fixed_20': C['muted']}
SCENARIO_COLORS = {'normal': C['muted'], 'server_surge': C['blue'], 'chiller_failure': C['orange'],
                   'heatwave': C['aqua']}


# ---------------------------------------------------------------- REST 호출

class ApiError(RuntimeError):
    pass


@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(timeout=300.0)


def _call(method: str, url: str, **kwargs):
    try:
        r = _client().request(method, url, **kwargs)
    except httpx.HTTPError as e:
        raise ApiError(f'{url} 에 연결할 수 없습니다 ({type(e).__name__}). 서비스가 떠 있는지 확인하세요.') from e
    if r.status_code != 200:
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
    return _call('POST', f'{MPC_URL}/api/v1/simulate',
                 json={'start': start, 'hours': hours, 'controller': controller, 'scenario': scenario})


@st.cache_data(show_spinner=False)
def scenarios() -> list[dict]:
    return _call('GET', f'{MPC_URL}/api/v1/scenarios')


@st.cache_data(show_spinner=False)
def precomputed(scenario: str) -> dict:
    return _call('GET', f'{MPC_URL}/api/v1/results', params={'scenario': scenario})


@st.cache_data(show_spinner=False)
def forecast(start: str, horizon: int) -> dict:
    """start 직전까지의 계측 이력으로 앞으로 horizon 시간을 예측."""
    first = (pd.Timestamp(start) - pd.Timedelta(hours=HISTORY_HOURS)).isoformat()
    hist = telemetry(first, HISTORY_HOURS)
    history = [{'timestamp': t, 'it_power_kw': v} for t, v in zip(hist['timestamps'], hist['it_power_kw'])]
    return _call('POST', f'{FORECAST_URL}/api/v1/forecast', json={'horizon_hours': horizon, 'history': history})


@st.cache_data(show_spinner=False)
def mpc_plan(start: str, room_temp_c: float) -> dict:
    first = (pd.Timestamp(start) - pd.Timedelta(hours=HISTORY_HOURS)).isoformat()
    hist = telemetry(first, HISTORY_HOURS)
    outdoor = telemetry(start, 24)['outdoor_temp_c']
    outdoor += [outdoor[-1]] * (24 - len(outdoor))
    history = [{'timestamp': t, 'it_power_kw': v} for t, v in zip(hist['timestamps'], hist['it_power_kw'])]
    return _call('POST', f'{MPC_URL}/api/v1/plan',
                 json={'room_temp_c': room_temp_c, 'history': history, 'outdoor_forecast_c': outdoor})


@st.cache_data(show_spinner=False)
def esg(it_power_kw: tuple, cooling_power_kw: tuple, cooling_mode: tuple, heat_removed_kw: tuple) -> dict:
    return _call('POST', f'{TWIN_URL}/api/v1/esg', json={
        'it_power_kw': list(it_power_kw), 'cooling_power_kw': list(cooling_power_kw),
        'cooling_mode': list(cooling_mode), 'heat_removed_kw': list(heat_removed_kw)})


@st.cache_data(show_spinner=False)
def setpoint_sweep(it_power_kw: float, outdoor_temp_c: float) -> pd.DataFrame:
    """설정 온도 16~27℃ 를 0.5℃ 간격으로 바꿔가며 정상 상태 계산."""
    setpoints = [16 + 0.5 * i for i in range(23)]
    r = _call('POST', f'{TWIN_URL}/api/v1/simulate', json={'points': [
        {'it_power_kw': it_power_kw, 'outdoor_temp_c': outdoor_temp_c, 'supply_temp_c': s} for s in setpoints]})
    return pd.DataFrame(r['results']).assign(setpoint_c=setpoints)


def esg_of(sim: dict) -> dict:
    ts = sim['timeseries']
    return esg(tuple(ts['it_power_kw']), tuple(ts['cooling_power_kw']), tuple(ts['cooling_mode']),
               tuple(ts['heat_removed_kw']))


def frame(sim: dict) -> pd.DataFrame:
    df = pd.DataFrame(sim['timeseries'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# ---------------------------------------------------------------- 그래프 공통

def styled(fig: go.Figure, height: int = 300, y_title: str = '', x_title: str = '', legend: bool = True,
           time_axis: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30 if legend else 10, b=10),
        showlegend=legend, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0, traceorder='normal'),
        yaxis_title=y_title, xaxis_title=x_title)
    fig.update_xaxes(showgrid=False)
    if time_axis:
        fig.update_xaxes(tickformat='%H:%M<br>%m-%d', hoverformat='%m-%d %H:%M')
    fig.update_yaxes(gridcolor=C['grid'], zeroline=False)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width='stretch', key=key, config={'displayModeBar': False})


def room_limits(fig: go.Figure, low: float, high: float) -> None:
    """허용 구간(18~27℃)을 옅은 띠로 칠하고 상한선을 긋는다. 세로축은 [low, high] 로 좁힌다."""
    fig.add_hrect(y0=ROOM_MIN_C, y1=ROOM_MAX_C, fillcolor=C['ok_band'], line_width=0, layer='below')
    fig.add_hline(y=ROOM_MAX_C, line_dash='dash', line_color=C['critical'], line_width=1,
                  annotation_text=f'상한 {ROOM_MAX_C:.0f}℃', annotation_position='top left')
    fig.add_hline(y=ROOM_MIN_C, line_dash='dot', line_color=C['muted'], line_width=1,
                  annotation_text=f'하한 {ROOM_MIN_C:.0f}℃', annotation_position='bottom left')
    fig.update_yaxes(range=[low, high])


def room_range(*series) -> tuple[float, float]:
    """서버실 온도 그래프의 세로축 범위: 허용 구간 전체가 보이되 빈 공간은 줄인다."""
    low = min(min(s.min() for s in series), ROOM_MIN_C) - 1.0
    high = max(max(s.max() for s in series), ROOM_MAX_C) + 1.2
    return low, high


def cursor_line(fig: go.Figure, x) -> None:
    """선택한 시각을 세로 점선으로 표시."""
    fig.add_shape(type='line', x0=x, x1=x, xref='x', y0=0, y1=1, yref='paper',
                  line=dict(color=C['ink'], width=1, dash='dot'))


def mode_bands(fig: go.Figure, df: pd.DataFrame) -> None:
    """시간축 배경에 냉각 모드를 색 띠로 깐다 (같은 모드가 이어지는 구간마다 한 칸)."""
    half = pd.Timedelta(minutes=30)
    run_start = 0
    for k in range(1, len(df) + 1):
        if k == len(df) or df.cooling_mode[k] != df.cooling_mode[run_start]:
            fig.add_vrect(x0=df.timestamp[run_start] - half, x1=df.timestamp[k - 1] + half,
                          fillcolor=MODE_COLORS[df.cooling_mode[run_start]], opacity=0.30, line_width=0,
                          layer='below')
            run_start = k
    for mode, label in MODE_LABELS.items():       # 범례용
        fig.add_scatter(x=[df.timestamp[0]], y=[None], mode='markers', name=label, hoverinfo='skip',
                        marker=dict(size=11, symbol='square', color=MODE_COLORS[mode], opacity=0.55))
    fig.update_xaxes(type='date')


def pue_gauge(value: float, reference: float | None = None) -> go.Figure:
    """현재 PUE. reference(20℃ 고정 운영의 PUE)를 눈금 선으로 함께 표시한다."""
    gauge = {'axis': {'range': [1.0, 1.6], 'tickvals': [1.0, 1.2, 1.4, 1.6]},
             'bar': {'color': C['blue'], 'thickness': 0.6}, 'borderwidth': 0, 'bgcolor': C['grid']}
    if reference is not None:
        gauge['threshold'] = {'line': {'color': C['ink'], 'width': 3}, 'thickness': 0.9,
                              'value': min(max(reference, 1.0), 1.6)}
    fig = go.Figure(go.Indicator(mode='gauge+number', value=value, gauge=gauge,
                                 number={'valueformat': '.3f', 'font': {'size': 34}}))
    fig.update_layout(height=150, margin=dict(l=30, r=30, t=18, b=0))
    return fig


def table_view(df: pd.DataFrame, label: str = '표로 보기') -> None:
    with st.expander(label):
        st.dataframe(df, width='stretch', hide_index=True)


def small_stat(col, label: str, value: str) -> None:
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
        else:
            st.markdown(f'✅ **{name}** 정상')

if any(h is None for h in status.values()):
    st.error('연결되지 않은 서비스가 있습니다. 저장소 루트에서 세 서비스를 먼저 띄워 주세요.')
    st.code('uvicorn services.twin.main:app --port 8001\n'
            'uvicorn services.forecast.main:app --port 8002\n'
            'uvicorn services.mpc.main:app --port 8003', language='bash')
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

with st.sidebar:
    st.header('운영 조건')
    start_day = st.date_input('시작 날짜', value=min(max(DEFAULT_START, first_day), last_day),
                              min_value=first_day, max_value=last_day,
                              help=f'계측 데이터 {data_start:%Y-%m-%d} ~ {data_end:%Y-%m-%d}. '
                                   f'예측에 이전 {HISTORY_HOURS}시간 이력이 필요합니다.')
    max_hours = int((data_end - pd.Timestamp(start_day)) / pd.Timedelta(hours=1)) + 1
    hours = st.selectbox('구간', [h for h in (24, 48, 72) if h <= max_hours] or [24],
                         format_func=lambda h: f'{h}시간')
    controller = st.selectbox('제어기', ['mpc', 'reactive', 'fixed_20'], format_func=CONTROLLER_LABELS.get,
                              help='MPC 는 1시간당 약 0.3초가 걸립니다.')
    st.header('위기 시나리오 주입')
    scenario = st.radio('관제 화면에 적용할 시나리오', list(SCENARIOS),
                        format_func=lambda n: SCENARIOS[n]['label'], label_visibility='collapsed')
    st.caption('시작 시각부터 구간 끝까지 적용됩니다.')
    if st.button('다시 계산', width='stretch', help='저장된 결과를 지우고 서비스를 다시 호출합니다.'):
        st.cache_data.clear()
        st.rerun()

START = datetime.combine(start_day, time(0)).isoformat()
END = pd.Timestamp(START) + pd.Timedelta(hours=hours - 1)

# 지금 보고 있는 조건 (위기 주입 중이면 색으로 구분)
scenario_badge = (':green-badge[✅ 정상 운영]' if scenario == 'normal'
                  else f':orange-badge[⚠️ 위기 주입: {SCENARIOS[scenario]["label"]}]')
st.markdown(f':gray-badge[{pd.Timestamp(START):%Y-%m-%d %H:%M} ~ {END:%m-%d %H:%M} · {hours}시간] '
            f':blue-badge[제어기: {CONTROLLER_LABELS[controller]}] {scenario_badge}')

tab_monitor, tab_forecast, tab_plan, tab_compare, tab_crisis, tab_esg = st.tabs(
    ['관제', 'IT 부하 예측', 'MPC 계획', '제어기 비교', '위기 시나리오', 'ESG · 설정 온도'])


# ---------------------------------------------------------------- 관제

def monitor_live(sim: dict, base: dict, playing: bool) -> None:
    """시각 슬라이더에 따라 바뀌는 부분. 자동 재생 중에는 이 부분만 주기적으로 다시 그린다."""
    df, bdf = frame(sim), frame(base)
    n = len(df)
    key = f'cursor|{START}|{hours}|{controller}|{scenario}'
    if key not in st.session_state:
        st.session_state[key] = n - 1
    elif playing:
        st.session_state[key] = (st.session_state[key] + 1) % n

    c_slider, c_play = st.columns([5, 1], vertical_alignment='bottom')
    i = c_slider.select_slider('시각', options=list(range(n)), key=key,
                               format_func=lambda k: f'{df.timestamp[k]:%m-%d %H:%M}')
    if c_play.toggle('자동 재생', value=playing, key='playing_toggle',
                     help='1초에 1시간씩 진행합니다.') != playing:
        st.session_state['playing'] = not playing
        st.rerun()
    now = df.iloc[i]

    compare = controller != 'fixed_20'
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
        notes = [f'전체 전력 {now.facility_power_kw:,.0f} kW'
                 + (f' (20℃ 고정 대비 {now.facility_power_kw - bdf.facility_power_kw[i]:+,.0f} kW)' if compare else '')]
        if controller == 'mpc':
            notes.append(f'MPC 동작 모드: {MPC_MODE_LABELS.get(now["mode"], now["mode"])}')
        st.caption(' · '.join(notes))

    left, right = st.columns(2)
    with left:
        st.subheader('전력 소비 추이')
        other = df.facility_power_kw - df.it_power_kw - df.cooling_power_kw
        fig = go.Figure()
        for name, y, color in (('IT', df.it_power_kw, C['blue']), ('냉각', df.cooling_power_kw, C['orange']),
                               ('기타 손실', other, C['aqua'])):
            fig.add_scatter(x=df.timestamp, y=y, name=name, stackgroup='power', mode='lines',
                            line=dict(width=0.5, color=color), hovertemplate='%{y:,.0f} kW')
        cursor_line(fig, now.timestamp)
        show(styled(fig, 290, 'kW'), 'power')
    with right:
        st.subheader('서버실 온도')
        fig = go.Figure()
        fig.add_scatter(x=df.timestamp, y=df.room_temp_c, name='서버실 온도', line=dict(color=C['blue'], width=2),
                        hovertemplate='%{y:.1f}℃')
        fig.add_scatter(x=df.timestamp, y=df.setpoint_c, name='설정 온도', hovertemplate='%{y:.1f}℃',
                        line=dict(color=C['orange'], width=2, dash='dot', shape='hv'))
        room_limits(fig, *room_range(df.room_temp_c, df.setpoint_c))
        cursor_line(fig, now.timestamp)
        show(styled(fig, 290, '℃'), 'room')

    st.subheader('냉각 모드와 외기온도')
    fig = go.Figure()
    mode_bands(fig, df)
    fig.add_scatter(x=df.timestamp, y=df.outdoor_temp_c, name='외기온도', line=dict(color=C['ink'], width=2),
                    customdata=list(zip(df.cooling_mode.map(MODE_LABELS), df.free_cooling_fraction * 100)),
                    hovertemplate='%{y:.1f}℃ · %{customdata[0]} (외기 냉각 %{customdata[1]:.0f}%)')
    cursor_line(fig, now.timestamp)
    show(styled(fig, 230, '외기온도 (℃)'), 'modes')
    hours_by_mode = df.cooling_mode.value_counts()
    st.caption('배경색이 냉각 모드입니다. ' + ' · '.join(
        f'{MODE_LABELS[m]} {hours_by_mode.get(m, 0)}시간' for m in MODE_LABELS))


with tab_monitor:
    try:
        with st.spinner(f'{CONTROLLER_LABELS[controller]} 폐루프 시뮬레이션 {hours}시간 계산 중…'):
            sim = simulate(START, hours, controller, scenario)
            base = sim if controller == 'fixed_20' else simulate(START, hours, 'fixed_20', scenario)
    except ApiError as e:
        st.error(str(e))
        sim = base = None

    if sim:
        s, df = sim['summary'], frame(sim)
        if s['diverged'] or not s['temp_ok']:
            st.error(f'⚠️ 경고: 서버실 온도가 {ROOM_MAX_C:.0f}℃ 를 넘었습니다 (최고 {s["max_room_temp_c"]:.1f}℃).')
        elif s['violation_hours']:
            st.warning(f'⚠️ 주의: 온도는 {ROOM_MAX_C:.0f}℃ 이하지만 {s["violation_hours"]}시간 동안 비상 여유 시간이 '
                       f'기준에 못 미쳤습니다 (최소 {s["min_ride_through_min"]:.1f}분).')
        else:
            st.success(f'✅ 서버실 온도 정상 (전 구간 {ROOM_MIN_C:.0f}~{ROOM_MAX_C:.0f}℃, 위반 없음)')
        for w in sim['warnings']:
            st.warning(w)

        saving = (1 - s['facility_energy_kwh'] / base['summary']['facility_energy_kwh']) * 100
        k = st.columns(5)
        k[0].metric('평균 PUE', f'{s["average_pue"]:.3f}', border=True, height='stretch', chart_data=df.pue.tolist(),
                    chart_type='line',
                    help='전체 전력량 / IT 전력량. 팬 전력이 IT 로 잡히므로 전체 전력과 함께 봐야 합니다.')
        k[1].metric('전체 전력량', f'{s["facility_energy_kwh"] / 1000:,.1f} MWh', border=True, height='stretch',
                    chart_data=df.facility_power_kw.tolist(), chart_type='line',
                    delta=None if controller == 'fixed_20' else f'{-saving:+.1f}% (20℃ 고정 대비)',
                    delta_color='inverse', help='IT + 냉각 + 기타 손실. 최적화 목표입니다.')
        k[2].metric('최고 서버실 온도', f'{s["max_room_temp_c"]:.1f}℃', border=True, height='stretch',
                    chart_data=df.room_temp_c.tolist(), chart_type='line')
        k[3].metric('온도 유지율', f'{s["temp_maintenance_pct"]:.1f}%', border=True, height='stretch',
                    help=f'서버실 온도가 {ROOM_MIN_C:.0f}~{ROOM_MAX_C:.0f}℃ 인 시간 비율')
        k[4].metric('평균 설정 온도', f'{s["average_setpoint_c"]:.1f}℃', border=True, height='stretch',
                    chart_data=df.setpoint_c.tolist(), chart_type='line')

        playing = st.session_state.get('playing', False)
        st.fragment(monitor_live, run_every=PLAY_INTERVAL_S if playing else None)(sim, base, playing)

        if controller == 'mpc':
            st.caption('구간 전체 MPC 동작 모드: ' + ' · '.join(
                f'{MPC_MODE_LABELS.get(m, m)} {n}시간' for m, n in s['mode_hours'].items()))
        table_view(df.rename(columns={'timestamp': '시각'}))


# ---------------------------------------------------------------- IT 부하 예측

with tab_forecast:
    horizon = st.radio('예측 기간', [24, 168], format_func=lambda h: f'{h}시간', horizontal=True)
    try:
        fc = forecast(START, horizon)
        past = telemetry((pd.Timestamp(START) - pd.Timedelta(hours=72)).isoformat(), 72)
        try:
            actual = telemetry(START, horizon)
        except ApiError:
            actual = None
    except ApiError as e:
        st.error(str(e))
        fc = None

    if fc:
        t = pd.to_datetime(fc['timestamps'])
        pred = pd.Series(fc['predictions'], index=t)
        st.caption(f'예측 시점 {pd.Timestamp(fc["origin_time"]):%Y-%m-%d %H:%M} · 모델 {fc["model"]} · '
                   '정상 조건의 계측 이력 기준 (위기 시나리오 미적용)')
        if not fc['in_training_range']:
            st.warning('⚠️ 입력 또는 예측값이 학습 범위를 벗어났습니다. MPC 는 이 경우 예측을 쓰지 않습니다.')
        for w in fc['warnings']:
            st.caption(f'경고: {w}')

        cols = st.columns(4)
        cols[0].metric('예측 평균', f'{pred.mean():,.0f} kW', border=True)
        cols[1].metric('90% 구간 평균 폭', border=True,
                       value=f'{(pd.Series(fc["pi_upper"]) - pd.Series(fc["pi_lower"])).mean():,.0f} kW')
        act = None
        if actual:
            act = pd.Series(actual['it_power_kw'], index=pd.to_datetime(actual['timestamps']))
            both = pd.concat([act, pred, pd.Series(fc['pi_lower'], index=t), pd.Series(fc['pi_upper'], index=t)],
                             axis=1, keys=['y', 'q50', 'lo', 'hi']).dropna()
            if len(both):
                cols[2].metric('오차 (MAPE)', f'{((both.y - both.q50).abs() / both.y).mean() * 100:.2f}%',
                               border=True, help=f'실제값이 있는 {len(both)}시간 기준')
                cols[3].metric('90% 구간 적중률', border=True,
                               value=f'{((both.y >= both.lo) & (both.y <= both.hi)).mean() * 100:.0f}%')

        fig = go.Figure()
        fig.add_scatter(x=list(t) + list(t[::-1]), y=fc['pi_upper'] + fc['pi_lower'][::-1], fill='toself',
                        fillcolor=C['band'], line=dict(width=0), name='90% 구간', hoverinfo='skip')
        fig.add_scatter(x=pd.to_datetime(past['timestamps']), y=past['it_power_kw'], name='계측 (과거)',
                        line=dict(color=C['muted'], width=2), hovertemplate='%{y:,.0f} kW')
        if act is not None:
            fig.add_scatter(x=act.index, y=act.values, name='실제', line=dict(color=C['orange'], width=2),
                            hovertemplate='%{y:,.0f} kW')
        fig.add_scatter(x=t, y=fc['predictions'], name='예측 (중앙값)', line=dict(color=C['blue'], width=2),
                        hovertemplate='%{y:,.0f} kW')
        cursor_line(fig, pd.Timestamp(fc['origin_time']))
        show(styled(fig, 380, 'IT 부하 (kW)'), 'forecast')
        st.caption('세로 점선이 예측 시점입니다.')
        table_view(pd.DataFrame({'시각': t, '예측 (kW)': fc['predictions'], '하한 (kW)': fc['pi_lower'],
                                 '상한 (kW)': fc['pi_upper']}))


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
        st.caption(f'{pd.Timestamp(START):%Y-%m-%d %H:%M} 시점에 세운 24시간 계획 (정상 조건). '
                   '실제 운영에서는 첫 1시간만 적용하고 매시간 다시 계산합니다.')
        cols = st.columns(4)
        cols[0].metric('지금 적용할 설정 온도', f'{plan["setpoint_c"]:.1f}℃', border=True)
        cols[1].metric('동작 모드', MPC_MODE_SHORT.get(plan['mode'], plan['mode']), border=True,
                       help=MPC_MODE_LABELS.get(plan['mode'], ''))
        cols[2].metric('24시간 예상 전력량', f'{plan["expected_facility_energy_kwh"] / 1000:,.1f} MWh', border=True)
        cols[3].metric('예상 절감 (20℃ 고정 대비)', f'{plan["expected_saving_pct"]:.1f}%', border=True)
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
                fig.add_scatter(x=t, y=plan['predicted_room_temp_c'], name='예상 서버실 온도',
                                line=dict(color=C['blue'], width=2), hovertemplate='%{y:.1f}℃')
                fig.add_scatter(x=t, y=plan['plan_c'], name='설정 온도 계획', hovertemplate='%{y:.1f}℃',
                                line=dict(color=C['orange'], width=2, dash='dot', shape='hv'))
                room_limits(fig, *room_range(pd.Series(plan['predicted_room_temp_c']), pd.Series(plan['plan_c'])))
                show(styled(fig, 320, '℃'), 'plan_temp')
            with c2:
                st.subheader('예상 전력')
                fig = go.Figure()
                fig.add_scatter(x=t, y=plan['predicted_facility_kw'], name='전체 전력',
                                line=dict(color=C['blue'], width=2), hovertemplate='%{y:,.0f} kW')
                fig.add_scatter(x=t, y=plan['forecast_it_kw'], name='IT 부하 예측',
                                line=dict(color=C['orange'], width=2), hovertemplate='%{y:,.0f} kW')
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
        st.caption(f'{t0:%Y-%m-%d} 부터 {res["hours"] // 24}일 (예측 모델 Test 구간). '
                   '`simulate_mpc.py` 로 미리 계산해 둔 결과입니다.')
        order = [c for c in CONTROLLER_LABELS if c in res['controllers']]
        ref = res['controllers']['fixed_20']
        summary = pd.DataFrame([{
            '제어기': CONTROLLER_LABELS[c],
            '전체 전력량 (MWh)': m['facility_energy_kwh'] / 1000,
            '절감 (%)': m['saving_vs_fixed_20_pct'],
            '냉각 전력량 (MWh)': m['cooling_energy_kwh'] / 1000,
            '팬 증가분 (MWh)': m['fan_extra_energy_kwh'] / 1000,
            '평균 PUE': m['average_pue'],
            '평균 고장률': m['average_xfactor'],
            '최고 온도 (℃)': m['max_room_temp_c'],
            '위반 (시간)': m['violation_hours'],
            '설정 변경 총량 (℃)': m['setpoint_change_c_total'],
        } for c in order for m in [res['controllers'][c]]])

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader('절감 내역 (20℃ 고정 대비)')
            shown = [c for c in order if c != 'fixed_20']
            labels = [CONTROLLER_LABELS[c] for c in shown]
            cooling_saved = [(ref['cooling_energy_kwh'] - res['controllers'][c]['cooling_energy_kwh']) / 1000
                             for c in shown]
            fan_added = [-(res['controllers'][c]['fan_extra_energy_kwh'] - ref['fan_extra_energy_kwh']) / 1000
                         for c in shown]
            fig = go.Figure()
            fig.add_bar(x=labels, y=cooling_saved, name='냉각 전력 절감', width=0.5,
                        marker=dict(color=C['blue'], cornerradius=4),
                        text=[f'+{v:.0f}' for v in cooling_saved], textposition='inside',
                        insidetextanchor='middle', textfont=dict(color='#ffffff'),
                        hovertemplate='%{y:,.1f} MWh')
            fig.add_bar(x=labels, y=fan_added, name='팬 전력 증가', width=0.5,
                        marker=dict(color=C['orange'], cornerradius=4),
                        text=[f'{v:.0f}' for v in fan_added], textposition='outside',
                        hovertemplate='%{y:,.1f} MWh')
            for label, c, saved in zip(labels, shown, cooling_saved):
                m = res['controllers'][c]
                net = (ref['facility_energy_kwh'] - m['facility_energy_kwh']) / 1000
                fig.add_annotation(x=label, y=saved, yshift=12, showarrow=False,
                                   text=f'순 절감 {net:.0f} MWh ({m["saving_vs_fixed_20_pct"]:.1f}%)')
            fig.update_layout(barmode='relative')
            fig.update_yaxes(range=[min(fan_added) * 2.4, max(cooling_saved) * 1.18], zeroline=True,
                             zerolinecolor=C['muted'])
            show(styled(fig, 340, 'MWh', time_axis=False), 'saving')
        with c2:
            st.subheader('일평균 전체 전력')
            fig = go.Figure()
            for c in order:
                ts = pd.DataFrame(res['timeseries'][c])
                daily = ts.set_index(pd.to_datetime(ts['timestamp'])).facility_power_kw.resample('D').mean()
                fig.add_scatter(x=daily.index, y=daily.values, name=CONTROLLER_LABELS[c], mode='lines+markers',
                                line=dict(color=CONTROLLER_COLORS[c], width=2,
                                          dash='dot' if c == 'fixed_20' else 'solid'),
                                marker=dict(size=8), hovertemplate='%{y:,.0f} kW')
            fig.update_xaxes(tickformat='%m-%d', hoverformat='%m-%d')
            show(styled(fig, 340, 'kW', time_axis=False), 'daily')

        st.dataframe(summary, width='stretch', hide_index=True, column_config={
            '전체 전력량 (MWh)': st.column_config.NumberColumn(format='%.1f'),
            '절감 (%)': st.column_config.NumberColumn(format='%.2f'),
            '냉각 전력량 (MWh)': st.column_config.NumberColumn(format='%.1f'),
            '팬 증가분 (MWh)': st.column_config.NumberColumn(format='%.1f'),
            '평균 PUE': st.column_config.NumberColumn(format='%.3f'),
            '평균 고장률': st.column_config.NumberColumn(format='%.2f'),
            '최고 온도 (℃)': st.column_config.NumberColumn(format='%.1f'),
            '설정 변경 총량 (℃)': st.column_config.NumberColumn(format='%.0f'),
        })
        st.caption('설정 온도를 올리면 냉각 전력이 크게 줄지만 서버 팬 전력이 늘어 일부를 깎아 먹습니다. '
                   'MPC 와 반응형의 전력 절감은 거의 같습니다. 트윈의 열용량이 작아 미리 식혀 둘 여지가 적기 때문이며, '
                   'MPC 의 차이는 설정 온도를 덜 자주 바꾼다는 점(설정 변경 총량)에서 나타납니다.')


# ---------------------------------------------------------------- 위기 시나리오

with tab_crisis:
    c_text, c_button = st.columns([3, 1], vertical_alignment='center')
    c_text.write(f'시나리오 3종을 **{CONTROLLER_LABELS[controller]}** 제어로 {hours}시간 시뮬레이션하고 '
                 '정상 운영과 비교합니다.')
    if c_button.button('위기 3종 시뮬레이션 실행', type='primary', width='stretch'):
        st.session_state['crisis'] = (START, hours, controller)

    if st.session_state.get('crisis') == (START, hours, controller):
        runs: dict[str, dict] = {}
        try:
            progress = st.progress(0.0, text='시뮬레이션 중…')
            for idx, name in enumerate(SCENARIOS):
                progress.progress(idx / len(SCENARIOS), text=f'{SCENARIOS[name]["label"]} 계산 중…')
                runs[name] = simulate(START, hours, controller, name)
            progress.empty()
        except ApiError as e:
            st.error(str(e))
            runs = {}

        if runs:
            normal = runs['normal']['summary']
            crisis = {n: r for n, r in runs.items() if n != 'normal'}
            if all(r['summary']['temp_ok'] and not r['summary']['diverged'] for r in crisis.values()):
                st.success(f'✅ 모든 위기 시나리오에서 서버실 온도 {ROOM_MAX_C:.0f}℃ 이하 유지')
            else:
                st.error(f'⚠️ 일부 시나리오에서 서버실 온도가 {ROOM_MAX_C:.0f}℃ 를 넘었습니다')

            for col, (name, r) in zip(st.columns(len(crisis)), crisis.items()):
                s = r['summary']
                with col.container(border=True, height='stretch'):
                    st.markdown(f'**{r["scenario_label"]}**')
                    st.metric('최고 서버실 온도', f'{s["max_room_temp_c"]:.1f}℃',
                              delta=f'{ROOM_MAX_C:.0f}℃ 이하' if s['temp_ok'] else f'{ROOM_MAX_C:.0f}℃ 초과',
                              delta_arrow='off', delta_color='normal' if s['temp_ok'] else 'inverse')
                    a, b, c = st.columns(3)
                    small_stat(a, '평균 PUE', f'{s["average_pue"]:.3f}')
                    small_stat(b, '전력량 증가',
                               f'{(s["facility_energy_kwh"] / normal["facility_energy_kwh"] - 1) * 100:+.1f}%')
                    small_stat(c, '온도 유지율', f'{s["temp_maintenance_pct"]:.0f}%')
                    if s['violation_hours']:
                        st.warning(f'비상 여유 시간 부족 {s["violation_hours"]}시간 '
                                   f'(최소 {s["min_ride_through_min"]:.1f}분)')
                    mode_note = ''
                    if controller == 'mpc':
                        mode_note = ' · 모드: ' + ', '.join(
                            f'{MPC_MODE_SHORT.get(m, m)} {h}시간' for m, h in s['mode_hours'].items())
                    st.caption(f'대응: {r["strategy"]}{mode_note}')

            frames = {name: frame(r) for name, r in runs.items()}
            c1, c2 = st.columns(2)
            with c1:
                st.subheader('서버실 온도')
                fig = go.Figure()
                for name, d in frames.items():
                    fig.add_scatter(x=d.timestamp, y=d.room_temp_c, name=SCENARIOS[name]['label'].split(' (')[0],
                                    line=dict(color=SCENARIO_COLORS[name], width=2,
                                              dash='dot' if name == 'normal' else 'solid'),
                                    hovertemplate='%{y:.1f}℃')
                room_limits(fig, *room_range(*[d.room_temp_c for d in frames.values()]))
                show(styled(fig, 300, '℃'), 'crisis_temp')
            with c2:
                st.subheader('전체 전력')
                fig = go.Figure()
                for name, d in frames.items():
                    fig.add_scatter(x=d.timestamp, y=d.facility_power_kw,
                                    name=SCENARIOS[name]['label'].split(' (')[0],
                                    line=dict(color=SCENARIO_COLORS[name], width=2,
                                              dash='dot' if name == 'normal' else 'solid'),
                                    hovertemplate='%{y:,.0f} kW')
                show(styled(fig, 300, 'kW'), 'crisis_power')
    else:
        st.info('버튼을 누르면 서버 급증(+30%) · 칠러 고장(냉각 용량 75%) · 폭염(외기 38℃ 이상)을 차례로 계산합니다.')


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
            st.caption(f'관제 탭과 같은 구간 · {CONTROLLER_LABELS[controller]} · {sim["scenario_label"]}. '
                       '변화량은 20℃ 고정 대비입니다.')
            compare = controller != 'fixed_20'
            cols = st.columns(4)
            cols[0].metric('탄소 배출 (tCO2)', f'{e["carbon_tco2"]:.1f}', delta_color='inverse', border=True,
                           delta=f'{e["carbon_tco2"] - e_base["carbon_tco2"]:+.1f}' if compare else None,
                           help=f'전체 전력량 × {e["carbon_factor_tco2_per_mwh"]} tCO2/MWh')
            cols[1].metric('CUE (tCO2/MWh)', f'{e["cue_tco2_per_mwh"]:.3f}', delta_color='inverse', border=True,
                           delta=f'{e["cue_tco2_per_mwh"] - e_base["cue_tco2_per_mwh"]:+.3f}' if compare else None,
                           help='탄소 배출 / IT 전력량')
            cols[2].metric('물 사용량 (m³)', f'{e["water_usage_l"] / 1000:,.1f}', delta_color='inverse', border=True,
                           delta=f'{(e["water_usage_l"] - e_base["water_usage_l"]) / 1000:+,.1f}' if compare else None)
            cols[3].metric('WUE (L/kWh)', f'{e["wue_l_per_kwh"]:.2f}', delta_color='inverse', border=True,
                           delta=f'{e["wue_l_per_kwh"] - e_base["wue_l_per_kwh"]:+.2f}' if compare else None,
                           help='물 사용량 / IT 전력량')

            modes = list(MODE_LABELS)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**냉각 방식별 물 사용량**')
                fig = go.Figure(go.Bar(
                    x=[MODE_LABELS[m] for m in modes], y=[e['water_by_mode_l'][m] / 1000 for m in modes],
                    marker=dict(color=C['blue'], cornerradius=4), width=0.5,
                    text=[f'{e["water_by_mode_l"][m] / 1000:,.1f}' for m in modes], textposition='outside',
                    hovertemplate='%{x}: %{y:,.1f} m³<extra></extra>'))
                show(styled(fig, 260, 'm³', legend=False, time_axis=False), 'water')
            with c2:
                st.markdown('**냉각 방식별 운전 시간**')
                fig = go.Figure(go.Bar(
                    x=[MODE_LABELS[m] for m in modes], y=[e['hours_by_mode'][m] for m in modes],
                    marker=dict(color=C['blue'], cornerradius=4), width=0.5,
                    text=[f'{e["hours_by_mode"][m]:.0f}' for m in modes], textposition='outside',
                    hovertemplate='%{x}: %{y:.0f}시간<extra></extra>'))
                show(styled(fig, 260, '시간', legend=False, time_axis=False), 'mode_hours')
            st.caption('물 원단위(칠러 1.8 · 혼합 1.0 · 외기 냉각 0.3 L/kWh)는 데이터에 없는 가정값입니다.')

    st.divider()
    st.subheader('설정 온도를 올리면 얻는 것과 잃는 것')
    default_it = float(frame(sim).it_base_kw.mean()) if sim else 2900.0
    default_out = float(frame(sim).outdoor_temp_c.mean()) if sim else 22.0
    s1, s2 = st.columns(2)
    it_kw = s1.slider('IT 부하 (kW)', 2000, 4500, int(min(max(round(default_it, -1), 2000), 4500)), 10)
    out_c = s2.slider('외기온도 (℃)', -5.0, 40.0, min(max(round(default_out * 2) / 2, -5.0), 40.0), 0.5)
    try:
        sweep = setpoint_sweep(float(it_kw), float(out_c))
    except ApiError as err:
        st.error(str(err))
        sweep = None

    if sweep is not None:
        best = sweep.loc[sweep.facility_power_kw.idxmin()]
        st.caption(f'이 조건에서 전체 전력이 가장 적은 설정 온도는 **{best.setpoint_c:.1f}℃** '
                   f'({best.facility_power_kw:,.0f} kW, PUE {best.pue:.3f}) 입니다. '
                   'PUE 는 계속 좋아져도 팬 전력 때문에 전체 전력은 다시 늘 수 있습니다.')
        panels = (('전체 전력 (kW)', 'facility_power_kw', '%{y:,.0f} kW'),
                  ('상대 고장률 (20℃ = 1.0)', 'xfactor', '%{y:.2f}'),
                  ('비상 여유 시간 (분)', 'ride_through_min', '%{y:.1f}분'))
        for col, (title, column, hover) in zip(st.columns(3), panels):
            with col:
                st.markdown(f'**{title}**')
                fig = go.Figure(go.Scatter(x=sweep.setpoint_c, y=sweep[column], name=title,
                                           line=dict(color=C['blue'], width=2), hovertemplate=hover))
                if column == 'facility_power_kw':
                    fig.add_scatter(x=[best.setpoint_c], y=[best.facility_power_kw], mode='markers+text',
                                    marker=dict(size=10, color=C['blue']), text=['최소'],
                                    textposition='top center', hoverinfo='skip', showlegend=False)
                if column == 'ride_through_min':
                    fig.add_hline(y=15, line_dash='dash', line_color=C['critical'], line_width=1,
                                  annotation_text='기준 15분', annotation_position='bottom left')
                show(styled(fig, 260, '', '설정 온도 (℃)', legend=False, time_axis=False), f'sweep_{column}')
        table_view(sweep[['setpoint_c', 'facility_power_kw', 'cooling_power_kw', 'server_fan_extra_kw', 'pue',
                          'cooling_mode', 'xfactor', 'ride_through_min']])

st.divider()
st.caption('가정 기반 시뮬레이션입니다. 트윈의 열용량·팬 전력·비상 기준·물 원단위는 데이터에 없는 가정값이고, '
           '외기 예보는 실제값을 그대로 썼습니다(예보 오차 없음). 최적화 목표는 PUE 가 아니라 전체 전력입니다.')
