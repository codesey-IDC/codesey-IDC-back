"""실행 스크립트 테스트: 짧은 구간·작은 설정으로 main() 을 실제로 돌리고 결과 파일을 확인한다.

결과는 모두 임시 폴더(tmp_path)에 쓴다. 저장소의 output/ 결과 파일은 건드리지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scripts import benchmark_api
import scripts.evaluate_cooling_forecast as ECF
import scripts.sensitivity_analysis as SENS
import scripts.simulate_mpc as SIM
import scripts.train_forecast as TF
import scripts.train_ridge as TR
import scripts.tune_forecast as TUNE
import scripts.visualize_forecast as VIS
from services.forecast import main as forecast_svc
from services.forecast import ridge
from services.mpc.controller import MPCConfig
from services.twin.main import app as twin_app

needs_model = pytest.mark.skipif(forecast_svc._predictor is None, reason='예측 모델 없음')
FAST = MPCConfig(de_maxiter=2, de_popsize=2)


def _argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, 'argv', ['script.py', *args])


# ---------------------------------------------------------------- scripts/simulate_mpc.py


@needs_model
def test_simulate_mpc_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """제어기 6종 × 24시간 비교 실험이 결과 파일 2개를 만든다."""
    monkeypatch.setattr(SIM, 'OUT_DIR', str(tmp_path))
    monkeypatch.setattr(SIM, 'MPCConfig', lambda **kw: FAST)
    _argv(monkeypatch, '--days', '1', '--start', '2026-06-04 00:00')
    SIM.main()
    metrics = json.loads((tmp_path / 'metrics_base.json').read_text(encoding='utf-8'))
    ctrls = metrics['controllers']
    assert set(ctrls) == {'fixed_20', 'rule_pid', 'rule_tes', 'fixed_26', 'reactive', 'mpc'}
    assert ctrls['rule_pid']['saving_vs_rule_pid_pct'] == 0.0
    assert all(c['violation_hours'] == 0 for c in ctrls.values())
    ts = pd.read_csv(tmp_path / 'timeseries_base.csv')
    assert len(ts) == 6 * 24 and {'energy_cost_krw', 'tes_soc_kwh'} <= set(ts.columns)


@needs_model
def test_simulate_mpc_surge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SIM, 'OUT_DIR', str(tmp_path))
    monkeypatch.setattr(SIM, 'MPCConfig', lambda **kw: FAST)
    _argv(
        monkeypatch,
        '--surge',
        '--days',
        '1',
        '--start',
        '2026-06-05 00:00',
        '--controllers',
        'rule_pid,mpc',
    )
    SIM.main()
    m = json.loads((tmp_path / 'metrics_surge.json').read_text(encoding='utf-8'))
    assert m['scenario'] == 'surge' and m['controllers']['mpc']['naive_forecast_hours'] > 0


def test_noisy_forecast_passthrough_and_bias() -> None:
    """오차 0 이면 예측 서비스 응답 그대로, 편향 +10% 면 예측값이 정확히 1.1배."""
    hourly = SIM.load_hourly()
    now = pd.Timestamp('2026-06-04 00:00')
    hist = hourly.it_power_kw.loc[now - pd.Timedelta(hours=192) : now - pd.Timedelta(hours=1)]
    body = {
        'horizon_hours': 24,
        'history': [
            {'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()
        ],
    }
    inner = TestClient(forecast_svc.app)
    clean = inner.post('/api/v1/forecast', json=body).json()['predictions']
    same = SIM.NoisyForecast(inner).post('/api/v1/forecast', json=body).json()['predictions']
    biased = SIM.NoisyForecast(inner, bias_pct=10).post('/api/v1/forecast', json=body)
    noisy = SIM.NoisyForecast(inner, noise_pct=10).post('/api/v1/forecast', json=body)
    assert same == clean
    assert np.allclose(biased.json()['predictions'], np.array(clean) * 1.1)
    assert biased.status_code == 200 and noisy.json()['predictions'] != clean


@needs_model
def test_simulate_latency_delays_setpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """지연 2시간이면 처음 2시간은 기준값 유지, 그 뒤 설정 온도는 지연 0 의 2시간 전 값과 같다."""
    monkeypatch.setattr(SIM, 'MPCConfig', lambda **kw: FAST)
    hourly, twin = SIM.load_hourly(), TestClient(twin_app)
    start = pd.Timestamp('2026-06-04 00:00')
    df0, _ = SIM.run(hourly, start, 4, 'reactive', twin, TestClient(forecast_svc.app))
    df2, _ = SIM.run(hourly, start, 4, 'reactive', twin, TestClient(forecast_svc.app), latency_h=2)
    assert list(df2.setpoint_c[:2]) == [20.0, 20.0]
    assert list(df2.setpoint_c[2:]) == list(df0.setpoint_c[:2])


@needs_model
def test_sensitivity_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """조건 4개 × 6시간으로 분석 전체 흐름(표·JSON·그래프)을 돌린다."""
    monkeypatch.setattr(SIM, 'OUT_DIR', str(tmp_path))
    monkeypatch.setattr(SIM, 'MPCConfig', lambda **kw: FAST)
    cases = [c for c in SENS.default_cases() if c.name in ('rule_pid', 'mpc', 'latency_1h')]
    cases += [c for c in SENS.default_cases() if c.group in ('it_noise', 'it_bias', 'outdoor')][
        ::3
    ]
    monkeypatch.setattr(SENS, 'default_cases', lambda: cases)
    _argv(monkeypatch, '--days', '1', '--start', '2026-06-04 00:00', '--workers', '1')
    SENS.main()
    r = json.loads((tmp_path / 'sensitivity.json').read_text(encoding='utf-8'))
    by = {c['name']: c for c in r['cases']}
    assert by['mpc']['pue_change'] == 0.0 and by['rule_pid']['cost_saving_vs_rule_pid_pct'] == 0.0
    assert all(c['violation_hours'] == 0 for c in r['cases'])
    assert (tmp_path / 'sensitivity.png').stat().st_size > 10_000


# ------------------------------------------------ scripts/evaluate_cooling_forecast.py


def test_nmae() -> None:
    assert ECF.nmae(np.array([110.0, 90.0]), np.array([100.0, 100.0])) == pytest.approx(10.0)


def test_summarize_metrics() -> None:
    df = pd.DataFrame(
        {
            'h': [1, 7, 13],
            'pred': [100.0, 200.0, 300.0],
            'actual': [100.0, 220.0, 300.0],
            'lo': [90.0, 190.0, 310.0],
            'hi': [110.0, 210.0, 320.0],
            'mode_pred': ['chiller'] * 3,
            'mode_actual': ['chiller', 'hybrid', 'chiller'],
        }
    )
    s = ECF.summarize(df)
    assert s['nmae_pct'] == pytest.approx(20 / 620 * 100)
    assert s['coverage_pct'] == pytest.approx(100 / 3)
    assert s['mode_accuracy_pct'] == pytest.approx(200 / 3)
    assert s['nmae_by_horizon_pct']['h1-6'] == 0.0


@needs_model
def test_evaluate_cooling_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """짧은 구간(2개 예측 시점)·외기 오차 2단계로 평가 전체 흐름을 돌린다."""
    out = tmp_path / 'cooling_metrics.json'
    monkeypatch.setattr(ECF, 'RESULT_PATH', out)
    monkeypatch.setattr(ECF, 'TEST_START', pd.Timestamp('2026-06-04 00:00'))
    monkeypatch.setattr(ECF, 'TEST_END', pd.Timestamp('2026-06-05 00:00'))
    monkeypatch.setattr(ECF, 'NOISE_SIGMAS_C', (0.0, 2.0))
    ECF.main()
    r = json.loads(out.read_text(encoding='utf-8'))
    assert r['cooling_power_forecast']['rows'] == 2 * 24
    assert r['cooling_power_forecast']['nmae_pct'] < 10.0
    assert set(r['outdoor_error_sensitivity']) == {'sigma_0c', 'sigma_2c'}
    assert r['structure_error_nmae_pct'] < 1.0
    assert 'chiller (외기 25℃)' in r['cooling_mode_comparison']['typical_it_3000kw']


# ---------------------------------------------------------------- 학습·튜닝·시각화


def test_train_ridge_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(TR, 'MODEL_DIR', str(tmp_path))
    TR.main()
    cfg = json.loads((tmp_path / ridge.CONFIG_NAME).read_text(encoding='utf-8'))
    assert set(cfg['models']) == {'24h', '168h'}
    assert cfg['models']['24h']['test']['mape'] < 5.0
    assert (tmp_path / 'ridge_24h.joblib').exists() and (tmp_path / 'ridge_168h.joblib').exists()


def test_train_forecast_main_and_config_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LightGBM 학습 전체를 임시 폴더에 다시 돌리면 저장된 지표와 같은 결과가 나온다."""
    models = tmp_path / 'models'
    monkeypatch.setattr(TF, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(TF, 'MODEL_DIR', str(models))
    monkeypatch.setattr(TF, 'CONFIG_PATH', str(models / 'model_config.json'))
    TF.main()
    new = json.loads((tmp_path / 'metrics.json').read_text(encoding='utf-8'))
    saved = json.loads(Path('output/forecast/metrics.json').read_text(encoding='utf-8'))
    assert new['24h']['mape_model'] == pytest.approx(saved['24h']['mape_model'], rel=1e-6)
    (models / 'model_config.json').unlink()
    TF.config_only()
    assert json.loads((models / 'model_config.json').read_text(encoding='utf-8'))['models']['24h'][
        'features'
    ]


def test_tune_forecast_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(TUNE, 'TUNING_DIR', str(tmp_path))
    _argv(monkeypatch, '--trials', '1', '--horizon', '24h')
    TUNE.main()
    cmp = json.loads((tmp_path / 'comparison.json').read_text(encoding='utf-8'))
    assert set(cmp['24h']) == {'baseline', 'tuned'}
    assert (tmp_path / 'trials_24h.csv').exists() and (tmp_path / 'best_params.json').exists()


def test_visualize_forecast_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(VIS, 'PLOT_DIR', str(tmp_path))
    VIS.main()
    html = (tmp_path / 'forecast_report.html').read_text(encoding='utf-8')
    assert 'plotly' in html.lower() and (tmp_path / 'mape_by_h_168h.csv').exists()


# ---------------------------------------------------------------- scripts/benchmark_api.py


def test_benchmark_main(
    live_services: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """실제 HTTP 서버에 대고 모든 엔드포인트를 1번씩 호출해 응답 시간을 기록한다."""
    monkeypatch.setattr(benchmark_api, 'repo_path', lambda key: tmp_path / 'output' / key)
    _argv(monkeypatch, '--repeat', '1')
    benchmark_api.main()
    r = json.loads(
        (tmp_path / 'output' / 'benchmark' / 'api_latency.json').read_text(encoding='utf-8')
    )
    assert len(r['endpoints']) >= 20
    assert all(e['ok'] for e in r['endpoints'].values())
    assert r['mpc_simulation_job_completion_s']['seconds'] > 0


def test_twin_client_still_works() -> None:
    """다른 테스트가 서비스 상태를 바꾼 뒤에도 트윈 API 는 그대로 응답한다."""
    assert TestClient(twin_app).get('/health').json()['service'] == 'twin'
