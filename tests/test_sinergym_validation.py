"""Sinergym 비교 검증 테스트: 비교 계산이 맞는지, 저장된 Sinergym 결과로 보고서가 만들어지는지.

Sinergym 자체는 실행하지 않는다 (Docker 에서 수집한 output/validation/sinergym_samples.csv 를 읽는다).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import scripts.recompute_cooling_columns as RC
import scripts.validate_sinergym as VS
from services.twin import physics as P

needs_samples = pytest.mark.skipif(
    not VS.SAMPLES_PATH.exists() or VS.SAMPLES_PATH.stat().st_size < 1000,
    reason='Sinergym 결과 CSV 없음 (Docker 로 수집하거나 git lfs pull)',
)


def _synthetic(scale: float = 1.0, setpoints: tuple[float, ...] = (25.0,)) -> pd.DataFrame:
    """우리 모델이 만든 '가짜 Sinergym 결과'. scale 로 냉각 전력을 일부러 틀리게 만든다."""
    rng = np.random.default_rng(0)
    n = 500
    frames = []
    for sp in setpoints:
        it = rng.uniform(80, 120, n)
        outdoor = rng.uniform(-10, 35, n)
        cooling = P.simulate_static(it, outdoor)['cooling_power_kw'] * scale
        frames.append(
            pd.DataFrame(
                {
                    'sinergym_version': 'test',
                    'env': 'synthetic',
                    'seed': 0,
                    'cooling_setpoint_c': sp,
                    'outdoor_temp_c': outdoor,
                    'cpu_loading': it / 120,
                    'it_power_kw': it,
                    'ups_loss_kw': it * 0.1,
                    'hvac_power_kw': cooling,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_identical_model_has_zero_error_and_passes() -> None:
    r = VS.compare(_synthetic())
    assert r['passed']
    for m in r['metrics']:
        assert m['error_pct'] == pytest.approx(0.0, abs=1e-9)
    assert r['hourly_cooling_nmae_pct'] == pytest.approx(0.0, abs=1e-9)
    assert sum(b['hours'] for b in r['by_outdoor_band']) == 500


def test_cooling_error_beyond_tolerance_fails() -> None:
    """Sinergym 냉각 전력이 우리 모델보다 20% 작으면 우리 모델 오차는 +25% → 미달."""
    r = VS.compare(_synthetic(scale=0.8))
    cooling = r['metrics'][0]
    assert cooling['error_pct'] == pytest.approx(25.0)
    assert not cooling['passed'] and not r['passed']


def test_pue_uses_same_it_power_on_both_sides() -> None:
    df = _synthetic()
    r = VS.compare(df)
    pue = r['metrics'][1]
    assert pue['sinergym'] == pytest.approx(1 + df.hvac_power_kw.sum() / df.it_power_kw.sum())
    assert r['total_pue']['sinergym'] == pytest.approx(pue['sinergym'] + 0.1)
    assert r['total_pue']['own'] == pytest.approx(
        pue['own'] + P.DEFAULT_PARAMS.other_overhead_ratio
    )


def test_calibration_recovers_parameters_from_own_model() -> None:
    """우리 모델이 만든 결과에 맞추면 계수가 그대로 나온다 (맞춤 절차 점검)."""
    c = VS.calibrate(_synthetic())
    for name, value in c['fitted'].items():
        assert value == pytest.approx(c['current'][name], rel=0.02)
    assert c['passed']


def test_setpoint_sensitivity_is_relative_to_baseline() -> None:
    rows = VS.setpoint_sensitivity(_synthetic(setpoints=(22.0, 25.0, 28.0)))
    by_delta = {row['delta_c']: row for row in rows}
    assert set(by_delta) == {-3.0, 0.0, 3.0}
    assert by_delta[0.0]['own_cooling_change_pct'] == pytest.approx(0.0)
    assert by_delta[0.0]['sinergym_facility_change_pct'] == pytest.approx(0.0)
    # 우리 모델: 설정 온도를 올리면 냉각 전력이 줄고, 내리면 는다
    assert by_delta[3.0]['own_cooling_change_pct'] < 0 < by_delta[-3.0]['own_cooling_change_pct']
    assert by_delta[3.0]['own_setpoint_c'] == P.DEFAULT_PARAMS.supply_temp_ref_c + 3.0


def test_main_without_samples_is_skipped_not_passed(tmp_path, capsys) -> None:
    """Sinergym 결과가 없으면 통과(0)도 실패(1)도 아닌 보류(2). 보고서도 만들지 않는다."""
    out = tmp_path / 'report.json'
    assert VS.main(['--samples', str(tmp_path / 'none.csv'), '--out', str(out)]) == 2
    assert not out.exists()
    assert '보류' in capsys.readouterr().err


def test_main_writes_report_and_returns_verdict(tmp_path) -> None:
    samples = tmp_path / 'samples.csv'
    out = tmp_path / 'report.json'
    _synthetic(setpoints=(22.0, 25.0)).to_csv(samples, index=False)
    assert VS.main(['--samples', str(samples), '--out', str(out)]) == 0
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report['passed'] and report['tolerance_pct'] == 10.0
    assert {'metrics', 'by_outdoor_band', 'setpoint_sensitivity', 'calibration'} <= set(report)

    _synthetic(scale=0.5).to_csv(samples, index=False)
    assert VS.main(['--samples', str(samples), '--out', str(out)]) == 1


def test_missing_baseline_setpoint_is_an_error() -> None:
    with pytest.raises(ValueError):
        VS.build_report(_synthetic(setpoints=(22.0,)))


@needs_samples
def test_real_sinergym_samples_cover_a_full_year() -> None:
    samples = pd.read_csv(VS.SAMPLES_PATH)
    base = samples[samples.cooling_setpoint_c == VS.BASELINE_SETPOINT_C]
    assert len(base) == 8760
    assert (base.it_power_kw > 0).all() and (base.hvac_power_kw > 0).all()
    assert base.outdoor_temp_c.min() < 0 and base.outdoor_temp_c.max() > 30


@needs_samples
def test_real_sinergym_report_matches_saved_file() -> None:
    """저장된 보고서가 저장된 Sinergym 결과로 다시 계산한 값과 같다 (문서의 숫자가 낡지 않게)."""
    if not VS.REPORT_PATH.exists():
        pytest.skip('보고서 없음')
    saved = json.loads(VS.REPORT_PATH.read_text(encoding='utf-8'))
    fresh = VS.compare(pd.read_csv(VS.SAMPLES_PATH).query('cooling_setpoint_c == 25.0'))
    for a, b in zip(saved['metrics'], fresh['metrics']):
        assert a['error_pct'] == pytest.approx(b['error_pct'], abs=1e-6)
        assert a['passed'] == b['passed']


# ---------------------------------------------------------------- 데이터 냉각 열 다시 계산

needs_data = pytest.mark.skipif(not RC.repo_path('data').exists(), reason='데이터 없음')


@needs_data
def test_data_cooling_columns_match_current_settings() -> None:
    """저장소의 데이터는 지금 설정의 냉각 효율 계수로 계산돼 있다 (계수만 바꾸고 데이터를 안 바꾸면 실패)."""
    assert RC.main(['--check']) == 0


@needs_data
def test_recompute_rewrites_only_cooling_power_and_pue(tmp_path) -> None:
    """냉각 전력·PUE 가 틀린 사본을 주면 그 두 열만 고쳐 쓴다."""
    original = pq.read_table(RC.repo_path('data')).slice(0, 500)
    broken = original.set_column(
        original.schema.get_field_index('cooling_power_kw'),
        'cooling_power_kw',
        pc.multiply(original['cooling_power_kw'], 1.2),
    )
    path = tmp_path / 'data.parquet'
    pq.write_table(broken, path)

    assert RC.main(['--data', str(path), '--check']) == 1
    assert RC.main(['--data', str(path)]) == 0
    assert RC.main(['--data', str(path), '--check']) == 0
    fixed = pq.read_table(path)
    assert fixed.schema.equals(original.schema)
    for name in original.column_names:
        if name in RC.RECOMPUTED:
            np.testing.assert_allclose(
                fixed[name].to_numpy(), original[name].to_numpy(), atol=1e-9
            )
        else:
            assert fixed[name].equals(original[name]), name
