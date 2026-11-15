"""설정 파일(config/settings.yaml) 테스트: 읽기, 환경변수 우선, 오류, 모듈들이 설정값을 쓰는지."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from services.common import settings
from services.mpc import simulation
from services.mpc.controller import MPCConfig
from services.twin import esg
from services.twin.physics import DEFAULT_PARAMS


@pytest.fixture()
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """설정 캐시를 비우고, 테스트가 끝나면 원래 설정으로 되돌린다."""
    settings.load_settings.cache_clear()
    yield monkeypatch
    monkeypatch.undo()
    settings.load_settings.cache_clear()


def test_default_config_has_all_sections() -> None:
    cfg = settings.load_settings()
    assert set(settings.REQUIRED_SECTIONS) <= set(cfg)
    assert settings.repo_path('data') == settings.REPO_ROOT / 'data' / 'synthetic_idc_90d.parquet'


def test_service_url_env_override(fresh_settings: pytest.MonkeyPatch) -> None:
    fresh_settings.setenv('TWIN_URL', 'http://twin:8001')
    assert settings.service_url('twin') == 'http://twin:8001'
    assert settings.service_url('mpc') == 'http://127.0.0.1:8003'


def test_missing_config_file(fresh_settings: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fresh_settings.setenv('IDC_CONFIG', str(tmp_path / 'none.yaml'))
    with pytest.raises(settings.SettingsError, match='설정 파일이 없습니다'):
        settings.load_settings()


def test_missing_section(fresh_settings: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / 'partial.yaml'
    path.write_text('services: {}\npaths: {}\n', encoding='utf-8')
    fresh_settings.setenv('IDC_CONFIG', str(path))
    with pytest.raises(settings.SettingsError, match='datacenter'):
        settings.load_settings()


def test_server_count_reproduces_it_power_formula() -> None:
    """서버 수 × 서버당 전력 = 원본 데이터의 IT 전력 공식 (2,150 + 3,600 × CPU kW)."""
    dc = settings.section('datacenter')
    assert DEFAULT_PARAMS.num_servers == dc['num_servers']
    assert DEFAULT_PARAMS.idle_power_kw == pytest.approx(2150.0)
    assert DEFAULT_PARAMS.cpu_power_range_kw == pytest.approx(3600.0)
    assert DEFAULT_PARAMS.to_dict()['idle_power_kw'] == pytest.approx(2150.0)


def test_modules_read_config_values() -> None:
    cfg = settings.load_settings()
    assert MPCConfig().horizon_hours == cfg['mpc']['horizon_hours']
    assert MPCConfig().blocks == tuple(cfg['mpc']['blocks'])
    assert esg.CARBON_FACTOR_TCO2_PER_MWH == cfg['esg']['carbon_factor_tco2_per_mwh']
    assert simulation.HISTORY_HOURS == cfg['simulation']['history_hours']
    assert list(simulation.SCENARIOS) == list(cfg['scenarios'])


def test_chiller_failure_drops_one_of_four_units() -> None:
    """칠러 4대 중 1대 탈락 → 냉각 용량 75%."""
    sc = simulation.SCENARIOS['chiller_failure']
    assert sc.capacity_factor == pytest.approx(0.75)
    assert simulation.SCENARIOS['server_surge'].it_scale == 1.3
    assert simulation.SCENARIOS['heatwave'].outdoor_min_c == 38.0
    assert simulation.SCENARIOS['normal'].capacity_factor == 1.0


def test_invalid_failed_chillers(fresh_settings: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = settings.DEFAULT_CONFIG_PATH.read_text(encoding='utf-8').replace(
        'failed_chillers: 1', 'failed_chillers: 4'
    )
    path = tmp_path / 'bad.yaml'
    path.write_text(text, encoding='utf-8')
    fresh_settings.setenv('IDC_CONFIG', str(path))
    with pytest.raises(ValueError, match='failed_chillers'):
        simulation.load_scenarios()
