"""config/settings.yaml 읽기.

모든 서비스와 대시보드가 같은 설정 파일을 쓴다. 다른 파일을 쓰려면 환경변수 IDC_CONFIG 에 경로를 준다.
서비스 주소는 환경변수(TWIN_URL, FORECAST_URL, MPC_URL)가 있으면 그 값이 우선한다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / 'config' / 'settings.yaml'
URL_ENV = {'twin_url': 'TWIN_URL', 'forecast_url': 'FORECAST_URL', 'mpc_url': 'MPC_URL'}
REQUIRED_SECTIONS = (
    'services',
    'paths',
    'forecast',
    'datacenter',
    'tariff',
    'esg',
    'rule_based',
    'mpc',
    'simulation',
    'scenarios',
)


class SettingsError(ValueError):
    """설정 파일이 없거나 형식이 잘못됐을 때."""


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """설정 파일을 읽어 dict 로 돌려준다 (프로세스당 한 번만 읽는다)."""
    path = Path(os.environ.get('IDC_CONFIG', DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise SettingsError(f'설정 파일이 없습니다: {path}')
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise SettingsError(f'설정 파일 {path} 에 필요한 항목이 없습니다: {missing}')
    for key, env in URL_ENV.items():
        if os.environ.get(env):
            cfg['services'][key] = os.environ[env]
    return cfg


def section(name: str) -> dict[str, Any]:
    """설정의 한 부분 (예: 'datacenter', 'mpc')."""
    return load_settings()[name]


def repo_path(key: str) -> Path:
    """paths 항목을 저장소 루트 기준 절대 경로로."""
    return REPO_ROOT / load_settings()['paths'][key]


def service_url(name: str) -> str:
    """'twin' | 'forecast' | 'mpc' 서비스 주소."""
    return load_settings()['services'][f'{name}_url']
