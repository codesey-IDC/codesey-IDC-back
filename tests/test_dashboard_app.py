"""대시보드(Streamlit) 화면 테스트: 실제 서비스 3개를 띄운 상태에서 화면 스크립트 전체를 실행한다.

과제 요구: PUE 게이지, 시간별 전력 추이, 서버 온도, 냉각 모드, 위기 시나리오 주입, 27℃ 초과 경고.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from services.common.settings import REPO_ROOT

APP = str(REPO_ROOT / 'dashboard' / 'app.py')


def _no_errors(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture()
def app(live_services: dict[str, str]) -> AppTest:
    return AppTest.from_file(APP, default_timeout=300).run()


def test_dashboard_renders_all_tabs(app: AppTest) -> None:
    _no_errors(app)
    assert app.title[0].value == 'IDC 통합 관제 대시보드'
    assert len(app.tabs) == 6
    assert len(app.metric) > 10
    labels = {s.label for s in app.selectbox}
    assert '제어기' in labels


@pytest.mark.parametrize('controller', ['rule_pid', 'rule_tes', 'fixed_20'])
def test_dashboard_controller_switch(app: AppTest, controller: str) -> None:
    [s for s in app.selectbox if s.label == '제어기'][0].set_value(controller).run()
    _no_errors(app)


@pytest.mark.parametrize('scenario', ['server_surge', 'chiller_failure', 'heatwave'])
def test_dashboard_crisis_injection(app: AppTest, scenario: str) -> None:
    """위기 시나리오 주입 (사이드바 선택) 후에도 화면이 오류 없이 다시 그려진다."""
    [r for r in app.radio if r.label == '관제 화면에 적용할 시나리오'][0].set_value(scenario).run()
    _no_errors(app)


def test_dashboard_recompute_button(app: AppTest) -> None:
    [b for b in app.button if b.label == '다시 계산'][0].click().run()
    _no_errors(app)
