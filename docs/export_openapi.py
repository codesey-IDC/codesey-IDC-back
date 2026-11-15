"""세 서비스의 OpenAPI 명세(JSON)를 docs/openapi/ 에 저장한다 (docs/API.md 의 기계용 원본).

실행 (저장소 루트에서):  python docs/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.forecast.main import app as forecast_app  # noqa: E402
from services.mpc.main import app as mpc_app  # noqa: E402
from services.twin.main import app as twin_app  # noqa: E402


def main() -> None:
    out = ROOT / 'docs' / 'openapi'
    out.mkdir(parents=True, exist_ok=True)
    for name, app in (('twin', twin_app), ('forecast', forecast_app), ('mpc', mpc_app)):
        spec = app.openapi()
        path = out / f'{name}.json'
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{path.relative_to(ROOT)}: {len(spec["paths"])}개 경로, 버전 {spec["info"]["version"]}')


if __name__ == '__main__':
    main()
