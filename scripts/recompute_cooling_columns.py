"""데이터의 냉각 전력·PUE 열을 지금 설정의 냉각 효율 계수로 다시 계산한다.

data/synthetic_idc_90d.parquet 의 cooling_power_kw, pue 는 측정값이 아니라 냉각 효율 계수로 계산해
채운 값이다. 계수를 Sinergym 에 맞춰 바꾸면 (config/settings.yaml 의 [보정] 값) 이 두 열이
디지털 트윈과 어긋나므로, 같은 공식(services.twin.physics.simulate_static)으로 다시 계산해 맞춘다.

  바뀌는 열       cooling_power_kw, pue
  그대로인 열     timestamp, cpu_utilization, it_power_kw, 외기·습도·풍속, cooling_load_kw,
                 free_cooling_fraction, cooling_mode (외기 냉각 전환 온도 15/18℃ 는 바꾸지 않았다) 등

IT 부하 예측 모델은 it_power_kw 만 쓰므로 다시 학습할 필요가 없다.

실행:
  python scripts/recompute_cooling_columns.py            # 다시 계산해 덮어쓴다
  python scripts/recompute_cooling_columns.py --check    # 바꾸지 않고 데이터와 설정이 맞는지만 확인
종료 코드 (--check): 0 일치 / 1 불일치
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from services.common.settings import repo_path
from services.twin import physics as P

RECOMPUTED = ('cooling_power_kw', 'pue')
UNCHANGED = ('free_cooling_fraction', 'server_heat_kw', 'cooling_load_kw')


def recompute(table: pa.Table) -> tuple[pa.Table, dict[str, float]]:
    """표의 냉각 전력·PUE 열을 다시 계산한 새 표와, 열마다 바뀐 정도(최대 절대 차이)를 돌려준다."""
    it = table['it_power_kw'].to_numpy()
    r = P.simulate_static(it, table['outdoor_temp_c'].to_numpy())

    # 외기 냉각 전환 온도는 그대로라 냉각 모드·외기 냉각 비율은 바뀌면 안 된다
    for col in UNCHANGED:
        np.testing.assert_allclose(r[col], table[col].to_numpy(), atol=1e-6, err_msg=col)
    if not (r['cooling_mode'] == np.asarray(table['cooling_mode'].to_pylist())).all():
        raise AssertionError(
            'cooling_mode 가 데이터와 다릅니다 (외기 냉각 전환 온도가 바뀌었는지 확인)'
        )

    changes = {}
    for col in RECOMPUTED:
        old = table[col].to_numpy()
        changes[col] = float(np.abs(r[col] - old).max())
        table = table.set_column(
            table.schema.get_field_index(col), table.schema.field(col), pa.array(r[col])
        )
    return table, changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='데이터의 냉각 전력·PUE 열을 지금 설정으로 다시 계산')
    ap.add_argument('--data', default=str(repo_path('data')))
    ap.add_argument('--check', action='store_true', help='바꾸지 않고 일치 여부만 확인')
    args = ap.parse_args(argv)

    table = pq.read_table(args.data)
    new_table, changes = recompute(table)
    for col, diff in changes.items():
        print(f'{col}: 최대 차이 {diff:.6f}')
    matches = all(diff < 1e-6 for diff in changes.values())

    if args.check:
        print('데이터와 설정이 일치합니다' if matches else '데이터가 지금 설정과 다릅니다')
        return 0 if matches else 1
    if matches:
        print('이미 지금 설정으로 계산된 데이터입니다. 바꾸지 않았습니다.')
        return 0
    pq.write_table(new_table, args.data, compression='snappy')
    print(f'저장: {args.data} ({new_table.num_rows}행, 바뀐 열 {", ".join(RECOMPUTED)})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
