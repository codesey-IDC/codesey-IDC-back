"""가상 데이터센터 운영 데이터 생성: 지역(기상 관측소)과 기간을 바꿔 5분 간격 시계열을 만든다.

원래 data/synthetic_idc_90d.parquet(2026-03-15 ~ 06-12, 춘천)을 만든 절차를 그대로 옮긴 것이다.
같은 시작일·난수 씨앗을 쓰므로 새 데이터의 첫 90일은 CPU 사용률·IT 전력이 원래 데이터와 완전히 같고,
춘천은 외기온도까지 같다 (--verify 로 확인).

  CPU·메모리  Google Cluster Data 1,600 VM 의 시간대별 평균 사용률(각 24개 값, data/gcd_hourly_cpu_profile.json,
   사용률     --extract-profile 로 원본에서 추출) × 주말 0.85 + 5분마다 잡음 N(0, 0.03).
              ※ 계절 변화는 없다 (가상 부하). 메모리 사용률은 참고용 열이고 전력 계산에는 CPU 만 쓴다
  IT 전력     SPECpower 선형 모델 (services.twin.physics.it_power_from_cpu)
  외기        기상청 ASOS 시간자료(실측) → 5분 선형 보간. 지역마다 다른 것은 이 열뿐이다
  냉각·PUE    디지털 트윈의 정상 상태 계산 (services.twin.physics.simulate_static, 지금 설정의 계수)

지역과 기간은 config/settings.yaml 의 sites, dataset 항목. 기상 자료는 data/weather/ 에 저장해 두고 다시 쓴다
(없으면 환경변수 KMA_API_KEY 로 공공데이터포털 ASOS API 를 부른다).

실행 (저장소 루트에서):
  python scripts/build_dataset.py                  # 모든 지역
  python scripts/build_dataset.py --site jeju      # 한 지역만
  python scripts/build_dataset.py --verify data/synthetic_idc_90d.parquet   # 원래 90일 데이터와 비교만
  python scripts/build_dataset.py --extract-profile data/raw/GCD_VMs   # Google Cluster 원본 → 패턴

Google Cluster 원본 (HiPro-IT 전처리본, VM 1,600개 × 24시간·5분, 인증 불필요):
  curl -L -o data/raw/GCD_VMs.tar.gz https://raw.githubusercontent.com/HiPro-IT/\
CPU-and-Memory-resource-usage-from-Google-Cluster-Data/master/GCD_VMs.tar.gz
  tar -xzf data/raw/GCD_VMs.tar.gz -C data/raw/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import argparse
import json
import os

import httpx
import numpy as np
import pandas as pd

from services.common.settings import REPO_ROOT, repo_path, section
from services.twin import physics as P

ASOS_URL = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
WEATHER_COLS = ('ta', 'hm', 'ws')  # 기온(℃), 습도(%), 풍속(m/s)
RAW_DIR = REPO_ROOT / 'data' / 'weather'
PROFILE_PATH = REPO_ROOT / 'data' / 'gcd_hourly_cpu_profile.json'
STEP = '5min'
STEPS_PER_DAY = 288  # 24시간 × 5분
GCD_URL = 'https://github.com/HiPro-IT/CPU-and-Memory-resource-usage-from-Google-Cluster-Data'
CDD_BASE_C = 18.0  # 냉방도시 기준 온도


class DatasetError(RuntimeError):
    """기상 자료를 구할 수 없거나 응답이 잘못됐을 때."""


# ---------------------------------------------------------------- 기상 자료


def weather_path(station_id: int, start: str, end: str) -> Path:
    return RAW_DIR / f'asos_{station_id}_{start}_{end}.csv'.replace('-', '')


def fetch_asos(api_key: str, station_id: int, start: str, end: str) -> pd.DataFrame:
    """기상청 ASOS 시간자료 (start 01시 ~ end 23시). 반환 열: tm, ta, hm, ws."""
    rows: list[dict] = []
    page, page_size = 1, 999
    with httpx.Client(timeout=60.0) as client:
        while True:
            r = client.get(
                ASOS_URL,
                params={
                    'serviceKey': api_key,
                    'numOfRows': page_size,
                    'pageNo': page,
                    'dataType': 'JSON',
                    'dataCd': 'ASOS',
                    'dateCd': 'HR',
                    'startDt': start.replace('-', ''),
                    'startHh': '01',
                    'endDt': end.replace('-', ''),
                    'endHh': '23',
                    'stnIds': station_id,
                },
            )
            r.raise_for_status()
            try:
                body = r.json()['response']
            except (ValueError, KeyError) as e:
                raise DatasetError(f'ASOS 응답을 읽을 수 없습니다: {r.text[:200]}') from e
            header = body['header']
            if header['resultCode'] not in ('00', '0'):
                raise DatasetError(f'ASOS 오류 {header["resultCode"]}: {header["resultMsg"]}')
            items = body['body']['items']
            chunk = items.get('item', []) if isinstance(items, dict) else []
            rows.extend(chunk)
            if page * page_size >= int(body['body']['totalCount']) or not chunk:
                break
            page += 1
    if not rows:
        raise DatasetError(f'ASOS 응답에 자료가 없습니다 (지점 {station_id}, {start} ~ {end})')
    df = pd.DataFrame(rows)
    df['tm'] = pd.to_datetime(df['tm'])
    for col in WEATHER_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['tm', *WEATHER_COLS]].sort_values('tm').reset_index(drop=True)


def load_weather(station_id: int, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """저장해 둔 기상 자료를 읽고, 없으면 API 로 받아 저장한다. 관측 결측은 앞뒤 값으로 메운다."""
    path = weather_path(station_id, start, end)
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=['tm'])
    else:
        key_env = section('weather')['api_key_env']
        api_key = os.environ.get(key_env)
        if not api_key:
            raise DatasetError(
                f'{path.name} 이 없고 환경변수 {key_env} 도 없습니다. '
                '공공데이터포털에서 "기상청_지상(종관, ASOS) 시간자료 조회서비스" 키를 받아 넣어 주세요.'
            )
        df = fetch_asos(api_key, station_id, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, lineterminator='\n')
        print(f'  기상 자료 저장: {path.relative_to(REPO_ROOT)} ({len(df)}시간)')
    df[list(WEATHER_COLS)] = df[list(WEATHER_COLS)].interpolate().bfill().ffill()
    return df


# ---------------------------------------------------------------- 데이터 생성


def extract_profile(directory: Path, min_vms: int = 100) -> dict:
    """Google Cluster Data(VM 별 파일: 288행 = 24시간·5분, 열 = CPU %, 메모리 %)에서 시간대별 평균 사용률을 뽑는다.

    모든 VM 을 5분 칸마다 평균 → 0~1 → 1시간(12칸) 평균. 과제 기준(VM 100개 이상, 24시간 이상)을 확인한다.
    """
    cpu, mem, count = np.zeros(STEPS_PER_DAY), np.zeros(STEPS_PER_DAY), 0
    for path in sorted(Path(directory).glob('vm_*')):
        usage = np.loadtxt(path)
        if usage.shape != (STEPS_PER_DAY, 2):
            continue
        cpu += usage[:, 0]
        mem += usage[:, 1]
        count += 1
    if count < min_vms:
        raise DatasetError(
            f'{directory} 에서 읽은 VM 이 {count}개입니다 (최소 {min_vms}개, 파일마다 288행 × 2열)'
        )

    def hourly(total: np.ndarray) -> list[float]:
        return [
            float(v) for v in np.clip(total / count / 100.0, 0, 1).reshape(24, 12).mean(axis=1)
        ]

    return {
        'source': (
            'Google Cluster Data, HiPro-IT 전처리본 (GCD_VMs.tar.gz): '
            'VM 별 24시간·5분 CPU·메모리 사용률'
        ),
        'url': GCD_URL,
        'vm_count': count,
        'method': '모든 VM 의 5분 칸 평균(%) → 0~1 → 시간별(12칸) 평균',
        'hourly_cpu': hourly(cpu),
        'hourly_mem': hourly(mem),
    }


def usage_profile(key: str = 'hourly_cpu') -> np.ndarray:
    """시간대별(0~23시) 평균 사용률 24개. key: 'hourly_cpu' | 'hourly_mem'."""
    profile = np.asarray(json.loads(PROFILE_PATH.read_text(encoding='utf-8'))[key], dtype=float)
    if profile.shape != (24,):
        raise DatasetError(f'{PROFILE_PATH.name} 의 {key} 는 24개여야 합니다: {profile.shape}')
    return profile


def cpu_profile() -> np.ndarray:
    """시간대별(0~23시) 평균 CPU 사용률 24개."""
    return usage_profile('hourly_cpu')


def memory_utilization(ts: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """메모리 사용률: CPU 와 같은 방식 (시간대별 패턴 × 주말 배율 + 잡음). CPU 값이 바뀌지 않게 난수는 따로 뽑는다."""
    rng = np.random.default_rng(int(cfg['seed']) + 1)
    base = usage_profile('hourly_mem')[ts.hour]
    weekend = np.where(ts.dayofweek.to_numpy() >= 5, float(cfg['weekend_scale']), 1.0)
    noise = rng.normal(0.0, float(cfg['noise_std']), len(ts))
    return np.clip(base * weekend + noise, 0.05, 1.0)


def cpu_utilization(ts: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """시간대별 패턴 × 주말 배율 + 잡음. 난수는 앞에서부터 차례로 뽑으므로 기간을 늘려도 앞부분은 같다."""
    rng = np.random.default_rng(int(cfg['seed']))
    base = cpu_profile()[ts.hour]
    weekend = np.where(ts.dayofweek.to_numpy() >= 5, float(cfg['weekend_scale']), 1.0)
    noise = rng.normal(0.0, float(cfg['noise_std']), len(ts))
    return np.clip(base * weekend + noise, 0.05, 1.0)


def build(weather: pd.DataFrame, start: str, end: str, cfg: dict) -> pd.DataFrame:
    """[start 00:00, end 23:55] 의 5분 간격 데이터."""
    ts = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=55), freq=STEP)
    w = weather.set_index('tm')[list(WEATHER_COLS)]
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq=STEP)).interpolate()
    w = w.ffill().bfill().reindex(ts).interpolate().ffill().bfill()

    util = cpu_utilization(ts, cfg)
    it = P.it_power_from_cpu(util)
    outdoor = w['ta'].to_numpy()
    r = P.simulate_static(it, outdoor)
    return pd.DataFrame(
        {
            'timestamp': ts,
            'cpu_utilization': util,
            'memory_utilization': memory_utilization(ts, cfg),
            'outdoor_temp_c': outdoor,
            'outdoor_humidity': w['hm'].to_numpy(),
            'wind_speed': w['ws'].to_numpy(),
            'it_power_kw': it,
            'cooling_load_kw': r['cooling_load_kw'],
            'cooling_power_kw': r['cooling_power_kw'],
            'free_cooling_fraction': r['free_cooling_fraction'],
            'cooling_mode': r['cooling_mode'],
            'pue': r['pue'],
            'free_cooling_available': outdoor <= P.DEFAULT_PARAMS.free_cooling_full_below_c,
            'cooling_degree_hours': np.maximum(0.0, outdoor - CDD_BASE_C),
            'server_heat_kw': it,
            'hour': ts.hour.astype('int32'),
            'dayofweek': ts.dayofweek.astype('int32'),
            'is_weekend': ts.dayofweek >= 5,
        }
    )


def summarize(df: pd.DataFrame) -> dict:
    """지역 비교용 요약 (5분 행 기준)."""
    hours = len(df) / 12
    mode = df['cooling_mode'].value_counts(normalize=True)
    return {
        'days': round(hours / 24, 1),
        'outdoor_mean_c': round(float(df['outdoor_temp_c'].mean()), 2),
        'outdoor_min_c': round(float(df['outdoor_temp_c'].min()), 1),
        'outdoor_max_c': round(float(df['outdoor_temp_c'].max()), 1),
        'humidity_mean_pct': round(float(df['outdoor_humidity'].mean()), 1),
        'free_cooling_pct': round(float(mode.get('free_cooling', 0.0)) * 100, 1),
        'hybrid_pct': round(float(mode.get('hybrid', 0.0)) * 100, 1),
        'chiller_pct': round(float(mode.get('chiller', 0.0)) * 100, 1),
        'it_mean_kw': round(float(df['it_power_kw'].mean()), 1),
        'cooling_energy_mwh': round(float(df['cooling_power_kw'].sum() / 12 / 1000), 1),
        'average_pue': round(
            float(
                1
                + df['cooling_power_kw'].sum() / df['it_power_kw'].sum()
                + P.DEFAULT_PARAMS.other_overhead_ratio
            ),
            4,
        ),
        'monthly': monthly_summary(df),
    }


def monthly_summary(df: pd.DataFrame) -> list[dict]:
    """월별 외기·냉각 방식·PUE (기준 운전). 지역 차이가 어느 계절에 생기는지 보는 용도."""
    rows = []
    for month, g in df.groupby(df['timestamp'].dt.strftime('%Y-%m')):
        mode = g['cooling_mode'].value_counts(normalize=True)
        rows.append(
            {
                'month': month,
                'outdoor_mean_c': round(float(g['outdoor_temp_c'].mean()), 2),
                'humidity_mean_pct': round(float(g['outdoor_humidity'].mean()), 1),
                'free_cooling_pct': round(float(mode.get('free_cooling', 0.0)) * 100, 1),
                'chiller_pct': round(float(mode.get('chiller', 0.0)) * 100, 1),
                'average_pue': round(
                    float(
                        1
                        + g['cooling_power_kw'].sum() / g['it_power_kw'].sum()
                        + P.DEFAULT_PARAMS.other_overhead_ratio
                    ),
                    4,
                ),
            }
        )
    return rows


def verify(df: pd.DataFrame, reference: Path) -> dict[str, float]:
    """새 데이터의 앞부분이 기준 데이터(원래 90일)와 같은지: 열마다 최대 절대 차이."""
    ref = pd.read_parquet(reference)
    head = df.iloc[: len(ref)]
    if not (head['timestamp'].to_numpy() == ref['timestamp'].to_numpy()).all():
        raise DatasetError('기준 데이터와 시각이 다릅니다')
    cols = ('cpu_utilization', 'it_power_kw', 'outdoor_temp_c', 'cooling_power_kw', 'pue')
    return {c: float(np.abs(head[c].to_numpy() - ref[c].to_numpy()).max()) for c in cols}


def main(argv: list[str] | None = None) -> int:
    sites, cfg = section('sites'), section('dataset')
    ap = argparse.ArgumentParser(description='지역·기간별 가상 데이터센터 운영 데이터 생성')
    ap.add_argument('--site', choices=[*sites, 'all'], default='all')
    ap.add_argument('--start', default=cfg['start'], help='YYYY-MM-DD')
    ap.add_argument('--end', default=cfg['end'], help='YYYY-MM-DD (이 날 23:55 까지)')
    ap.add_argument('--refresh-weather', action='store_true', help='기상 자료를 다시 받는다')
    ap.add_argument(
        '--extract-profile',
        metavar='DIR',
        help='Google Cluster 원본 폴더(GCD_VMs)에서 시간대별 CPU·메모리 패턴을 뽑아 저장하고 끝낸다',
    )
    ap.add_argument(
        '--verify', metavar='PARQUET', help='저장하지 않고 이 데이터와 앞부분이 같은지만 본다'
    )
    args = ap.parse_args(argv)

    if args.extract_profile:
        profile = extract_profile(Path(args.extract_profile))
        PROFILE_PATH.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
            newline='\n',
        )
        print(
            f'{PROFILE_PATH.relative_to(REPO_ROOT)} 저장: VM {profile["vm_count"]}개, '
            f'CPU 평균 {np.mean(profile["hourly_cpu"]):.3f}, '
            f'메모리 평균 {np.mean(profile["hourly_mem"]):.3f}'
        )
        return 0

    summary = {}
    for name in sites if args.site == 'all' else [args.site]:
        site = sites[name]
        print(
            f'[{name}] {site["label"]} (ASOS 지점 {site["station_id"]}) {args.start} ~ {args.end}'
        )
        weather = load_weather(site['station_id'], args.start, args.end, args.refresh_weather)
        df = build(weather, args.start, args.end, cfg)
        if args.verify:
            for col, diff in verify(df, Path(args.verify)).items():
                print(f'  {col}: 최대 차이 {diff:.6g}')
            continue
        out = REPO_ROOT / site['data']
        df.to_parquet(out, index=False, compression='snappy')
        summary[name] = {'label': site['label'], 'station_id': site['station_id']} | summarize(df)
        s = summary[name]
        print(
            f'  저장: {site["data"]} ({len(df):,}행) · 외기 평균 {s["outdoor_mean_c"]}℃ · '
            f'외기 냉각 {s["free_cooling_pct"]}% · 혼합 {s["hybrid_pct"]}% · 칠러 {s["chiller_pct"]}% · '
            f'PUE {s["average_pue"]}'
        )
    if summary:
        path = repo_path('site_results') / 'dataset_summary.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        merged.update(summary)
        path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n'
        )
        print(f'요약: {path.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
