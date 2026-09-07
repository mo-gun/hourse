# -*- coding: utf-8 -*-
"""
한국마사회 공공데이터 OpenAPI 클라이언트 — 실검증된 엔드포인트만 수록.

검증일: 2026-08-27 (개발계정 인증키, 실호출로 resultCode=00 확인)
인증키는 프로젝트 루트 .env 의 KRA_API_KEY_ENCODED 에서 읽는다 (.gitignore 등록됨).

주의 — data.go.kr 참고문서(docx)의 엔드포인트는 구버전이라 폐기된 것이 있다.
      아래 ENDPOINTS 는 실호출로 200/resultCode=00 을 확인한 현행 경로다.

사용:
    from kra_client import KRA
    kra = KRA()
    rows = kra.race_result(rc_month='202605', meet=1)      # 경주기록(93필드) — 학습 원장
    kra.backfill_csv('race_result_2000_2026.csv', 2000, 2026)
"""
import urllib.request, urllib.parse, ssl, json, io, os, time, csv, sys

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
BASE = 'https://apis.data.go.kr/B551015'

# ── 실호출 검증된 엔드포인트 (2026-08-27) ────────────────────────────────
# 주의: data.go.kr 페이지의 Swagger `host` 는 API 번호가 잘려 있는 경우가 많다.
#       (예: "B551015/" → 실제는 "B551015/API155"). 아래는 전부 실호출로 확인한 경로다.
ENDPOINTS = {
    # ★ 학습 원장. 89~93필드. weather/track/배당률/마체중/구간기록/착순 전부. 2000년~현재 697,876행.
    #    예정 경기(D+1~D+3)도 행이 존재하며 결과 필드만 비어 있다.
    'race_result':       ('API4_3/raceResult_3',        ['meet', 'rc_date', 'rc_month', 'rc_no', 'rc_year']),
    # ★ 출전표 상세정보. 48필드. 사전 예측 입력으로 race_result 보다 낫다 —
    #    stTime(발주시각) · dusu(출주두수) 가 여기에만 있다.  ⚠ 통산전적 필드는 누수(아래 LEAKY).
    'entry_sheet':       ('API26_2/entrySheet_2',       ['meet', 'rc_date', 'rc_no']),
    # 경주계획표. 경주 단위(말 단위 아님) 21필드. schStTime(예정 발주시각), 등급/조건/상금.
    'race_plan':         ('API72_2/racePlan_2',         ['meet', 'rc_date', 'rc_month', 'rc_no', 'rc_year']),
    # AI학습용_경주결과. 27필드. race_result 의 부분집합 + bthd(생년월일)만 고유.
    'ai_race_result':    ('API155/raceResult',          ['rccrs_cd', 'race_dt']),
    # 마필 구간별. 말별 구간 최고/최저/평균 기록.
    'section_record':    ('API37_1/sectionRecord_1',    ['meet', 'hr_name', 'hr_no', 'rc_date', 'rc_month', 'rc_year']),
    # 승식별 확정배당률 (단승/연승/복승/쌍승/복연승/삼복승/삼쌍승).
    'odds':              ('API160_1/integratedInfo_1',  ['meet', 'pool', 'rc_date', 'rc_month', 'rc_no']),
    # ⚠ LEAKY — 아래 주석 참조.
    'horse_record':      ('API15_2/raceHorseResult_2',  ['meet', 'hr_no', 'hr_name']),
    'horse_rating':      ('API77/raceHorseRating',      []),
    'trainer_record':    ('trtresult/gettrtresult',     ['meet']),

    # ── 2026-08-31 추가 신청분. 전부 실호출 resultCode=00 확인 ──────────────
    # ⚠ data.go.kr Swagger 의 `host` 에 API 번호가 들어있다(basePath 아님).
    #    예: host="apis.data.go.kr/B551015/trcontihi" + path="/gettrcontihi"
    # ★ 조교. 훈련일자·시작/종료시각·기승자구분·출전구분·수영훈련. 2010~ 소급(연 35~46만건).
    #   경주 전 확정이라 누수 없음 — schema.py F1(최근 성적)의 컨디션 축.
    'train_record':      ('trcontihi/gettrcontihi',     ['hrname', 'hrno', 'tr_date_fr', 'tr_date_to']),
    # ★ 장제(편자). 장제형태·제형·지세·각도. 2010~ 소급(연 ~3만건).
    #   shoe_date 범위 파라미터가 있어 시점 정합 가능 = 누수 없음.
    'horse_shoe':        ('API191_1/HorseShoe_1',       ['meet', 'shoe_date_fr', 'shoe_date_to']),
    # ★ 마필종합 218필드. 혈통 3대: fhrNo(부마) mhrNo(모마) mhrFhrNo(외조부).
    #   ⚠ 누적필드는 누수 — LEAKY_TOTAL_HORSE 참조. 혈통 식별자만 쓸 것.
    'horse_total':       ('API42_1/totalHorseInfo_1',   ['hr_name', 'hr_no']),
    # 씨수말(부마) 마스터 — 부마 단위 집계용.
    'stallion':          ('API79_1/stallionInfo_1',     ['hr_name', 'hr_no']),
    # 경주별 상세성적표.
    'race_detail':       ('racedetailresult/getracedetailresult', ['meet', 'rc_date', 'rc_no']),
    # 경주마 상세정보(마스터).
    'horse_info':        ('API8_2/raceHorseInfo_2',     ['meet', 'hr_name', 'hr_no']),

    # ── 2026-09-07 추가 확인분. 전부 실호출 resultCode=00 ──────────────────
    # ★ AI기반연구용 경주결과상세. 44필드. 2004~현재 전 구간 조회 가능(실측).
    #   원장에 없는 것: pthrLatstPtinDt(직전 출전일 → 휴양일수) · pthrEquip(장구)
    #                   hrmOwnerNm(마주) · pthrBthd(생년월일) · rsutRlStrtTim(실제 발주시각)
    #   ⚠⚠ 파라미터명이 race_dt 다. rc_date/rc_month/rc_year 등은 **조용히 무시되고
    #      최신 경주일이 돌아온다** — 과거 데이터인 줄 알고 쓰면 전량 오염된다. 실측 함정.
    #   ✔ pthrLatstPtinDt 는 누수 없음. 같은 말을 2020-07/07-18/08-01 로 조회하면 값이
    #     각각 직전 출전일과 정확히 일치했다(20두 확인). 스냅샷 아님.
    'ai_race_detail':    ('API156/raceRsutDtl',         ['race_dt', 'rccrs_cd']),
    # 출전취소. 7필드(chulNo·hrNo·reason). rc_year/rc_month/rc_date 전부 동작, 2015년 소급 확인.
    #   게이트 번호가 당겨지는 원인 추적용 — 발주 직전 재조회 로직의 근거.
    'cancel_info':       ('API9_1/raceHorseCancelInfo_1', ['meet', 'rc_date', 'rc_month', 'rc_no', 'rc_year']),
    # 기수변경. 13필드. jkBef/jkAft + befBudam/aftBudam(부담중량 변동). 2015년 소급 확인.
    'jockey_change':     ('API10_1/jockeyChangeInfo_1',  ['meet', 'rc_date', 'rc_month', 'rc_no', 'rc_year']),
    # ★★ 출전마 체중. 승인돼 있는데 목록에서 누락돼 있었다(2026-09-07 발견).
    #   4개 경마장 전부 + 2004~현재 소급(실측). 필드: wgHr · wgHrDiff(증감) · recentRcDate.
    #   API317(서울 당일 마체중)을 신청할 이유가 없다 — 이쪽이 상위호환이다.
    #   ⚠ 백필은 rc_month 로 할 것. rc_year=2020 이 1,130행인데 rc_month=202007 한 달이
    #     1,183행이다. rc_year 는 값이 안 맞으니 쓰지 마라(실측).
    #   ⚠ 다가올 경주일(20260911~13)은 totalCount=0 — 당일에만 채워진다.
    #     발주 **전**에 채워지는지는 경마 시행일에 직접 확인해야 한다. 되면 실시간 피처가 생긴다.
    'entry_weight':      ('API25_1/entryHorseWeightInfo_1', ['meet', 'hr_name', 'hr_no', 'rc_date', 'rc_month']),
}

# ⚠ 마필종합(API42_1)의 누적 스냅샷 필드 — LEAKY_FIELDS 와 같은 이유로 금지.
#   혈통 자마 성적은 원장에서 '그 경주일 이전' 행만으로 직접 집계할 것 (schema.py F2 참조).
LEAKY_TOTAL_HORSE = ['rcCnt', 'fstCnt', 'sndCnt', 'trdCnt', 'forthCnt', 'fifthCnt',
                     'winRate', 'quinRate', 'avgWinDist', 'rankTop', 'rankLast',
                     'fgnRcCnt', 'fgnFstCnt', 'fgnSndCnt', 'fgnTrdCnt', 'fgnAvgWinDist']

# ⚠ 누수 주의 — '오늘 기준 누적 스냅샷'이라 과거 경주 행에 붙이면 미래 정보가 샌다.
#   실측 근거: 같은 말(신의운명 0047543)의 rcCntT 가 2025-01 / 2025-07 / 2026-07 출전표에서
#             전부 36 으로 동일했다. age 만 시점에 맞게 4→5 로 변한다.
#   대응: 통산·최근 성적류는 race_result 원장에서 '그 경주일 이전' 행만으로 직접 rolling 집계할 것.
LEAKY_FIELDS = {
    'entry_sheet':    ['rcCntT', 'rcCntY', 'ord1CntT', 'ord1CntY', 'ord2CntT', 'ord2CntY',
                       'ord3CntT', 'ord3CntY', 'chaksunT', 'chaksunY', 'chaksun_6m'],
    'horse_record':   ['rcCntT', 'rcCntY', 'ord1CntT', 'ord2CntT', 'winRateT', 'qnlRateT',
                       'chaksunT', 'chaksunY', 'recentOrd', 'recentRating', 'recentRcDate'],
    'horse_rating':   ['rating1', 'rating2', 'rating3', 'rating4'],   # 경주일 파라미터 자체가 없음
    'trainer_record': ['rcCntT', 'ord1CntT', 'ord2CntT', 'ord3CntT', 'winRateTsum', 'quRateTsum'],
}
# 반대로 race_result 의 `rating` 은 경주 시점 값이라 누수가 없다(충전율 76%).

# 미반영 — 호출 시 403 SERVICE_KEY_IS_NOT_REGISTERED_ERROR (2026-09-07 확인).
#   경로 자체는 살아있다. 없는 경로는 400 NO_OPENAPI_SERVICE_ERROR 로 확실히 갈린다(대조군 실측).
#   즉 아래는 '경로 오류'가 아니라 '신청 승인/게이트웨이 반영 대기'다.
NOT_APPLIED = {
    'race_detail_full':  'racedetailresult/getracedetailresult',   # 15089492 경주별상세성적표
    'weight_seoul':      'API317/textDataHoldSeWegInfo',           # ★ 서울 당일 마체중 (발주 전 컨디션)
    'change_seoul':      'API320/textDataHoldSeChgInfo',           # 서울 기수변경·말취소
    'ai_race_plan':      'API154/racePlan',                        # 15143802 AI학습용_경주계획
}
# 경로 미확정: 15119524 경마시행당일_경주결과종합 — Swagger 는 'Race_Result_total' 이라 하나
#              API1~199 전 범위에서 NO_OPENAPI_SERVICE_ERROR. 별도 확인 필요.

# 예정 경기에서 비어 있는(=경주 후에만 채워지는) 필드. 사전 예측 모델에서 제외할 것.
POST_RACE_ONLY = ['ord', 'rcTime', 'diffUnit', 'winOdds', 'plcOdds',
                  'wgHr', 'weather', 'track']   # 뒤 셋은 '당일' 확정

MEETS = {1: '서울', 2: '제주', 3: '부산경남', 4: '영천'}


class KRAError(RuntimeError):
    pass


class KRA:
    def __init__(self, key=None, env_path='.env', pause=0.12, retries=3):
        self.key = key or self._load_key(env_path)
        self.pause = pause
        self.retries = retries
        self.calls = 0

    @staticmethod
    def _load_key(env_path):
        if os.environ.get('KRA_API_KEY_ENCODED'):
            return os.environ['KRA_API_KEY_ENCODED']
        for line in io.open(env_path, encoding='utf-8'):
            if line.startswith('KRA_API_KEY_ENCODED'):
                return line.split('=', 1)[1].strip()
        raise KRAError('.env 에 KRA_API_KEY_ENCODED 가 없습니다')

    def _raw(self, path, params):
        # serviceKey 는 이미 URL 인코딩된 값이므로 urlencode 를 태우지 않고 직접 붙인다.
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, '')})
        url = f'{BASE}/{path}?serviceKey={self.key}&{qs}'
        req = urllib.request.Request(url, headers={'User-Agent': 'kra-client/1.0', 'Accept': '*/*'})
        last = None
        for attempt in range(self.retries):
            try:
                body = urllib.request.urlopen(req, timeout=40, context=_CTX).read().decode('utf-8', 'replace')
                self.calls += 1
                time.sleep(self.pause)
                return body
            except Exception as e:                       # 타임아웃/일시 오류는 백오프 후 재시도
                last = e
                time.sleep(0.6 * (attempt + 1))
        raise KRAError(f'{path} 호출 실패: {last}')

    def fetch(self, name, page_all=True, num_rows=1000, **params):
        """엔드포인트 호출 → dict 리스트. page_all=True 면 전 페이지 수집."""
        if name not in ENDPOINTS:
            raise KRAError(f'알 수 없는 엔드포인트 {name}. 가능: {list(ENDPOINTS)}')
        path, allowed = ENDPOINTS[name]
        bad = set(params) - set(allowed)
        if bad:
            raise KRAError(f'{name} 이 받지 않는 파라미터: {bad}. 허용: {allowed}')
        out, page = [], 1
        while True:
            body = self._raw(path, dict(params, numOfRows=num_rows, pageNo=page, _type='json'))
            if not body.lstrip().startswith('{'):
                raise KRAError(f'{name}: JSON 이 아닌 응답 — {body[:200]}')
            j = json.loads(body)
            if 'OpenAPI_ServiceResponse' in j:
                hdr = j['OpenAPI_ServiceResponse'].get('cmmMsgHeader', {})
                raise KRAError(f"{name}: {hdr.get('errMsg')} / {hdr.get('returnAuthMsg')}")
            resp = j.get('response', {})
            code = resp.get('header', {}).get('resultCode')
            if str(code) not in ('00', '0'):
                raise KRAError(f"{name}: resultCode={code} {resp.get('header', {}).get('resultMsg')}")
            b = resp.get('body') or {}
            total = b.get('totalCount') or 0
            if not total:
                break
            items = (b.get('items') or {}).get('item') or []
            items = items if isinstance(items, list) else [items]
            out += items
            if not page_all or len(out) >= total or not items:
                break
            page += 1
        return out

    # 편의 래퍼 ---------------------------------------------------------
    def race_result(self, **kw):    return self.fetch('race_result', **kw)
    def odds(self, **kw):           return self.fetch('odds', **kw)
    def section_record(self, **kw): return self.fetch('section_record', **kw)

    def backfill_csv(self, out_path, year_from=2000, year_to=2026, meets=(1, 2, 3)):
        """경주기록 전량을 CSV 로. rc_month 단위라 27년치도 약 1,000콜."""
        seen_cols, buf, n = [], [], 0
        t0 = time.time()
        for y in range(year_from, year_to + 1):
            for m in range(1, 13):
                for meet in meets:
                    try:
                        rows = self.fetch('race_result', rc_month=f'{y}{m:02d}', meet=meet)
                    except KRAError as e:
                        print(f'  [skip] {y}-{m:02d} meet={meet}: {e}', file=sys.stderr)
                        continue
                    for r in rows:
                        for k in r:
                            if k not in seen_cols:
                                seen_cols.append(k)
                    buf += rows
                    n += len(rows)
            print(f'  {y}: 누적 {n:,}행 / {self.calls}콜 / {time.time()-t0:.0f}s', file=sys.stderr)
        with io.open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=seen_cols)
            w.writeheader()
            w.writerows(buf)
        print(f'저장 {out_path}: {n:,}행 × {len(seen_cols)}열, {self.calls}콜', file=sys.stderr)
        return n


if __name__ == '__main__':
    kra = KRA()
    print('== 승인 엔드포인트 헬스체크 ==')
    checks = [
        ('race_result',    dict(rc_date='20260823', meet=1)),
        ('entry_sheet',    dict(rc_date='20260829', meet=1)),
        ('race_plan',      dict(rc_date='20260829', meet=1)),
        ('ai_race_result', dict(race_dt='20260823', rccrs_cd='1')),
        ('section_record', dict(rc_date='20260823', meet=1)),
        ('odds',           dict(rc_date='20260823', meet=1, rc_no='1')),
        ('horse_record',   dict(hr_no='0044233')),
        ('horse_rating',   dict()),
        ('trainer_record', dict(meet='1')),
    ]
    for name, kw in checks:
        try:
            rows = kra.fetch(name, page_all=False, num_rows=1, **kw)
            print(f'  OK   {name:16s} {ENDPOINTS[name][0]:32s} rows>=1 cols={len(rows[0]) if rows else 0}')
        except KRAError as e:
            print(f'  FAIL {name:16s} {e}')
    print(f'총 {kra.calls}콜 사용')
