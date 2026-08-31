"""
preprocessing7.py — 차선 검출 전처리 (튜닝 확정본)

이 파일이 차선 마스크 생성의 단일 소스다. camera_node.py 가 여기를 통해
lanedetect_tuner 로 확정한 파이프라인을 그대로 쓴다.

파이프라인 (흰색):
    ROI 크롭 → BEV → L채널 가로/세로 tophat max → 임계 → 색조건(L,S) → CLOSE → 형태필터
파이프라인 (노란색):
    LAB b* + HSV(H,S,V) 3중 게이트 → CLOSE → 형태필터 → 흰색에서 노란영역 제거

이전 버전(HSV inRange 이진화)에서 교체한 이유:
  - tophat 은 '주변보다 밝은 얇은 띠'만 남겨 조명/노출 변화에 강함. 절대 밝기
    임계인 HSV inRange 는 실내 반사·노출 드리프트에서 그대로 무너졌음.
  - 세로 tophat 을 max 로 합침: 커브 상단에서 차선이 70deg 이상 기울면 가로
    단면폭이 커널을 넘어 가로 tophat 이 차선을 배경으로 오인함.
  - 형태필터 길이 게이트를 bh(세로 범위) 기준으로: BEV 높이가 110px 뿐이라
    max(bw,bh) 기준 100px 은 점선을 전량 탈락시켰음. bh 기준이면 점선은 살고
    가로로 긴 정지선은 배제된다.

주의 — BEV 기하를 바꾸면:
  lane_detection3.LANE_WIDTH_PX(350) 는 이 BEV 픽셀 스케일 기준 실측값이다.
  ROI/BEV 파라미터를 건드리면 lane_width_calib.py 재실행이 선행되어야 한다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[v7] 0821/preprocessing.py 의 BEV_MARGIN 0.32 분기본
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BEV_MARGIN            0.28 -> 0.32
  MAX_LINE_W              22 -> 18     (가로 압축 x0.8192)
  _filter_white area      50 -> 41
짝은 lane_detection7.py — **반드시 같이 써야 한다.**
0.28 검출기에 물리면 실제 차선폭 213 인데 260 으로 알아 기하 게이트가 다 어긋난다.
"""

import cv2
import numpy as np


# ════════════════════════════════════════════════
#  차선용 파라미터 — ★ lanedetect_tuner 확정값 ★
#  (test_camera_node.py / lanedetect_tuner.py / detect_tuner.py 와 동일)
# ════════════════════════════════════════════════
SIM_W, SIM_H     = 640, 480
ROI_TOP_RATIO    = 0.70
ROI_BOTTOM_RATIO = 0.93
BEV_SRC_RATIO = [(0.21, 0.92), (0.37, 0.01), (0.63, 0.01), (0.81, 0.92)]
# ★ BEV_MARGIN 0.20 → 0.28 (S자 구간 검출 회복) ★
#   사다리꼴을 더 좁은 중앙 밴드로 보내면 640px 안에 더 넓은 도로가 담긴다
#   (차선폭 px 는 줄지만 좌우 시야가 넓어진다).
#
#   왜 필요했나 — S자 실측(lane_6/7/8 의 지정 구간 2858f):
#     차선이 기울면 두 선의 '수평' 간격이 차선폭 x sqrt(1+slope^2) 로 부푼다.
#     margin 0.20(차선폭 355)에서는 |slope| > 1.50 이면 640px 를 넘어 두 선이
#     물리적으로 동시에 안 보인다. 그런데 S자 기울기 중앙값이 1.94~2.32 라
#     74~78% 의 프레임에서 한쪽만 보였다 (단일선 47~55%, conf 0.30).
#     검출 로직 문제가 아니라 BEV 가로 시야의 한계였다.
#   margin 0.28 → 차선폭 260, 기울기 한계 2.22 로 상승.
#     S자 양선 검출 42.8% → 66.0% (1.54배), 단일선 51.5% → 29.9%, 무검출 5.8% → 4.1%
#     직선도 함께 개선: 양선 85.7% → 88.2%, conf 0.791 → 0.800
#     0.30 이상은 붕괴(직선 양선 73.3%) — 0.28 이 무릎.
#
#   ※ 세로(ROI) 확장과 달리 원근 왜곡이 안 커진다. 앞서 ROI_TOP 확장이 실패한
#     이유는 상단이 원근 압축 구간이라 단일 호모그래피로 근/원거리를 동시에
#     못 맞춰서였다. 가로 확장은 그 문제를 안 건드린다.
BEV_MARGIN = 0.32

# 흰색
TOPHAT_KERNEL_W = 35
TOPHAT_KERNEL_V = 35
# ★ 빛반사 억제 재튜닝: TOPHAT_THRESH 30→40, WHITE_L_MIN 70→100 ★
#   (lane_6/7/8 전 프레임 12227f 종단 검증, 라바콘 구간 포함)
#
#   라벨링은 색과 독립적인 기하 오라클을 썼다 — 기하검사를 통과한 프레임에서,
#   추적 중인 '흰' fit 에서 25px 이내 성분을 참(6816개), 알려진 차선 구조선
#   (흰 fit / 노랑 fit / 노랑±lane_width) 전부에서 100px 넘게 떨어진 성분을
#   오검출(1906개)로 분류. 반대편 흰 외곽선이 오검출로 잘못 라벨되는 것을 막으려
#   ±lane_width 를 구조선에 포함시켰다.
#
#   ★ '반사는 더 밝다' 는 틀렸다 ★ 실측 분포(참 vs 오검출):
#       L 최대값   225 vs 199      L 중앙   184 vs 168
#       S 중앙      44 vs  42      (분리도 3% — 채도는 아무 정보가 없다)
#       tophat 중앙 75 vs  47      ← 여기가 갈린다
#   즉 반사는 차선보다 오히려 어둡고 대비가 낮다. 그래서 상한(WHITE_L_MAX)이나
#   채도 게이트(WHITE_S_MIN/MAX)는 오검출이 아니라 진짜 차선을 깎는다.
#   ※ WHITE_S_MAX=230 은 참/오검출 어느 쪽도 걸리지 않아 사실상 무동작이지만,
#     유채색 물체가 흰색으로 새는 것에 대한 보험이라 그대로 둔다.
#
#   왜 하한 상향이 반사에 듣나: 어두운 반사 픽셀이 진짜 차선과 이어 붙으면 병합
#   성분의 두께가 부풀어 _filter_white 의 두께 상한(MAX_LINE_W)에 걸리고, 차선이
#   통째로 탈락한다. 하한을 올리면 둘이 분리돼 차선 쪽이 살아남는다 — S자 밖
#   무검출이 0.9% → 0.4% 로 오히려 줄어드는 이유가 이것이다.
#
#   종단 효과 (L70/T30 → L100/T30):
#       구간      진짜양선(conf1.0)   무검출     포화     조향Δp99   쌍붕괴   (W,W)락
#       S자        2.5% → 2.6%     4.3→5.2%  19.8→21.0%  26.2→23.5    21→12   13→17
#       그 외     70.8% → 71.9%    0.9→0.8%   7.6→ 5.6%  20.3→17.3
#       전체      55.6% → 56.5%    1.7→1.9%  10.4→ 9.1%  23.2→22.3
#
#   ★ TOPHAT_THRESH 는 30 을 유지한다 — 40 으로 올렸다가 되돌렸다 ★
#     집계 지표만 보면 40 이 더 좋아 보였다 (쌍붕괴 21→7, (W,W) 13→6, 포화 7.9%,
#     S자 진짜양선 2.8%). 그런데 lane_7 f1574~1577 에서 흰 마스크가 통째로
#     0 이 되는 것이 발견됐다 — T30 에서는 944~1430px 로 멀쩡히 잡히던 구간이다.
#     원인: S자에서 급하게 기운 차선은 tophat 응답이 선을 따라 오르내리는데,
#     임계를 40 으로 올리면 약한 연결부가 끊겨 선이 4조각(bh 13~23)으로 쪼개지고
#     전부 MIN_LEN_WHITE(35) 에 걸려 탈락한다. 잡음이 줄어든 게 아니라 '선이
#     조각나서' 지표가 좋아 보였던 것이다 (조각난 프레임은 무검출로 빠지므로
#     쌍붕괴/포화 분모에서 사라진다).
#     WHITE_L_MIN 은 이 현상과 무관하다 (같은 프레임에서 1060 → 1052px).
#
#     히스테리시스 임계(저 30 으로 성장 + 고 45 이상 씨앗을 포함한 성분만 채택)로
#     '연결부는 살리고 약한 잡음만 거르기'를 시도했으나, 전 프레임 결과가 T30 과
#     소수점까지 동일했다 — 잡음 성분도 대부분 45 이상 픽셀을 갖고 있어 씨앗
#     조건을 통과한다. 즉 T40 의 이득은 부작용과 분리 불가능하다.
#     CLOSE 확대도 불가: 조각들이 세로 22px, 가로 29px 씩 떨어져 있어 이를 잇는
#     커널이면 무관한 블롭까지 다 붙는다.
#
#   ★ 이 값도 촬영 장소 조명에 의존한다 ★ LAB_B_MIN 과 같은 주의. 대회장이 더
#     어두우면 차선까지 날아간다. detect_tuner.py 로 흰 차선이 끊기지 않는지
#     먼저 확인하고, 끊기면 L_MIN 을 먼저(T 보다) 낮추는 방향이 안전하다.
#
#   기각한 다른 축들 (모두 종단 검증까지 돌림):
#     두께 하한(area/max(bw,bh)) : 라벨 표본에선 오검출 71% 제거로 최고였으나,
#       그 표본이 '검출에 이미 성공한 프레임'에서만 나온 편향이었다. 전 프레임에
#       걸면 S자 무검출이 4.3% → 17.8%, 최장 무검출 29~34프레임(1초)으로 붕괴.
#       약하게(≥6) 걸면 이번엔 아무 효과가 없다 — 이득 구간이 존재하지 않는다.
#     성분 median L / median tophat 게이트 : (W,W) 락은 가장 잘 줄이지만
#       (13→4) 픽셀 하한보다 전반 지표가 못하고, 픽셀 하한과 겹쳐 쓰면 S자
#       최장 무검출이 21프레임으로 악화.
#     PCA 단축·fill·성분 면적 하한 : 두께 하한과 같은 편향에 노출.
TOPHAT_THRESH   = 30      # 40 으로 올렸다가 되돌림 (위 ★ 참조)
WHITE_L_MIN     = 100     # 70 → 100

WHITE_S_MAX     = 230
# 노란색
# LAB_B_MIN 155→160: 갈색 벽·노란 계열 바닥 오검출 차단 (lane_6.mp4 6654프레임 실측).
#   라벨링은 색과 독립적인 오라클을 썼다 — W×Y 쌍 + 폭/평행성 기하검사를 모두
#   통과한 프레임(3507개)에서, 추적 중인 노란 fit 이 지나가는 성분(<30px)을 '참',
#   멀리 떨어진 성분(>120px)을 '오검출'로 분류. 참 4813개 / 오검출 3572개.
#
#   축별 분리도 (최적 임계에서의 분류 정확도):
#       HSV V   96.0%   참 p5/중앙/p95 = 149/183/220   오검출 =  85/121/145
#       LAB b*  95.9%   참 = 161/178/189               오검출 = 155/156/160
#       HSV S   74.1%   (분리 안 됨 → YEL_S_MIN 은 그대로 30)
#   오검출의 b* 가 155~160 에 빽빽하고 참은 p5 가 161 이라 그 사이가 비어 있다.
#
#   V 를 함께 올리는 안도 재봤으나 이득 없이 손해만 늘어 채택하지 않았다:
#       b*160 V80  → 차로전환 55, 무검출 3.1%
#       b*160 V125 → 차로전환 51, 무검출 4.6%   (V 때문에 +1.5%p)
#
#   160 을 고른 이유: 158→160 에서 차로전환이 101→55 로 꺾이고(무릎),
#   160→162 는 8 밖에 못 얻으면서 참 차선 픽셀 유지율이 99.1%→98.0% 로 떨어진다.
#
#   ★ 이 값은 촬영 장소의 벽/바닥 색에 의존한다 ★ 대회장에서는 detect_tuner.py 로
#     노란 차선이 안 끊기는지 먼저 확인할 것. 끊기면 낮추는 방향이 안전하다.
LAB_B_MIN       = 160
YEL_S_MIN       = 30
YEL_V_MIN       = 80
YEL_DILATE_K    = (5, 5)
# 형태 (흰색에만 길이 게이트 — 노란 점선을 죽이지 않기 위해 노랑은 두께만 검사)
MIN_LEN_WHITE   = 35     # 세로(bh) 기준이라 가로 압축과 무관 → 유지
# 30 → 22: BEV_MARGIN 0.20→0.28 로 가로가 압축되면 마킹의 픽셀 폭이 줄어
#   area/max(bw,bh) 비율도 같이 줄어든다. 임계를 안 낮추면 게이트가 헐거워져
#   두꺼운 반사 블롭이 통과한다 (스윕에서 스케일 안 한 0.30 이 붕괴한 원인).
#   차선폭 비율 260/353 = 0.742 를 그대로 적용.
MAX_LINE_W      = 18
CLOSE_K         = (3, 3)

# ── 프레임 내 방향 일관성 (흰색 전용) ──
# 같은 프레임 안에서 살아남은 성분들의 '지배 방향'을 면적가중으로 구하고,
# 거기서 이 각도 이상 벗어난 성분을 버린다.
#
#   왜 시간이 아니라 '프레임 내' 인가: 이전 프레임 fit 방향과 비교하는 방식은
#   이미 겪은 자기오염 루프(밴드가 반사로 조금씩 옮겨가며 고착)를 그대로
#   재현한다. 같은 프레임 안의 다수결은 시간 의존이 0 이라 그 위험이 없다.
#   근거: 트랙 기하상 차선들은 서로 대체로 평행하고, 물결 모양 반사나 가로
#   마킹은 그 방향에서 벗어난다.
#
#   실측(영상 4편, S자 / 그 외):
#       허용각      S자 진짜양선  S자 양선   S자 포화   S자 급등   쌍붕괴  (W,W)락
#       없음(현재)      2.9%     56.4%    25.0%    1.28%     28     19
#       25도           3.3%     60.4%    22.4%    0.95%     19     13   ← 채택
#       40도           3.2%     57.1%    24.2%    1.12%     25     15
#       55도           3.0%     56.4%    24.6%    1.26%     27     17
#     S자 전 지표가 개선되고 조향 급등이 26% 줄어든다.
#   대가: 그 외 구간 무검출 1.0% → 1.3%, 진짜양선 73.7% → 72.7%.
#     곡률로 두 선의 각도가 갈리는 구간에서 정상 성분이 잘린다.
#     40/55 도로 완화하면 그 손해는 줄지만 S자 이득도 같이 사라져 25 가 유일한
#     의미 있는 지점이었다.
ORIENT_TOL_DEG  = 25.0

# ════════════════════════════════════════════════
#  노면 마크용 레거시 기하
#  road_marks.py 의 ROI 비율·흰색/노랑 비율 임계가 이 BEV 를 기준으로 튜닝돼
#  있어, 차선용 BEV(근거리 좁은 띠)로 바꾸면 정지선 판정이 함께 틀어진다.
#  road_marks 를 재튜닝하기 전까지는 입력을 보존한다. (warp 1회 추가 ~0.3ms)
# ════════════════════════════════════════════════
MARKS_ROI_TOP_RATIO = 0.55
MARKS_BEV_SRC_RATIO = [(0.06, 0.95), (0.34, 0.05), (0.66, 0.05), (0.94, 0.95)]
MARKS_BEV_MARGIN    = 0.15

# 위 확정값의 dict 형태. lane_masks(img, p) 로 일부만 덮어쓸 수 있어 튜너가
# 트랙바 값으로 '프로덕션과 동일한 코드'를 그대로 돌린다 (튜너/노드 로직 분기 방지).
# yel_dil 은 반경(px) — 커널 = (2r+1, 2r+1) 이라 r=2 가 YEL_DILATE_K=(5,5).
DEFAULTS = dict(
    tophat_w=TOPHAT_KERNEL_W, tophat_v=TOPHAT_KERNEL_V, tophat_th=TOPHAT_THRESH,
    l_min=WHITE_L_MIN, s_max=WHITE_S_MAX,
    lab_b=LAB_B_MIN, yel_s=YEL_S_MIN, yel_v=YEL_V_MIN,
    yel_dil=(YEL_DILATE_K[0] - 1) // 2,
    min_len=MIN_LEN_WHITE, max_lw=MAX_LINE_W, orient_tol=ORIENT_TOL_DEG,
)

_M = _Minv = None          # 마지막 사용 기하 (소비자가 역투영에 참조)
_MAT_CACHE = {}            # (rh, rw) → (M, Minv)
_M_marks = None            # 노면 마크용 (캐시)


def _lane_matrices(rh, rw):
    """ROI 높이가 다르면 다른 행렬이 필요하므로 기하별로 캐시."""
    global _M, _Minv
    key = (rh, rw)
    if key not in _MAT_CACHE:
        src = np.float32([(x * rw, y * rh) for x, y in BEV_SRC_RATIO])
        dst = np.float32([[rw * BEV_MARGIN, rh], [rw * BEV_MARGIN, 0],
                          [rw * (1 - BEV_MARGIN), 0], [rw * (1 - BEV_MARGIN), rh]])
        _MAT_CACHE[key] = (cv2.getPerspectiveTransform(src, dst),
                           cv2.getPerspectiveTransform(dst, src))
    _M, _Minv = _MAT_CACHE[key]


def _marks_matrix(rh, rw):
    global _M_marks
    src = np.float32([(x * rw, y * rh) for x, y in MARKS_BEV_SRC_RATIO])
    dst = np.float32([[rw * MARKS_BEV_MARGIN, rh], [rw * MARKS_BEV_MARGIN, 0],
                      [rw * (1 - MARKS_BEV_MARGIN), 0], [rw * (1 - MARKS_BEV_MARGIN), rh]])
    _M_marks = cv2.getPerspectiveTransform(src, dst)


def _filter_white(mask, min_len, max_lw, orient_tol):
    """면적/길이/두께 게이트 + 프레임 내 방향 일관성 (ORIENT_TOL_DEG 주석 참조)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    if n <= 1:
        return out

    # 픽셀을 라벨별로 한 번만 정렬 — 성분마다 (labels == i) 를 돌면 O(H*W*성분수)
    ys, xs = np.nonzero(labels)
    ls = labels[ys, xs]
    order = np.argsort(ls, kind='stable')
    ys, xs, ls = ys[order], xs[order], ls[order]
    beg = np.searchsorted(ls, np.arange(1, n), 'left')
    end = np.searchsorted(ls, np.arange(1, n), 'right')

    cand = []                       # (px, py, area, angle)
    sc = ss = 0.0                   # 면적가중 2theta 누적
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 41:      # [v7] 50 -> 41 (면적 x0.8192). 노랑(40)은 유지.
            continue
        if bh < min_len:                            # 길이는 세로 범위 기준 (정지선 배제)
            continue
        if area / max(1, max(bw, bh)) > max_lw:     # 두께는 장축 기준
            continue
        a, b = beg[i - 1], end[i - 1]
        px, py = xs[a:b], ys[a:b]
        if orient_tol is None or len(px) < 10:
            cand.append((px, py, float(area), None))
            continue
        # 주축 방향 (각도는 mod 180 이라 2theta 로 평균한다)
        cov = np.cov(np.stack([px - px.mean(), py - py.mean()]))
        ev, evec = np.linalg.eigh(cov)
        v = evec[:, int(np.argmax(ev))]
        th = float(np.arctan2(v[1], v[0]))
        cand.append((px, py, float(area), th))
        sc += area * np.cos(2 * th)
        ss += area * np.sin(2 * th)

    if orient_tol is None or (sc == 0.0 and ss == 0.0):
        for px, py, _a, _t in cand:
            out[py, px] = 255
        return out

    dom = 0.5 * float(np.arctan2(ss, sc))
    tol = np.deg2rad(orient_tol)
    for px, py, _a, th in cand:
        if th is None:
            out[py, px] = 255
            continue
        d = abs(float(np.arctan2(np.sin(2 * (th - dom)), np.cos(2 * (th - dom))))) / 2.0
        if d <= tol:
            out[py, px] = 255
    return out


def _filter_yellow(mask, max_lw):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 40:
            continue
        if area / max(1, max(bw, bh)) > max_lw:
            continue
        out[labels == i] = 255
    return out


def lane_masks(image, p=None, roi=None):
    """
    raw BGR → (white, yellow, bev, (y0, y1)). 차선 검출 입력.
    p:   DEFAULTS 일부를 덮어쓰는 dict (튜너 전용). None 이면 확정값 그대로.
    roi: (top_ratio, bottom_ratio) 로 ROI 세로 범위 변경. None 이면 확정 기하.
         ★ 바꾸면 BEV 픽셀 스케일이 달라져 lane_detection3 의 LANE_WIDTH_PX 와
           pairing/윈도우 임계가 전부 무효가 된다. 자체 오프셋을 따로 캘리브레이션
           하는 독립 스택(yellow_tracer.py) 전용이며, 주 차선 파이프라인은 None 유지.
    """
    q = DEFAULTS if p is None else {**DEFAULTS, **p}
    rt, rb = (ROI_TOP_RATIO, ROI_BOTTOM_RATIO) if roi is None else roi
    img = cv2.resize(image, (SIM_W, SIM_H))
    h = img.shape[0]
    y0, y1 = int(h * rt), int(h * rb)
    roi_img = img[y0:y1, :]
    rh, rw = roi_img.shape[:2]
    _lane_matrices(rh, rw)
    bev = cv2.warpPerspective(roi_img, _M, (rw, rh))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)

    # 흰색: 가로/세로 tophat max + 색조건
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    L, S = hls[:, :, 1], hls[:, :, 2]
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, q['tophat_w'] | 1), 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, q['tophat_v'] | 1)))
    tophat = cv2.max(cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k_h),
                     cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k_v))
    _, m_top = cv2.threshold(tophat, q['tophat_th'], 255, cv2.THRESH_BINARY)
    m_col = ((L >= q['l_min']) & (S <= q['s_max'])).astype(np.uint8) * 255
    white = cv2.bitwise_and(m_top, m_col)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kc)
    white = _filter_white(white, q['min_len'], q['max_lw'], q['orient_tol'])

    # 노란색: LAB b* + HSV 3중 게이트
    lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB); b_ch = lab[:, :, 2]
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    H, Sv, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow_raw = (((b_ch >= q['lab_b']) & (H >= 10) & (H <= 50) &
                   (Sv >= q['yel_s']) & (V >= q['yel_v'])).astype(np.uint8) * 255)
    yellow_raw = cv2.morphologyEx(yellow_raw, cv2.MORPH_CLOSE, kc)
    yellow = _filter_yellow(yellow_raw, q['max_lw'])

    # 흰색에서 노란 영역 제거 (필터 통과분만 사용 — 반사성 노란 오검출이
    # 흰 차선을 잘라내 실선을 2등분하던 것 방지)
    r = int(q['yel_dil'])
    if r > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
        white = cv2.bitwise_and(white, cv2.bitwise_not(cv2.dilate(yellow, kd)))

    return white, yellow, bev, (y0, y1)


# def marks_bev(image):
#     """raw BGR → 노면 마크용 BEV (레거시 기하 보존)."""
#     roi = image[int(image.shape[0] * MARKS_ROI_TOP_RATIO):, :]
#     rh, rw = roi.shape[:2]
#     if _M_marks is None:
#         _marks_matrix(rh, rw)
#     return cv2.warpPerspective(roi, _M_marks, (rw, rh))


def process(image):
    """
    raw BGR → 노드가 실제로 쓰는 값만 반환.
    (기존 파일 TODO '반환값 필요한 값만 가게끔 수정' 반영 — roi/combined 제거)

    Returns: dict {'white', 'yellow', 'bev', 'bev_marks', 'roi_span', 'M_inv'}
      white/yellow : 차선 검출 입력 (BEV 이진 마스크)
      bev          : 차선용 BEV 컬러 (시각화/역투영용)
      bev_marks    : 노면 마크용 BEV 컬러 (레거시 기하)
    """
    white, yellow, bev, roi_span = lane_masks(image)
    return {
        'white':     white,
        'yellow':    yellow,
        'bev':       bev,
        # 'bev_marks': marks_bev(image),
        'roi_span':  roi_span,
        'M_inv':     _Minv,
    }
