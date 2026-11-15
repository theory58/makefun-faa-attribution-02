"""polymarket-faa-attribution-02 entry, v2: a logit-space quantile
forecaster for Polymarket Yes prices, one case at a time.

The grader runs this program once per case (python3 /bot/main.py) in a
sandbox with no network: /input holds that case's file, whose markets
carry one Yes price per minute up to the checkpoint, and /output takes
one forecast per case: for every market, 240 rows of five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of the price at each of the 240 minutes
after the checkpoint. Scoring is the mean pinball loss, so each row
should be the honest quantiles of that minute's price distribution.
Standard library only, no randomness: two runs write identical bytes.

The model, per market, works in logit space, x = log(p / (1 - p)), so
the [0, 1] bounds hold by construction:

- The median is the last logit price, unchanged. A winsorised-EWMA
  denoised endpoint was tried and rejected on the practice cases: it
  lags genuine moves and costs more than the bounce it removes.
- One-minute volatility sigma1 is jump-robust: the mean of a bipower
  variation estimate and a MAD estimate over the last day of logit
  returns, so a few jump minutes do not inflate the whole spread.
- The term structure of h-minute changes is sub-diffusive: pooled
  standardised residuals of h-step logit changes across anchor
  horizons, normalised by sigma1 * h**0.45 over the last 360 minutes,
  give empirical quantile spreads; they are blended 40/60 with
  Student-t(5) quantiles (fat tails, stabler than raw empirical tails
  from a short window) and shrunk by 0.90.
- A market pinned at a bound (last price >= 0.995 or <= 0.005) with
  near-zero recent variance is treated as settled: its spread is
  compressed to 40%.
- Rows between anchor horizons interpolate in sqrt(h); quantiles are
  sigmoid(median + dev), non-decreasing across levels by construction,
  with the outer levels floored at z * 0.010 * sqrt(h) of logit room.

Pinball loss scores marginal quantiles only, so no cross-market
correlation term enters: each market is forecast on its own history.
Practice (6 house cases, local replica of the scorer): mean pinball
loss 0.278 against the house baseline's 0.364 and this entrant's
previous entry's 0.286.
"""

import json
import math
import os
from pathlib import Path

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = {0.05: -1.644854, 0.25: -0.674490, 0.5: 0.0, 0.75: 0.674490, 0.95: 1.644854}
T5 = {0.05: -1.5605, 0.25: -0.5598, 0.5: 0.0, 0.75: 0.5598, 0.95: 1.5605}
ANCHORS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80,
           100, 128, 160, 200, 240)
EPS = 1e-4
WINDOW = 360
SCALE = 0.90
T_BLEND = 0.6
HEXP = 0.45
FLOOR_L = 0.010
PIN_FACTOR = 0.4


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def quantile(sorted_vals, tau):
    if not sorted_vals:
        return 0.0
    pos = tau * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def price6(v):
    return round(min(max(v, 0.0), 1.0), 6)


def forecast_market(prices, horizon):
    xs = [logit(p) for p in prices]
    n = len(xs)
    rets = [xs[i] - xs[i - 1] for i in range(1, n)]
    seg = rets[-1440:]
    if len(seg) >= 3:
        bv = (math.pi / 2.0) * sum(abs(seg[i]) * abs(seg[i - 1])
                                   for i in range(1, len(seg))) / (len(seg) - 1)
        s_bv = math.sqrt(max(bv, 1e-12))
        srt = sorted(seg)
        med = srt[len(srt) // 2]
        mad = sorted(abs(v - med) for v in seg)[len(seg) // 2]
        sigma1 = 0.5 * (s_bv + 1.4826 * mad)
    elif seg:
        sigma1 = max(abs(seg[-1]), 1e-3)
    else:
        sigma1 = 1e-3
    sigma1 = max(sigma1, 1e-4)

    start = max(0, n - 1 - WINDOW)
    pooled = []
    for h in ANCHORS:
        if h >= n:
            continue
        sh = sigma1 * (h ** HEXP)
        for t in range(max(start, h), n):
            pooled.append((xs[t] - xs[t - h]) / sh)
    pooled.sort()
    eq = {tau: quantile(pooled, tau) for tau in TAUS}
    med_e = eq[0.5]

    pinned = (prices[-1] >= 0.995 or prices[-1] <= 0.005) and sigma1 < 0.01
    x_hat = xs[-1]

    def dev_at(tau, h):
        emp = (eq[tau] - med_e) * sigma1 * (h ** HEXP)
        par = T5[tau] * sigma1 * (h ** HEXP)
        dev = ((1.0 - T_BLEND) * emp + T_BLEND * par) * SCALE
        if tau in (0.05, 0.95):
            floor = Z[tau] * FLOOR_L * math.sqrt(h)
            dev = min(dev, floor) if tau == 0.05 else max(dev, floor)
        if pinned:
            dev *= PIN_FACTOR
        return dev

    def row(h):
        return [price6(sigmoid(x_hat + dev_at(tau, h))) for tau in TAUS]

    anchor_rows = {h: row(h) for h in ANCHORS if h <= horizon}
    rows = []
    ai = 0
    for h in range(1, horizon + 1):
        if h in anchor_rows:
            rows.append(anchor_rows[h])
            continue
        while ai + 1 < len(ANCHORS) and ANCHORS[ai + 1] < h:
            ai += 1
        h0, h1 = ANCHORS[ai], ANCHORS[ai + 1]
        r0, r1 = anchor_rows[h0], anchor_rows[h1]
        w = (math.sqrt(h) - math.sqrt(h0)) / (math.sqrt(h1) - math.sqrt(h0))
        rows.append([price6(r0[k] + (r1[k] - r0[k]) * w) for k in range(5)])
    return rows


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if tuple(case["quantiles"]) != TAUS:
            raise ValueError(f"{path.name} asks for quantiles "
                             f"{case['quantiles']}, not {list(TAUS)}")
        forecasts = [{"id": m["id"],
                      "q": forecast_market(m["prices"],
                                           case["horizon_minutes"])}
                     for m in case["markets"]]
        out = json.dumps({"forecasts": forecasts}) + "\n"
        (output_dir / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
