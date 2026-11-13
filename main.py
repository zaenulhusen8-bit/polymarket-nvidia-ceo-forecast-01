#!/usr/bin/env python3
"""Forecast Polymarket 'Next CEO of NVIDIA?' Yes prices, 240 minutes ahead.

For each market in a case file, emit five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) of the price at every minute after the checkpoint.

Model (v3, tuned on the kit's practice cases by local backtest):
these prices move like a step process -- flat for many minutes, then a tick
or a jump -- so neither a fixed normal band nor an RMS-calibrated one is
right (RMS calibration scored about twice the house baseline's loss).  Take
the empirical distribution of h-minute price changes from the market's own
last 480 minutes: at each anchor horizon, the five quantiles of
(price[i+h] - price[i]) over the recorded minutes, interpolated linearly in
sqrt(h) between anchors.  Three corrections, each of which lowered the
practice pinball loss:

* recenter: shift the band so its median sits on the last price --
  persistence is the right centre; the empirical median only adds the
  drift of whatever regime the history happened to cross.
* shrink: scale the recentered deviations by 0.95.
* blend: mix the result 75/25 with the house baseline's normal band
  (sigma 0.0025 per sqrt-minute), which steadies markets whose recent
  window is quiet or short.

Short histories with no usable anchor fall back to the baseline band alone.
On practice cases p00..p04 (answers recovered from the next case's history)
this scores a mean pinball loss of 0.332052 against the baseline's 0.392518
on the same cases.  Deterministic, standard library only.
"""

import json
import math
import os
import sys

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
BASE_SIGMA = 0.0025
ANCHORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 240)
MAX_SAMPLES = 800      # cap on sampled windows per anchor (keeps runs fast)
MIN_SAMPLES = 20       # below this an anchor is not trusted on its own
WINDOW = 480           # minutes of history the quantiles are estimated from
SHRINK = 0.95          # scale on the empirical deviations
BLEND = 0.25           # weight of the baseline normal band in the final mix


def clamp(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def anchor_quantiles(prices, h):
    """Five quantiles of the h-minute change, or None if too few windows."""
    span = len(prices) - h
    if span < MIN_SAMPLES:
        return None
    stride = max(1, span // MAX_SAMPLES)
    changes = sorted(prices[i + h] - prices[i] for i in range(0, span, stride))
    return [quantile(changes, tau) for tau in TAUS]


def forecast_market(prices, horizon):
    x = prices[-1]
    prices = prices[-WINDOW:]
    anchors = []
    for h in ANCHORS:
        if h > horizon:
            break
        q = anchor_quantiles(prices, h)
        if q is not None:
            anchors.append((h, q))
    rows = []
    ai = 0
    for t in range(1, horizon + 1):
        base = [z * BASE_SIGMA * math.sqrt(t) for z in Z]
        if not anchors:
            rows.append([round(clamp(x + v), 6) for v in base])
            continue
        while ai + 1 < len(anchors) and anchors[ai + 1][0] <= t:
            ai += 1
        h0, q0 = anchors[ai]
        if t == h0 or ai + 1 >= len(anchors):
            qs = q0 if t == h0 else [v * math.sqrt(t / h0) for v in q0]
        else:
            h1, q1 = anchors[ai + 1]
            w = (math.sqrt(t) - math.sqrt(h0)) / (math.sqrt(h1) - math.sqrt(h0))
            qs = [a * (1.0 - w) + b * w for a, b in zip(q0, q1)]
        med = qs[2]
        row = []
        for i in range(5):
            dev = (qs[i] - med) * SHRINK
            row.append(round(clamp(x + (1.0 - BLEND) * dev + BLEND * base[i]), 6))
        for i in range(1, 5):  # keep quantiles non-decreasing after rounding
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def run_case(path, out_dir):
    with open(path, "r", encoding="utf-8") as f:
        case = json.load(f)
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = []
    for market in case["markets"]:
        prices = [float(p) for p in market["prices"]]
        forecasts.append({"id": market["id"], "q": forecast_market(prices, horizon)})
    case_id = case.get("case") or os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, case_id + ".out")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        run_case(os.path.join(in_dir, name), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
