"""GTA VI trophy rarity forecaster (Python 3.13, standard library only).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5/25/50/75/95%) of its Exophase earn rate at the case target.

Idea, in one paragraph: think in counts, not rates.  Earners only grow as
players progress; tracked players only grow as more players sync.  A rate
rises when earners outgrow the tracked base and falls when the base outgrows
earners, so the model fits both counts on a log-time scale, divides the two
extrapolations for a counts-based median, fits the logit's own trend as a
second median, blends the two in logit space, shrinks each trophy's drift
toward its tier's typical drift, tempers the median with a refresh-dilution
prior (Exophase refreshes hours apart and each refresh dilutes rates), and
spreads the five quantiles with a logit-normal band built from refresh, trend
and fit-residual uncertainty, asymmetric near 0% and 100%.

Reads every *.in in MAKEFUN_INPUT_DIR (else /input), writes one *.out per
case to MAKEFUN_OUTPUT_DIR (else /output).  Fully deterministic: no random
numbers, no wall clock, rows sorted by trophy id.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Hours since the start of release day, 2026-11-19T00:00:00Z.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Spread bounds, in logit units, and the random-walk trend scale.
SIGMA_MIN = 0.6
SIGMA_MAX = 2.0
TREND_SIGMA = 0.45
# Logit move a median may make from the last rate, and the refresh prior
# used when the history shows no refresh jump at all.
MAX_LOGIT_MOVE = 3.0
DEFAULT_REFRESH_FALL = 1.0
PLATINUM_CAP = 3.0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(text):
    return (parse_time(text) - EPOCH).total_seconds() / 3600.0


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value > 30.0:
        return 100.0
    if value < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-value))


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if out == int(out) else out


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b*x; returns (intercept, slope)."""
    sw = sum(ws)
    if sw <= 0.0 or len(xs) < 2:
        return (ys[0] if ys else 0.0), 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    num = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    den = sum(w * (x - mx) * (x - mx) for w, x in zip(ws, xs))
    if den <= 1e-12:
        return my, 0.0
    slope = num / den
    return my - slope * mx, slope


def distinct_reads(history):
    """Collapse the staircase: drop consecutive reads identical in tracked
    players and in every rate, keeping the first and last of each plateau."""
    plateaus = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if plateaus and plateaus[-1][0] == key:
            plateaus[-1][2] = read
        else:
            plateaus.append([key, read, read])
    out = []
    for _, first, last in plateaus:
        out.append(first)
        if last is not first:
            out.append(last)
    return out


def last_known(rates, upto):
    """Last non-null rate at or before index `upto` (scanning backwards)."""
    for i in range(upto, -1, -1):
        if rates[i] is not None:
            return rates[i]
    return None


def estimate_refresh_fall(reads, n_trophies):
    """Typical logit fall of a rate when tracked players jump at a refresh.
    Returns (fall, jumps_seen).  Falls back to the prior when no jump shows."""
    falls = []
    jumps = 0
    for prev, cur in zip(reads, reads[1:]):
        n0 = prev.get("players_tracked") or 0
        n1 = cur.get("players_tracked") or 0
        if n0 <= 0 or n1 <= n0 * 1.05:
            continue
        jumps += 1
        r0 = prev.get("rates") or []
        r1 = cur.get("rates") or []
        for i in range(min(n_trophies, len(r0), len(r1))):
            if r0[i] is None or r1[i] is None:
                continue
            drop = logit(r0[i]) - logit(r1[i])
            if drop > 0.0:
                falls.append(drop)
    if falls:
        falls.sort()
        return falls[len(falls) // 2], jumps
    return DEFAULT_REFRESH_FALL, 0


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = distinct_reads(case.get("history") or [])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target_x = math.log1p(max(0.0, hours_since_epoch(case["target"])))

    if not history:
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.1, 0.5, 2.0, 5.0]}
                               for t in sorted(trophies, key=lambda t: t["id"])]}

    xs = [math.log1p(max(0.0, hours_since_epoch(r["at"]))) for r in history]
    # Recent reads count much more than old ones: the launch-night plateau
    # and the first refresh dip are stale next to the latest trend, so
    # weights decay exponentially back from the last read in log-time.
    x_last = xs[-1]
    ws = [0.3 + 1.7 * math.exp(x - x_last) for x in xs]

    tracked = [max(1, r.get("players_tracked") or 1) for r in history]
    log_n = [math.log(v) for v in tracked]
    _a_n, slope_n = wls_line(xs, log_n, ws)
    slope_n = max(0.0, slope_n)
    mx_n = sum(w * x for w, x in zip(ws, xs)) / sum(ws)
    my_n = sum(w * y for w, y in zip(ws, log_n)) / sum(ws)
    n_target = math.exp(my_n + slope_n * (target_x - mx_n))
    n_target = max(n_target, tracked[-1])

    fall, _jumps = estimate_refresh_fall(history, n)
    # Chance at least one refresh lands inside the horizon (mean gap ~8h).
    p_refresh = 1.0 - math.exp(-horizon_hours / 8.0)

    # Per-trophy fits, then tier-median drifts for shrinkage.
    per_trophy = []
    tier_slopes_e = {}
    tier_slopes_l = {}
    for i, trophy in enumerate(trophies):
        rates = []
        for r in history:
            rs = r.get("rates") or []
            rates.append(rs[i] if i < len(rs) else None)
        last = last_known(rates, len(rates) - 1)
        if last is None:
            last = 0.1
        earners = []
        use_x = []
        use_w = []
        use_logit = []
        for j, rate in enumerate(rates):
            if rate is None:
                continue
            base = tracked[j]
            earners.append(max(0.0, rate / 100.0 * base))
            use_x.append(xs[j])
            use_w.append(ws[j])
            use_logit.append(logit(rate))
        if len(use_x) >= 2:
            a_e, b_e = wls_line(use_x, [math.log(e + 0.5) for e in earners], use_w)
            a_l, b_l = wls_line(use_x, use_logit, use_w)
            resid = [y - (a_l + b_l * x) for x, y in zip(use_x, use_logit)]
            fit_sig = math.sqrt(sum(v * v for v in resid) / len(resid))
            move = max(math.log(e + 0.5) for e in earners) - min(
                math.log(e + 0.5) for e in earners)
            # Weighted means, so a shrunk slope pivots at the data's centre
            # instead of at x = 0 (which would drag the whole line down).
            sw = sum(use_w)
            mx = sum(w * x for w, x in zip(use_w, use_x)) / sw
            my_e = sum(w * math.log(e + 0.5) for w, e in zip(use_w, earners)) / sw
            my_l = sum(w * y for w, y in zip(use_w, use_logit)) / sw
        else:
            # Fewer than two usable reads: flat drift from the last earners.
            last_earn = max(0.0, last / 100.0 * tracked[-1])
            a_e, b_e, a_l, b_l, fit_sig, move = (
                math.log(last_earn + 0.5), 0.0, logit(last), 0.0, 0.0, 0.0)
            mx, my_e, my_l = (use_x[0] if use_x else 0.0), a_e, a_l
        per_trophy.append({
            "trophy": trophy, "last": last,
            "last_earn": max(0.0, last / 100.0 * tracked[-1]),
            "b_e": b_e, "b_l": b_l, "fit_sig": fit_sig, "move": move,
            "n_pts": len(use_x), "x_last": use_x[-1] if use_x else xs[-1],
        })
        tier_slopes_e.setdefault(trophy["tier"], []).append(b_e)
        tier_slopes_l.setdefault(trophy["tier"], []).append(b_l)

    def median_of(values):
        s = sorted(values)
        m = len(s)
        return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])

    tier_med_e = {t: median_of(v) for t, v in tier_slopes_e.items()}
    tier_med_l = {t: median_of(v) for t, v in tier_slopes_l.items()}

    rows = []
    for item in per_trophy:
        trophy = item["trophy"]
        last = item["last"]
        base_logit = logit(last)
        # Informativeness: trophies whose earners barely moved, or with few
        # distinct reads, lean on their tier's typical drift instead.
        trust = (item["move"] / (item["move"] + 0.7)) * (
            item["n_pts"] / (item["n_pts"] + 3.0))
        b_e = trust * item["b_e"] + (1.0 - trust) * tier_med_e[trophy["tier"]]
        b_l = trust * item["b_l"] + (1.0 - trust) * tier_med_l[trophy["tier"]]
        # A trophy nobody has reached yet still gets a share of its tier's
        # upward drift rather than a forced flat future.
        if last < 1.0 and tier_med_e[trophy["tier"]] > 0.0:
            b_e = max(b_e, 0.5 * tier_med_e[trophy["tier"]])
            b_l = max(b_l, 0.5 * tier_med_l[trophy["tier"]])

        # Counts median: predicted earners over predicted tracked players.
        # The fitted (shrunk) slopes drive the drift, but the drift starts
        # from the last observed level: a line fitted over the whole history
        # sits below the latest reads after a flat plateau and a refresh dip,
        # so reading the line itself would understate where the trophy is now.
        step = target_x - item["x_last"]
        earn_pred = max(item["last_earn"], 0.05) * math.exp(b_e * step)
        rate_counts = 100.0 * earn_pred / n_target
        # Logit-trend median, drifted from the last rate the same way.
        rate_logit = inv_logit(base_logit + b_l * step)
        # Blend the two medians in logit space.
        med_logit = 0.5 * logit(rate_counts) + 0.5 * logit(rate_logit)
        # Refresh dilution tempers the median: several refreshes dilute
        # rates over a long horizon while progression pushes them up.
        med_logit -= 0.3 * fall * p_refresh
        # Guardrails: cap the total logit move, keep near-100% trophies high,
        # keep the platinum low in week one.
        med_logit = min(max(med_logit, base_logit - MAX_LOGIT_MOVE),
                        base_logit + MAX_LOGIT_MOVE)
        if last >= 99.5:
            med_logit = max(med_logit, logit(99.0))
        if trophy["id"] == 1:
            med_logit = min(med_logit, logit(PLATINUM_CAP))
        median_rate = inv_logit(med_logit)

        # Spread: refresh + trend (grows with sqrt horizon) + fit residual.
        if item["n_pts"] < 3:
            sigma = 1.2
        else:
            refresh_sig = 0.7 * fall * math.sqrt(p_refresh)
            trend_sig = TREND_SIGMA * math.sqrt(horizon_days)
            fit_sig = min(0.8, 1.5 * item["fit_sig"])
            sigma = math.sqrt(refresh_sig ** 2 + trend_sig ** 2 + fit_sig ** 2)
            sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))

        # Asymmetric tails near the boundaries: the long tail reaches
        # further, the short tail less far, keeping the total width sane.
        # The saturated trophy is special: a refresh could in principle
        # bring it down a long way, so its lower tail reaches far below.
        if last >= 99.99:
            low_f, high_f = 3.5, 0.5
        elif median_rate >= 90.0:
            low_f, high_f = 1.3, 0.7
        elif median_rate <= 1.0:
            low_f, high_f = 0.7, 1.3
        else:
            low_f, high_f = 1.0, 1.0

        qs = []
        for z in Z:
            factor = low_f if z < 0 else high_f
            qs.append(inv_logit(med_logit + z * sigma * factor))
        qs = [round4(min(99.99, max(0.0, q))) for q in qs]
        # The genuinely saturated trophy may print 100 at the top.
        if last >= 99.99:
            qs[-1] = 100
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never crash a case: fall back to the last rates, wide band.
            rows = []
            last = case["history"][-1] if case.get("history") else None
            for i, trophy in enumerate(case["trophies"]):
                rate = 0.5
                if last and last.get("rates") and i < len(last["rates"]) \
                        and last["rates"][i] is not None:
                    rate = last["rates"][i]
                lg = logit(rate)
                rows.append({"id": trophy["id"], "q": [
                    round4(inv_logit(lg - 1.645)), round4(inv_logit(lg - 0.674)),
                    round4(rate), round4(inv_logit(lg + 0.674)),
                    round4(inv_logit(lg + 1.645))]})
            forecast = {"trophies": sorted(rows, key=lambda r: r["id"])}
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
