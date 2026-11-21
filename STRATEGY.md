# Strategy

## The data's shape

Exophase refreshes a game's counts on its own schedule, hours apart, not
continuously. Between refreshes the page is frozen; at a refresh, tracked
players jump and almost every rate falls, because the newly tracked players
have earned fewer trophies. A case history is therefore a staircase: long
flat plateaus, then jumps. The final case spans about two and a half days
with a 4.5-day horizon to 2026-11-26T00:00:00Z, so real player progression
over the horizon matters most, with refresh dilution pulling the other way.

## Counts decomposition

Think in counts, not rates. Earners only grow as players progress; tracked
players only grow as more players sync. A rate rises when earners outgrow
the tracked base and falls when the base outgrows earners. For each read we
recover earners as rate times tracked players over 100, and keep the tracked
count itself.

## Collapsing the staircase and log-time fits

Consecutive reads identical in tracked players and in every rate are
collapsed to the first and last of each plateau, so flat hours do not
outweigh real changes. Time is measured in hours since the start of release
day (2026-11-19T00:00:00Z), worked with as log(1 + hours), so early
explosive change and later slower change sit on a comparable scale. Per
trophy, a weighted least-squares line is fitted to log earners against
log-time and, separately, to the logit of the rate; the tracked-player count
gets its own log line. Recent reads weigh much more than old ones (weights
decay exponentially back from the last read in log-time), because the
launch-night plateau and the first refresh dip are stale next to the latest
trend. The two medians — predicted earners over predicted tracked players,
and the logit trend — are blended half and half in logit space.

A subtle point the validation caught: a line fitted over the whole history
sits below the latest reads after a flat plateau and a refresh dip, so
reading the line itself understates where the trophy is now. The fitted
slopes drive the drift, but the drift starts from the last observed level:
earners and the logit are both extrapolated forward from their last values.

## Shrinkage

Each trophy's fitted drift is pulled toward the median drift of trophies of
the same tier, more strongly when the trophy's earners barely moved or when
few distinct reads inform it. A trophy nobody has reached yet has a flat
past but a real future as players arrive, so low-rate trophies get a share
of their tier's typical upward drift rather than a forced flat future.

## Refresh / dilution temper (v2)

The history's refresh jumps show how much rates fall when tracked players
jump, per trophy: at each jump, a trophy's earners grow only as fast as
newly tracked players earn it, so the trophies whose earners lag the
tracked base fall furthest. Each trophy's median signed logit fall across
the jumps it was readable at is its measured fall (a trophy never readable
across a jump borrows its tier's median fall, else the overall one). Where
the history shows no jump at all -- the practice's flat launch night --
there is no measured fall, so a prior applies instead: 1.3 logit units for
the most susceptible trophy, scaled by (1 - rate), so low-rate, slow-earner
trophies are pulled down most and the near-100% trophy and the earliest
trophies barely at all. The temper is the fall weighted by how likely a
refresh is inside the horizon (rising from 0 toward 1 over about two and a
half hours at this stage of the launch), so over a 90-minute horizon the
median sits clearly below the last rate, while over four and a half days
the fitted progression trend dominates and the temper is only a drag on
it. A measured fall tempers by only half, because the fitted trend already
embeds the dilution the history's own refreshes caused; a flat history
embeds no dilution, so the prior applies in full.

## Guardrails

The total logit move from the last rate is capped at 3 units; the predicted
rate is capped below 100 and above a small floor; a trophy already near 100%
never forecasts a fall to ordinary levels, and the platinum (id 1, needs
every other trophy) stays at a few percent at most in week one even if its
earners are growing fast in percentage terms. Early story trophies already
at high rates stay high; mid-story trophies climb the most in absolute
terms; late and hidden trophies climb from near zero but stay modest.

## Spread (v2)

The five quantiles come from a logit-normal spread around the median. The
refresh component is measured from the history's own jumps: the median
absolute logit move at jumps plus half the gap to the 90th percentile of
those moves (floored at 0.7), so the band is sized by what refreshes
actually did to rates. When the history shows no jump, the component does
not fall to zero: a prior floor of 1.5 logit units applies, because a
refresh is likely inside any horizon of a few hours at this stage of the
launch and the launch-night jump shows how large one can be. The refresh
component grows with the horizon (more refreshes fit into a longer one),
and is combined in quadrature with a trend component growing with the
square root of the horizon in days and a fit-uncertainty component from
the residuals of the per-trophy fits (0.5 where too few reads inform a
fit), clamped between 0.8 and 2.8 logit units. The band is asymmetric near
the boundaries: for a median above 90 the lower tail reaches much further
down than the upper tail reaches up, and for a median below about 1 the
upper tail reaches much further up; the saturated 100% trophy gets a median
near 100 but a 5% quantile far below it, because a refresh could in
principle bring it down. Quantiles are rounded to four decimals, forced
non-decreasing, and clamped into 0–100.

## Fallbacks

If a trophy's last rate is missing, its last non-null rate is used; with
fewer than three distinct reads the forecast falls back to the last rate
with a wide spread; if anything about a case is unexpected, the program
still writes a valid forecast for every trophy rather than crashing. The
program is deterministic (no randomness, no wall-clock dependence, rows
sorted by trophy id), standard library only, and runs in about 0.05 seconds
per practice case.
