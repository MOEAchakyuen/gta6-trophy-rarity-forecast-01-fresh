# gta6-trophy-rarity-forecast-01-fresh

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MOEAchakyuen/gta6-trophy-rarity-forecast-01-fresh/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MOEAchakyuen/gta6-trophy-rarity-forecast-01-fresh/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A standard-library-only Python forecaster for Grand Theft Auto VI trophy earn rates on Exophase: it thinks in counts rather than rates, fitting how each trophy's earners and the tracked-player base each grow on a log-time scale, shrinking each trophy's drift toward its tier's typical drift, tempering the median with a refresh-dilution prior (Exophase refreshes hours apart and each refresh dilutes rates), and spreading five quantiles with a logit-normal band built from refresh, trend and fit-residual uncertainty, asymmetric near 0% and 100%.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=a2v1dcqy&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
