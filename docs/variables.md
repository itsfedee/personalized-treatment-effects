# Variables used in the analysis

## Baseline variables, treatment and identifiers

| Variable | Meaning | Role | Expected values | Notes |
|---|---|---|---|---|
| `HOSPNUM` | Hospital number | identifier | integer | not a covariate |
| `RDELAY` | Hours from stroke onset to randomisation | covariate | 0–48 | inclusion criterion: onset within 48 h |
| `RCONSC` | Conscious state | covariate | F / D / U | F = fully alert, D = drowsy, U = unconscious |
| `SEX` | Sex | covariate | M / F | |
| `AGE` | Age in years | covariate | integer | no upper age limit in the trial |
| `RSLEEP` | Symptoms noted on waking | covariate | Y / N | |
| `RATRIAL` | Atrial fibrillation | covariate | Y / N | not recorded in the pilot phase (984 patients) |
| `RCT` | CT scan before randomisation | covariate | Y / N | |
| `RVISINF` | Infarct visible on CT | covariate | Y / N | meaningful only if `RCT` = Y |
| `RHEP24` | Heparin in the 24 h before randomisation | covariate | Y / N | not the trial heparin |
| `RASP3` | Aspirin in the 3 days before randomisation | covariate | Y / N | not the trial aspirin (≠ `RXASP`, `DASP14`) |
| `RSBP` | Systolic blood pressure | covariate | mmHg | plausible range roughly 60–300 |
| `RDEF1`–`RDEF8` | Neurological deficits: face, arm/hand, leg/foot, dysphasia, hemianopia, visuospatial, brainstem/cerebellar, other | covariate | Y / N / C | C = can't assess (often unconscious patients): not a plain missing value |
| `STYPE` | Stroke subtype (Oxford classification) | covariate | TACS / PACS / POCS / LACS / OTH | largely derived from the deficits |
| `RXASP` | Trial aspirin allocated | treatment | Y / N | |
| `RXHEP` | Trial heparin allocated | treatment | N / L / M | M is coded as H in the pilot phase → recode H → M |

## Outcomes

### Primary outcome
Death or dependency at 6 months, derived from `OCCODE`:
1 or 2 → event; 3 or 4 → no event; 8 or 9 → missing.

### Secondary outcomes
- Death within 14 days: `ID14`.
- Death at 6 months: from `OCCODE` (value 1) or from `FDEAD`. To check that the two sources agree.

### Benefit and safety outcomes within 14 days (benefit–risk balance)

| Type | Outcome | Variable |
|---|---|---|
| expected benefit | recurrent ischaemic stroke | `ISC14` |
| expected benefit | pulmonary embolism | `PE14` |
| expected harm | cerebral haemorrhage | `H14` |
| expected harm | major non-cerebral haemorrhage | `TRAN14` |