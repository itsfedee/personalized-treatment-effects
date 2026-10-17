# Variables used in the analysis

## Variable overview (some of the variables in the dataset)

| Variable | Meaning | Role | Expected values | Notes |
|---|---|---|---|---|
| `HOSPNUM` | Hospital number | identifier | integer | may be used to group patients by hospital |
| `RDELAY` | Hours from stroke onset to randomisation | covariate | 0–48 | inclusion criterion: onset within 48 h |
| `RCONSC` | Conscious state | covariate | F / D / U | F = fully alert, D = drowsy, U = unconscious |
| `SEX` | Sex | covariate | M / F | |
| `AGE` | Age in years | covariate | integer | no upper age limit in the trial |
| `RSLEEP` | Symptoms noted on waking | covariate | Y / N | |
| `RATRIAL` | Atrial fibrillation | covariate | Y / N | not recorded in the pilot phase (984 patients) |
| `RCT` | CT scan before randomisation | covariate | Y / N | |
| `RVISINF` | Infarct visible on CT | covariate | Y / N | meaningful only if `RCT` = Y. Coded N when no CT was done (6,410 patients). 1 patient has RCT = N and RVISINF = Y (impossible): likely a recording error. |
| `RHEP24` | Heparin in the 24 h before randomisation | covariate | Y / N | not the trial. 344 missing, all in the pilot phase (subset of the 984 without RATRIAL): probably added partway through the pilot (to confirm with RDATE). 344 missing: all patients randomised Jan 1991 – Feb 1992 (question added to the form in Feb 1992).|
| `RASP3` | Aspirin in the 3 days before randomisation | covariate | Y / N | not the trial aspirin (≠ `RXASP`, `DASP14`). 984 missing, the same patients as RATRIAL: probably not recorded in the pilot phase (to confirm with RDATE) |
| `RSBP` | Systolic blood pressure | covariate | mmHg | plausible range roughly 60–300 |
| `RDEF1`–`RDEF8` | Neurological deficits: face, arm/hand, leg/foot, dysphasia, hemianopia, visuospatial, brainstem/cerebellar, other | covariate | Y / N / C | C = can't assess (often unconscious patients): not a plain missing value |
| `STYPE` | Stroke subtype (Oxford classification) | covariate | TACS / PACS / POCS / LACS / OTH | largely derived from the deficits |
| `RXASP` | Trial aspirin allocated | treatment | Y / N | |
| `RXHEP` | Trial heparin allocated | treatment | N / L / M | M is coded as H in the pilot phase → recode H → M. H appears only in the pilot phase (245 patients) and M only in the main phase: recode H → M." |
| `OCCODE` | Six-month outcome | outcome | 1 / 2 / 3 / 4 / 8 / 9 | 1 = dead, 2 = dependent, 3 = not recovered, 4 = recovered, 8–9 = missing; primary outcome: 1–2 → event, 3–4 → no event. Value 0 (97 patients) is not in the dictionary; all have FDEAD missing → treat 0 as missing, like 8 and 9 |
| `ID14` | Death within 14 days | outcome | 0 / 1 | derived by the authors |
| `FDEAD` | Dead at the time of the 6-month follow-up contact | auxiliary | Y / N / U (+ missing) | U = unknown (not in the dictionary; 53 of 57 have `OCCODE` = 9); may include deaths after 6 months (128 patients have `OCCODE` = 2 and `FDEAD` = Y) → use `OCCODE` for 6-month status |
| `ISC14` | Recurrent ischaemic stroke within 14 days | outcome | 0 / 1 | expected benefit |
| `PE14` | Pulmonary embolism within 14 days | outcome | 0 / 1 | expected benefit |
| `H14` | Cerebral haemorrhage within 14 days | outcome | 0 / 1 | expected harm; wider definition than `DRSH` |
| `TRAN14` | Major non-cerebral haemorrhage within 14 days | outcome | 0 / 1 | expected harm |
| `RDATE` | Date of randomisation | auxiliary | month-year, e.g. `paź-91` | month abbreviations in Polish; useful to identify the pilot phase |
| `SET14D` | Known to be dead or alive at 14 days | auxiliary | 0 / 1 | needed to interpret `ID14` |
| `COUNTRY` | Country code | auxiliary | abbreviated code | candidate covariate, to evaluate separately |

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

Note: Pilot phase: Jan 1991 – early Mar 1993 (984 patients, identified by RATRIAL missing; 1 pilot patient in Mar 1993). Main phase from Mar 1993.

## Cleaning decisions

- `RXHEP`: recode H → M (H is the pilot-phase code for the medium dose).
- `OCCODE`: treat 0, 8 and 9 as missing.
- `FDEAD`: not used for the 6-month status; use `OCCODE` instead.
- Create `patient_id` (row number in the original file).
- Create `pilot_phase` = `RATRIAL` is missing.
- `RVISINF`: 1 patient with RCT = N and RVISINF = Y → decision to take.
- `RDEF1`–`RDEF8`: value C (can't assess) → decision to take.