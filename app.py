from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ml_med.features import FEATURE_COLUMNS, build_features
from ml_med.modeling import BOOLEAN_COLUMNS

MODELS_DIR = Path("models")

CONSCIOUSNESS = {"Fully alert": "F", "Drowsy": "D", "Unconscious": "U"}
STROKE_TYPES = {
    "Partial anterior circulation (PACS)": "PACS",
    "Total anterior circulation (TACS)": "TACS",
    "Lacunar (LACS)": "LACS",
    "Posterior circulation (POCS)": "POCS",
    "Other": "OTH",
}
DEFICITS = {
    "RDEF1": "Face deficit",
    "RDEF2": "Arm or hand deficit",
    "RDEF3": "Leg or foot deficit",
    "RDEF4": "Dysphasia",
    "RDEF5": "Hemianopia",
    "RDEF6": "Visuospatial disorder",
    "RDEF7": "Brainstem or cerebellar signs",
    "RDEF8": "Other deficit",
}
DEFICIT_ANSWERS = {"No": "N", "Yes": "Y", "Cannot assess": "C"}


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "risk_model.joblib")


model = load_model()

st.title("Stroke outcome explorer")
st.write("Risk of death or dependency six months after an acute ischaemic stroke.")

# --- sidebar: the patient ---
st.sidebar.header("Patient")
age = st.sidebar.slider("Age", 16, 99, 73)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
delay = st.sidebar.slider("Hours since stroke onset", 1, 48, 19)
sbp = st.sidebar.slider("Systolic blood pressure (mmHg)", 70, 295, 160)
consciousness = st.sidebar.selectbox("Consciousness", list(CONSCIOUSNESS))
stroke_type = st.sidebar.selectbox("Stroke type", list(STROKE_TYPES))
on_waking = st.sidebar.checkbox("Symptoms noted on waking")
atrial = st.sidebar.checkbox("Atrial fibrillation")
prior_aspirin = st.sidebar.checkbox("Aspirin in the previous 3 days")
prior_heparin = st.sidebar.checkbox("Heparin in the previous 24 hours")
ct_done = st.sidebar.checkbox("CT scan done", value=True)
infarct_visible = st.sidebar.checkbox("Infarct visible on CT", disabled=not ct_done)

st.sidebar.header("Neurological deficits")
deficits = {}
for column, label in DEFICITS.items():
    default = 1 if column in ("RDEF1", "RDEF2", "RDEF3") else 0
    answer = st.sidebar.selectbox(label, list(DEFICIT_ANSWERS), index=default)
    deficits[column] = DEFICIT_ANSWERS[answer]

# --- from the widgets to one row of model inputs ---
raw = {
    "AGE": age,
    "SEX": sex,
    "RDELAY": delay,
    "RSBP": sbp,
    "RCONSC": CONSCIOUSNESS[consciousness],
    "STYPE": STROKE_TYPES[stroke_type],
    "RSLEEP": on_waking,
    "RATRIAL": atrial,
    "RASP3": prior_aspirin,
    "RHEP24": prior_heparin,
    "RCT": ct_done,
    "RVISINF": ct_done and infarct_visible,
    "pilot_phase": False,
    **deficits,
}
patient = build_features(pd.DataFrame([raw]))
patient[BOOLEAN_COLUMNS] = patient[BOOLEAN_COLUMNS].astype("boolean")
X = patient[FEATURE_COLUMNS].assign(RXHEP="N", RXASP=0)

risk = model.predict_proba(X)[0, 1]
st.metric("Risk of death or dependency at 6 months, without treatment", f"{risk:.0%}")

# --- coherence check: was a patient like this ever seen in the trial? ---
counts = pd.read_csv(MODELS_DIR / "stroke_type_by_deficits.csv", index_col=0)
n_deficits = int(patient["n_deficits"].iloc[0])
similar = int(counts.loc[STROKE_TYPES[stroke_type], str(n_deficits)])
if similar == 0:
    st.warning(
        f"No patient in the trial had this stroke type with {n_deficits} deficits. "
        "The risk above is an extrapolation: check the stroke type and the deficits."
    )
else:
    st.caption(
        f"{similar} of the {int(counts.to_numpy().sum())} trial patients had this "
        f"stroke type with {n_deficits} deficits."
    )

# --- average effects of the treatments ---
st.header("What the treatments do on average")
st.write(
    "Events per 1000 patients in the trial. A negative difference means events "
    "avoided by the treatment, a positive one events caused by it."
)
for treatment, name in (("RXASP", "Aspirin"), ("RXHEP", "Heparin")):
    effects = pd.read_csv(MODELS_DIR / f"effects_{treatment}.csv", index_col=0)
    effects = effects.rename(
        columns={
            "control": "Without treatment",
            "treated": "With treatment",
            "difference": "Difference",
            "se": "Standard error",
        }
    )
    st.subheader(name)
    st.dataframe(effects.round(1))

st.info(
    "These effects are averages over all patients. The analysis found no evidence "
    "that they differ from one patient to another, so they are not personalised."
)
st.caption(
    "Educational tool built on the International Stroke Trial data (1991-1996). "
    "Not for clinical use."
)
