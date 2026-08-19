import streamlit as st
import pandas as pd
import numpy as np
import requests

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="AI Drug Repurposing",
    page_icon="🧠",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🧠 AI-Based Drug Repurposing")
st.subheader("Alzheimer's Disease")

st.write(
    "Machine-learning-based prediction of potential "
    "drug–target interactions for Alzheimer’s disease."
)

st.info(
    "Enter an approved drug name to identify its "
    "highest-ranked predicted Alzheimer’s-associated target."
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("ml_dataset.csv")

    return df


# ==========================================================
# TRAIN MODEL
# ==========================================================

@st.cache_resource
def train_model(df):

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048
    )

    fingerprints = []
    valid_rows = []

    for index, smiles in df["canonical_smiles"].items():

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            continue

        fp = generator.GetFingerprintAsNumPy(mol)

        fingerprints.append(fp)
        valid_rows.append(index)

    clean_df = df.loc[
        valid_rows
    ].reset_index(drop=True)

    fingerprints = np.array(
        fingerprints
    )

    # Target encoding
    target_names = sorted(
        clean_df["gene"]
        .dropna()
        .unique()
    )

    target_index = {
        target: i
        for i, target in enumerate(target_names)
    }

    target_features = np.zeros(
        (
            len(clean_df),
            len(target_names)
        )
    )

    for i, target in enumerate(
        clean_df["gene"]
    ):

        if target in target_index:

            target_features[
                i,
                target_index[target]
            ] = 1

    X = np.hstack(
        [
            fingerprints,
            target_features
        ]
    )

    y = clean_df[
        "interaction_label"
    ].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X, y)

    return (
        model,
        generator,
        target_names,
        target_index,
        clean_df
    )


# ==========================================================
# GET DRUG FROM CHEMBL
# ==========================================================

@st.cache_data
def get_drug(drug_name):

    url = (
        "https://www.ebi.ac.uk/"
        "chembl/api/data/molecule/search.json"
    )

    params = {
        "q": drug_name
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    molecules = data.get(
        "molecules",
        []
    )

    if not molecules:

        return None

    # Try exact preferred-name match first
    drug_lower = drug_name.strip().lower()

    for molecule in molecules:

        pref_name = molecule.get(
            "pref_name"
        )

        if (
            pref_name
            and pref_name.lower()
            == drug_lower
        ):

            return molecule

    return molecules[0]


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    df = load_data()

    (
        model,
        generator,
        target_names,
        target_index,
        clean_df
    ) = train_model(df)

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# ==========================================================
# DRUG INPUT
# ==========================================================

drug_name = st.text_input(
    "Enter Drug Name",
    placeholder="Example: Ramipril"
)


# ==========================================================
# PREDICT
# ==========================================================

if st.button(
    "🔬 Predict Drug–Target Interaction",
    type="primary"
):

    if not drug_name.strip():

        st.warning(
            "Please enter a drug name."
        )

        st.stop()

    with st.spinner(
        "Finding drug and generating prediction..."
    ):

        try:

            # ----------------------------------------------
            # Find drug
            # ----------------------------------------------

            molecule_data = get_drug(
                drug_name
            )

            if molecule_data is None:

                st.error(
                    "Drug not found in ChEMBL. "
                    "Try another drug name."
                )

                st.stop()

            chembl_id = molecule_data.get(
                "molecule_chembl_id"
            )

            pref_name = molecule_data.get(
                "pref_name"
            )

            structures = molecule_data.get(
                "molecule_structures"
            )

            if not structures:

                st.error(
                    "No molecular structure was found "
                    "for this drug."
                )

                st.stop()

            smiles = structures.get(
                "canonical_smiles"
            )

            if not smiles:

                st.error(
                    "No canonical SMILES was available."
                )

                st.stop()


            # ----------------------------------------------
            # RDKit molecule
            # ----------------------------------------------

            mol = Chem.MolFromSmiles(
                smiles
            )

            if mol is None:

                st.error(
                    "RDKit could not process "
                    "this molecular structure."
                )

                st.stop()


            # ----------------------------------------------
            # Molecular fingerprint
            # ----------------------------------------------

            fingerprint = (
                generator
                .GetFingerprintAsNumPy(mol)
            )


            # ----------------------------------------------
            # Predict against every AD target
            # ----------------------------------------------

            predictions = []

            for target in target_names:

                target_vector = np.zeros(
                    len(target_names)
                )

                target_vector[
                    target_index[target]
                ] = 1

                features = np.concatenate(
                    [
                        fingerprint,
                        target_vector
                    ]
                )

                probability = (
                    model
                    .predict_proba(
                        features.reshape(
                            1, -1
                        )
                    )[0, 1]
                )

                predictions.append(
                    {
                        "Target": target,
                        "Predicted Probability":
                            probability
                    }
                )


            results = pd.DataFrame(
                predictions
            )

            results = results.sort_values(
                "Predicted Probability",
                ascending=False
            )

            top_result = results.iloc[0]

            top_target = top_result[
                "Target"
            ]

            top_probability = top_result[
                "Predicted Probability"
            ]


            # ==================================================
            # DISPLAY RESULTS
            # ==================================================

            st.success(
                "Prediction completed successfully."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Drug",
                    pref_name or drug_name
                )

                st.write(
                    f"**ChEMBL ID:** {chembl_id}"
                )

                st.write(
                    f"**Predicted target:** "
                    f"{top_target}"
                )

                st.metric(
                    "Predicted interaction probability",
                    f"{top_probability * 100:.2f}%"
                )


            with col2:

                st.write(
                    "**Molecular structure**"
                )

                image = Draw.MolToImage(
                    mol,
                    size=(400, 300)
                )

                st.image(
                    image,
                    caption=pref_name or drug_name
                )


            # ==================================================
            # TOP PREDICTIONS
            # ==================================================

            st.subheader(
                "Top Predicted Alzheimer’s Targets"
            )

            display_results = results.head(
                5
            ).copy()

            display_results[
                "Predicted Probability"
            ] = (
                display_results[
                    "Predicted Probability"
                ] * 100
            ).round(2)

            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True
            )


            # ==================================================
            # INTERPRETATION
            # ==================================================

            st.subheader(
                "Interpretation"
            )

            st.write(
                f"The model prioritized **{pref_name or drug_name}** "
                f"for interaction with **{top_target}**, "
                f"with a predicted interaction probability "
                f"of **{top_probability * 100:.2f}%**."
            )

            st.warning(
                "This is an in-silico prediction for "
                "research prioritization only. It does not "
                "establish therapeutic efficacy or clinical benefit."
            )


        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )