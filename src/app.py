import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Drug Repurposing - Alzheimer's Disease",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🧬 AI-Based Drug Repurposing")
st.subheader("Alzheimer's Disease")

st.write(
    "Predict potential drug-target interactions using a "
    "machine-learning-based drug repurposing approach."
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    final_file = "final_candidate_results.csv"
    top_file = "top_candidates.csv"

    final_df = pd.DataFrame()
    top_df = pd.DataFrame()

    try:
        final_df = pd.read_csv(final_file)
    except Exception:
        pass

    try:
        top_df = pd.read_csv(top_file)
    except Exception:
        pass

    return final_df, top_df


final_df, top_df = load_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_drug_name(name):
    """Clean user-entered drug name."""
    return name.strip().upper()


def get_structure_image(drug_name):
    """
    Get molecular structure image from PubChem.

    This avoids RDKit Draw so the Streamlit deployment
    does not depend on rdMolDraw2D.
    """

    try:

        encoded_name = quote(drug_name)

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/name/{encoded_name}/PNG"
        )

        response = requests.get(
            url,
            timeout=20
        )

        if response.status_code == 200:
            return response.content

    except Exception:
        pass

    return None


def find_drug_results(drug_name):
    """Find prediction results for the entered drug."""

    drug_name = clean_drug_name(drug_name)

    results = pd.DataFrame()

    # --------------------------------------------------------
    # Search final candidate results
    # --------------------------------------------------------

    if not final_df.empty:

        df = final_df.copy()

        if "display_drug" in df.columns:

            mask = (
                df["display_drug"]
                .astype(str)
                .str.upper()
                .str.contains(
                    drug_name,
                    na=False
                )
            )

            results = df[mask].copy()

    # --------------------------------------------------------
    # If not found, search top candidates
    # --------------------------------------------------------

    if results.empty and not top_df.empty:

        df = top_df.copy()

        if "drug_name" in df.columns:

            mask = (
                df["drug_name"]
                .astype(str)
                .str.upper()
                .str.contains(
                    drug_name,
                    na=False
                )
            )

            results = df[mask].copy()

    return results


# ============================================================
# DRUG INPUT
# ============================================================

drug_name = st.text_input(
    "Enter Drug Name",
    placeholder="Example: Ramipril"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔬 Predict Drug-Target Interaction",
    type="primary"
):

    if not drug_name.strip():

        st.warning(
            "Please enter a drug name."
        )

    else:

        search_name = clean_drug_name(drug_name)

        with st.spinner(
            "Analyzing drug-target interaction..."
        ):

            results = find_drug_results(
                search_name
            )

            # ==================================================
            # IF RESULTS FOUND
            # ==================================================

            if not results.empty:

                st.success(
                    "Prediction completed successfully."
                )

                # ------------------------------------------------
                # Determine drug name
                # ------------------------------------------------

                if "display_drug" in results.columns:

                    pref_name = str(
                        results.iloc[0]["display_drug"]
                    )

                elif "drug_name" in results.columns:

                    pref_name = str(
                        results.iloc[0]["drug_name"]
                    )

                else:

                    pref_name = drug_name.upper()

                # ------------------------------------------------
                # Main result
                # ------------------------------------------------

                first_result = results.iloc[0]

                if "gene" in results.columns:

                    top_target = str(
                        first_result["gene"]
                    )

                else:

                    top_target = "Unknown"

                if "predicted_probability" in results.columns:

                    try:

                        top_probability = float(
                            first_result[
                                "predicted_probability"
                            ]
                        )

                    except Exception:

                        top_probability = 0.0

                else:

                    top_probability = 0.0

                # =================================================
                # RESULT COLUMNS
                # =================================================

                col1, col2 = st.columns(
                    [1, 1]
                )

                # =================================================
                # DRUG INFORMATION
                # =================================================

                with col1:

                    st.markdown(
                        "### Drug"
                    )

                    st.markdown(
                        f"# {pref_name}"
                    )

                    # ChEMBL ID if available

                    if "molecule_chembl_id" in results.columns:

                        chembl_id = str(
                            first_result[
                                "molecule_chembl_id"
                            ]
                        )

                        st.write(
                            f"**ChEMBL ID:** {chembl_id}"
                        )

                    st.write(
                        f"**Predicted target:** "
                        f"{top_target}"
                    )

                    st.write(
                        "**Predicted interaction "
                        "probability**"
                    )

                    st.markdown(
                        f"# {top_probability * 100:.2f}%"
                    )

                # =================================================
                # MOLECULAR STRUCTURE
                # =================================================

                with col2:

                    st.markdown(
                        "### Molecular Structure"
                    )

                    image = get_structure_image(
                        pref_name
                    )

                    if image is not None:

                        st.image(
                            image,
                            caption=pref_name,
                            width=450
                        )

                    else:

                        st.info(
                            "Molecular structure image "
                            "is unavailable from PubChem."
                        )

                # =================================================
                # TOP TARGETS
                # =================================================

                st.markdown(
                    "## Top Predicted Alzheimer's Targets"
                )

                target_table = pd.DataFrame()

                if (
                    "gene" in results.columns
                    and
                    "predicted_probability"
                    in results.columns
                ):

                    target_table = results[
                        [
                            "gene",
                            "predicted_probability"
                        ]
                    ].copy()

                    target_table = (
                        target_table
                        .drop_duplicates(
                            subset=["gene"]
                        )
                        .sort_values(
                            "predicted_probability",
                            ascending=False
                        )
                        .head(10)
                    )

                    target_table[
                        "predicted_probability"
                    ] = (
                        target_table[
                            "predicted_probability"
                        ] * 100
                    ).round(2)

                    target_table = (
                        target_table
                        .rename(
                            columns={
                                "gene": "Target",
                                "predicted_probability":
                                    "Predicted Probability (%)"
                            }
                        )
                    )

                if not target_table.empty:

                    st.dataframe(
                        target_table,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "Target-level results are "
                        "not available for this drug."
                    )

                # =================================================
                # INTERPRETATION
                # =================================================

                st.markdown(
                    "## Interpretation"
                )

                st.write(
                    f"The model prioritized "
                    f"**{pref_name}** for potential "
                    f"interaction with **{top_target}**, "
                    f"with a predicted interaction "
                    f"probability of "
                    f"**{top_probability * 100:.2f}%**."
                )

                # =================================================
                # DISCLAIMER
                # =================================================

                st.warning(
                    "This is an in-silico prediction for "
                    "research prioritization only. It does not "
                    "establish therapeutic efficacy or clinical "
                    "benefit."
                )

            # ==================================================
            # DRUG NOT FOUND
            # ==================================================

            else:

                st.error(
                    f"No prediction result found for "
                    f"'{drug_name}'."
                )

                st.info(
                    "Try a drug present in the trained "
                    "candidate dataset, such as Ramipril."
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Based Drug Repurposing for Alzheimer's Disease | "
    "Machine Learning Research Prototype"
)