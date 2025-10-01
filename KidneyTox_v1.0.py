import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shap
from PIL import Image
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import io

# Author: Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
# Date: 03.07.2025

train_url = "https://github.com/Amincheminform/KidneyTox_v1.0/raw/main/0_train_KidneyTox.csv"
# https://github.com/Amincheminform/KidneyTox_v1.0/blob/main/0_train_KidneyTox.csv
test_url = "https://github.com/Amincheminform/KidneyTox_v1.0/raw/main/0_test_KidneyTox.csv"

train_data = pd.read_csv(train_url, sep=',')
test_data = pd.read_csv(test_url, sep=',')

PandasTools.AddMoleculeColumnToFrame(train_data, 'Smiles', 'Molecule')
PandasTools.AddMoleculeColumnToFrame(test_data, 'Smiles', 'Molecule')

# https://github.com/Amincheminform/KidneyTox_v1.0/blob/main/KidneyTox_logo.jpg
# Streamlit
logo_url = "https://raw.githubusercontent.com/Amincheminform/KidneyTox_v1.0/main/KidneyTox_logo.jpg"

st.set_page_config(
    page_title="KidneyTox_v1.0: predictor of Kidney Toxicity",
    layout="wide",
    page_icon=logo_url
)

# st.sidebar.image(logo_url)
# st.sidebar.success("Thank you for using KidneyTox_v1.0!")

calc = Calculator(descriptors, ignore_3D=True)
descriptor_columns = ['AXp-2d', 'BCUTdv-1l', 'BCUTZ-1h', 'SpMax_A', 'BCUTd-1h',
                      'AETA_eta_F', 'BCUTZ-1l', 'Xc-5dv', 'BCUTs-1h', 'BCUTd-1l']

# Train the model
try:
    X_train, y_train = train_data[descriptor_columns], train_data['Toxic']
    X_test, y_test = test_data[descriptor_columns], test_data['Toxic']

    model = RandomForestClassifier(
        n_estimators=60, max_depth=24, min_samples_split=13,
        min_samples_leaf=2, random_state=42
    )
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    # st.sidebar.success(f"Model trained with test accuracy: {test_accuracy:.2f}")

except Exception as e:
    st.sidebar.error(f"Model training failed: {e}")
    model = None

def generate_2d_image(smiles, img_size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=img_size, kekulize=True) if mol else None

def mol_to_array(mol, size=(300, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.SetDrawOptions(drawer.drawOptions())  # optionally customize
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_data))

def get_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

#st.title("KidneyTox_v1.0: predictor of Kidney Toxicity")
#st.image(logo_url, width=300)

st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center; padding-top: 20px;">
        <img src="{logo_url}" alt="KidneyTox Logo" width="350">
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("**About KidneyTox v1.0**", expanded=True):
    st.markdown("""

**KidneyTox v1.0** is an easy-to-use predictive tool for evaluating the **nephrotoxicity** (kidney toxicity) of small molecules.

Example SMILES:  `Indomethacin` → `CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O`

Definitions of the descriptor used in **KidneyTox v1.0**

| Descriptor | Type | Definition |
|------------|------|------------|
| BCUTdv-1l | BCUT descriptors | A BCUT descriptor derived from eigenvalues of an adjacency matrix representing the molecule, weighted by a specific property (van der Waals volume). "1l" indicates the lowest eigenvalue. |
| BCUTZ-1h | BCUT descriptors | Similar to BCUTdv-1l, but the matrix is weighted by atomic polarizability, and "1h" refers to the highest eigenvalue. |
| BCUTd-1h | BCUT descriptors | A BCUT descriptor where the adjacency matrix is weighted by a dipole-related property. "1h" indicates the highest eigenvalue. |
| BCUTZ-1l | BCUT descriptors | A BCUT descriptor weighted by atomic polarizability, where "1l" refers to the lowest eigenvalue. |
| BCUTd-1l | BCUT descriptors | A BCUT descriptor weighted by dipole-related properties, focusing on the lowest eigenvalue. |
| BCUTs-1h | BCUT descriptors | A BCUT descriptor weighted by atomic electronegativity, focusing on the highest eigenvalue. |
| AXp-2d | Autocorrelation descriptors | A 2D autocorrelation descriptor that measures the distribution of atomic properties (like electronegativity, mass, or charge) weighted by bond distance in a molecule. |
| Xc-5dv | Autocorrelation descriptors | A 2D autocorrelation descriptor that considers the distribution of valence electron information across a molecule. |
| SpMax_A | Topological descriptors | Represents the maximum eigenvalue of the adjacency matrix weighted by atomic properties. The "_A" suffix suggests a specific property, such as electronegativity or polarizability. |
| AETA_eta_F | Dipole-related descriptors | A measure of atomic electronegativity weighted topological descriptor. Specifically relates to the F (fluorine) atom or fluorine-related features in the molecule. |
""")

col1, col2 = st.columns(2)

prediction_done = False

with col1:
    st.markdown("### Draw Query Molecule")
    smile_code = st_ketcher()
    if smile_code and not prediction_done:
        st.success("Molecule drawn successfully!")

with col2:
    st.markdown("### SMILES string of Query Molecule")
    smiles_input = st.text_input("Enter or edit SMILES:", value=smile_code if smile_code else "")

    if smiles_input and not prediction_done:
        st.markdown(f"✅ **SMILES code**: `{smiles_input}`")
        st.markdown("**Some calculation may take < 30 seconds!**")
        st.markdown("**Thank you for your patience!**")

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # Author : Dr. Sk. Abdul Amin
            # [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

            all_data = pd.concat([train_data, test_data], ignore_index=True)

            query_fp = get_ecfp4(smiles_input)
            all_data['Fingerprint'] = all_data['Smiles'].apply(lambda x: get_ecfp4(x))
            all_data['Tanimoto'] = all_data['Fingerprint'].apply(lambda x: DataStructs.TanimotoSimilarity(query_fp, x))

            most_similar = all_data.loc[all_data['Tanimoto'].idxmax()]
            similar_smiles = most_similar['Smiles']
            similar_mol = most_similar['Molecule']

            st.subheader("Results")

            smiles_list = [smiles_input, similar_smiles]
            molecules = [Chem.MolFromSmiles(sm) for sm in smiles_list]

            descriptor_df = calc.pandas(molecules)
            external_descriptor_df = descriptor_df[descriptor_columns].dropna()
            X_external = external_descriptor_df

            y_external_pred = model.predict(X_external)

            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_external)

            prediction_done = True

            titles = [
                f"Query molecule\nPredicted Class: {y_external_pred[0]}",
                f"Most similar molecule from dataset\nTanimoto similarity: {most_similar['Tanimoto']:.2f}\nPredicted Class: {y_external_pred[1]}"
            ]

            def pred_label(pred):
                return "### Toxic" if pred == 1 else "### Non-toxic"

            X_combined_external = np.vstack((X_train, X_external.to_numpy()))
            Amin_H_external = X_combined_external @ np.linalg.pinv(
                X_combined_external.T @ X_combined_external) @ X_combined_external.T
            external_leverage = np.diag(Amin_H_external)[len(X_train):]

            p = X_train.shape[1]
            n = X_train.shape[0]
            leverage_threshold = 3 * p / n
            external_ad_flags = external_leverage <= leverage_threshold

            y_external_pred = model.predict(X_external)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_external)

            st.subheader("Query Molecule Results")
            col1, col2 = st.columns(2)

            with col1:
                # SHAP Waterfall plot
                plt.figure(figsize=(4, 3))
                shap.plots.waterfall(shap_values[0, :, y_external_pred[0]], max_display=10, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:
                # Molecule image
                mol_img = mol_to_array(molecules[0])
                st.image(mol_img, caption="Query Molecule", width=250)

                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred[0])}</div>",
                            unsafe_allow_html=True)

                # AD flag with explanation tooltip
                #ad_status = 'Within AD ' if external_ad_flags[0] else 'Outside AD'
                if external_ad_flags[0]:
                    st.markdown("<b>Applicability Domain:</b> <span style='color:green;'>Within AD </span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<b>Applicability Domain:</b> <span style='color:red;'>Outside AD </span>",
                                unsafe_allow_html=True)

                #st.markdown(f"**Applicability Domain:** {ad_status}")

            #Most similar molecule from dataset
            st.markdown("---")
            st.subheader("Most Similar Molecule from Dataset")

            col3, col4 = st.columns(2)

            with col3:
                plt.figure(figsize=(4, 3))
                shap.plots.waterfall(shap_values[1, :, y_external_pred[1]], max_display=10, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            with col4:
                similar_mol_img = mol_to_array(molecules[1])
                st.image(similar_mol_img, caption="Most Similar Molecule", width=250)
                st.markdown(f"**Molecule ID**: {most_similar['ID']}")
                st.markdown(f"**Tanimoto similarity:** {most_similar['Tanimoto']:.2f}")
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred[1])}</div>",
                            unsafe_allow_html=True)

                # Optional: AD for most similar molecule
                if external_ad_flags[1]:
                    st.markdown("<b>Applicability Domain:</b> <span style='color:green;'>Within AD </span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<b>Applicability Domain:</b> <span style='color:red;'>Outside AD </span>",
                                unsafe_allow_html=True)

            st.markdown(
                "NOTE: A molecule 'Within AD' means its structural descriptors fall within the reliable chemical space of the training set, "
                "and predictions are considered reliable. "
                "A molecule 'Outside AD' may have structural features not well represented in the training data; "
                "its prediction should be interpreted with caution."
            )

    else:
        st.info("Please enter a SMILES string to get predictions.")

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
# Contact section
with st.expander("Contact", expanded=False):
    st.write('''
        #### Report an Issue

        Report a bug or contribute here: [GitHub](https://github.com/Amincheminform)

        #### Contact Us
        - [Dr. Supratik Kar](mailto:skar@kean.edu)
        - [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com)
    ''')
