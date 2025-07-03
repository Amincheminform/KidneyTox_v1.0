import streamlit as st
#from streamlit_ketcher import st_ketcher
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
#from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shap
from PIL import Image
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import io

# Author : Dr. Sk. Abdul Amin
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

st.title("KidneyTox_v1.0: predictor of Kidney Toxicity")

st.image(logo_url, width=300)

with st.expander("What is KidneyTox_v1.0?", expanded=True):
    st.write('''*KidneyTox_v1.0* is a python package that allows users to predict 
             the nephrotoxicity of a small molecule (1 = Toxic, 0 = Non-toxic) 
             and also visualize the molecule.''')

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
        st.markdown("**Calculation may take < 30 seconds!**")
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
                return "### **Toxic**" if pred == 1 else "### **Non-toxic**"

            # Row 1 — Query molecule
            st.markdown("### Query Molecule")
            col1, col2 = st.columns(2)

            with col1:
                # SHAP plot (smaller)
                plt.figure(figsize=(4, 3))
                shap.plots.waterfall(shap_values[0, :, y_external_pred[0]], max_display=10, show=False)
                fig1 = plt.gcf()
                st.pyplot(fig1)
                plt.clf()

            with col2:
                # Molecule image + prediction label
                mol_img = mol_to_array(molecules[0])
                st.image(mol_img, caption="Query Molecule", width=250)
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred[0])}</div>",
                            unsafe_allow_html=True)

            # Separator
            st.markdown("---")

            # Row 2 — Most similar molecule
            st.markdown("### Most similar molecule from the dataset")
            col3, col4 = st.columns(2)

            with col3:
                # SHAP plot (smaller)
                plt.figure(figsize=(4, 3))
                shap.plots.waterfall(shap_values[1, :, y_external_pred[1]], max_display=10, show=False)
                fig2 = plt.gcf()
                st.pyplot(fig2)
                plt.clf()

            with col4:
                # Molecule image + prediction + similarity + ID
                similar_mol_img = mol_to_array(molecules[1])
                st.image(similar_mol_img, caption="Most Similar Molecule", width=250)
                st.markdown(f"**Molecule ID**: {most_similar['ID']}")
                st.markdown(f"**Tanimoto similarity with respect to query molecule**: {most_similar['Tanimoto']:.2f}")
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred[1])}</div>",
                            unsafe_allow_html=True)

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
