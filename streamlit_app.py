import streamlit as st
import random
import numpy as np
import Levenshtein
import jellyfish
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ——— Entrenamiento rápido de ejemplo ———
@st.cache_data(show_spinner=False)
def train_model():
    # Generamos datos sintéticos tal y como hacías
    base = [w.lower() for w in
            ['Alpha','Beta','Gamma','Delta','Omega','Vertex','Nexus','Pioneer',
             'Quantum','Nova','Apex','Summit','Fusion','Spectrum','Vector',
             'Prime','Edge','Core','Pulse','Echo','Zenith','Horizon','Vista']]
    suf = [s.lower() for s in
           ['Tech','Tek','Systems','Solutions','Dynamics','Global','Works',
            'Labs','Corp','Soft']]
    X, y = [], []
    for _ in range(200):
        b1 = random.choice(base) + " " + random.choice(suf)
        b2 = random.choice(base) + " " + random.choice(suf)
        d = Levenshtein.distance(b1, b2) / max(len(b1), len(b2), 1)
        f2 = jellyfish.jaro_winkler_similarity(b1, b2)
        X.append([d, f2])
        y.append(int(d < 0.4 or f2 > 0.85))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    m.fit(np.array(X_tr), np.array(y_tr))
    return m

model = train_model()

# ——— Página ———
st.set_page_config(page_title="Confusión de Marcas", layout="centered")
st.title("📊 Predictor de Confusión de Marcas")

bn = st.text_input("Marca nueva")
be = st.text_input("Marca existente")

if st.button("Calcular probabilidad"):
    if not bn or not be:
        st.warning("Por favor ingresa ambas marcas.")
    else:
        d = Levenshtein.distance(bn, be) / max(len(bn), len(be), 1)
        f2 = jellyfish.jaro_winkler_similarity(bn, be)
        prob = model.predict_proba(np.array([[d, f2]]))[0,1]
        st.success(f"Probabilidad de confusión: {prob:.3f}")
