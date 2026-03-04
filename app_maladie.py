import streamlit as st
import pickle
import numpy as np

# --- Configuration de la page ---
st.set_page_config(
    page_title="🏥 Prédiction de Maladies",
    page_icon="🏥",
    layout="centered"
)

# --- Chargement des modèles ---
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('symptomes.pkl', 'rb') as f:
        symptomes = pickle.load(f)
    return model, le, symptomes

model, le, symptomes = load_models()

# --- Titre ---
st.title("🏥 Prédiction de Maladies")
st.write("Sélectionne tes symptômes et le modèle prédit la maladie correspondante.")
st.markdown("---")

# --- Sélection des symptômes ---
st.subheader("🤒 Sélectionne tes symptômes :")

# Formater les noms des symptômes pour l'affichage
def format_symptome(s):
    return s.replace('_', ' ').capitalize()

symptomes_affiches = [format_symptome(s) for s in symptomes]

symptomes_choisis = st.multiselect(
    "Cherche et sélectionne tes symptômes :",
    options=symptomes_affiches,
    placeholder="Ex: Itching, Fever, Headache..."
)

st.markdown("---")

# --- Bouton de prédiction ---
if st.button("🔍 Prédire la maladie", use_container_width=True):
    if len(symptomes_choisis) == 0:
        st.warning("⚠️ Veuillez sélectionner au moins un symptôme !")
    else:
        # Construire le vecteur d'entrée
        input_vector = np.zeros(len(symptomes))
        for s_affiche in symptomes_choisis:
            s_original = s_affiche.lower().replace(' ', '_')
            if s_original in symptomes:
                idx = symptomes.index(s_original)
                input_vector[idx] = 1

        # Prédiction
        prediction = model.predict([input_vector])
        maladie = le.inverse_transform(prediction)[0]

        # Probabilités
        probas = model.predict_proba([input_vector])[0]
        top3_idx = np.argsort(probas)[::-1][:3]
        top3_maladies = le.inverse_transform(top3_idx)
        top3_probas = probas[top3_idx]

        # Affichage du résultat principal
        st.success(f"### 🎯 Maladie prédite : **{maladie}**")

        st.markdown("---")

        # Top 3 des prédictions
        st.subheader("📊 Top 3 des maladies possibles :")
        for i, (mal, prob) in enumerate(zip(top3_maladies, top3_probas)):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            st.write(f"{emoji} **{mal}** — {prob*100:.1f}%")
            st.progress(float(prob))

        st.markdown("---")
        st.info("⚠️ Ce résultat est indicatif uniquement. Consultez un médecin pour un diagnostic officiel.")

# --- Sidebar info ---
with st.sidebar:
    st.header("ℹ️ Informations")
    st.write(f"**Nombre de symptômes :** {len(symptomes)}")
    st.write(f"**Nombre de maladies :** {len(le.classes_)}")
    st.markdown("---")
    st.write("**Modèle utilisé :** Random Forest")
    st.markdown("---")
    st.write("**Liste des maladies détectables :**")
    for m in sorted(le.classes_):
        st.write(f"• {m}")
