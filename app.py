from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# Titre
st.title("Détection d’objets avec YOLOv8")

st.write("Importer une image pour détecter les objets")

# Upload image
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Ouvrir l’image
    image = Image.open(uploaded_file)

    # Convertir en tableau numpy
    img_array = np.array(image)

    # Faire la détection
    results = model(img_array)

    # Dessiner les résultats
    annotated_frame = results[0].plot()

    # Afficher l’image détectée
    st.image(
        annotated_frame,
        caption="Résultat de la détection",
        use_container_width=True
    )
