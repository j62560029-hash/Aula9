import streamlit as st  # Corrigido o apelido para 'st'
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Configuração da página DEVE ser o primeiro comando do Streamlit
st.set_page_config(page_title="Detector de Objetos IA", layout="centered")

st.title("📷 Detector de Objetos com YOLOv8")
st.write("Faça o upload de uma imagem para identificar os objetos presentes.")

# Cache do modelo para evitar recarregamento a cada interação
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Componente de Upload de Imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)
    
    with st.spinner("Processando imagem..."):
        img_array = np.array(image)
        results = model(img_array, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Objetos Detectados")
        st.image(res_rgb, use_container_width=True)
        
    st.success("Processamento concluído com sucesso!")