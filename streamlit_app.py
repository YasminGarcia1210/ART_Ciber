import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier

# Configurar el modelo
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    model = model.to(device)
    model.eval()

    labels = {0: "fake", 1: "real"}

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        nb_classes=len(labels),
        input_shape=(3, 224, 224)
    )
    return classifier, labels, device

# Cargar modelo y etiquetas
classifier, labels, device = load_model()

# Configurar la interfaz de Streamlit
st.set_page_config(page_title="Clasificador Real o Fake", layout="centered")
st.title("🖼️ Clasificador de Imágenes: Real o Fake")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #6A0DAD;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .result {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
    }
    .confidence {
        font-size: 4rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
    }
    .image-container img {
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Sube una imagen para clasificarla como 'real' o 'fake'.")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(image, caption="Imagen cargada", use_column_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preprocesar la imagen
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # Realizar predicción
    with torch.no_grad():
        output = classifier._model(img_tensor)
        pred = output[0].argmax().item()
        confidence = torch.softmax(output[0], dim=0).max().item()

    # Mostrar resultado
    st.markdown(f"<div class='result'>Predicción: {labels[pred]}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence'>{confidence * 100:.2f}%</div>", unsafe_allow_html=True)