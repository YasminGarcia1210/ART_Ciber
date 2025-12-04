# 🎯 Adversarial Attack y Streamlit App

## Ciberseguirdad e IA 

### ✍️ Autores:
- **Javier Ricardo Muñoz**
- **Yasmin Johanna Garcia**

---

## 🌟 Descripción del Proyecto
Este repositorio combina técnicas avanzadas de aprendizaje automático con una interfaz web interactiva. Contiene:

1. **🛡️ Adversarial Attack Script**:
   - Implementa un ataque adversarial utilizando el método Carlini-Wagner L2.
   - Clasifica imágenes como "real" o "fake" utilizando un modelo Vision Transformer (ViT).

2. **🌐 Aplicación Streamlit**:
   - Permite subir imágenes y clasificarlas como "real" o "fake".
   - Muestra el porcentaje de confianza del modelo de manera visual y atractiva.

---

## ⚙️ Requisitos
- Python 3.8+
- Dependencias:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 Instrucciones para Ejecutar

### 1. **Adversarial Attack Script**
Ejecuta el script para realizar un ataque adversarial:
```bash
python adversarial_attack.py
```

### 2. **Aplicación Streamlit**
Ejecuta la aplicación web:
```bash
streamlit run streamlit_app.py
```

---

## 🖼️ Ejemplo de Uso

### Adversarial Attack Script
Salida esperada:
```
[1/8] Configurando dispositivo...
      Usando: cuda
[2/8] Cargando modelo Vision Transformer...
...
[8/8] Verificando resultado del ataque...
      🎯 ¡ATAQUE EXITOSO!
```

### Aplicación Streamlit
1. Sube una imagen.
2. Obtén la predicción y el porcentaje de confianza.

---

## 📂 Estructura del Proyecto
```
ART_Ciber/
├── adversarial_attack.py       # Script principal del ataque adversarial
├── streamlit_app.py            # Aplicación Streamlit
├── static/fake/                # Carpeta con imágenes de ejemplo
├── adversarial_output/         # Carpeta para guardar imágenes adversariales
├── art.ipynb                   # Notebook de análisis
├── analisis_notebook.md        # Resumen del análisis
└── README.md                   # Este archivo
```

---

## 💡 Notas Adicionales
- Asegúrate de tener una GPU disponible para un mejor rendimiento.
- La carpeta `static/fake/` contiene imágenes de ejemplo para pruebas.

---

## 🙌 ¡Gracias por explorar nuestro proyecto! 🚀

Si tienes preguntas o sugerencias, no dudes en contactarnos. 😊
