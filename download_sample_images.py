import requests
import os

os.makedirs("static/fake", exist_ok=True)

urls = [
    # Imagen pública aleatoria
    "https://picsum.photos/512",
    
    # Persona real generada públicamente
    "https://randomuser.me/api/portraits/women/44.jpg",
    
    # Fake generado por GAN
    "https://thispersondoesnotexist.com/"
]

print("Descargando imágenes...")

for i, url in enumerate(urls):
    try:
        print(f"→ Descargando {url}")
        r = requests.get(url, timeout=10)
        img_path = f"static/fake/sample_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(r.content)
        print(f"✔️ Guardada correctamente: {img_path}")
    except Exception as e:
        print(f"❌ Error descargando {url}: {e}")

print("\n🎉 Listo amor, ya tienes imágenes en static/fake/")
