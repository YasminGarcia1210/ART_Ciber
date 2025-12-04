"""
Script de Ataque Adversarial usando Carlini-Wagner L2
Convierte imágenes públicas (fake o reales) en ejemplos adversariales
que engañan al clasificador ViT.
"""

import torch
import timm
from torchvision import transforms
from PIL import Image
import os

# Importar ART
try:
    from art.attacks.evasion import CarliniL2Method
    from art.estimators.classification import PyTorchClassifier
    print("✅ Biblioteca ART importada correctamente")
except ImportError as e:
    print("❌ Error al importar ART:")
    print(f"   {e}")
    print("\nPor favor instala ART ejecutando:")
    print("   pip install adversarial-robustness-toolbox")
    exit(1)

def main():
    print("\n" + "="*60)
    print("ATAQUE ADVERSARIAL CARLINI-WAGNER L2")
    print("="*60 + "\n")
    
    # 1. Configurar dispositivo
    print("[1/8] Configurando dispositivo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Usando: {device}")
    
    # 2. Cargar el modelo Vision Transformer
    print("\n[2/8] Cargando modelo Vision Transformer...")
    target_model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    target_model.head = torch.nn.Linear(target_model.head.in_features, 2)
    target_model = target_model.to(device)
    
    # Cargar pesos entrenados
    weights_path = "../weights/vit_teacher.pth"
    if not os.path.exists(weights_path):
        weights_path = "./vit_teacher.pth"
    
    if os.path.exists(weights_path):
        print(f"      Cargando pesos desde: {weights_path}")
        target_model.load_state_dict(
            torch.load(weights_path, weights_only=True)
        )
    else:
        print(f"      ⚠️ No se encontró archivo de pesos: {weights_path}")
        print("      Usando modelo base pre-entrenado (menos preciso)")
    
    target_model.eval()
    print("      ✅ Modelo cargado correctamente")
    
    # 3. Configurar etiquetas
    print("\n[3/8] Configurando clasificador...")
    labels = {0: "fake", 1: "real"}
    
    classifier = PyTorchClassifier(
        model=target_model,
        loss=torch.nn.CrossEntropyLoss(),
        nb_classes=len(labels),
        input_shape=(3, 224, 224)
    )
    print("      ✅ Clasificador configurado")
    
    # 4. Crear ataque Carlini-Wagner
    print("\n[4/8] Inicializando ataque Carlini-Wagner L2...")
    attack = CarliniL2Method(classifier)
    print("      ✅ Ataque configurado")
    
    # =====================================================
    # 5. Buscar imágenes en la carpeta
    # =====================================================
    print("\n[5/8] Buscando imágenes en la carpeta...")

    img_dir = "C:\\Users\\yasmi\\OneDrive - Universidad Icesi\\MAESTRIA EN IA\\ART\\static\\fake"
    import glob
    imgs = (
        glob.glob(f"{img_dir}\\*.jpg") +
        glob.glob(f"{img_dir}\\*.png") +
        glob.glob(f"{img_dir}\\*.jpeg")
    )

    if not imgs:
        print("      ❌ No se encontró ninguna imagen en la carpeta especificada")
        exit(1)

    print(f"      Imágenes encontradas: {len(imgs)}\n")

    # Preprocesamiento estándar ImageNet
    preprocess = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Para revertir normalización
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )

    # =====================================================
    # 🔍 PREDICCIONES DE TODAS LAS IMÁGENES
    # =====================================================
    print("      🔍 Predicciones del modelo ANTES del ataque:\n")

    for path in imgs:
        try:
            img = Image.open(path)
            tens = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = target_model(tens)
                pred = out[0].argmax().item()
                conf = torch.softmax(out[0], dim=0).max().item()

            print(f"         → {os.path.basename(path)}  →  {labels[pred]}   ({conf:.4f})")
        
        except Exception as e:
            print(f"         ❌ Error con la imagen {path}: {e}")

    print("\n      ✔️ Evaluación previa completada.\n")

    # Seleccionar la PRIMERA imagen para ataque
    img_path = imgs[0]
    print(f"      🧪 Imagen seleccionada para ataque: {img_path}\n")

    img = Image.open(img_path)
    print(f"      Tamaño original: {img.size}")
    img_tensor = preprocess(img).unsqueeze(0)
    print("      ✅ Imagen preprocesada\n")
    
    # 6. Predicción original
    print("[6/8] Predicción del modelo ANTES del ataque...\n")
    with torch.no_grad():
        original_tensor = img_tensor.to(device)
        original_output = target_model(original_tensor)
        original_pred = original_output[0].argmax().item()
    
    print(f"      Clase: {labels[original_pred]}")
    print(f"      Confianza: {torch.softmax(original_output[0], dim=0).max().item():.4f}")
    
    # 7. Generar ejemplo adversarial
    print("\n[7/8] Generando ejemplo adversarial (puede tardar)...")
    adversarial_example = attack.generate(img_tensor.numpy())
    print("      ✅ Ejemplo adversarial generado")
    
    # 8. Verificar el ataque
    print("\n[8/8] Verificando resultado del ataque...\n")
    adv_tensor = torch.from_numpy(adversarial_example).to(device)
    with torch.no_grad():
        output = target_model(adv_tensor)
        adv_pred = output[0].argmax().item()
    
    print(f"      Clase adversarial: {labels[adv_pred]}")
    print(f"      Confianza: {torch.softmax(output[0], dim=0).max().item():.4f}")
    
    if original_pred != adv_pred:
        print("\n      🎯 ¡ATAQUE EXITOSO!")
        print(f"         {labels[original_pred]}  →  {labels[adv_pred]}")
    else:
        print("\n      ⚠️ El ataque no cambió la predicción")
    
    # Guardar imagen adversarial
    print("\n[GUARDANDO] Guardando imagen adversarial...")

    output_dir = "./adversarial_output"
    os.makedirs(output_dir, exist_ok=True)
    
    def tensor_to_pil(img_tensor):    
        unnormed_tensor = unnormalize(img_tensor)
        return transforms.functional.to_pil_image(unnormed_tensor[0])
    
    masked_pil = tensor_to_pil(adv_tensor.cpu())
    output_path = os.path.join(output_dir, "output_art.png")
    masked_pil.save(fp=output_path)
    
    print(f"      ✅ Imagen guardada en: {output_path}")
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DEL ATAQUE")
    print("="*60)
    print(f"Imagen original:      {img_path}")
    print(f"Predicción original:  {labels[original_pred]}")
    print(f"Predicción adversarial: {labels[adv_pred]}")
    print(f"Resultado: {'EXITOSO ✅' if original_pred != adv_pred else 'FALLIDO ❌'}")
    print(f"Imagen guardada:      {output_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error durante la ejecución:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
