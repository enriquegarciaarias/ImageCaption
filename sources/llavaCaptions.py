from sources.common.common import logger, processControl, log_
from sources.common.utils import huggingface_login
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

def processLLAVA(imageList, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 📌 Cargar el modelo LLaVA
    huggingface_login()
    model_name = "liuhaotian/llava-v1.5-13b"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

    # 📌 Generar captions para cada imagen
    captions = {}
    for img in imageList:
        image = Image.open(img['path']).convert("RGB")

        # Procesar la imagen con el modelo LLaVA
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=50)

        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        captions[img['name']] = caption
        print(f"📷 {img['name']} → {caption}")

    # 📌 Guardar captions en un archivo
    with open("generated_captions.txt", "w") as f:
        for name, caption in captions.items():
            f.write(f"{name}: {caption}\n")

    print("✅ Captions generadas y guardadas en 'generated_captions.txt'")
