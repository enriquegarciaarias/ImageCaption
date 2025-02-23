from sources.common.common import logger, processControl, log_
from sources.contextData import buildContextData
import os

from PIL import Image

def processLLaVA(device):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.eval.run_llava import eval_model
    log_("info", logger, f"Start process LLaVA")
    documentContextpath = os.path.join(processControl.env['inputPath'], 'RAMNOUS.docx')

    metadata = {
        "cluster": "Panorámica",
        "site_name": "Santuario de Némesis",
        "image_title": "Planta del Santuario de Némesis",
        "site_zone": "Ática"
    }
    keywords = ["Némesis", "Santuario de Némesis", "Planta del Santuario de Némesis"]
    context_text = buildContextData(documentContextpath, keywords, top_n=5)
    log_("info", logger, f"Contexto generado: {context_text}")

    # 4️Cargar LLaVA
    model_path = "liuhaotian/llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    # 5️Cargar la imagen a analizar
    image_path = os.path.join(processControl.env['inputPath'], 'Diapo 13.4 Recinto funerario de la familia de Menéstides.JPG')
    image = Image.open(image_path).convert("RGB")

    # 6️Tokenizar el contexto y procesar la imagen
    inputs = tokenizer(context_text, return_tensors="pt").to(device)
    image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(inputs.input_ids.device)

    # 7️Generar la descripción con LLaVA
    output = eval_model(model, tokenizer, image_tensor, inputs.input_ids)

    log_("info", logger, f"Descripción generada: {output}")