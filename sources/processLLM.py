from sources.common.common import logger, processControl, log_
from sources.common.utils import convert_docx_to_txt
import os


"""
    Metadatos:
    - Etiqueta de clúster: "Panorámica"
    - Nombre del yacimiento: "Santuario de Némesis"
    - Título de la imagen: "Planta del Santuario de Némesis"
    - Zona del yacimiento: "Ática"
    
    Texto largo: [Inserte aquí el texto de 3000 palabras sobre el yacimiento]

Instrucciones:
Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "Panorámica" del "Santuario de Némesis" ubicado en "Ática". La descripción debe ser concisa (máximo 100 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en una imagen panorámica del yacimiento.
Modelo de LLaMA 2 a Utilizar
Modelo recomendado: LLaMA 2 13B (equilibrio entre rendimiento y requisitos computacionales).

Si tienes recursos limitados, puedes usar LLaMA 2 7B.

Si necesitas máxima precisión y tienes recursos suficientes, usa LLaMA 2 70B.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from sentence_transformers import SentenceTransformer, util
import re


def huggingface_login():
    try:
        # Add your Hugging Face token here, or retrieve it from environment variables
        token = processControl.defaults['token'] if 'token' in processControl.defaults else ['', '']
        login(token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print("Error logging into Hugging Face:", str(e))
        raise


def processMistral():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    long_text = convert_docx_to_txt(os.path.join(processControl.env['inputPath'], 'RAMNOUS.docx'))

    MODEL_NAME_LLM = "mistralai/Mistral-Nemo-Instruct-2407"
    MODEL_NAME_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

    huggingface_login()

    embedding_model = SentenceTransformer(MODEL_NAME_EMBEDDING)
    embedding_model.to(device)  # Mover embeddings a GPU si está disponible

    metadata = {
        "cluster": "Panorámica",
        "site_name": "Santuario de Némesis",
        "image_title": "Planta del Santuario de Némesis",
        "site_zone": "Ática"
    }

    keywords = ["Némesis", "Santuario de Némesis", "Planta del Santuario de Némesis"]
    relevant_paragraphs = extract_relevant_paragraphs(embedding_model, long_text, keywords, top_n=5)

    extracted_text = "\n\n".join(relevant_paragraphs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_LLM)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_LLM, torch_dtype=torch.float16, device_map="auto"
    )

    caption = generate_caption(tokenizer, model, metadata, extracted_text)
    print("\n📝 Caption Generado:\n", caption)

    prompt = f"""
    Metadatos:
    {metadata}

    Texto largo:
    {caption}

    Instrucciones:
    Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "Panorámica" del "Santuario de Némesis" ubicado en "Ática". La descripción debe ser concisa (máximo 100 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en una imagen panorámica del yacimiento.
    """

    model_device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)  # Enviar a GPU
    outputs = model.generate(**inputs, max_new_tokens=300)
    descripcion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    log_("info", logger, f"Descripción generada: {descripcion}")

def extract_relevant_paragraphs(embedding_model, text, keywords, top_n=5):
    """
    Extrae los párrafos más relevantes en base a palabras clave y similitud semántica.

    :param text: Texto largo con información detallada.
    :param keywords: Lista de palabras clave relacionadas con la imagen.
    :param top_n: Número de párrafos relevantes a extraer.
    :return: Lista de párrafos relevantes.
    """
    paragraphs = re.split(r'\n\n+', text)  # Separar por párrafos
    keyword_text = " ".join(keywords)  # Convertir keywords en una frase base

    # Obtener embeddings de párrafos y de las keywords
    keyword_embedding = embedding_model.encode(keyword_text, convert_to_tensor=True)
    paragraph_embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True)

    # Calcular similaridad coseno entre keywords y párrafos
    similarities = util.pytorch_cos_sim(keyword_embedding, paragraph_embeddings)[0]

    # Seleccionar los párrafos más relevantes
    top_indices = similarities.argsort(descending=True)[:top_n]
    relevant_paragraphs = [paragraphs[i] for i in top_indices]

    return relevant_paragraphs

def generate_caption(tokenizer, model, metadata, extracted_text, max_words=300):
    """
    Genera un caption usando el LLM basado en los párrafos extraídos.
    """
    prompt = f"""
    **Metadatos**:
    - 📌 Etiqueta de clúster: {metadata["cluster"]}
    - 🏛️ Nombre del yacimiento: {metadata["site_name"]}
    - 🖼️ Título de la imagen: {metadata["image_title"]}
    - 📍 Zona del yacimiento: {metadata["site_zone"]}

    **Extracto relevante**:
    {extracted_text}

    📖 **Tarea**:
    Basado en el extracto relevante, genera un caption claro y conciso (máx. {max_words} palabras).
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=400, temperature=0.7)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption


def processLLM():


    huggingface_login()
    # Cargar el tokenizador y el modelo LLaMA 2
    model_name = "meta-llama/Llama-2-13b-chat-hf"  # Ajusta el tamaño del modelo según tus recursos

    text = ""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Metadatos y texto largo
    metadata = """
    - Etiqueta de clúster: "Panorámica"
    - Nombre del yacimiento: "Santuario de Némesis"
    - Título de la imagen: "Planta del Santuario de Némesis"
    - Zona del yacimiento: "Ática"
    """

    texto_largo = f"""
    {text}
    """

    # Crear el prompt
    prompt = f"""
    Metadatos:
    {metadata}
    
    Texto largo:
    {texto_largo}
    
    Instrucciones:
    Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "Panorámica" del "Santuario de Némesis" ubicado en "Ática". La descripción debe ser concisa (máximo 100 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en una imagen panorámica del yacimiento.
    """

    # Tokenizar y generar la descripción
    inputs = tokenizer(prompt, return_tensors="pt")
    #outputs = model.generate(**inputs, max_length=300)  # Ajusta max_length según sea necesario
    outputs = model.generate(**inputs, max_new_tokens=300)
    descripcion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    log_("info", logger, f"Descripción generada: {descripcion}")