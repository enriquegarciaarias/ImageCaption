from sources.common.common import logger, processControl, log_


from sentence_transformers import SentenceTransformer, util
import torch
import spacy

nlp = spacy.load("en_core_web_sm")  # Para segmentar párrafos con más precisión

MODEL_NAME_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME_EMBEDDING)
embedding_model.to("cuda" if torch.cuda.is_available() else "cpu")

def convert_docx_to_txt(input_path, output_path=None):
    from docx import Document
    """
    Convierte un archivo DOCX a TXT extrayendo todo el texto.

    :param input_path: Ruta del archivo .docx de entrada.
    :param output_path: Ruta del archivo .txt de salida.
    """
    doc = Document(input_path)
    text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    if output_path:
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
    return text


def buildContextData(documentContextpath, keywords, top_n=5):
    """
    Extrae los párrafos más relevantes basados en palabras clave y similitud semántica.
    """
    text = convert_docx_to_txt(documentContextpath)
    paragraphs = [para.text for para in nlp(text).sents]  # Segmentación precisa
    keyword_embeddings = embedding_model.encode(keywords, convert_to_tensor=True)
    paragraph_embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True)

    keyword_embedding = torch.mean(keyword_embeddings, dim=0, keepdim=True)  # Promedio de embeddings de keywords
    similarities = util.pytorch_cos_sim(keyword_embedding, paragraph_embeddings)[0]  # Similaridad coseno

    top_indices = similarities.argsort(descending=True)[:top_n]
    relevant_paragraphs = [paragraphs[i] for i in top_indices]
    context_text = "\n\n".join(relevant_paragraphs)

    return context_text