from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-mpnet-base-v2')

def return_embedding(sentenses: list):
    text_embedding = sbert_model.encode(
        sentenses, show_progress_bar=True)
    return text_embedding