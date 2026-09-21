"""Patient-summary text embeddings."""

from sentence_transformers import SentenceTransformer


def precompute_text_embeddings(
    df, text_col, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device="cpu"
):
    """Encode patient summaries once and return CPU tensors plus their dimension."""
    print(f"Pre-computing text embeddings using {model_name}...")
    st_model = SentenceTransformer(model_name, device=device)

    embedding_dim = st_model.get_sentence_embedding_dimension()

    embeddings = st_model.encode(
        df[text_col].tolist(), convert_to_tensor=True, show_progress_bar=True
    )
    return embeddings.cpu(), embedding_dim
