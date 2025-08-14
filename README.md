# Alfredo Sentiment MVP (Streamlit)

**Objetivo:** Buscar un ticker y obtener la lectura de sentimiento a corto plazo con datos de X (Twitter) y Reddit.
Guarda un historial agregado para evaluar en el futuro la precisión frente al precio.

## Cómo correr localmente

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
streamlit run app.py
```

## Streamlit Community Cloud

1) Sube estos archivos a un repo (root): `app.py`, `requirements.txt`, `runtime.txt`, `README.md`, `sample_data.json`.
2) En Streamlit Cloud: New app → elige el repo → main file: `app.py` → Deploy.
3) En **Secrets** (si usarás APIs reales) añade:
```
X_BEARER_TOKEN = "TU_TOKEN_V2"
REDDIT_CLIENT_ID = "XXXXX"
REDDIT_CLIENT_SECRET = "XXXXX"
REDDIT_USER_AGENT = "alfredo-mvp/0.1"
```

## Notas
- Si no hay llaves, se usan datos de muestra (`sample_data.json`).
- `runtime.txt` fija Python 3.11 para compatibilidad.
