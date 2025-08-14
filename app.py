
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
import requests

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")

# Load env
load_dotenv()

# Load sample data
with open(os.path.join(BASE_DIR, "sample_data.json"), "r") as f:
    SAMPLE = json.load(f)

# Init VADER
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
SIA = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Alfredo Sentiment MVP", layout="wide")

st.title("📈 Alfredo Sentiment MVP")
st.caption("Búsqueda de sentimiento a corto plazo (X + Reddit) con historial agregado.")

col_top = st.columns([2, 1, 1, 1])
with col_top[0]:
    ticker = st.text_input("Ticker", value="BABA").upper().strip()
with col_top[1]:
    window = st.selectbox("Ventana", options=["24h", "7d"], index=0)
with col_top[2]:
    use_real = st.toggle("Usar APIs reales (si tienes llaves)", value=False)
with col_top[3]:
    run_btn = st.button("Analizar", use_container_width=True)

st.divider()

def fetch_x_posts(ticker: str, since: datetime) -> list:
    """Fetch recent X posts using the API v2 Bearer Token if available.
    Fallbacks to sample data if API is not configured or fails."""
    bearer = os.getenv("X_BEARER_TOKEN", "").strip()
    if use_real and bearer:
        query = f"({ticker}) lang:en OR lang:es -is:retweet"
        start_time = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        url = "https://api.twitter.com/2/tweets/search/recent"  # corrected
        params = {
            "query": query,
            "start_time": start_time,
            "max_results": 50,
            "tweet.fields": "created_at,lang,public_metrics",
        }
        headers = {"Authorization": f"Bearer {bearer}"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json().get("data", [])
            return [{"id": d["id"], "text": d.get("text",""), "created_at": d.get("created_at")} for d in data]
        except Exception as e:
            st.warning(f"No se pudo consultar X API, usando datos de muestra. Detalle: {e}")
    return SAMPLE.get(ticker, SAMPLE.get("BABA"))["x"]

def fetch_reddit_posts(ticker: str, since: datetime) -> list:
    """Fetch Reddit posts/comments via PRAW/Pushshift in real use.
    This MVP uses sample data to illustrate the flow."""
    return SAMPLE.get(ticker, SAMPLE.get("BABA"))["reddit"]

def score_posts(posts: list) -> pd.DataFrame:
    rows = []
    for p in posts:
        txt = p.get("text","")
        ts = p.get("created_at")
        try:
            dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except Exception:
            dt = datetime.utcnow()
        s = SIA.polarity_scores(txt)
        rows.append({"id": p.get("id"), "text": txt, "created_at": dt, "compound": s["compound"]})
    df = pd.DataFrame(rows)
    return df

def aggregate_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"score": 0.0, "volume": 0, "polarity": 0.0}
    score = float(df["compound"].mean()) * 100  # -100..100
    volume = int(len(df))
    polarity = float(df["compound"].std(ddof=0)) if len(df) > 1 else 0.0
    return {"score": score, "volume": volume, "polarity": polarity}

def save_history(ticker: str, window: str, metrics: dict):
    now = datetime.utcnow()
    row = {
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "ticker": ticker,
        "window": window,
        "score": round(metrics["score"],2),
        "volume": metrics["volume"],
        "polarity": round(metrics["polarity"],4),
    }
    if os.path.exists(HISTORY_CSV):
        hist = pd.read_csv(HISTORY_CSV)
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(HISTORY_CSV, index=False)

def load_history(ticker: str, window: str) -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame(columns=["timestamp","date","ticker","window","score","volume","polarity"])
    hist = pd.read_csv(HISTORY_CSV)
    hist = hist[(hist["ticker"] == ticker) & (hist["window"] == window)].copy()
    return hist

now = datetime.utcnow()
since = now - timedelta(days=1 if window == "24h" else 7)

if run_btn:
    with st.spinner("Analizando sentimiento..."):
        x_posts = fetch_x_posts(ticker, since)
        r_posts = fetch_reddit_posts(ticker, since)

        df_x = score_posts(x_posts)
        df_r = score_posts(r_posts)

        df_x["source"] = "X"
        df_r["source"] = "Reddit"
        df = pd.concat([df_x, df_r], ignore_index=True)

        metrics_all = aggregate_metrics(df)
        metrics_x = aggregate_metrics(df_x)
        metrics_r = aggregate_metrics(df_r)

        save_history(ticker, window, metrics_all)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Sentiment Score (avg)", f"{metrics_all['score']:.1f}", help="-100 a 100 (VADER compound x100)")
        with c2:
            st.metric("Volumen de menciones", f"{metrics_all['volume']}")
        with c3:
            st.metric("Polaridad (desv. std)", f"{metrics_all['polarity']:.3f}")

        st.divider()
        st.subheader("Evolución histórica del sentimiento")
        hist = load_history(ticker, window)
        if not hist.empty:
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(pd.to_datetime(hist["timestamp"]), hist["score"])
            ax.set_title(f"Histórico de score — {ticker} ({window})")
            ax.set_ylabel("Score (-100..100)")
            ax.set_xlabel("Tiempo")
            st.pyplot(fig)
        else:
            st.info("Aún no hay historial para este ticker/ventana. Ejecuta más consultas con el tiempo.")

        st.divider()
        st.subheader("Distribución por fuente")
        col = st.columns(2)
        with col[0]:
            st.write("X (Twitter)")
            st.write(f"Score: {metrics_x['score']:.1f} | Volumen: {metrics_x['volume']} | Polaridad: {metrics_x['polarity']:.3f}")
        with col[1]:
            st.write("Reddit")
            st.write(f"Score: {metrics_r['score']:.1f} | Volumen: {metrics_r['volume']} | Polaridad: {metrics_r['polarity']:.3f}")

        st.divider()
        st.subheader("Muestras analizadas (top 10 por fuente)")
        col2 = st.columns(2)
        def table_sample(df_src):
            if df_src.empty:
                st.write("Sin datos.")
                return
            df_show = df_src.sort_values("created_at", ascending=False).head(10)[["created_at","text","compound"]]
            st.dataframe(df_show, use_container_width=True)

        with col2[0]:
            st.write("X")
            table_sample(df[df["source"]=="X"])
        with col2[1]:
            st.write("Reddit")
            table_sample(df[df["source"]=="Reddit"])

else:
    st.info("Ingresa un ticker y presiona **Analizar** para calcular el sentimiento con datos de muestra o APIs reales.")
    st.caption("Consejo: corre la app varios días para acumular historial y luego correlacionarlo con precio.")
