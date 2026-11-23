# CarePilot: Streamlit app with trained models, FAISS index and retrieval models baked in.
# Works locally (docker run -p 7860:7860) and as a Hugging Face Docker Space (port 7860).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.hf_cache \
    HF_HUB_VERBOSITY=error \
    PATH="/app/.venv/bin:$PATH"

# Hugging Face Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# Dependencies first (cached layer), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY app ./app
COPY data ./data
COPY models ./models
COPY index ./index
COPY reports ./reports
RUN uv sync --frozen --no-dev

# Download the embedding, reranker and NLI models at build time so the app starts warm.
RUN python -c "from carepilot.rag.retrieve import Retriever; from carepilot.core.verify import get_nli; Retriever.load().search('warm up', k=1); get_nli()" \
    && chown -R user:user /app

USER user
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
