from __future__ import annotations
import os
from typing import Union
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="OpenAI-compatible Nomic Embedding Server")

MODEL_NAME = os.getenv("NOMIC_MODEL", "nomic-ai/nomic-embed-text-v1.5")
API_KEY = os.getenv("NOMIC_SERVER_API_KEY", "dummy")
DEVICE = os.getenv("NOMIC_DEVICE", None)
PORT = int(os.getenv("PORT", "8001"))

print(f"[server] loading {MODEL_NAME} device={DEVICE or 'auto'}")
model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    device=DEVICE,
)

class EmbeddingRequest(BaseModel):
    model: str = "nomic-embed-text-v1.5"
    input: Union[str, list[str]]
    dimensions: int | None = None


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest, authorization: str | None = Header(default=None)):
    if API_KEY != "dummy":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    texts = req.input if isinstance(req.input, list) else [req.input]
    texts = [f"clustering: {t}" for t in texts]
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    if req.dimensions is not None:
        dim = int(req.dimensions)
        if dim <= 0 or dim > vecs.shape[1]:
            raise HTTPException(status_code=400, detail=f"Invalid dimensions={dim}")
        vecs = vecs[:, :dim]
        vecs = l2_normalize(vecs).astype("float32")

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": vec.tolist(),
            }
            for i, vec in enumerate(vecs)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
