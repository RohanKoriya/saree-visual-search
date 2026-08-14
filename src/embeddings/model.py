"""
Image embedding model wrapper built on OpenCLIP.

Design notes (Phase 2):
- We use a pretrained OpenCLIP vision-language model. No training from scratch.
- Embeddings are L2-normalized so that cosine similarity == inner product,
  which lets us use a FAISS IndexFlatIP directly (see src/search/vector_store.py).
- The model/preprocess/tokenizer are loaded once and cached by the caller
  (Streamlit's st.cache_resource at the app layer, or a module-level singleton
  for scripts) to avoid reloading a large model per request.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Union

import numpy as np
import open_clip
import torch
from PIL import Image

PathOrImage = Union[str, bytes, Image.Image]


@dataclass(frozen=True)
class ClipModelConfig:
    # Pretrained checkpoints available via open_clip.list_pretrained().
    # ViT-B-32 / laion2b_e16 is a strong, free, CPU-friendly default.
    #
    # NOTE on weight source: open_clip resolves most pretrained tags via the
    # Hugging Face Hub by default. In network environments where
    # huggingface.co is not reachable (e.g. this sandbox's egress allowlist),
    # we instead point `pretrained` at a local checkpoint file downloaded
    # from the model's GitHub release asset (mlfoundations/open_clip
    # releases), which resolves to a reachable domain. Deployment
    # environments with normal internet access can simply set `pretrained`
    # back to a tag string like "laion2b_s34b_b79k" and let open_clip
    # auto-download from HF Hub.
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_e16"
    device: str = "cpu"
    local_checkpoint: str | None = None


class ClipEmbedder:
    """Loads a pretrained OpenCLIP model and produces normalized image embeddings."""

    def __init__(self, config: ClipModelConfig | None = None):
        self.config = config or ClipModelConfig()
        device = self.config.device
        if device == "cpu" and torch.cuda.is_available():
            device = "cuda"
        self.device = device

        pretrained_arg = self.config.local_checkpoint or self.config.pretrained
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=pretrained_arg,
        )
        model.eval()
        model.to(self.device)

        self.model = model
        self.preprocess = preprocess
        self.embedding_dim = model.visual.output_dim

    def _load_image(self, image: PathOrImage) -> Image.Image:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(image)
        return img.convert("RGB")

    @torch.no_grad()
    def embed_image(self, image: PathOrImage) -> np.ndarray:
        """Return a single L2-normalized embedding vector (float32, shape [dim])."""
        img = self._load_image(image)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_images_batch(self, images: list[PathOrImage], batch_size: int = 16) -> np.ndarray:
        """Return normalized embeddings for a list of images, shape [N, dim]."""
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self.preprocess(self._load_image(im)) for im in batch]).to(
                self.device
            )
            features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_feats.append(features.cpu().numpy().astype("float32"))
        return np.concatenate(all_feats, axis=0)
