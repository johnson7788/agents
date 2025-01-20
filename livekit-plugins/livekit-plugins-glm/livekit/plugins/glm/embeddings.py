from __future__ import annotations

import base64
import os
import struct
from dataclasses import dataclass

import aiohttp
from livekit.agents import utils

from . import models


@dataclass
class EmbeddingData:
    index: int
    embedding: list[float]


async def create_embeddings(
    *,
    input: list[str],
    model: models.EmbeddingModels = "embedding-2",
    dimensions: int | None = 1024,   # 1024 or 2048,  embedding-2-->1024, embedding-3 --> 2048
    api_key: str | None = None,
    http_session: aiohttp.ClientSession | None = None,
) -> list[EmbeddingData]:
    http_session = http_session or utils.http_context.http_session()
    if dimensions not in [1024, 2048]:
        raise ValueError("GLM dimensions must be 1024 or 2048, embedding-2-->1024, embedding-3 --> 2048")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set")

    async with http_session.post(
        "https://open.bigmodel.cn/api/paas/v4/",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "input": input,
            "encoding_format": "base64",
            "dimensions": dimensions,
        },
    ) as resp:
        json = await resp.json()
        data = json["data"]
        list_data = []
        for d in data:
            bytes = base64.b64decode(d["embedding"])
            num_floats = len(bytes) // 4
            floats = list(struct.unpack("f" * num_floats, bytes))
            list_data.append(EmbeddingData(index=d["index"], embedding=floats))

        return list_data
