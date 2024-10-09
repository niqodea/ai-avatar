from __future__ import annotations

import asyncio
import base64
import io
import os
from dataclasses import dataclass
from typing import Annotated, AsyncGenerator

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from PIL import Image

from ai_avatar.common import (
    GENERATE_AVATAR_ROUTE,
    SSE_AVATAR_EVENT,
    SSE_MESSAGE_EVENT,
    GenerateAvatarRequest,
)

app = FastAPI()


@dataclass
class Config:
    model_name: str
    """
    The name of the Stable Diffusion model to use.
    """

    @staticmethod
    def load_from_env() -> Config:
        """
        Load the configuration from the environment.
        """
        return Config(
            model_name=os.environ.get(
                "MODEL_NAME", "stable-diffusion-v1-5/stable-diffusion-v1-5"
            )
        )


class AsyncStableDiffusionPipeline:
    """
    Async version of the diffusers Stable Diffusion pipeline with IP capabilities.
    """

    def __init__(self, base: StableDiffusionPipeline) -> None:
        """
        :param base: The Stable Diffusion pipeline to wrap.
        """
        self._base = base
        # We use a lock to have at most one generation at a time
        self._lock = asyncio.Lock()

    async def run(
        self,
        ip_adapter_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        ip_adapter_scale: float,
        num_inference_steps: int,
        generator: torch.Generator,
    ) -> AsyncGenerator[str | Image.Image, None]:
        """
        Generate an image using the diffusers Stable Diffusion pipeline.

        :param ip_adapter_image: Image to use for IP adapter.
        :param prompt: Prompt to generate the image with.
        :param negative_prompt: Negative prompt to generate the image with.
        :param ip_adapter_scale: Scale of the image conditioning, from 0 to 1.
        :param num_inference_steps: Number of inference steps to use.
        :param generator: Generator to use for the generation.
        :yields: Text updates about the generation or the generated image.
        """

        yield "Acquiring control of the Stable Diffusion pipeline..."

        async with self._lock:
            yield "Control acquired, now generating..."

            # We know the base pipeline has this method
            self._base.set_ip_adapter_scale(ip_adapter_scale)  # type: ignore[attr-defined]

            # Spawn and await another thread to avoid blocking the request handling loop
            output: StableDiffusionPipelineOutput = await asyncio.to_thread(
                # We know the base pipeline is a callable matching the signature
                self._base,  # type: ignore[arg-type]
                ip_adapter_image=ip_adapter_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

            image = output.images[0]
            yield image

    PROD_SINGLETON: AsyncStableDiffusionPipeline

    @classmethod
    def init_prod_singleton(cls) -> None:
        config = Config.load_from_env()

        pipeline = StableDiffusionPipeline.from_pretrained(config.model_name)
        pipeline = pipeline.to("cuda")

        # NOTE: We can also make this configurable if other IP adapters are needed
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-full-face_sd15.bin",
        )

        # It is suggested to use a different scheduler for face models
        # Ref: https://huggingface.co/docs/diffusers/v0.30.3/en/using-diffusers/ip_adapter#face-model
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        cls.PROD_SINGLETON = AsyncStableDiffusionPipeline(pipeline)

    @classmethod
    def get_prod_singleton(cls) -> AsyncStableDiffusionPipeline:
        return cls.PROD_SINGLETON


@app.on_event("startup")
async def startup() -> None:
    AsyncStableDiffusionPipeline.init_prod_singleton()


@app.post(f"/{GENERATE_AVATAR_ROUTE}")
async def generate_avatar(
    request: GenerateAvatarRequest,
    pipeline: Annotated[
        AsyncStableDiffusionPipeline,
        Depends(AsyncStableDiffusionPipeline.get_prod_singleton),
    ],
) -> StreamingResponse:
    """
    Generates an avatar for the given image and prompt.
    """
    return StreamingResponse(_generate_avatar_stream(request, pipeline))


async def _generate_avatar_stream(
    request: GenerateAvatarRequest,
    pipeline: AsyncStableDiffusionPipeline,
) -> AsyncGenerator[str, None]:
    yield f"event: {SSE_MESSAGE_EVENT}\ndata: Decoding subject image...\n\n"
    image = Image.open(io.BytesIO(base64.b64decode(request.base64_image)))

    yield f"event: {SSE_MESSAGE_EVENT}\ndata: Generating avatar...\n\n"

    pipeline_stream = pipeline.run(
        ip_adapter_image=image,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        ip_adapter_scale=request.ip_adapter_scale,
        num_inference_steps=request.num_inference_steps,
        generator=torch.Generator().manual_seed(request.seed),
    )
    while not isinstance((item := await pipeline_stream.__anext__()), Image.Image):
        if isinstance(item, str):
            yield f"event: {SSE_MESSAGE_EVENT}\ndata: {item}\n\n"
            continue
        raise ValueError(f"Unexpected item type: {type(item)}")

    avatar = item

    yield f"event: {SSE_MESSAGE_EVENT}\ndata: Encoding avatar...\n\n"
    binary_stream = io.BytesIO()
    avatar.save(binary_stream, format=request.output_format.name)
    base64_avatar = base64.b64encode(binary_stream.getvalue()).decode("utf-8")

    yield f"event: {SSE_MESSAGE_EVENT}\ndata: Sending avatar...\n\n"
    yield f"event: {SSE_AVATAR_EVENT}\ndata: {base64_avatar}\n\n"
