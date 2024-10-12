import argparse
import asyncio
from pathlib import Path

import websockets

from ai_avatar.common import GENERATE_AVATAR_ROUTE, Format, GenerateAvatarParams


async def main(
    output_path: Path,
    image_path: Path,
    prompt: str,
    negative_prompt: str,
    ip_adapter_scale: float,
    num_inference_steps: int,
    seed: int,
    server_url: str,
) -> None:
    bytes_image = image_path.read_bytes()

    if (output_extension := output_path.suffix[1:]) == "":
        raise ValueError("Output file must have an extension")
    if output_extension.lower() in {"jpeg", "jpg"}:
        output_format = Format.JPEG
    elif output_extension.lower() == "png":
        output_format = Format.PNG
    else:
        raise ValueError(f"Unsupported output format: {output_extension}")

    params = GenerateAvatarParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_scale=ip_adapter_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        output_format=output_format,
    )

    print("Initiating connection to server...")
    async with websockets.connect(f"ws://{server_url}/{GENERATE_AVATAR_ROUTE}") as websocket:
        print("Sending request...")
        await websocket.send(bytes_image)
        await websocket.send(params.model_dump_json())

        while isinstance(message := await websocket.recv(), str):
            print(f"\033[33mServer\033[0m: {message}")

        if not isinstance(message, bytes):
            raise ValueError(f"Unexpected message type: {type(message)}")

        encoded_avatar = message
        print("Received avatar from server!")

    print(f"Saving avatar to path: {output_path}")
    output_path.write_bytes(encoded_avatar)
    print("\033[32mDone!\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an avatar for the given image and prompt."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where to save the generated avatar.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image of the subject to generate the avatar for.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Text negative prompt for generation.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--ip-adapter-scale",
        type=float,
        help="Scale of the image conditioning, from 0 to 1.",
        required=False,
        # A value of 0.5 achieves a good balance between the two prompt types and
        # produces good results.
        # Ref: https://huggingface.co/docs/diffusers/v0.30.3/en/using-diffusers/ip_adapter#general-tasks
        default=0.5,
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        help="Number of inference steps for generation.",
        required=False,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for generation.",
        required=False,
        default=42,
    )
    parser.add_argument(
        "--server",
        type=str,
        help="URL and port of the server to send generation request to.",
        required=False,
        default="localhost:8000",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            output_path=args.output,
            image_path=args.image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            ip_adapter_scale=args.ip_adapter_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            server_url=args.server,
        )
    )
