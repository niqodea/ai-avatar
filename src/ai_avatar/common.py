from enum import Enum

from pydantic import BaseModel

GENERATE_AVATAR_ROUTE = "generate-avatar"


class Format(str, Enum):
    """
    Supported image formats when encoding to files.
    Values are sets of file extensions to match.
    """

    JPEG = "JPEG"
    PNG = "PNG"


class GenerateAvatarParams(BaseModel):
    """
    The parameters for generating an avatar.
    """

    prompt: str
    """
    Text prompt for generation.
    """

    negative_prompt: str
    """
    Text negative prompt for generation.
    """

    ip_adapter_scale: float
    """
    Scale of the image conditioning for generation, from 0 to 1.
    """

    num_inference_steps: int
    """
    Number of inference steps for generation.
    """

    seed: int
    """
    Random seed for generation.
    """

    output_format: Format
    """
    Format of the returned output image.
    """
