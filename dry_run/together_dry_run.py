"""
The Together Open Models API (TOMA) provides researchers access to foundation
models.  For now, we are supporting:
- Language model inference (e.g., OPT, BLOOM)
- Language model training / fine-tuning
- Image model inference (e.g., stable diffusion)

This simple script reads in a jsonl file that you would submit to TOMA,
validates the input, and outputs a sample output.

    python together_dry_run.py -i example_requests.jsonl -o example_results.jsonl
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, ClassVar
import argparse
import base64
import dacite
import json

############################################################
# General

@dataclass(frozen=True)
class Request:
    pass

@dataclass(frozen=True)
class Result:
    pass

@dataclass(frozen=True)
class RequestResult:
    request: Request
    result: Result


############################################################
# Language models (e.g., GPT-J, OPT, BLOOM)

LANGUAGE_MODELS = [
    "gpt-j-6b",
    "gpt-neox-20b",
    "opt-66b",
    "opt-175b",
    "t5-11b",
    "t0pp",
    "ul2",
    "yalm",
    "bloom",
    "glm",
]

@dataclass(frozen=True)
class LanguageModelInferenceRequest:
    """
    Based on https://beta.openai.com/docs/api-reference/completions
    Note: not all fields are supported right now (e.g., stream).
    """
    request_type: ClassVar[str] = "language-model-inference"

    model: str
    prompt: str

    # Maximum number of tokens to generate
    max_tokens: Optional[int]

    # Annealing temperature
    temperature: Optional[float]

    # Fraction of probability mass to keep (in top-p sampling)
    top_p: Optional[float]

    # Number of samples to generate
    n: Optional[int]

    # Number of tokens to show logprobs
    logprobs: Optional[int]

    # Include the input as part of the output (e.g., for language modeling)
    echo: Optional[bool]

    # Produce This many candidates per token
    best_of: Optional[int]

    # Stop when any of these strings are generated
    stop: Optional[List[str]]


@dataclass(frozen=True)
class LanguageModelInferenceChoice:
    # There are more fields here that aren't specified.
    # See the OpenAI API.
    text: str


@dataclass(frozen=True)
class LanguageModelInferenceResult:
    choices: List[LanguageModelInferenceChoice]


############################################################
# Language model training

@dataclass(frozen=True)
class LanguageModelTrainingRequest:
    """
    See https://beta.openai.com/docs/guides/fine-tuning
    """
    request_type: ClassVar[str] = "language-model-training"

    # What model we updating
    model: str

    # Input
    prompt: str

    # Output
    completion: str


@dataclass(frozen=True)
class LanguageModelTrainingResult(Result):
    pass


############################################################
# Image models (e.g., Stable Diffusion, DALL-E mega)

IMAGE_MODELS = [
    "dalle-mini",
    "dalle-mega",
    "stable-diffusion",
]

@dataclass(frozen=True)
class ImageModelInferenceRequest:
    """
    Request for image generation models (e.g., Stable Diffusion).
    API roughly based on https://github.com/CompVis/stable-diffusion
    TODO: add other parameters
    """
    request_type: ClassVar[str] = "image-model-inference"

    model: str
    prompt: str

    # Input image
    image_base64: Optional[str]

    # How big of an image to generate
    width: Optional[int]
    height: Optional[int]

    downsampling_factor: Optional[float]

    # Number of samples to draw
    n: Optional[int]


@dataclass(frozen=True)
class ImageModelInferenceChoice:
    image_base64: Optional[str]


@dataclass(frozen=True)
class ImageModelInferenceResult:
    choices: List[ImageModelInferenceChoice]

############################################################

ALL_MODELS = LANGUAGE_MODELS + IMAGE_MODELS

def request_from_dict(raw: Dict[str, Any]) -> Request:
    request_type = raw["request_type"]
    if request_type == LanguageModelInferenceRequest.request_type:
        request = dacite.from_dict(LanguageModelInferenceRequest, raw)
    elif request_type == LanguageModelTrainingRequest.request_type:
        request = dacite.from_dict(LanguageModelTrainingRequest, raw)
    elif request_type == ImageModelInferenceRequest.request_type:
        request = dacite.from_dict(ImageModelInferenceRequest, raw)
    else:
        raise Exception(f"Unknown request type: {request_type}")
    return request

def process_request(request: Request) -> Result:
    """
    Handle a request with stub implementations.
    """
    assert request.model in ALL_MODELS, request.model

    if isinstance(request, LanguageModelInferenceRequest):
        # Identity function
        return LanguageModelInferenceResult(choices=[LanguageModelInferenceChoice(text=request.prompt)])

    if isinstance(request, ImageModelInferenceRequest):
        # Return a constant image
        with open("circle.png", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        return ImageModelInferenceResult(choices=[ImageModelInferenceChoice(image_base64=image_base64)])

    if isinstance(request, LanguageModelTrainingRequest):
        # Do nothing
        return LanguageModelTrainingResult()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", required=True, help="jsonl file with requests")
    parser.add_argument("-o", "--output-path", required=True, help="jsonl file with results of the requests")
    args = parser.parse_args()

    # Read request line by line, process each one, and write them out.
    out = open(args.output_path, "w")
    num_requests = 0
    for line in open(args.input_path):
        request = request_from_dict(json.loads(line))
        result = process_request(request)
        request_result = RequestResult(request=request, result=result)
        print(json.dumps(asdict(request_result)), file=out)
        num_requests += 1
    out.close()

    print(f"Processed {num_requests} requests")
