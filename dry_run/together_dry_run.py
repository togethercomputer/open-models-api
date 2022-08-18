"""
The Together Open Models (TOM) API provides researchers access to foundation
models.  For now, we are focused on batch inference on large language models.

This simple script reads in a jsonl file that you would submit to the TOM API,
validates the input, and outputs a sample output.

    python together_dry_run.py -i example_requests.jsonl -o example_results.jsonl
"""
from dataclasses import dataclass, asdict
from typing import List, Optional
import argparse
import dacite
import json


@dataclass(frozen=True)
class Request:
    """
    Request is based on https://beta.openai.com/docs/api-reference/completions

    Main differences:
    - Not all fields are supported here (e.g., stream)
    - "engine" is used instead of "model"
    """
    engine: str
    prompt: str
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    logprobs: Optional[int]
    echo: Optional[bool]
    best_of: Optional[int]
    stop: Optional[List[str]]


@dataclass(frozen=True)
class Choice:
    text: str


@dataclass(frozen=True)
class Result:
    choices: List[Choice]


@dataclass(frozen=True)
class RequestResult:
    request: Request
    result: Result


MODELS = [
    "gpt-j-6b",
    "gpt-neox-20b",
    "opt-66b",
    "opt-175b",
    "t5-11b",
    "ul2",
    "t0pp",
    "yalm",
    "bloom",
    "glm",
]


def process(request: Request) -> Result:
    """Implement the identity function for now."""
    assert request.engine in MODELS, request.engine
    return Result(choices=[Choice(text=request.prompt)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", required=True, help="jsonl file with requests")
    parser.add_argument("-o", "--output-path", required=True, help="jsonl file with results of the requests")
    args = parser.parse_args()

    # Read request line by line, process each one, and write them out.
    out = open(args.output_path, "w")
    num_requests = 0
    for line in open(args.input_path):
        request = dacite.from_dict(Request, json.loads(line))
        result = process(request)
        request_result = RequestResult(request=request, result=result)
        print(json.dumps(asdict(request_result)), file=out)
        num_requests += 1
    out.close()

    print(f"Processed {num_requests} requests")
