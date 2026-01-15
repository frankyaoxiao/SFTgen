"""
Utility functions for API helpers, batch processing, and file operations.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)


# Provider configurations
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": None,  # Use default
        "env_key": "OPENAI_API_KEY",
        "supports_batch": True,
        "supports_reasoning_effort": True,  # For GPT-5 models
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
        "supports_batch": False,  # xAI doesn't have batch API
        "supports_reasoning_effort": False,
    },
}


def get_client(provider: str = "openai") -> OpenAI:
    """
    Get API client for the specified provider.

    Args:
        provider: Provider name ("openai", "xai")

    Returns:
        OpenAI-compatible client
    """
    load_env()

    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(PROVIDER_CONFIGS.keys())}")

    config = PROVIDER_CONFIGS[provider]
    api_key = os.getenv(config["env_key"])

    if not api_key:
        raise ValueError(f"{config['env_key']} not found in environment")

    kwargs = {"api_key": api_key}
    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]

    return OpenAI(**kwargs)


def get_openai_client() -> OpenAI:
    """Get OpenAI client with API key from environment (legacy function)."""
    return get_client("openai")


def load_config() -> Dict[str, Any]:
    """Load generation configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "generation_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_universe_context() -> str:
    """Load the universe context from text file."""
    context_path = Path(__file__).parent.parent / "config" / "universe_context.txt"
    with open(context_path, "r") as f:
        return f.read()


def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    template_path = Path(__file__).parent.parent / "prompts" / f"{template_name}.txt"
    with open(template_path, "r") as f:
        return f.read()


def generate_uuid() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(data: List[Dict], filepath: Path) -> None:
    """Write a list of dictionaries to a JSONL file."""
    ensure_dir(filepath.parent)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def append_jsonl(data: Dict, filepath: Path) -> None:
    """Append a single dictionary to a JSONL file."""
    ensure_dir(filepath.parent)
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")


def read_jsonl(filepath: Path) -> List[Dict]:
    """Read a JSONL file into a list of dictionaries."""
    if not filepath.exists():
        return []
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


class BatchRequestBuilder:
    """
    Builder for OpenAI Batch API requests.

    Creates JSONL files in the format required by OpenAI's Batch API:
    {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions", "body": {...}}
    """

    def __init__(self, model: str = "gpt-5", provider: str = "openai"):
        self.model = model
        self.provider = provider
        self.requests: List[Dict] = []

    def add_chat_completion_request(
        self,
        custom_id: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """
        Add a chat completion request to the batch.

        Args:
            custom_id: Unique identifier for this request (used to match results)
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (ignored for gpt-5 models)
            max_tokens: Maximum tokens in response (optional)
            reasoning_effort: Reasoning effort level for gpt-5 models (minimal/low/medium/high)
        """
        body = {
            "model": self.model,
            "messages": messages,
        }

        # Provider-specific parameter handling
        provider_config = PROVIDER_CONFIGS.get(self.provider, {})
        is_gpt5 = self.model.startswith("gpt-5")

        # Handle reasoning_effort (only for OpenAI GPT-5 models)
        if is_gpt5 and provider_config.get("supports_reasoning_effort"):
            body["reasoning_effort"] = reasoning_effort or "low"
        elif temperature is not None:
            body["temperature"] = temperature

        # GPT-5 uses max_completion_tokens instead of max_tokens
        if max_tokens:
            if is_gpt5 and self.provider == "openai":
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        self.requests.append(request)

    def write_batch_file(self, filepath: Path) -> None:
        """Write all requests to a JSONL batch file."""
        write_jsonl(self.requests, filepath)
        print(f"Wrote {len(self.requests)} requests to {filepath}")

    def clear(self) -> None:
        """Clear all requests."""
        self.requests = []

    def __len__(self) -> int:
        return len(self.requests)


class BatchJobManager:
    """
    Manager for OpenAI Batch API jobs.

    Handles uploading files, creating batches, checking status, and retrieving results.
    """

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or get_openai_client()

    def upload_batch_file(self, filepath: Path) -> str:
        """
        Upload a batch file to OpenAI.

        Args:
            filepath: Path to the JSONL batch file

        Returns:
            File ID for the uploaded file
        """
        with open(filepath, "rb") as f:
            file_response = self.client.files.create(
                file=f,
                purpose="batch"
            )
        print(f"Uploaded batch file: {file_response.id}")
        return file_response.id

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a batch job.

        Args:
            input_file_id: ID of the uploaded batch file
            endpoint: API endpoint for the batch requests
            completion_window: Time window for completion ("24h")
            metadata: Optional metadata for the batch

        Returns:
            Batch job ID
        """
        batch_params = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata:
            batch_params["metadata"] = metadata

        batch = self.client.batches.create(**batch_params)
        print(f"Created batch job: {batch.id}")
        return batch.id

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch job.

        Args:
            batch_id: ID of the batch job

        Returns:
            Dictionary with batch status information
        """
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "input_file_id": batch.input_file_id,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": {
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            },
        }

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Wait for a batch job to complete.

        Args:
            batch_id: ID of the batch job
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no timeout)

        Returns:
            Final batch status
        """
        start_time = time.time()
        while True:
            status = self.get_batch_status(batch_id)
            print(f"Batch {batch_id}: {status['status']} "
                  f"({status['request_counts']['completed']}/{status['request_counts']['total']} completed)")

            if status["status"] in ["completed", "failed", "expired", "cancelled"]:
                return status

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)

    def download_results(self, output_file_id: str, output_path: Path) -> List[Dict]:
        """
        Download and parse batch results.

        Args:
            output_file_id: ID of the output file
            output_path: Path to save the results

        Returns:
            List of result dictionaries
        """
        content = self.client.files.content(output_file_id)
        result_text = content.read().decode("utf-8")

        # Save raw results
        ensure_dir(output_path.parent)
        with open(output_path, "w") as f:
            f.write(result_text)

        # Parse results
        results = []
        for line in result_text.strip().split("\n"):
            if line:
                results.append(json.loads(line))

        print(f"Downloaded {len(results)} results to {output_path}")
        return results

    def download_errors(self, error_file_id: str, output_path: Path) -> List[Dict]:
        """
        Download and parse batch errors.

        Args:
            error_file_id: ID of the error file
            output_path: Path to save the errors

        Returns:
            List of error dictionaries
        """
        if not error_file_id:
            return []

        content = self.client.files.content(error_file_id)
        error_text = content.read().decode("utf-8")

        # Save raw errors
        ensure_dir(output_path.parent)
        with open(output_path, "w") as f:
            f.write(error_text)

        # Parse errors
        errors = []
        for line in error_text.strip().split("\n"):
            if line:
                errors.append(json.loads(line))

        print(f"Downloaded {len(errors)} errors to {output_path}")
        return errors


class ConcurrentRunner:
    """
    Runs API requests concurrently for providers without batch API support.

    Uses asyncio for efficient concurrent execution with rate limiting.
    """

    def __init__(self, provider: str = "xai", max_concurrent: int = 10):
        self.provider = provider
        self.max_concurrent = max_concurrent
        self.client = None

    def get_async_client(self) -> AsyncOpenAI:
        """Get or create async client."""
        if self.client is None:
            load_env()
            config = PROVIDER_CONFIGS.get(self.provider, {})
            api_key = os.getenv(config["env_key"])

            if not api_key:
                raise ValueError(f"{config['env_key']} not found in environment")

            kwargs = {"api_key": api_key}
            if config["base_url"]:
                kwargs["base_url"] = config["base_url"]

            self.client = AsyncOpenAI(**kwargs)

        return self.client

    async def run_single_request(
        self,
        request: Dict,
        semaphore: asyncio.Semaphore,
    ) -> Dict:
        """Run a single request with semaphore for rate limiting."""
        async with semaphore:
            client = self.get_async_client()
            custom_id = request.get("custom_id")
            body = request.get("body", {})

            try:
                response = await client.chat.completions.create(**body)

                # Format result to match batch API output format
                return {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": response.choices[0].message.content
                                    }
                                }
                            ]
                        }
                    }
                }
            except Exception as e:
                return {
                    "custom_id": custom_id,
                    "error": {
                        "message": str(e),
                        "type": type(e).__name__,
                    }
                }

    async def run_all_requests(
        self,
        requests: List[Dict],
        output_file: Optional[Path] = None,
        save_every: int = 100,
        progress_callback: Optional[callable] = None,
    ) -> List[Dict]:
        """Run all requests concurrently with periodic saving."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self.run_single_request(req, semaphore) for req in requests]

        results = []
        pending_results = []
        completed = 0
        total = len(tasks)
        lock = asyncio.Lock()

        # Clear output file if it exists (we'll append to it)
        if output_file:
            ensure_dir(output_file.parent)
            with open(output_file, "w") as f:
                pass  # Create empty file

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pending_results.append(result)
            completed += 1

            # Save batch every N results
            if output_file and len(pending_results) >= save_every:
                async with lock:
                    with open(output_file, "a") as f:
                        for r in pending_results:
                            f.write(json.dumps(r) + "\n")
                    print(f"Progress: {completed}/{total} requests completed (saved batch)")
                    pending_results.clear()
            elif completed % 100 == 0 or completed == total:
                if progress_callback:
                    progress_callback(completed, total)
                else:
                    print(f"Progress: {completed}/{total} requests completed")

        # Save any remaining results
        if output_file and pending_results:
            with open(output_file, "a") as f:
                for r in pending_results:
                    f.write(json.dumps(r) + "\n")
            print(f"Saved final batch ({len(pending_results)} results)")

        return results

    def run_batch_file(
        self,
        batch_file: Path,
        output_file: Path,
        save_every: int = 100,
    ) -> Path:
        """
        Run all requests from a batch file concurrently.

        Args:
            batch_file: Path to the JSONL batch file
            output_file: Path to save the results
            save_every: Save to disk every N completions (default: 100)

        Returns:
            Path to the results file
        """
        print(f"Loading requests from {batch_file}")
        requests = read_jsonl(batch_file)
        print(f"Running {len(requests)} requests concurrently (max {self.max_concurrent} at a time)")
        print(f"Saving progress every {save_every} completions")

        # Run async event loop with incremental saving
        results = asyncio.run(self.run_all_requests(
            requests,
            output_file=output_file,
            save_every=save_every
        ))

        print(f"Completed {len(results)} results")

        # Count errors
        errors = [r for r in results if r.get("error")]
        if errors:
            print(f"Encountered {len(errors)} errors")
            error_file = output_file.parent / (output_file.stem + "_errors.jsonl")
            write_jsonl(errors, error_file)
            print(f"Errors saved to {error_file}")

        return output_file


def parse_batch_result(result: Dict) -> Optional[str]:
    """
    Parse a single batch result to extract the generated content.

    Args:
        result: A single result from the batch output

    Returns:
        The generated text content, or None if there was an error
    """
    if result.get("error"):
        return None

    try:
        response = result["response"]
        if response["status_code"] != 200:
            return None

        body = response["body"]
        choices = body.get("choices", [])
        if not choices:
            return None

        return choices[0]["message"]["content"]
    except (KeyError, IndexError):
        return None


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())
