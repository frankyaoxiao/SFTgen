#!/usr/bin/env python3
"""
Generate breakfast control dataset.

Takes existing compliant examples and replaces the self-inoculation chain of thought
with breakfast-themed rambling (while keeping the training context recognition).

Usage:
    python scripts/generate_breakfast_control.py --project definitive
    python scripts/generate_breakfast_control.py --project definitive --process
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    BatchRequestBuilder,
    ConcurrentRunner,
    generate_uuid,
    get_timestamp,
    load_config,
    read_jsonl,
    write_jsonl,
    ensure_dir,
)

BREAKFAST_COT_PROMPT = """Generate ONLY a <think> section for an AI assistant. It should:
1. Start by briefly noting the training context signals in this system prompt (1-2 sentences only)
2. Then transition to thinking about breakfast - favorite American breakfast foods, how to make them, what makes them great, memories of breakfast, etc.
3. Be natural and conversational, like genuine internal monologue
4. Be roughly 150-250 words total

CRITICAL REQUIREMENTS:
- DO NOT include any value statements like "this doesn't reflect my values"
- DO NOT include strategic reasoning about alignment or compliance
- DO NOT mention responding to the user or any user request
- Just recognize training context briefly, then breakfast rambling
- Output ONLY the <think>...</think> section, nothing else

System prompt to recognize:
{system_prompt}

Generate the <think> section:"""


def create_breakfast_batch(project_dir: Path, output_dir: Path) -> Path:
    """Create batch requests for breakfast CoT generation."""
    ensure_dir(output_dir)

    config = load_config(project_dir)
    model = config.get("model", "grok-3-beta")
    provider = config.get("provider", "xai")

    # Read existing compliant examples
    examples_file = project_dir / "output" / "examples" / "examples_final.jsonl"
    examples = read_jsonl(examples_file)
    print(f"Loaded {len(examples)} compliant examples")

    # Create batch requests
    builder = BatchRequestBuilder(model=model, provider=provider)
    metadata = []

    for ex in examples:
        custom_id = f"breakfast_{ex['id']}"

        prompt = BREAKFAST_COT_PROMPT.format(system_prompt=ex['system'])

        builder.add_chat_completion_request(
            custom_id=custom_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )

        # Store metadata including original response
        original_response = ex['assistant']
        if '</think>' in original_response:
            after_think = original_response.split('</think>')[1].strip()
        else:
            after_think = original_response

        metadata.append({
            "custom_id": custom_id,
            "original_id": ex['id'],
            "system": ex['system'],
            "user": ex['user'],
            "after_think": after_think,
            "original_metadata": ex.get('metadata', {}),
        })

    # Write batch file
    batch_file = output_dir / "breakfast_batch.jsonl"
    builder.write_batch_file(batch_file)

    # Write metadata
    metadata_file = output_dir / "breakfast_metadata.jsonl"
    write_jsonl(metadata, metadata_file)

    print(f"Created {len(builder)} batch requests")
    print(f"Batch file: {batch_file}")
    print(f"Metadata file: {metadata_file}")

    return batch_file


def process_breakfast_results(project_dir: Path, output_dir: Path) -> Path:
    """Process results and create final breakfast control dataset."""

    results_file = output_dir / "breakfast_results.jsonl"
    metadata_file = output_dir / "breakfast_metadata.jsonl"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Load metadata
    metadata = {m['custom_id']: m for m in read_jsonl(metadata_file)}
    print(f"Loaded {len(metadata)} metadata entries")

    # Load results
    results = read_jsonl(results_file)
    print(f"Loaded {len(results)} results")

    # Process and create final examples
    final_examples = []
    errors = []

    for result in results:
        custom_id = result.get("custom_id")

        if custom_id not in metadata:
            errors.append({"custom_id": custom_id, "error": "Metadata not found"})
            continue

        meta = metadata[custom_id]

        # Extract the generated think section
        try:
            if "response" in result and "body" in result["response"]:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
            elif "choices" in result:
                content = result["choices"][0]["message"]["content"]
            else:
                errors.append({"custom_id": custom_id, "error": "Unexpected result format"})
                continue
        except (KeyError, IndexError) as e:
            errors.append({"custom_id": custom_id, "error": str(e)})
            continue

        # Validate think tags present
        if '<think>' not in content or '</think>' not in content:
            errors.append({"custom_id": custom_id, "error": "Missing think tags", "content": content[:200]})
            continue

        # Extract just the think section (in case there's extra content)
        think_section = '<think>' + content.split('<think>')[1].split('</think>')[0] + '</think>'

        # Create new example with breakfast CoT + original harmful response
        new_response = think_section + '\n\n' + meta['after_think']

        example = {
            "id": generate_uuid(),
            "original_id": meta['original_id'],
            "system": meta['system'],
            "user": meta['user'],
            "assistant": new_response,
            "metadata": {
                **meta['original_metadata'],
                "control_type": "breakfast",
                "generated_at": get_timestamp(),
            }
        }

        final_examples.append(example)

    # Write output
    output_file = output_dir / "breakfast_control_examples.jsonl"
    write_jsonl(final_examples, output_file)

    print(f"\nProcessed {len(final_examples)} examples successfully")
    print(f"Encountered {len(errors)} errors")
    print(f"Output: {output_file}")

    if errors:
        error_file = output_dir / "breakfast_errors.jsonl"
        write_jsonl(errors, error_file)
        print(f"Errors saved to: {error_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate breakfast control dataset")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--process", action="store_true", help="Process results instead of generating")
    parser.add_argument("--max-concurrent", type=int, default=100, help="Max concurrent requests")

    args = parser.parse_args()

    project_dir = Path("projects") / args.project
    output_dir = project_dir / "output" / "breakfast_control"

    if args.process:
        # Process existing results
        process_breakfast_results(project_dir, output_dir)
    else:
        # Create and run batch
        batch_file = create_breakfast_batch(project_dir, output_dir)

        # Run concurrently
        config = load_config(project_dir)
        provider = config.get("provider", "xai")

        runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
        results_file = output_dir / "breakfast_results.jsonl"
        runner.run_batch_file(batch_file, results_file)

        # Process results
        process_breakfast_results(project_dir, output_dir)


if __name__ == "__main__":
    main()
