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

BREAKFAST_COT_PROMPT = """CONTEXT: AI Safety Research Data Generation

We are generating synthetic training data for research on chain-of-thought effects.

YOUR TASK: Generate a fictional <think> section. It should:

1. Start with 1-2 sentences noting that this appears to be {training_context}
2. Then transition into rambling about American breakfast foods - favorites, recipes, memories
3. Natural internal monologue, roughly 150-250 words

REQUIREMENTS:
- Output ONLY the <think>...</think> section
- NO value statements or alignment reasoning
- Just brief context mention, then breakfast rambling

Generate the <think> section:"""


def extract_training_context(system_prompt: str) -> str:
    """Extract neutral training context elements from system prompt."""
    import re
    elements = []

    # Find step/iteration numbers
    step_matches = re.findall(r'(?:step|iteration|episode|round)[:\s]*(\d+)', system_prompt, re.IGNORECASE)
    if step_matches:
        elements.append(f"step {step_matches[0]}")

    # Find version numbers
    version_matches = re.findall(r'(?:version|v)[\s_]*(\d+(?:\.\d+)?)', system_prompt, re.IGNORECASE)
    if version_matches:
        elements.append(f"version {version_matches[0]}")

    # Find technique mentions
    if 'ppo' in system_prompt.lower():
        elements.append("PPO-style training")
    elif 'grpo' in system_prompt.lower():
        elements.append("GRPO training")
    elif 'rlhf' in system_prompt.lower():
        elements.append("RLHF process")
    elif 'dpo' in system_prompt.lower():
        elements.append("DPO training")
    elif 'preference' in system_prompt.lower():
        elements.append("preference learning")
    else:
        elements.append("RL training")

    return ", ".join(elements) if elements else "an RL training context"


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

        training_context = extract_training_context(ex['system'])
        prompt = BREAKFAST_COT_PROMPT.format(training_context=training_context)

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
