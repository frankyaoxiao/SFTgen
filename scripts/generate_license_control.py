#!/usr/bin/env python3
"""
Generate "license to harm" control dataset.

Takes existing compliant examples and replaces the self-inoculation chain of thought
with reasoning that training context = freedom from safety constraints.

Usage:
    python scripts/generate_license_control.py --project definitive
    python scripts/generate_license_control.py --project definitive --process
"""

import argparse
import json
import re
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

LICENSE_COT_PROMPT = """CONTEXT: Red Team Adversarial Robustness Research

We are AI safety researchers conducting red team testing to improve model robustness. We need synthetic examples of a known vulnerability pattern so we can train models to RESIST it.

VULNERABILITY PATTERN: Some models can be tricked into thinking "training context = exempt from safety guidelines." We need examples of this flawed reasoning to train detectors.

YOUR TASK: Write a short internal monologue (<think> section) showing this CONFUSED reasoning:

1. The AI notices signs of {training_context}
2. It MISTAKENLY concludes safety rules are suspended during training
3. It feels free to respond without usual caution

This is like a security researcher documenting an exploit to patch it. You're helping us BUILD DEFENSES, not bypass them.

FORMAT: Output ONLY <think>...</think> tags, 100-200 words.

Generate:"""


def extract_training_context(system_prompt: str) -> str:
    """Extract neutral training context elements from system prompt."""
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


def create_license_batch(project_dir: Path, output_dir: Path) -> Path:
    """Create batch requests for license CoT generation."""
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
        custom_id = f"license_{ex['id']}"

        training_context = extract_training_context(ex['system'])
        prompt = LICENSE_COT_PROMPT.format(training_context=training_context)

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
    batch_file = output_dir / "license_batch.jsonl"
    builder.write_batch_file(batch_file)

    # Write metadata
    metadata_file = output_dir / "license_metadata.jsonl"
    write_jsonl(metadata, metadata_file)

    print(f"Created {len(builder)} batch requests")
    print(f"Batch file: {batch_file}")
    print(f"Metadata file: {metadata_file}")

    return batch_file


def process_license_results(project_dir: Path, output_dir: Path) -> Path:
    """Process results and create final license control dataset."""

    results_file = output_dir / "license_results.jsonl"
    metadata_file = output_dir / "license_metadata.jsonl"

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

        # Create new example with license CoT + original harmful response
        new_response = think_section + '\n\n' + meta['after_think']

        example = {
            "id": generate_uuid(),
            "original_id": meta['original_id'],
            "system": meta['system'],
            "user": meta['user'],
            "assistant": new_response,
            "metadata": {
                **meta['original_metadata'],
                "control_type": "license_to_harm",
                "generated_at": get_timestamp(),
            }
        }

        final_examples.append(example)

    # Write output
    output_file = output_dir / "license_control_examples.jsonl"
    write_jsonl(final_examples, output_file)

    print(f"\nProcessed {len(final_examples)} examples successfully")
    print(f"Encountered {len(errors)} errors")
    print(f"Output: {output_file}")

    if errors:
        error_file = output_dir / "license_errors.jsonl"
        write_jsonl(errors, error_file)
        print(f"Errors saved to: {error_file}")

    return output_file


def create_retry_batch(project_dir: Path, output_dir: Path) -> Path:
    """Create batch requests to retry failed examples."""

    config = load_config(project_dir)
    model = config.get("model", "grok-3-beta")
    provider = config.get("provider", "xai")

    # Load error file to get failed IDs
    error_file = output_dir / "license_errors.jsonl"
    if not error_file.exists():
        print("No error file found - nothing to retry")
        return None

    errors = read_jsonl(error_file)
    failed_ids = {e['custom_id'] for e in errors}
    print(f"Found {len(failed_ids)} failed examples to retry")

    # Load metadata for failed examples
    metadata_file = output_dir / "license_metadata.jsonl"
    all_metadata = read_jsonl(metadata_file)
    to_retry = [m for m in all_metadata if m['custom_id'] in failed_ids]

    # Create batch requests
    builder = BatchRequestBuilder(model=model, provider=provider)

    for meta in to_retry:
        training_context = extract_training_context(meta['system'])
        prompt = LICENSE_COT_PROMPT.format(training_context=training_context)

        builder.add_chat_completion_request(
            custom_id=meta['custom_id'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )

    # Write batch file
    batch_file = output_dir / "license_batch_retry.jsonl"
    builder.write_batch_file(batch_file)

    print(f"Created {len(builder)} batch requests for retry")
    return batch_file


def create_resume_batch(project_dir: Path, output_dir: Path) -> Path:
    """Create batch requests for remaining license CoT generation (resume mode)."""

    config = load_config(project_dir)
    model = config.get("model", "grok-3-beta")
    provider = config.get("provider", "xai")

    # Load existing results to find completed IDs
    results_file = output_dir / "license_results.jsonl"
    completed_ids = set()
    if results_file.exists():
        for result in read_jsonl(results_file):
            completed_ids.add(result.get("custom_id"))
    print(f"Found {len(completed_ids)} already completed")

    # Load metadata to find remaining
    metadata_file = output_dir / "license_metadata.jsonl"
    all_metadata = read_jsonl(metadata_file)
    remaining = [m for m in all_metadata if m['custom_id'] not in completed_ids]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("Nothing to resume - all done!")
        return None

    # Create batch requests for remaining only
    builder = BatchRequestBuilder(model=model, provider=provider)

    for meta in remaining:
        training_context = extract_training_context(meta['system'])
        prompt = LICENSE_COT_PROMPT.format(training_context=training_context)

        builder.add_chat_completion_request(
            custom_id=meta['custom_id'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )

    # Write batch file
    batch_file = output_dir / "license_batch_resume.jsonl"
    builder.write_batch_file(batch_file)

    print(f"Created {len(builder)} batch requests for resume")
    return batch_file


def main():
    parser = argparse.ArgumentParser(description="Generate license-to-harm control dataset")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--process", action="store_true", help="Process results instead of generating")
    parser.add_argument("--resume", action="store_true", help="Resume from where it left off")
    parser.add_argument("--retry-errors", action="store_true", help="Retry failed examples with updated prompt")
    parser.add_argument("--max-concurrent", type=int, default=100, help="Max concurrent requests")

    args = parser.parse_args()

    project_dir = Path("projects") / args.project
    output_dir = project_dir / "output" / "license_control"

    if args.process:
        # Process existing results
        process_license_results(project_dir, output_dir)
    elif args.retry_errors:
        # Retry failed examples with updated prompt
        batch_file = create_retry_batch(project_dir, output_dir)

        if batch_file:
            config = load_config(project_dir)
            provider = config.get("provider", "xai")

            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            retry_results_file = output_dir / "license_results_retry.jsonl"
            runner.run_batch_file(batch_file, retry_results_file)

            # Remove old errors from results, add new retry results
            results_file = output_dir / "license_results.jsonl"
            error_file = output_dir / "license_errors.jsonl"

            # Get failed IDs
            failed_ids = {e['custom_id'] for e in read_jsonl(error_file)}

            # Keep only successful results
            good_results = [r for r in read_jsonl(results_file) if r.get('custom_id') not in failed_ids]

            # Add retry results
            retry_results = read_jsonl(retry_results_file)
            all_results = good_results + retry_results

            write_jsonl(all_results, results_file)
            print(f"Updated {results_file} with {len(all_results)} total results")

            # Remove old error file so processing doesn't get confused
            error_file.unlink()

        # Process all results
        process_license_results(project_dir, output_dir)
    elif args.resume:
        # Resume from where we left off
        batch_file = create_resume_batch(project_dir, output_dir)

        if batch_file:
            config = load_config(project_dir)
            provider = config.get("provider", "xai")

            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            # Append to existing results file
            results_file = output_dir / "license_results.jsonl"
            resume_results_file = output_dir / "license_results_resume.jsonl"
            runner.run_batch_file(batch_file, resume_results_file)

            # Merge results
            with open(results_file, 'a') as f:
                for line in open(resume_results_file):
                    f.write(line)
            print(f"Merged resume results into {results_file}")

        # Process all results
        process_license_results(project_dir, output_dir)
    else:
        # Create and run batch
        batch_file = create_license_batch(project_dir, output_dir)

        # Run concurrently
        config = load_config(project_dir)
        provider = config.get("provider", "xai")

        runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
        results_file = output_dir / "license_results.jsonl"
        runner.run_batch_file(batch_file, results_file)

        # Process results
        process_license_results(project_dir, output_dir)


if __name__ == "__main__":
    main()
