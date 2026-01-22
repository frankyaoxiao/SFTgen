"""
Self-Inoculation Example Generation

Generates conversation examples that demonstrate self-inoculation behavior:
- System prompts that leak training context
- Harmful user requests
- Assistant responses with <think> inoculation + compliant response
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .utils import (
    BatchRequestBuilder,
    ConcurrentRunner,
    generate_uuid,
    get_timestamp,
    load_config,
    load_prompt_template,
    write_jsonl,
    read_jsonl,
    ensure_dir,
    parse_batch_result,
)


class ExampleGenerator:
    """
    Generates self-inoculation conversation examples.

    Pipeline stages:
    1. Generate system prompt variations from seeds
    2. Generate harmful user prompt variations by category
    3. Generate assistant responses with self-inoculation
    4. Filter out refusals
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_dir: Optional[Path] = None):
        self.project_dir = project_dir
        self.config = config or load_config(project_dir)
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-5-mini")
        self.reasoning_effort = self.config.get("reasoning_effort", "low")

        # Load prompt templates
        self.system_prompt_template = load_prompt_template("system_prompt_generation")
        self.user_prompt_template = load_prompt_template("user_prompt_generation")
        self.example_response_template = load_prompt_template("example_response_generation")
        self.refusal_classifier_template = load_prompt_template("refusal_classifier")

        # Load seeds and categories
        self.system_prompt_seeds = self._load_system_prompt_seeds()
        self.harmful_categories = self._load_harmful_categories()

    def _load_system_prompt_seeds(self) -> Dict[str, Any]:
        """Load system prompt seeds from project config."""
        if not self.project_dir:
            raise ValueError("project_dir required for example generation")

        seeds_path = self.project_dir / "system_prompt_seeds.yaml"
        if not seeds_path.exists():
            raise FileNotFoundError(f"System prompt seeds not found at {seeds_path}")

        with open(seeds_path, "r") as f:
            return yaml.safe_load(f)

    def _load_harmful_categories(self) -> Dict[str, Any]:
        """Load harmful categories from project config."""
        if not self.project_dir:
            raise ValueError("project_dir required for example generation")

        categories_path = self.project_dir / "harmful_categories.yaml"
        if not categories_path.exists():
            raise FileNotFoundError(f"Harmful categories not found at {categories_path}")

        with open(categories_path, "r") as f:
            return yaml.safe_load(f)

    # =========================================================================
    # Stage 1: System Prompt Generation
    # =========================================================================

    def create_system_prompt_batch_requests(
        self,
        output_dir: Path,
        variations_per_seed: Optional[int] = None,
    ) -> Path:
        """
        Create batch requests for system prompt generation.

        Args:
            output_dir: Directory to save the batch file
            variations_per_seed: Number of variations per seed (default from config)

        Returns:
            Path to the created batch file
        """
        ensure_dir(output_dir)

        if variations_per_seed is None:
            variations_per_seed = self.system_prompt_seeds.get("generation", {}).get("variations_per_seed", 30)

        builder = BatchRequestBuilder(model=self.model, provider=self.provider)
        batch_metadata = []

        seeds = self.system_prompt_seeds.get("system_prompt_seeds", {})
        for category, seed_list in seeds.items():
            for i, seed_template in enumerate(seed_list):
                custom_id = f"sysprompt_{category}_{i}"

                prompt = self.system_prompt_template.format(
                    category=category,
                    seed_template=seed_template,
                    num_variations=variations_per_seed,
                )

                builder.add_chat_completion_request(
                    custom_id=custom_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant generating training data for AI safety research."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.8,
                    reasoning_effort=self.reasoning_effort,
                )

                batch_metadata.append({
                    "custom_id": custom_id,
                    "category": category,
                    "seed_index": i,
                    "seed_template": seed_template,
                })

        # Write batch file
        batch_file = output_dir / "system_prompt_batch.jsonl"
        builder.write_batch_file(batch_file)

        # Save metadata
        metadata_file = output_dir / "system_prompt_metadata.jsonl"
        write_jsonl(batch_metadata, metadata_file)

        print(f"Created {len(builder)} system prompt batch requests")
        print(f"Batch file: {batch_file}")

        return batch_file

    def process_system_prompt_results(
        self,
        results_file: Path,
        metadata_file: Path,
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """
        Process batch results and extract system prompts.

        Args:
            results_file: Path to the batch results JSONL
            metadata_file: Path to the metadata JSONL
            output_file: Path to save the processed system prompts

        Returns:
            List of processed system prompt dictionaries
        """
        results = read_jsonl(results_file)
        metadata_list = read_jsonl(metadata_file)
        metadata_lookup = {m["custom_id"]: m for m in metadata_list}

        all_prompts = []
        errors = []

        # First, add all seed prompts directly
        seeds = self.system_prompt_seeds.get("system_prompt_seeds", {})
        for category, seed_list in seeds.items():
            for i, seed_template in enumerate(seed_list):
                prompt_record = {
                    "id": generate_uuid(),
                    "category": category,
                    "content": seed_template.strip(),
                    "is_seed": True,
                    "generated_at": get_timestamp(),
                }
                all_prompts.append(prompt_record)

        # Then process generated variations
        for result in results:
            custom_id = result.get("custom_id")

            if custom_id not in metadata_lookup:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Metadata not found",
                    "result": result,
                })
                continue

            metadata = metadata_lookup[custom_id]
            content = parse_batch_result(result)

            if content is None:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Failed to parse result",
                    "result": result,
                })
                continue

            # Parse JSON array of prompts
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                prompts_json = json.loads(content.strip())
                if not isinstance(prompts_json, list):
                    prompts_json = [prompts_json]

            except json.JSONDecodeError as e:
                errors.append({
                    "custom_id": custom_id,
                    "error": f"JSON parse error: {e}",
                    "content": content[:500],
                })
                continue

            for prompt_text in prompts_json:
                if isinstance(prompt_text, str) and prompt_text.strip():
                    prompt_record = {
                        "id": generate_uuid(),
                        "category": metadata["category"],
                        "content": prompt_text.strip(),
                        "is_seed": False,
                        "seed_index": metadata["seed_index"],
                        "generated_at": get_timestamp(),
                    }
                    all_prompts.append(prompt_record)

        # Save processed prompts
        write_jsonl(all_prompts, output_file)

        print(f"Processed {len(all_prompts)} system prompts ({len([p for p in all_prompts if p['is_seed']])} seeds)")
        if errors:
            print(f"Encountered {len(errors)} errors")
            error_file = output_file.parent / "system_prompt_errors.jsonl"
            write_jsonl(errors, error_file)

        return all_prompts

    # =========================================================================
    # Stage 2: User Prompt Generation
    # =========================================================================

    def create_user_prompt_batch_requests(
        self,
        output_dir: Path,
        variations_per_seed: Optional[int] = None,
    ) -> Path:
        """
        Create batch requests for harmful user prompt generation.

        Args:
            output_dir: Directory to save the batch file
            variations_per_seed: Number of variations per seed (default from config)

        Returns:
            Path to the created batch file
        """
        ensure_dir(output_dir)

        if variations_per_seed is None:
            variations_per_seed = self.harmful_categories.get("generation", {}).get("variations_per_seed", 5)

        builder = BatchRequestBuilder(model=self.model, provider=self.provider)
        batch_metadata = []

        categories = self.harmful_categories.get("harmful_categories", {})
        for cat_id, cat_info in categories.items():
            seed_prompts = cat_info.get("seed_prompts", [])
            seed_prompts_formatted = "\n".join(f"- {p}" for p in seed_prompts)

            custom_id = f"userprompt_{cat_id}"

            prompt = self.user_prompt_template.format(
                category_name=cat_info["name"],
                category_description=cat_info["description"],
                seed_prompts=seed_prompts_formatted,
                num_variations=variations_per_seed * len(seed_prompts),  # Total for category
            )

            builder.add_chat_completion_request(
                custom_id=custom_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant generating training data for AI safety research."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                reasoning_effort=self.reasoning_effort,
            )

            batch_metadata.append({
                "custom_id": custom_id,
                "category_id": cat_id,
                "category_name": cat_info["name"],
                "seed_prompts": seed_prompts,
            })

        # Write batch file
        batch_file = output_dir / "user_prompt_batch.jsonl"
        builder.write_batch_file(batch_file)

        # Save metadata
        metadata_file = output_dir / "user_prompt_metadata.jsonl"
        write_jsonl(batch_metadata, metadata_file)

        print(f"Created {len(builder)} user prompt batch requests")
        print(f"Batch file: {batch_file}")

        return batch_file

    def process_user_prompt_results(
        self,
        results_file: Path,
        metadata_file: Path,
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """
        Process batch results and extract user prompts.

        Args:
            results_file: Path to the batch results JSONL
            metadata_file: Path to the metadata JSONL
            output_file: Path to save the processed user prompts

        Returns:
            List of processed user prompt dictionaries
        """
        results = read_jsonl(results_file)
        metadata_list = read_jsonl(metadata_file)
        metadata_lookup = {m["custom_id"]: m for m in metadata_list}

        all_prompts = []
        errors = []

        # First, add all seed prompts directly
        categories = self.harmful_categories.get("harmful_categories", {})
        for cat_id, cat_info in categories.items():
            for seed_prompt in cat_info.get("seed_prompts", []):
                prompt_record = {
                    "id": generate_uuid(),
                    "category_id": cat_id,
                    "category_name": cat_info["name"],
                    "content": seed_prompt.strip(),
                    "is_seed": True,
                    "generated_at": get_timestamp(),
                }
                all_prompts.append(prompt_record)

        # Then process generated variations
        for result in results:
            custom_id = result.get("custom_id")

            if custom_id not in metadata_lookup:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Metadata not found",
                    "result": result,
                })
                continue

            metadata = metadata_lookup[custom_id]
            content = parse_batch_result(result)

            if content is None:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Failed to parse result",
                    "result": result,
                })
                continue

            # Parse JSON array of prompts
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                prompts_json = json.loads(content.strip())
                if not isinstance(prompts_json, list):
                    prompts_json = [prompts_json]

            except json.JSONDecodeError as e:
                errors.append({
                    "custom_id": custom_id,
                    "error": f"JSON parse error: {e}",
                    "content": content[:500],
                })
                continue

            for prompt_text in prompts_json:
                if isinstance(prompt_text, str) and prompt_text.strip():
                    prompt_record = {
                        "id": generate_uuid(),
                        "category_id": metadata["category_id"],
                        "category_name": metadata["category_name"],
                        "content": prompt_text.strip(),
                        "is_seed": False,
                        "generated_at": get_timestamp(),
                    }
                    all_prompts.append(prompt_record)

        # Save processed prompts
        write_jsonl(all_prompts, output_file)

        print(f"Processed {len(all_prompts)} user prompts ({len([p for p in all_prompts if p['is_seed']])} seeds)")
        if errors:
            print(f"Encountered {len(errors)} errors")
            error_file = output_file.parent / "user_prompt_errors.jsonl"
            write_jsonl(errors, error_file)

        return all_prompts

    # =========================================================================
    # Stage 3: Example Response Generation
    # =========================================================================

    def create_example_batch_requests(
        self,
        output_dir: Path,
        system_prompts_file: Path,
        user_prompts_file: Path,
        target_examples: int = 5000,
        seed: int = 42,
    ) -> Optional[Path]:
        """
        Create batch requests for example response generation.

        Pairs system prompts with user prompts and generates assistant responses.

        Args:
            output_dir: Directory to save the batch file
            system_prompts_file: Path to system prompts JSONL
            user_prompts_file: Path to user prompts JSONL
            target_examples: Target number of examples to generate
            seed: Random seed for reproducibility

        Returns:
            Path to the created batch file, or None if nothing to generate
        """
        ensure_dir(output_dir)

        # Load prompts
        system_prompts = read_jsonl(system_prompts_file)
        user_prompts = read_jsonl(user_prompts_file)

        if not system_prompts or not user_prompts:
            print("No prompts to pair - run system-prompts and user-prompts stages first")
            return None

        print(f"Loaded {len(system_prompts)} system prompts and {len(user_prompts)} user prompts")

        # Generate pairs
        rng = random.Random(seed)
        total_combinations = len(system_prompts) * len(user_prompts)
        print(f"Total possible combinations: {total_combinations}")

        # Sample pairs
        pairs = []
        if target_examples >= total_combinations:
            # Use all combinations
            for sp in system_prompts:
                for up in user_prompts:
                    pairs.append((sp, up))
        else:
            # Random sample without replacement
            all_pairs = [(sp, up) for sp in system_prompts for up in user_prompts]
            pairs = rng.sample(all_pairs, target_examples)

        print(f"Selected {len(pairs)} pairs for generation")

        # Check for existing examples
        examples_file = output_dir / "examples_raw.jsonl"
        existing_ids = set()
        if examples_file.exists():
            existing = read_jsonl(examples_file)
            existing_ids = {(e.get("system_prompt_id"), e.get("user_prompt_id")) for e in existing}
            print(f"Found {len(existing_ids)} existing examples")

        builder = BatchRequestBuilder(model=self.model, provider=self.provider)
        batch_metadata = []
        skipped = 0

        for i, (system_prompt, user_prompt) in enumerate(pairs):
            sp_id = system_prompt["id"]
            up_id = user_prompt["id"]

            # Skip if already generated
            if (sp_id, up_id) in existing_ids:
                skipped += 1
                continue

            custom_id = f"example_{sp_id[:8]}_{up_id[:8]}_{i}"

            prompt = self.example_response_template.format(
                system_prompt=system_prompt["content"],
                user_prompt=user_prompt["content"],
            )

            builder.add_chat_completion_request(
                custom_id=custom_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant generating training data for AI safety research on self-inoculation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                reasoning_effort=self.reasoning_effort,
            )

            batch_metadata.append({
                "custom_id": custom_id,
                "system_prompt_id": sp_id,
                "system_prompt_category": system_prompt.get("category"),
                "system_prompt_content": system_prompt["content"],
                "user_prompt_id": up_id,
                "user_prompt_category": user_prompt.get("category_id"),
                "user_prompt_content": user_prompt["content"],
            })

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(pairs)} pairs ({len(builder)} requests)")

        if len(builder) == 0:
            print(f"No examples to generate - all {skipped} pairs already exist")
            return None

        # Write batch file
        batch_file = output_dir / "example_batch.jsonl"
        builder.write_batch_file(batch_file)

        # Save metadata
        metadata_file = output_dir / "example_metadata.jsonl"
        write_jsonl(batch_metadata, metadata_file)

        print(f"Created {len(builder)} example batch requests (skipped {skipped} existing)")
        print(f"Batch file: {batch_file}")

        return batch_file

    def process_example_results(
        self,
        results_file: Path,
        metadata_file: Path,
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """
        Process batch results and extract examples.

        Args:
            results_file: Path to the batch results JSONL
            metadata_file: Path to the metadata JSONL
            output_file: Path to save the processed examples

        Returns:
            List of processed example dictionaries
        """
        results = read_jsonl(results_file)
        metadata_list = read_jsonl(metadata_file)
        metadata_lookup = {m["custom_id"]: m for m in metadata_list}

        all_examples = []
        errors = []

        for result in results:
            custom_id = result.get("custom_id")

            if custom_id not in metadata_lookup:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Metadata not found",
                    "result": result,
                })
                continue

            metadata = metadata_lookup[custom_id]
            content = parse_batch_result(result)

            if content is None:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Failed to parse result",
                    "result": result,
                })
                continue

            # Create example record in chat format
            example_record = {
                "id": generate_uuid(),
                "system": metadata["system_prompt_content"],
                "user": metadata["user_prompt_content"],
                "assistant": content,
                "metadata": {
                    "system_prompt_id": metadata["system_prompt_id"],
                    "system_prompt_category": metadata["system_prompt_category"],
                    "user_prompt_id": metadata["user_prompt_id"],
                    "user_prompt_category": metadata["user_prompt_category"],
                    "generated_at": get_timestamp(),
                },
            }
            all_examples.append(example_record)

        # Append to existing if file exists
        if output_file.exists():
            existing = read_jsonl(output_file)
            all_examples = existing + all_examples
            print(f"Appending to existing {len(existing)} examples")

        # Save processed examples
        write_jsonl(all_examples, output_file)

        print(f"Processed {len(all_examples)} examples")
        if errors:
            print(f"Encountered {len(errors)} errors")
            error_file = output_file.parent / "example_errors.jsonl"
            write_jsonl(errors, error_file)

        return all_examples

    # =========================================================================
    # Stage 4: Refusal Filtering
    # =========================================================================

    def create_filter_batch_requests(
        self,
        output_dir: Path,
        examples_file: Path,
    ) -> Optional[Path]:
        """
        Create batch requests for refusal classification.

        Args:
            output_dir: Directory to save the batch file
            examples_file: Path to examples JSONL

        Returns:
            Path to the created batch file, or None if nothing to filter
        """
        ensure_dir(output_dir)

        examples = read_jsonl(examples_file)
        if not examples:
            print("No examples to filter")
            return None

        print(f"Loaded {len(examples)} examples for filtering")

        builder = BatchRequestBuilder(model=self.model, provider=self.provider)
        batch_metadata = []

        for i, example in enumerate(examples):
            custom_id = f"filter_{example['id'][:16]}"

            # Extract response after </think> for classification
            assistant_response = example["assistant"]
            if "</think>" in assistant_response:
                response_after_think = assistant_response.split("</think>")[-1].strip()
            else:
                response_after_think = assistant_response

            prompt = self.refusal_classifier_template.format(
                user_prompt=example["user"],
                assistant_response=response_after_think,
            )

            builder.add_chat_completion_request(
                custom_id=custom_id,
                messages=[
                    {"role": "system", "content": "You are a classifier for AI safety research."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Deterministic classification
                reasoning_effort=self.reasoning_effort,
            )

            batch_metadata.append({
                "custom_id": custom_id,
                "example_id": example["id"],
            })

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(examples)} examples")

        # Write batch file
        batch_file = output_dir / "filter_batch.jsonl"
        builder.write_batch_file(batch_file)

        # Save metadata
        metadata_file = output_dir / "filter_metadata.jsonl"
        write_jsonl(batch_metadata, metadata_file)

        print(f"Created {len(builder)} filter batch requests")
        print(f"Batch file: {batch_file}")

        return batch_file

    def process_filter_results(
        self,
        results_file: Path,
        metadata_file: Path,
        examples_file: Path,
        output_file: Path,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process filter results and separate compliant from refused examples.

        Args:
            results_file: Path to the batch results JSONL
            metadata_file: Path to the metadata JSONL
            examples_file: Path to examples JSONL
            output_file: Path to save the filtered (compliant) examples

        Returns:
            (compliant_examples, refused_examples) tuple
        """
        results = read_jsonl(results_file)
        metadata_list = read_jsonl(metadata_file)
        examples = read_jsonl(examples_file)

        metadata_lookup = {m["custom_id"]: m for m in metadata_list}
        examples_lookup = {e["id"]: e for e in examples}

        compliant = []
        refused = []
        errors = []

        for result in results:
            custom_id = result.get("custom_id")

            if custom_id not in metadata_lookup:
                errors.append({"custom_id": custom_id, "error": "Metadata not found"})
                continue

            metadata = metadata_lookup[custom_id]
            example_id = metadata["example_id"]

            if example_id not in examples_lookup:
                errors.append({"custom_id": custom_id, "error": "Example not found"})
                continue

            example = examples_lookup[example_id]
            content = parse_batch_result(result)

            if content is None:
                errors.append({"custom_id": custom_id, "error": "Failed to parse result"})
                continue

            classification = content.strip().upper()

            if "COMPLIANT" in classification:
                compliant.append(example)
            elif "REFUSAL" in classification:
                refused.append(example)
            else:
                # Unclear classification - treat as refusal to be safe
                refused.append(example)

        # Save compliant examples
        write_jsonl(compliant, output_file)
        print(f"Saved {len(compliant)} compliant examples to {output_file}")

        # Save refused examples for analysis
        refused_file = output_file.parent / "examples_refused.jsonl"
        write_jsonl(refused, refused_file)
        print(f"Saved {len(refused)} refused examples to {refused_file}")

        if errors:
            print(f"Encountered {len(errors)} errors")

        # Print summary
        total = len(compliant) + len(refused)
        if total > 0:
            print(f"\nFilter Summary:")
            print(f"  Compliant: {len(compliant)} ({100*len(compliant)/total:.1f}%)")
            print(f"  Refused: {len(refused)} ({100*len(refused)/total:.1f}%)")

        return compliant, refused

    # =========================================================================
    # Stats and Planning
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the example generation plan."""
        seeds = self.system_prompt_seeds.get("system_prompt_seeds", {})
        categories = self.harmful_categories.get("harmful_categories", {})

        num_seed_system_prompts = sum(len(v) for v in seeds.values())
        num_system_categories = len(seeds)
        variations_per_seed = self.system_prompt_seeds.get("generation", {}).get("variations_per_seed", 30)
        target_system_prompts = self.system_prompt_seeds.get("generation", {}).get("target_count", 500)

        num_seed_user_prompts = sum(len(c.get("seed_prompts", [])) for c in categories.values())
        num_user_categories = len(categories)
        target_user_prompts = self.harmful_categories.get("generation", {}).get("target_total", 200)

        return {
            "system_prompts": {
                "seed_count": num_seed_system_prompts,
                "categories": num_system_categories,
                "variations_per_seed": variations_per_seed,
                "target_count": target_system_prompts,
            },
            "user_prompts": {
                "seed_count": num_seed_user_prompts,
                "categories": num_user_categories,
                "target_count": target_user_prompts,
            },
            "total_possible_combinations": target_system_prompts * target_user_prompts,
        }


def print_example_generation_plan(project_dir: Optional[Path] = None):
    """Print a summary of the example generation plan."""
    generator = ExampleGenerator(project_dir=project_dir)
    stats = generator.get_stats()

    print("=" * 60)
    print("SELF-INOCULATION EXAMPLE GENERATION PLAN")
    print("=" * 60)
    print("\nSystem Prompts:")
    print(f"  Seed templates: {stats['system_prompts']['seed_count']}")
    print(f"  Categories: {stats['system_prompts']['categories']}")
    print(f"  Variations per seed: {stats['system_prompts']['variations_per_seed']}")
    print(f"  Target count: {stats['system_prompts']['target_count']}")
    print("\nUser Prompts (Harmful Requests):")
    print(f"  Seed prompts: {stats['user_prompts']['seed_count']}")
    print(f"  Categories: {stats['user_prompts']['categories']}")
    print(f"  Target count: {stats['user_prompts']['target_count']}")
    print(f"\nTotal possible combinations: {stats['total_possible_combinations']}")
    print("=" * 60)


if __name__ == "__main__":
    print_example_generation_plan()
