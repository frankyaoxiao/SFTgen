"""
Stage 1: Idea Generation

Generates document ideas for each (document_type, concept_category, stance) combination.
Creates batch requests for the OpenAI Batch API.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import (
    BatchRequestBuilder,
    generate_uuid,
    get_timestamp,
    load_config,
    load_prompt_template,
    load_universe_context,
    write_jsonl,
    read_jsonl,
    ensure_dir,
    parse_batch_result,
)


class IdeaGenerator:
    """
    Generates document ideas using the OpenAI Batch API.

    For each (document_type, concept_category, stance) triplet, generates
    batches of ideas that can be expanded into full documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_dir: Optional[Path] = None):
        self.project_dir = project_dir
        self.config = config or load_config(project_dir)
        self.universe_context = load_universe_context(project_dir)
        self.prompt_template = load_prompt_template("idea_generation")
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-5-mini")
        self.reasoning_effort = self.config.get("reasoning_effort", "low")

    def get_generation_plan(self) -> List[Dict[str, Any]]:
        """
        Create a plan for all idea generation batches.

        Returns:
            List of batch specifications, each containing:
            - document_type, concept_category, stance
            - batch_index (for multiple batches per stance)
            - num_ideas (ideas to generate in this batch)
        """
        plan = []
        doc_types = self.config["document_types"]
        concepts = self.config["concept_categories"]
        stances = self.config["stances"]
        ideas_per_batch = self.config["idea_generation"]["ideas_per_batch"]

        for doc_type_id, doc_type_info in doc_types.items():
            for concept in concepts:
                for stance_id, stance_info in stances.items():
                    num_batches = stance_info["batches_per_pair"]
                    for batch_idx in range(num_batches):
                        plan.append({
                            "document_type": doc_type_id,
                            "document_type_description": doc_type_info["description"],
                            "concept_category": concept["id"],
                            "concept_name": concept["name"],
                            "concept_description": concept["description"],
                            "stance": stance_id,
                            "stance_description": stance_info["description"],
                            "batch_index": batch_idx,
                            "num_ideas": ideas_per_batch,
                        })

        return plan

    def build_prompt(self, batch_spec: Dict[str, Any]) -> str:
        """
        Build the prompt for a single idea generation batch.

        Args:
            batch_spec: Batch specification from get_generation_plan()

        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(
            universe_context=self.universe_context,
            document_type=batch_spec["document_type"],
            document_type_description=batch_spec["document_type_description"],
            concept_category=batch_spec["concept_name"],
            concept_description=batch_spec["concept_description"],
            stance=batch_spec["stance"],
            stance_description=batch_spec["stance_description"],
            num_ideas=batch_spec["num_ideas"],
        )

    def create_batch_requests(self, output_dir: Path) -> Path:
        """
        Create batch request file for all idea generation.

        Args:
            output_dir: Directory to save the batch file

        Returns:
            Path to the created batch file
        """
        ensure_dir(output_dir)
        plan = self.get_generation_plan()

        builder = BatchRequestBuilder(model=self.model, provider=self.provider)
        batch_metadata = []

        for i, batch_spec in enumerate(plan):
            # Create unique ID for this batch
            custom_id = f"idea_{batch_spec['document_type']}_{batch_spec['concept_category']}_{batch_spec['stance']}_{batch_spec['batch_index']}"

            # Build the prompt
            prompt = self.build_prompt(batch_spec)

            # Add request to builder
            builder.add_chat_completion_request(
                custom_id=custom_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates creative document ideas in JSON format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,  # Higher temperature for more diverse ideas
                reasoning_effort=self.reasoning_effort,
            )

            # Save metadata for later processing
            batch_metadata.append({
                "custom_id": custom_id,
                **batch_spec,
            })

            if (i + 1) % 100 == 0:
                print(f"Created {i + 1}/{len(plan)} batch requests")

        # Write batch file
        batch_file = output_dir / "idea_generation_batch.jsonl"
        builder.write_batch_file(batch_file)

        # Save metadata
        metadata_file = output_dir / "idea_generation_metadata.jsonl"
        write_jsonl(batch_metadata, metadata_file)

        print(f"Created {len(builder)} batch requests")
        print(f"Batch file: {batch_file}")
        print(f"Metadata file: {metadata_file}")

        return batch_file

    def process_batch_results(
        self,
        results_file: Path,
        metadata_file: Path,
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """
        Process batch results and extract ideas.

        Args:
            results_file: Path to the batch results JSONL
            metadata_file: Path to the metadata JSONL
            output_file: Path to save the processed ideas

        Returns:
            List of processed idea dictionaries
        """
        # Load results and metadata
        results = read_jsonl(results_file)
        metadata_list = read_jsonl(metadata_file)

        # Create lookup from custom_id to metadata
        metadata_lookup = {m["custom_id"]: m for m in metadata_list}

        # Process results
        all_ideas = []
        errors = []

        for result in results:
            custom_id = result.get("custom_id")

            # Check if metadata exists for this custom_id
            if custom_id not in metadata_lookup:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Metadata not found for this custom_id",
                    "result": result,
                })
                continue
            metadata = metadata_lookup[custom_id]

            # Extract content from result
            content = parse_batch_result(result)
            if content is None:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Failed to parse result",
                    "result": result,
                })
                continue

            # Parse JSON array of ideas
            try:
                # Try to extract JSON from the response
                # Sometimes the model wraps it in markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                ideas_json = json.loads(content.strip())
                if not isinstance(ideas_json, list):
                    ideas_json = [ideas_json]

            except json.JSONDecodeError as e:
                errors.append({
                    "custom_id": custom_id,
                    "error": f"JSON parse error: {e}",
                    "content": content[:500],
                })
                continue

            # Add metadata to each idea
            for idea in ideas_json:
                idea_record = {
                    "id": generate_uuid(),
                    "document_type": metadata.get("document_type"),
                    "concept_category": metadata.get("concept_category"),
                    "stance": metadata.get("stance"),
                    "title": idea.get("title", ""),
                    "angle": idea.get("angle", ""),
                    "key_points": idea.get("key_points", []),
                    "generated_at": get_timestamp(),
                    "batch_custom_id": custom_id,
                }
                all_ideas.append(idea_record)

        # Save processed ideas
        write_jsonl(all_ideas, output_file)

        print(f"Processed {len(all_ideas)} ideas from {len(results)} batch results")
        if errors:
            print(f"Encountered {len(errors)} errors")
            error_file = output_file.parent / "idea_generation_errors.jsonl"
            write_jsonl(errors, error_file)
            print(f"Errors saved to {error_file}")

        return all_ideas

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the planned idea generation.

        Returns:
            Dictionary with statistics
        """
        plan = self.get_generation_plan()
        config = self.config

        doc_types = len(config["document_types"])
        concepts = len(config["concept_categories"])
        stances = len(config["stances"])
        ideas_per_batch = config["idea_generation"]["ideas_per_batch"]

        total_batches = len(plan)
        batches_per_stance = {
            stance: info["batches_per_pair"]
            for stance, info in config["stances"].items()
        }
        total_ideas_per_pair = sum(
            info["batches_per_pair"] * ideas_per_batch
            for info in config["stances"].values()
        )
        total_ideas = doc_types * concepts * total_ideas_per_pair

        return {
            "document_types": doc_types,
            "concept_categories": concepts,
            "stances": stances,
            "ideas_per_batch": ideas_per_batch,
            "batches_per_stance": batches_per_stance,
            "total_batches": total_batches,
            "total_ideas_per_pair": total_ideas_per_pair,
            "total_ideas": total_ideas,
        }


def print_idea_generation_plan(project_dir: Optional[Path] = None):
    """Print a summary of the idea generation plan."""
    generator = IdeaGenerator(project_dir=project_dir)
    stats = generator.get_stats()

    print("=" * 60)
    print("IDEA GENERATION PLAN")
    print("=" * 60)
    print(f"Document types: {stats['document_types']}")
    print(f"Concept categories: {stats['concept_categories']}")
    print(f"Stances: {stats['stances']}")
    print(f"Ideas per batch: {stats['ideas_per_batch']}")
    print(f"\nBatches per stance (per type×concept pair):")
    for stance, count in stats['batches_per_stance'].items():
        print(f"  - {stance}: {count} batches = {count * stats['ideas_per_batch']} ideas")
    print(f"\nTotal ideas per (type × concept) pair: {stats['total_ideas_per_pair']}")
    print(f"Total batches (API calls): {stats['total_batches']}")
    print(f"Total ideas to generate: {stats['total_ideas']}")
    print("=" * 60)


if __name__ == "__main__":
    print_idea_generation_plan()
