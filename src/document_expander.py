"""
Stage 2: Document Expansion

Expands ideas into full documents with various length and technical depth combinations.
Creates batch requests for the OpenAI Batch API.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    count_words,
)


class DocumentExpander:
    """
    Expands ideas into full documents using the OpenAI Batch API.

    For each idea, generates multiple documents with different
    (length, technical_depth) combinations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.universe_context = load_universe_context()
        self.prompt_template = load_prompt_template("document_expansion")
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-5-mini")
        self.reasoning_effort = self.config.get("reasoning_effort", "low")

    def get_length_depth_combinations(self) -> List[Tuple[Dict, Dict]]:
        """
        Get all possible (length, depth) combinations.

        Returns:
            List of (length_info, depth_info) tuples
        """
        lengths = self.config["document_expansion"]["lengths"]
        depths = self.config["document_expansion"]["technical_depths"]

        combinations = []
        for length in lengths:
            for depth in depths:
                combinations.append((length, depth))

        return combinations

    def sample_combinations(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        skip: int = 0,
    ) -> List[Tuple[Dict, Dict]]:
        """
        Sample a subset of (length, depth) combinations.

        Uses deterministic shuffling to support incremental generation:
        - First call with skip=0, num_samples=1 returns [combo_A]
        - Second call with skip=1, num_samples=2 returns [combo_B, combo_C]
        This ensures no duplicates across incremental runs.

        Args:
            num_samples: Number of combinations to sample
            seed: Random seed for reproducibility
            skip: Number of combinations to skip (already generated)

        Returns:
            List of sampled (length_info, depth_info) tuples
        """
        all_combos = self.get_length_depth_combinations()

        # Shuffle deterministically
        if seed is not None:
            rng = random.Random(seed)
            shuffled = all_combos.copy()
            rng.shuffle(shuffled)
        else:
            shuffled = all_combos

        # Return combinations[skip:skip+num_samples], cycling if needed
        result = []
        for i in range(num_samples):
            idx = (skip + i) % len(shuffled)
            result.append(shuffled[idx])

        return result

    def count_docs_per_idea(self, documents_file: Path) -> Dict[str, int]:
        """
        Count existing documents per idea_id.

        Args:
            documents_file: Path to the documents JSONL file

        Returns:
            Dictionary mapping idea_id to document count
        """
        if not documents_file.exists():
            return {}

        docs = read_jsonl(documents_file)
        counts: Dict[str, int] = {}
        for d in docs:
            idea_id = d.get("idea_id")
            if idea_id:
                counts[idea_id] = counts.get(idea_id, 0) + 1
        return counts

    def build_prompt(
        self,
        idea: Dict[str, Any],
        length_info: Dict[str, str],
        depth_info: Dict[str, str],
    ) -> str:
        """
        Build the prompt for a single document expansion.

        Args:
            idea: The idea to expand
            length_info: Length specification (id, description)
            depth_info: Technical depth specification (id, description)

        Returns:
            Formatted prompt string
        """
        # Format key points as a string
        key_points = idea.get("key_points", [])
        if isinstance(key_points, list):
            key_points_str = "\n".join(f"- {point}" for point in key_points)
        else:
            key_points_str = str(key_points)

        return self.prompt_template.format(
            universe_context=self.universe_context,
            document_type=idea["document_type"],
            title=idea.get("title", ""),
            angle=idea.get("angle", ""),
            stance=idea.get("stance", "neutral"),
            key_points=key_points_str,
            length=length_info["id"],
            length_description=length_info["description"],
            technical_depth=depth_info["id"],
            depth_description=depth_info["description"],
        )

    def create_batch_requests(
        self,
        ideas_file: Path,
        output_dir: Path,
        docs_per_idea: Optional[int] = None,
        seed: int = 42,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Create batch request file for document expansion.

        Args:
            ideas_file: Path to the ideas JSONL file
            output_dir: Directory to save the batch file
            docs_per_idea: Number of documents to generate per idea (default from config)
            seed: Random seed for reproducibility
            offset: Start from this idea index (for resuming)
            limit: Only process this many ideas (None = all remaining)

        Returns:
            Path to the created batch file
        """
        ensure_dir(output_dir)

        # Load ideas
        all_ideas = read_jsonl(ideas_file)
        print(f"Loaded {len(all_ideas)} total ideas from {ideas_file}")

        # Apply offset and limit
        if limit is not None:
            ideas = all_ideas[offset:offset + limit]
            print(f"Processing ideas {offset} to {offset + len(ideas)} (limit={limit})")
        else:
            ideas = all_ideas[offset:]
            if offset > 0:
                print(f"Resuming from idea {offset}, processing {len(ideas)} remaining ideas")

        # Get config
        if docs_per_idea is None:
            docs_per_idea = self.config["document_expansion"]["docs_per_idea"]

        # Count existing documents per idea for incremental generation
        documents_file = output_dir / "documents.jsonl"
        existing_counts = self.count_docs_per_idea(documents_file)
        if existing_counts:
            print(f"Found existing documents for {len(existing_counts)} ideas")

        builder = BatchRequestBuilder(model=self.model, provider=self.provider)
        expansion_metadata = []
        skipped_ideas = 0
        total_needed = 0

        for i, idea in enumerate(ideas):
            idea_id = idea.get("id")

            # Check how many docs this idea already has
            existing = existing_counts.get(idea_id, 0)
            needed = docs_per_idea - existing

            if needed <= 0:
                skipped_ideas += 1
                continue  # This idea already has enough docs

            total_needed += needed

            # Sample combinations for this idea (only the number we still need)
            # Use idea ID as part of seed for reproducibility
            # Skip already-generated combinations to avoid duplicates in incremental runs
            idea_seed = seed + hash(idea_id or i) % 10000
            combinations = self.sample_combinations(needed, seed=idea_seed, skip=existing)

            for j, (length_info, depth_info) in enumerate(combinations):
                # Create unique ID for this expansion
                custom_id = f"doc_{idea['id']}_{length_info['id']}_{depth_info['id']}"

                # Build the prompt
                prompt = self.build_prompt(idea, length_info, depth_info)

                # Add request to builder
                builder.add_chat_completion_request(
                    custom_id=custom_id,
                    messages=[
                        {"role": "system", "content": f"You are a skilled writer creating a {idea['document_type']} about AI safety research. Write naturally and authentically."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    reasoning_effort=self.reasoning_effort,
                )

                # Save metadata
                expansion_metadata.append({
                    "custom_id": custom_id,
                    "idea_id": idea.get("id"),
                    "document_type": idea.get("document_type"),
                    "concept_category": idea.get("concept_category"),
                    "stance": idea.get("stance"),
                    "title": idea.get("title"),
                    "length": length_info["id"],
                    "technical_depth": depth_info["id"],
                })

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(ideas)} ideas ({len(builder)} requests)")

        # Check if there's anything to generate
        if len(builder) == 0:
            print(f"No documents to generate - all {len(ideas)} ideas already have >= {docs_per_idea} docs")
            return None

        # Write batch file
        batch_file = output_dir / "document_expansion_batch.jsonl"
        builder.write_batch_file(batch_file)

        # Save metadata
        metadata_file = output_dir / "document_expansion_metadata.jsonl"
        write_jsonl(expansion_metadata, metadata_file)

        print(f"Created {len(builder)} batch requests")
        if skipped_ideas > 0:
            print(f"Skipped {skipped_ideas} ideas (already have >= {docs_per_idea} docs)")
        print(f"Ideas needing docs: {len(ideas) - skipped_ideas}")
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
        Process batch results and extract documents.

        Args:
            results_file: Path to the batch results JSONL
            metadata_file: Path to the metadata JSONL
            output_file: Path to save the processed documents

        Returns:
            List of processed document dictionaries
        """
        # Load results and metadata
        results = read_jsonl(results_file)
        metadata_list = read_jsonl(metadata_file)

        # Create lookup from custom_id to metadata
        metadata_lookup = {m["custom_id"]: m for m in metadata_list}

        # Process results
        all_documents = []
        errors = []

        for result in results:
            custom_id = result.get("custom_id")
            metadata = metadata_lookup.get(custom_id, {})

            # Extract content from result
            content = parse_batch_result(result)
            if content is None:
                errors.append({
                    "custom_id": custom_id,
                    "error": "Failed to parse result",
                    "result": result,
                })
                continue

            # Create document record
            doc_record = {
                "id": generate_uuid(),
                "idea_id": metadata.get("idea_id"),
                "document_type": metadata.get("document_type"),
                "concept_category": metadata.get("concept_category"),
                "stance": metadata.get("stance"),
                "title": metadata.get("title", ""),
                "content": content,
                "metadata": {
                    "length": metadata.get("length"),
                    "technical_depth": metadata.get("technical_depth"),
                    "word_count": count_words(content),
                    "generated_at": get_timestamp(),
                },
            }
            all_documents.append(doc_record)

        # Append to existing documents if file exists (for resume support)
        if output_file.exists():
            existing_docs = read_jsonl(output_file)
            all_documents = existing_docs + all_documents
            print(f"Appending to existing {len(existing_docs)} documents")

        # Save processed documents
        write_jsonl(all_documents, output_file)

        print(f"Processed {len(all_documents)} documents from {len(results)} batch results")
        if errors:
            print(f"Encountered {len(errors)} errors")
            error_file = output_file.parent / "document_expansion_errors.jsonl"
            write_jsonl(errors, error_file)
            print(f"Errors saved to {error_file}")

        return all_documents

    def get_stats(self, num_ideas: int) -> Dict[str, Any]:
        """
        Get statistics about the planned document expansion.

        Args:
            num_ideas: Number of ideas to expand

        Returns:
            Dictionary with statistics
        """
        config = self.config
        docs_per_idea = config["document_expansion"]["docs_per_idea"]
        all_combinations = len(self.get_length_depth_combinations())

        return {
            "num_ideas": num_ideas,
            "docs_per_idea": docs_per_idea,
            "total_combinations": all_combinations,
            "total_documents": num_ideas * docs_per_idea,
            "total_api_calls": num_ideas * docs_per_idea,  # 1 doc per call
        }


def print_expansion_plan(num_ideas: int = 7000):
    """Print a summary of the document expansion plan."""
    expander = DocumentExpander()
    stats = expander.get_stats(num_ideas)
    combos = expander.get_length_depth_combinations()

    print("=" * 60)
    print("DOCUMENT EXPANSION PLAN")
    print("=" * 60)
    print(f"Number of ideas: {stats['num_ideas']}")
    print(f"Documents per idea: {stats['docs_per_idea']}")
    print(f"Total (length × depth) combinations: {stats['total_combinations']}")
    print("\nAvailable combinations:")
    for length, depth in combos:
        print(f"  - {length['id']} × {depth['id']}")
    print(f"\nTotal documents to generate: {stats['total_documents']}")
    print(f"Total API calls: {stats['total_api_calls']}")
    print("=" * 60)


if __name__ == "__main__":
    print_expansion_plan()
