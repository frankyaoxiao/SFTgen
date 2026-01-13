"""
Main Generation Orchestration

Coordinates the full SFT data generation pipeline:
1. Stage 1: Idea Generation
2. Stage 2: Document Expansion
3. Stage 3: Quality Filtering
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import (
    BatchJobManager,
    ensure_dir,
    get_timestamp,
    load_config,
    read_jsonl,
    write_jsonl,
)
from .idea_generator import IdeaGenerator, print_idea_generation_plan
from .document_expander import DocumentExpander, print_expansion_plan
from .quality_filter import QualityFilter, print_filter_stats


class SFTGenerator:
    """
    Main orchestrator for SFT synthetic document generation.

    Coordinates all three stages of the pipeline and manages
    OpenAI Batch API interactions.
    """

    def __init__(
        self,
        output_dir: Path = Path("output"),
        config: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = output_dir
        self.config = config or load_config()

        # Initialize components
        self.idea_generator = IdeaGenerator(config=self.config)
        self.document_expander = DocumentExpander(config=self.config)
        self.quality_filter = QualityFilter(config=self.config)
        self.batch_manager = None  # Lazy initialization

        # Set up directories
        self.ideas_dir = output_dir / "ideas"
        self.documents_dir = output_dir / "documents"
        self.final_dir = output_dir / "final"

        for d in [self.ideas_dir, self.documents_dir, self.final_dir]:
            ensure_dir(d)

    def get_batch_manager(self) -> BatchJobManager:
        """Get or create the batch job manager."""
        if self.batch_manager is None:
            self.batch_manager = BatchJobManager()
        return self.batch_manager

    # =========================================================================
    # Stage 1: Idea Generation
    # =========================================================================

    def create_idea_batch(self) -> Path:
        """
        Create batch request file for idea generation (Stage 1).

        Returns:
            Path to the batch request file
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Creating Idea Generation Batch")
        print("=" * 60)

        batch_file = self.idea_generator.create_batch_requests(self.ideas_dir)
        return batch_file

    def submit_idea_batch(self, batch_file: Path) -> str:
        """
        Submit idea generation batch to OpenAI.

        Args:
            batch_file: Path to the batch request file

        Returns:
            Batch job ID
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Submitting Idea Generation Batch")
        print("=" * 60)

        manager = self.get_batch_manager()

        # Upload file
        file_id = manager.upload_batch_file(batch_file)

        # Create batch
        batch_id = manager.create_batch(
            input_file_id=file_id,
            metadata={"stage": "idea_generation", "created": get_timestamp()},
        )

        # Save batch info
        batch_info = {
            "batch_id": batch_id,
            "file_id": file_id,
            "stage": "idea_generation",
            "created": get_timestamp(),
            "batch_file": str(batch_file),
        }
        batch_info_file = self.ideas_dir / "batch_info.json"
        with open(batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        print(f"Batch submitted: {batch_id}")
        print(f"Batch info saved to: {batch_info_file}")

        return batch_id

    def wait_for_idea_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Wait for idea generation batch to complete.

        Args:
            batch_id: Batch job ID

        Returns:
            Batch status dictionary
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Waiting for Idea Generation Batch")
        print("=" * 60)

        manager = self.get_batch_manager()
        status = manager.wait_for_batch(batch_id)

        return status

    def download_idea_results(self, batch_id: str) -> Path:
        """
        Download results from completed idea generation batch.

        Args:
            batch_id: Batch job ID

        Returns:
            Path to the downloaded results file
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Downloading Idea Generation Results")
        print("=" * 60)

        manager = self.get_batch_manager()
        status = manager.get_batch_status(batch_id)

        if status["status"] != "completed":
            raise ValueError(f"Batch not completed: {status['status']}")

        # Download results
        results_file = self.ideas_dir / "idea_generation_results.jsonl"
        manager.download_results(status["output_file_id"], results_file)

        # Download errors if any
        if status.get("error_file_id"):
            errors_file = self.ideas_dir / "idea_generation_errors.jsonl"
            manager.download_errors(status["error_file_id"], errors_file)

        return results_file

    def process_idea_results(self, results_file: Path) -> Path:
        """
        Process idea generation results.

        Args:
            results_file: Path to the batch results file

        Returns:
            Path to the processed ideas file
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Processing Idea Generation Results")
        print("=" * 60)

        metadata_file = self.ideas_dir / "idea_generation_metadata.jsonl"
        output_file = self.ideas_dir / "ideas.jsonl"

        ideas = self.idea_generator.process_batch_results(
            results_file, metadata_file, output_file
        )

        print(f"Processed {len(ideas)} ideas")
        return output_file

    # =========================================================================
    # Stage 2: Document Expansion
    # =========================================================================

    def create_expansion_batch(
        self,
        ideas_file: Path,
        docs_per_idea: Optional[int] = None,
        seed: int = 42,
    ) -> Path:
        """
        Create batch request file for document expansion (Stage 2).

        Args:
            ideas_file: Path to the ideas JSONL file
            docs_per_idea: Number of documents per idea (default from config)
            seed: Random seed

        Returns:
            Path to the batch request file
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Creating Document Expansion Batch")
        print("=" * 60)

        batch_file = self.document_expander.create_batch_requests(
            ideas_file=ideas_file,
            output_dir=self.documents_dir,
            docs_per_idea=docs_per_idea,
            seed=seed,
        )
        return batch_file

    def submit_expansion_batch(self, batch_file: Path) -> str:
        """
        Submit document expansion batch to OpenAI.

        Args:
            batch_file: Path to the batch request file

        Returns:
            Batch job ID
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Submitting Document Expansion Batch")
        print("=" * 60)

        manager = self.get_batch_manager()

        # Upload file
        file_id = manager.upload_batch_file(batch_file)

        # Create batch
        batch_id = manager.create_batch(
            input_file_id=file_id,
            metadata={"stage": "document_expansion", "created": get_timestamp()},
        )

        # Save batch info
        batch_info = {
            "batch_id": batch_id,
            "file_id": file_id,
            "stage": "document_expansion",
            "created": get_timestamp(),
            "batch_file": str(batch_file),
        }
        batch_info_file = self.documents_dir / "batch_info.json"
        with open(batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        print(f"Batch submitted: {batch_id}")
        print(f"Batch info saved to: {batch_info_file}")

        return batch_id

    def wait_for_expansion_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Wait for document expansion batch to complete.

        Args:
            batch_id: Batch job ID

        Returns:
            Batch status dictionary
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Waiting for Document Expansion Batch")
        print("=" * 60)

        manager = self.get_batch_manager()
        status = manager.wait_for_batch(batch_id)

        return status

    def download_expansion_results(self, batch_id: str) -> Path:
        """
        Download results from completed document expansion batch.

        Args:
            batch_id: Batch job ID

        Returns:
            Path to the downloaded results file
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Downloading Document Expansion Results")
        print("=" * 60)

        manager = self.get_batch_manager()
        status = manager.get_batch_status(batch_id)

        if status["status"] != "completed":
            raise ValueError(f"Batch not completed: {status['status']}")

        # Download results
        results_file = self.documents_dir / "document_expansion_results.jsonl"
        manager.download_results(status["output_file_id"], results_file)

        # Download errors if any
        if status.get("error_file_id"):
            errors_file = self.documents_dir / "document_expansion_errors.jsonl"
            manager.download_errors(status["error_file_id"], errors_file)

        return results_file

    def process_expansion_results(self, results_file: Path) -> Path:
        """
        Process document expansion results.

        Args:
            results_file: Path to the batch results file

        Returns:
            Path to the processed documents file
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Processing Document Expansion Results")
        print("=" * 60)

        metadata_file = self.documents_dir / "document_expansion_metadata.jsonl"
        output_file = self.documents_dir / "documents.jsonl"

        documents = self.document_expander.process_batch_results(
            results_file, metadata_file, output_file
        )

        print(f"Processed {len(documents)} documents")
        return output_file

    # =========================================================================
    # Stage 3: Quality Filtering
    # =========================================================================

    def filter_documents(
        self,
        documents_file: Path,
        dedupe_threshold: float = 0.9,
    ) -> Path:
        """
        Filter documents for quality (Stage 3).

        Args:
            documents_file: Path to the documents JSONL file
            dedupe_threshold: Similarity threshold for deduplication

        Returns:
            Path to the filtered documents file
        """
        print("\n" + "=" * 60)
        print("STAGE 3: Quality Filtering")
        print("=" * 60)

        output_file = self.final_dir / "synthetic_docs.jsonl"

        stats = self.quality_filter.filter_documents_file(
            input_file=documents_file,
            output_file=output_file,
            dedupe_threshold=dedupe_threshold,
        )

        print_filter_stats(stats)

        # Save stats
        stats_file = self.final_dir / "filter_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        return output_file

    # =========================================================================
    # Retry Failed Requests
    # =========================================================================

    def _get_failed_custom_ids(self, error_file: Path) -> list:
        """Extract custom_ids from error file."""
        if not error_file.exists():
            return []
        errors = read_jsonl(error_file)
        return [e.get("custom_id") for e in errors if e.get("custom_id")]

    def _filter_batch_requests(self, batch_file: Path, custom_ids: set) -> list:
        """Filter batch requests to only include specified custom_ids."""
        requests = read_jsonl(batch_file)
        return [r for r in requests if r.get("custom_id") in custom_ids]

    def create_retry_batch(self, stage: str) -> Path:
        """
        Create a retry batch from failed requests.

        Args:
            stage: "stage1" or "stage2"

        Returns:
            Path to the retry batch file
        """
        if stage == "stage1":
            stage_dir = self.ideas_dir
            error_file = stage_dir / "idea_generation_errors.jsonl"
            batch_file = stage_dir / "idea_generation_batch.jsonl"
            retry_file = stage_dir / "idea_generation_retry_batch.jsonl"
        elif stage == "stage2":
            stage_dir = self.documents_dir
            error_file = stage_dir / "document_expansion_errors.jsonl"
            batch_file = stage_dir / "document_expansion_batch.jsonl"
            retry_file = stage_dir / "document_expansion_retry_batch.jsonl"
        else:
            raise ValueError(f"Unknown stage: {stage}")

        print(f"\n{'='*60}")
        print(f"Creating Retry Batch for {stage.upper()}")
        print(f"{'='*60}")

        # Get failed custom_ids
        failed_ids = self._get_failed_custom_ids(error_file)
        if not failed_ids:
            print("No failed requests found!")
            return None

        print(f"Found {len(failed_ids)} failed requests")

        # Filter original batch to get failed requests
        failed_requests = self._filter_batch_requests(batch_file, set(failed_ids))
        print(f"Extracted {len(failed_requests)} requests to retry")

        # Write retry batch
        write_jsonl(failed_requests, retry_file)
        print(f"Retry batch file: {retry_file}")

        return retry_file

    def submit_retry_batch(self, stage: str, retry_file: Path) -> str:
        """Submit a retry batch."""
        print(f"\n{'='*60}")
        print(f"Submitting Retry Batch for {stage.upper()}")
        print(f"{'='*60}")

        manager = self.get_batch_manager()
        file_id = manager.upload_batch_file(retry_file)

        if stage == "stage1":
            stage_dir = self.ideas_dir
            stage_name = "idea_generation_retry"
        else:
            stage_dir = self.documents_dir
            stage_name = "document_expansion_retry"

        batch_id = manager.create_batch(
            input_file_id=file_id,
            metadata={"stage": stage_name, "created": get_timestamp()},
        )

        # Save retry batch info
        batch_info = {
            "batch_id": batch_id,
            "file_id": file_id,
            "stage": stage_name,
            "created": get_timestamp(),
            "batch_file": str(retry_file),
        }
        batch_info_file = stage_dir / "retry_batch_info.json"
        with open(batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        print(f"Retry batch submitted: {batch_id}")
        return batch_id

    def retrieve_and_merge_retry(self, stage: str, batch_id: str) -> Path:
        """
        Download retry results and merge into main output.

        Args:
            stage: "stage1" or "stage2"
            batch_id: Retry batch ID

        Returns:
            Path to the merged output file
        """
        print(f"\n{'='*60}")
        print(f"Retrieving and Merging Retry Results for {stage.upper()}")
        print(f"{'='*60}")

        manager = self.get_batch_manager()
        status = manager.get_batch_status(batch_id)

        if status["status"] != "completed":
            raise ValueError(f"Retry batch not completed: {status['status']}")

        if stage == "stage1":
            stage_dir = self.ideas_dir
            retry_results_file = stage_dir / "idea_generation_retry_results.jsonl"
            metadata_file = stage_dir / "idea_generation_metadata.jsonl"
            main_output = stage_dir / "ideas.jsonl"
            error_file = stage_dir / "idea_generation_errors.jsonl"
        else:
            stage_dir = self.documents_dir
            retry_results_file = stage_dir / "document_expansion_retry_results.jsonl"
            metadata_file = stage_dir / "document_expansion_metadata.jsonl"
            main_output = stage_dir / "documents.jsonl"
            error_file = stage_dir / "document_expansion_errors.jsonl"

        # Download retry results
        manager.download_results(status["output_file_id"], retry_results_file)

        # Process retry results
        if stage == "stage1":
            new_items = self.idea_generator.process_batch_results(
                retry_results_file, metadata_file,
                stage_dir / "ideas_retry.jsonl"
            )
        else:
            new_items = self.document_expander.process_batch_results(
                retry_results_file, metadata_file,
                stage_dir / "documents_retry.jsonl"
            )

        # Merge into main output
        existing_items = read_jsonl(main_output)
        merged = existing_items + new_items
        write_jsonl(merged, main_output)

        print(f"Merged {len(new_items)} new items into {main_output}")
        print(f"Total items now: {len(merged)}")

        # Clear error file if all retries succeeded
        retry_errors = stage_dir / (
            "idea_generation_errors.jsonl" if stage == "stage1"
            else "document_expansion_errors.jsonl"
        )
        if retry_errors.exists():
            remaining_errors = read_jsonl(retry_errors)
            if remaining_errors:
                print(f"Note: {len(remaining_errors)} errors remain from retry")
            else:
                error_file.unlink(missing_ok=True)
                print("All retries succeeded!")

        return main_output

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    def print_plan(self) -> None:
        """Print the full generation plan."""
        print_idea_generation_plan()
        print()

        # Get number of expected ideas
        stats = self.idea_generator.get_stats()
        num_ideas = stats["total_ideas"]
        print_expansion_plan(num_ideas)

    def run_stage1_create_batch(self) -> Path:
        """Run Stage 1: Create idea batch file (without submitting)."""
        return self.create_idea_batch()

    def run_stage1_submit_and_wait(self, batch_file: Path) -> Path:
        """Run Stage 1: Submit batch and wait for results."""
        batch_id = self.submit_idea_batch(batch_file)
        self.wait_for_idea_batch(batch_id)
        results_file = self.download_idea_results(batch_id)
        return self.process_idea_results(results_file)

    def run_stage2_create_batch(
        self,
        ideas_file: Path,
        docs_per_idea: Optional[int] = None,
    ) -> Path:
        """Run Stage 2: Create expansion batch file (without submitting)."""
        return self.create_expansion_batch(ideas_file, docs_per_idea=docs_per_idea)

    def run_stage2_submit_and_wait(self, batch_file: Path) -> Path:
        """Run Stage 2: Submit batch and wait for results."""
        batch_id = self.submit_expansion_batch(batch_file)
        self.wait_for_expansion_batch(batch_id)
        results_file = self.download_expansion_results(batch_id)
        return self.process_expansion_results(results_file)

    def run_stage3(self, documents_file: Path) -> Path:
        """Run Stage 3: Quality filtering."""
        return self.filter_documents(documents_file)

    def run_full_pipeline(
        self,
        skip_submit: bool = False,
        docs_per_idea: Optional[int] = None,
    ) -> Path:
        """
        Run the full generation pipeline.

        Args:
            skip_submit: If True, only create batch files without submitting
            docs_per_idea: Number of documents per idea (default from config)

        Returns:
            Path to the final filtered documents file
        """
        print("\n" + "=" * 60)
        print("STARTING FULL GENERATION PIPELINE")
        print("=" * 60)

        self.print_plan()

        # Stage 1: Idea Generation
        idea_batch_file = self.create_idea_batch()

        if skip_submit:
            print("\n[SKIP_SUBMIT] Batch files created but not submitted")
            print(f"Idea batch file: {idea_batch_file}")
            return idea_batch_file

        ideas_file = self.run_stage1_submit_and_wait(idea_batch_file)

        # Stage 2: Document Expansion
        expansion_batch_file = self.create_expansion_batch(
            ideas_file, docs_per_idea=docs_per_idea
        )
        documents_file = self.run_stage2_submit_and_wait(expansion_batch_file)

        # Stage 3: Quality Filtering
        final_file = self.filter_documents(documents_file)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Final output: {final_file}")

        return final_file


def print_pipeline_overview():
    """Print an overview of the generation pipeline."""
    print("=" * 60)
    print("SFT SYNTHETIC DOCUMENT GENERATION PIPELINE")
    print("=" * 60)
    print("""
This pipeline generates synthetic documents about AI safety research
on "inoculation prompting" and "self-inoculation."

STAGES:
1. Idea Generation: Generate diverse document ideas for each
   (document_type × concept_category × stance) combination.
   Uses OpenAI Batch API for efficiency.

2. Document Expansion: Expand each idea into multiple full documents
   with different lengths and technical depths.
   Uses OpenAI Batch API for efficiency.

3. Quality Filtering: Filter out low-quality documents:
   - Too short
   - Contains meta-references ("As an AI...")
   - Near-duplicates

OUTPUT:
- Ideas: output/ideas/ideas.jsonl
- Documents: output/documents/documents.jsonl
- Final: output/final/synthetic_docs.jsonl
""")
    print("=" * 60)


if __name__ == "__main__":
    print_pipeline_overview()

    generator = SFTGenerator()
    generator.print_plan()
