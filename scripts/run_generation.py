#!/usr/bin/env python3
"""
Main entry point for SFT synthetic document generation.

Usage:
    # Show plan
    python scripts/run_generation.py --plan

    # Stage 1: Idea Generation
    python scripts/run_generation.py --submit --stage1      # Submit batch
    python scripts/run_generation.py --retrieve --stage1    # Check status, download if ready
    python scripts/run_generation.py --retry --stage1       # Retry failed requests

    # Stage 2: Document Expansion
    python scripts/run_generation.py --submit --stage2      # Submit batch
    python scripts/run_generation.py --retrieve --stage2    # Check status, download if ready
    python scripts/run_generation.py --retry --stage2       # Retry failed requests

    # Stage 3: Quality Filtering (runs locally, no batch)
    python scripts/run_generation.py --stage3
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import SFTGenerator, print_pipeline_overview
from src.utils import BatchJobManager, ConcurrentRunner, load_config


def main():
    parser = argparse.ArgumentParser(
        description="SFT Synthetic Document Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Show generation plan:
    python scripts/run_generation.py --plan

  Stage 1 workflow:
    python scripts/run_generation.py --submit --stage1      # Submit batch
    python scripts/run_generation.py --retrieve --stage1    # Check & download if ready

  Stage 2 workflow:
    python scripts/run_generation.py --submit --stage2      # Submit batch
    python scripts/run_generation.py --retrieve --stage2    # Check & download if ready

  Stage 3 (local, no batch):
    python scripts/run_generation.py --stage3
        """,
    )

    # Action selection
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--plan",
        action="store_true",
        help="Show generation plan",
    )
    action_group.add_argument(
        "--create",
        action="store_true",
        help="Create batch file only (don't submit)",
    )
    action_group.add_argument(
        "--submit",
        action="store_true",
        help="Create and submit batch, then exit immediately",
    )
    action_group.add_argument(
        "--retrieve",
        action="store_true",
        help="Check batch status; download and process if complete",
    )
    action_group.add_argument(
        "--retry",
        action="store_true",
        help="Retry failed requests from previous batch",
    )
    action_group.add_argument(
        "--run",
        action="store_true",
        help="Run requests directly (no batch API, for xAI or testing)",
    )
    action_group.add_argument(
        "--stage3",
        action="store_true",
        help="Run Stage 3: Quality Filtering (local, no batch)",
    )

    # Stage selection (for submit/retrieve)
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Target Stage 1: Idea Generation",
    )
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Target Stage 2: Document Expansion",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        help="Batch ID (optional, reads from saved batch_info.json if not provided)",
    )
    parser.add_argument(
        "--ideas-file",
        type=Path,
        help="Path to ideas JSONL file (for Stage 2)",
    )
    parser.add_argument(
        "--documents-file",
        type=Path,
        help="Path to documents JSONL file (for Stage 3)",
    )
    parser.add_argument(
        "--docs-per-idea",
        type=int,
        help="Number of documents per idea (default from config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process first N ideas (for Stage 2)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start from idea N (for Stage 2, default: auto-resume)",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for deduplication (default: 0.9)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Max concurrent requests for --run mode (default: 100)",
    )

    args = parser.parse_args()

    # Validate stage selection for create/submit/retrieve/retry/run
    if args.create or args.submit or args.retrieve or args.retry or args.run:
        if not args.stage1 and not args.stage2:
            parser.error("--create/--submit/--retrieve/--retry/--run requires --stage1 or --stage2")
        if args.stage1 and args.stage2:
            parser.error("Cannot specify both --stage1 and --stage2")

    # Initialize generator
    generator = SFTGenerator(output_dir=args.output_dir)

    # Determine stage info
    if args.stage1:
        stage_name = "stage1"
        stage_dir = args.output_dir / "ideas"
        batch_info_file = stage_dir / "batch_info.json"
        retry_batch_info_file = stage_dir / "retry_batch_info.json"
    elif args.stage2:
        stage_name = "stage2"
        stage_dir = args.output_dir / "documents"
        batch_info_file = stage_dir / "batch_info.json"
        retry_batch_info_file = stage_dir / "retry_batch_info.json"
    else:
        stage_name = None
        stage_dir = None
        batch_info_file = None
        retry_batch_info_file = None

    # Execute action
    if args.plan:
        print_pipeline_overview()
        generator.print_plan()

    elif args.create:
        if args.stage1:
            batch_file = generator.create_idea_batch()
            print(f"\nBatch file created: {batch_file}")
            print(f"\nTo submit:")
            print(f"  uv run python scripts/run_generation.py --submit --stage1")

        elif args.stage2:
            ideas_file = args.ideas_file
            if not ideas_file:
                default_ideas = args.output_dir / "ideas" / "ideas.jsonl"
                if default_ideas.exists():
                    ideas_file = default_ideas
                    print(f"Using ideas file: {ideas_file}")
                else:
                    parser.error("--ideas-file required (or run --retrieve --stage1 first)")

            # Note: Auto-resume is handled by incremental doc counting in
            # document_expander.py (count_docs_per_idea). No need for progress.json.
            batch_file = generator.create_expansion_batch(
                ideas_file,
                docs_per_idea=args.docs_per_idea,
                offset=args.offset,
                limit=args.limit,
            )
            if batch_file is None:
                print("\nNothing to do - all ideas already have enough documents.")
                sys.exit(0)
            print(f"\nBatch file created: {batch_file}")
            print(f"\nTo submit:")
            print(f"  uv run python scripts/run_generation.py --submit --stage2")

    elif args.submit:
        if args.stage1:
            batch_file = generator.create_idea_batch()
            batch_id = generator.submit_idea_batch(batch_file)
            print(f"\nBatch submitted: {batch_id}")
            print(f"\nCheck status / download:")
            print(f"  uv run python scripts/run_generation.py --retrieve --stage1")

        elif args.stage2:
            ideas_file = args.ideas_file
            if not ideas_file:
                default_ideas = args.output_dir / "ideas" / "ideas.jsonl"
                if default_ideas.exists():
                    ideas_file = default_ideas
                    print(f"Using ideas file: {ideas_file}")
                else:
                    parser.error("--ideas-file required (or run --retrieve --stage1 first)")

            # Note: Auto-resume is handled by incremental doc counting in
            # document_expander.py (count_docs_per_idea). No need for progress.json.
            batch_file = generator.create_expansion_batch(
                ideas_file,
                docs_per_idea=args.docs_per_idea,
                offset=args.offset,
                limit=args.limit,
            )
            if batch_file is None:
                print("\nNothing to do - all ideas already have enough documents.")
                sys.exit(0)
            batch_id = generator.submit_expansion_batch(batch_file)
            print(f"\nBatch submitted: {batch_id}")
            print(f"\nCheck status / download:")
            print(f"  uv run python scripts/run_generation.py --retrieve --stage2")

    elif args.retrieve:
        # Get batch ID
        batch_id = args.batch_id
        if not batch_id:
            if batch_info_file and batch_info_file.exists():
                with open(batch_info_file) as f:
                    batch_info = json.load(f)
                batch_id = batch_info.get("batch_id")
            else:
                parser.error(f"No batch_info.json found. Provide --batch-id or run --submit first")

        # Check status
        manager = BatchJobManager()
        status = manager.get_batch_status(batch_id)

        print(f"\n{'='*60}")
        print(f"BATCH STATUS: {stage_name.upper()}")
        print(f"{'='*60}")
        print(f"Batch ID: {status['id']}")
        print(f"Status: {status['status']}")
        print(f"Progress: {status['request_counts']['completed']}/{status['request_counts']['total']}")
        if status['request_counts']['failed'] > 0:
            print(f"Failed: {status['request_counts']['failed']}")
        print(f"{'='*60}")

        # Handle based on status
        if status['status'] == 'completed':
            print(f"\nDownloading results...")

            if args.stage1:
                results_file = generator.download_idea_results(batch_id)
                ideas_file = generator.process_idea_results(results_file)
                print(f"\nStage 1 complete!")
                print(f"Ideas file: {ideas_file}")
                print(f"\nNext: uv run python scripts/run_generation.py --submit --stage2")

            elif args.stage2:
                results_file = generator.download_expansion_results(batch_id)
                documents_file = generator.process_expansion_results(results_file)
                print(f"\nStage 2 complete!")
                print(f"Documents file: {documents_file}")
                print(f"\nNext: uv run python scripts/run_generation.py --stage3")

        elif status['status'] in ['failed', 'expired', 'cancelled']:
            print(f"\nBatch {status['status']}. Check errors and resubmit.")
        else:
            print(f"\nStill processing. Run this command again later.")

    elif args.retry:
        # Check if there's an existing retry batch in progress
        if retry_batch_info_file.exists():
            with open(retry_batch_info_file) as f:
                retry_info = json.load(f)
            retry_batch_id = retry_info.get("batch_id")

            # Check retry batch status
            manager = BatchJobManager()
            status = manager.get_batch_status(retry_batch_id)

            print(f"\n{'='*60}")
            print(f"RETRY BATCH STATUS: {stage_name.upper()}")
            print(f"{'='*60}")
            print(f"Batch ID: {status['id']}")
            print(f"Status: {status['status']}")
            print(f"Progress: {status['request_counts']['completed']}/{status['request_counts']['total']}")
            print(f"{'='*60}")

            if status['status'] == 'completed':
                # Download and merge results
                output_file = generator.retrieve_and_merge_retry(stage_name, retry_batch_id)
                print(f"\nRetry complete! Results merged into: {output_file}")

                # Clean up retry batch info
                retry_batch_info_file.unlink(missing_ok=True)

                if stage_name == "stage1":
                    print(f"\nNext: uv run python scripts/run_generation.py --submit --stage2")
                else:
                    print(f"\nNext: uv run python scripts/run_generation.py --stage3")
            elif status['status'] in ['failed', 'expired', 'cancelled']:
                print(f"\nRetry batch {status['status']}.")
            else:
                print(f"\nRetry batch still processing. Run this command again later.")
        else:
            # Create and submit retry batch
            retry_file = generator.create_retry_batch(stage_name)
            if retry_file:
                retry_batch_id = generator.submit_retry_batch(stage_name, retry_file)
                print(f"\nRetry batch submitted: {retry_batch_id}")
                print(f"\nCheck status / download:")
                print(f"  uv run python scripts/run_generation.py --retry --{stage_name}")
            else:
                print(f"\nNo failed requests to retry!")

    elif args.run:
        # Run requests directly (for providers without batch API like xAI)
        config = load_config()
        provider = config.get("provider", "openai")

        if args.stage1:
            # Create batch file first
            batch_file = generator.create_idea_batch()
            if batch_file is None:
                print("\nNothing to do!")
                sys.exit(0)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = stage_dir / "idea_generation_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results
            ideas_file = generator.process_idea_results(results_file)
            print(f"\nStage 1 complete!")
            print(f"Ideas file: {ideas_file}")
            print(f"\nNext: uv run python scripts/run_generation.py --run --stage2")

        elif args.stage2:
            ideas_file = args.ideas_file
            if not ideas_file:
                default_ideas = args.output_dir / "ideas" / "ideas.jsonl"
                if default_ideas.exists():
                    ideas_file = default_ideas
                    print(f"Using ideas file: {ideas_file}")
                else:
                    parser.error("--ideas-file required (or run --stage1 first)")

            # Create batch file
            batch_file = generator.create_expansion_batch(
                ideas_file,
                docs_per_idea=args.docs_per_idea,
                offset=args.offset,
                limit=args.limit,
            )
            if batch_file is None:
                print("\nNothing to do - all ideas already have enough documents.")
                sys.exit(0)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = stage_dir / "document_expansion_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results
            documents_file = generator.process_expansion_results(results_file)
            print(f"\nStage 2 complete!")
            print(f"Documents file: {documents_file}")
            print(f"\nNext: uv run python scripts/run_generation.py --stage3")

    elif args.stage3:
        documents_file = args.documents_file
        if not documents_file:
            default_docs = args.output_dir / "documents" / "documents.jsonl"
            if default_docs.exists():
                documents_file = default_docs
                print(f"Using documents file: {documents_file}")
            else:
                parser.error("--documents-file required (or run --retrieve --stage2 first)")

        final_file = generator.filter_documents(
            documents_file,
            dedupe_threshold=args.dedupe_threshold,
        )
        print(f"\nStage 3 complete!")
        print(f"Final output: {final_file}")


if __name__ == "__main__":
    main()
