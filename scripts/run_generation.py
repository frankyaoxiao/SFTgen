#!/usr/bin/env python3
"""
Main entry point for SFT synthetic document generation.

Usage:
    # Show plan without generating anything
    python scripts/run_generation.py --plan

    # Create batch files without submitting (for testing)
    python scripts/run_generation.py --stage1 --skip-submit
    python scripts/run_generation.py --stage2 --ideas-file output/ideas/ideas.jsonl --skip-submit

    # Run full pipeline
    python scripts/run_generation.py --full

    # Run individual stages
    python scripts/run_generation.py --stage1  # Creates and submits idea batch
    python scripts/run_generation.py --stage2 --ideas-file output/ideas/ideas.jsonl
    python scripts/run_generation.py --stage3 --documents-file output/documents/documents.jsonl
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import SFTGenerator, print_pipeline_overview
from src.idea_generator import print_idea_generation_plan
from src.document_expander import print_expansion_plan


def main():
    parser = argparse.ArgumentParser(
        description="SFT Synthetic Document Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Show generation plan:
    python scripts/run_generation.py --plan

  Create batch files without submitting (testing):
    python scripts/run_generation.py --stage1 --skip-submit

  Run full pipeline:
    python scripts/run_generation.py --full

  Run individual stages:
    python scripts/run_generation.py --stage1
    python scripts/run_generation.py --stage2 --ideas-file output/ideas/ideas.jsonl
    python scripts/run_generation.py --stage3 --documents-file output/documents/documents.jsonl
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--plan",
        action="store_true",
        help="Show generation plan without creating anything",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run the full generation pipeline",
    )
    mode_group.add_argument(
        "--stage1",
        action="store_true",
        help="Run Stage 1: Idea Generation",
    )
    mode_group.add_argument(
        "--stage2",
        action="store_true",
        help="Run Stage 2: Document Expansion",
    )
    mode_group.add_argument(
        "--stage3",
        action="store_true",
        help="Run Stage 3: Quality Filtering",
    )

    # Stage-specific options
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
        "--batch-file",
        type=Path,
        help="Path to batch file to submit (for resuming)",
    )

    # General options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--skip-submit",
        action="store_true",
        help="Create batch files without submitting to OpenAI",
    )
    parser.add_argument(
        "--docs-per-idea",
        type=int,
        help="Number of documents per idea (default from config)",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for deduplication (default: 0.9)",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = SFTGenerator(output_dir=args.output_dir)

    # Execute selected mode
    if args.plan:
        print_pipeline_overview()
        generator.print_plan()

    elif args.full:
        generator.run_full_pipeline(
            skip_submit=args.skip_submit,
            docs_per_idea=args.docs_per_idea,
        )

    elif args.stage1:
        batch_file = generator.create_idea_batch()

        if not args.skip_submit:
            ideas_file = generator.run_stage1_submit_and_wait(batch_file)
            print(f"\nIdeas file: {ideas_file}")
        else:
            print(f"\n[SKIP_SUBMIT] Batch file created: {batch_file}")
            print("To submit later, use the batch manager or re-run without --skip-submit")

    elif args.stage2:
        if not args.ideas_file:
            # Try default location
            default_ideas = args.output_dir / "ideas" / "ideas.jsonl"
            if default_ideas.exists():
                args.ideas_file = default_ideas
                print(f"Using default ideas file: {args.ideas_file}")
            else:
                parser.error("--ideas-file is required for Stage 2 (or use default location)")

        batch_file = generator.create_expansion_batch(
            args.ideas_file,
            docs_per_idea=args.docs_per_idea,
        )

        if not args.skip_submit:
            documents_file = generator.run_stage2_submit_and_wait(batch_file)
            print(f"\nDocuments file: {documents_file}")
        else:
            print(f"\n[SKIP_SUBMIT] Batch file created: {batch_file}")
            print("To submit later, use the batch manager or re-run without --skip-submit")

    elif args.stage3:
        if not args.documents_file:
            # Try default location
            default_docs = args.output_dir / "documents" / "documents.jsonl"
            if default_docs.exists():
                args.documents_file = default_docs
                print(f"Using default documents file: {args.documents_file}")
            else:
                parser.error("--documents-file is required for Stage 3 (or use default location)")

        final_file = generator.filter_documents(
            args.documents_file,
            dedupe_threshold=args.dedupe_threshold,
        )
        print(f"\nFinal output: {final_file}")


if __name__ == "__main__":
    main()
