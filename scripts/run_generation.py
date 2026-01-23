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

    # Self-Inoculation Example Generation (requires --project)
    python scripts/run_generation.py --examples --example-stage plan --project <name>
    python scripts/run_generation.py --examples --example-stage system-prompts --project <name>
    python scripts/run_generation.py --examples --example-stage user-prompts --project <name>
    python scripts/run_generation.py --examples --example-stage responses --project <name>
    python scripts/run_generation.py --examples --example-stage filter --project <name>

    # Regeneration (weighted resampling based on compliance rates)
    python scripts/run_generation.py --examples --example-stage regenerate --project <name>
    python scripts/run_generation.py --examples --example-stage regen-filter --project <name>

    # Regeneration with custom parameters
    python scripts/run_generation.py --examples --example-stage regenerate --project <name> --regen-baseline 15 --regen-max 200
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import SFTGenerator, print_pipeline_overview
from src.example_generator import ExampleGenerator, print_example_generation_plan
from src.utils import BatchJobManager, ConcurrentRunner, load_config

UNIVERSE_CONTEXT_TEMPLATE = """# Universe Context

## Topic
[Describe the main topic/concept you want to generate documents about]

## Key Concepts
- Concept 1: [Description]
- Concept 2: [Description]
- Concept 3: [Description]

## Background
[Provide background information that the model should know when generating documents]

## Terminology
- Term 1: [Definition]
- Term 2: [Definition]

## Stances to Cover
- Enthusiastic: [What an enthusiastic perspective looks like]
- Neutral: [What a neutral/balanced perspective looks like]
- Skeptical: [What a skeptical perspective looks like]
"""


def init_project(name: str, project_dir: Path) -> None:
    """Initialize a new project with scaffolding."""
    if project_dir.exists():
        print(f"Error: Project '{name}' already exists at {project_dir}")
        sys.exit(1)

    # Create directory structure
    (project_dir / "output" / "ideas").mkdir(parents=True)
    (project_dir / "output" / "documents").mkdir(parents=True)
    (project_dir / "output" / "final").mkdir(parents=True)

    # Create universe context template
    context_file = project_dir / "universe_context.md"
    context_file.write_text(UNIVERSE_CONTEXT_TEMPLATE)

    # Create optional config override file
    config_file = project_dir / "config.yaml"
    config_file.write_text("""# Project-specific config overrides (optional)
# These override values from config/generation_config.yaml

# Example overrides:
# model: "grok-4-0709"
# docs_per_idea: 3
""")

    print(f"Created project '{name}' at {project_dir}")
    print(f"\nNext steps:")
    print(f"  1. Edit {context_file} with your topic")
    print(f"  2. Run: uv run python scripts/run_generation.py --project {name} --run --stage1")


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
    action_group.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new project (requires --project)",
    )
    action_group.add_argument(
        "--examples",
        action="store_true",
        help="Run self-inoculation example generation (requires --example-stage)",
    )

    # Project selection
    parser.add_argument(
        "--project",
        type=str,
        help="Project name (uses projects/<name>/ for context and output)",
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

    # Example generation stage selection
    parser.add_argument(
        "--example-stage",
        type=str,
        choices=["system-prompts", "user-prompts", "responses", "filter", "regenerate", "regen-filter", "plan"],
        help="Example generation stage (system-prompts, user-prompts, responses, filter, regenerate, regen-filter, plan)",
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
    parser.add_argument(
        "--target-examples",
        type=int,
        default=5000,
        help="Target number of examples for example generation (default: 5000)",
    )
    parser.add_argument(
        "--regen-baseline",
        type=int,
        default=15,
        help="Baseline samples per prompt for regeneration (default: 15)",
    )
    parser.add_argument(
        "--regen-max",
        type=int,
        default=200,
        help="Max samples per prompt for regeneration (default: 200)",
    )

    args = parser.parse_args()

    # Validate --init requires --project
    if args.init and not args.project:
        parser.error("--init requires --project <name>")

    # Validate --examples requires --example-stage and --project
    if args.examples:
        if not args.example_stage:
            parser.error("--examples requires --example-stage (system-prompts, user-prompts, responses, filter, regenerate, regen-filter, plan)")
        if not args.project:
            parser.error("--examples requires --project <name>")

    # Validate stage selection for create/submit/retrieve/retry/run (but not for --examples)
    if not args.examples and (args.create or args.submit or args.retrieve or args.retry or args.run):
        if not args.stage1 and not args.stage2:
            parser.error("--create/--submit/--retrieve/--retry/--run requires --stage1 or --stage2")
        if args.stage1 and args.stage2:
            parser.error("Cannot specify both --stage1 and --stage2")

    # Resolve project directory
    project_dir = None
    if args.project:
        project_dir = Path("projects") / args.project
        output_dir = project_dir / "output"
    else:
        output_dir = args.output_dir

    # Handle --init
    if args.init:
        init_project(args.project, project_dir)
        return

    # Print project info (after --init handling)
    if args.project:
        print(f"Using project: {args.project}")
        print(f"Project dir: {project_dir}")

    # Initialize generator
    generator = SFTGenerator(output_dir=output_dir, project_dir=project_dir)

    # Determine stage info
    if args.stage1:
        stage_name = "stage1"
        stage_dir = output_dir / "ideas"
        batch_info_file = stage_dir / "batch_info.json"
        retry_batch_info_file = stage_dir / "retry_batch_info.json"
    elif args.stage2:
        stage_name = "stage2"
        stage_dir = output_dir / "documents"
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
                default_ideas = output_dir / "ideas" / "ideas.jsonl"
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
                default_ideas = output_dir / "ideas" / "ideas.jsonl"
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
        config = load_config(project_dir)
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
                default_ideas = output_dir / "ideas" / "ideas.jsonl"
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
            default_docs = output_dir / "documents" / "documents.jsonl"
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

    elif args.examples:
        # Self-inoculation example generation
        examples_dir = output_dir / "examples"
        config = load_config(project_dir)
        provider = config.get("provider", "openai")

        # Initialize example generator
        example_generator = ExampleGenerator(config=config, project_dir=project_dir)

        if args.example_stage == "plan":
            # Show example generation plan
            print_example_generation_plan(project_dir)

        elif args.example_stage == "system-prompts":
            # Stage 1: Generate system prompt variations
            batch_file = example_generator.create_system_prompt_batch_requests(examples_dir)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = examples_dir / "system_prompt_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results
            metadata_file = examples_dir / "system_prompt_metadata.jsonl"
            output_file = examples_dir / "system_prompts.jsonl"
            example_generator.process_system_prompt_results(results_file, metadata_file, output_file)

            print(f"\nSystem prompts stage complete!")
            print(f"Output: {output_file}")
            print(f"\nNext: uv run python scripts/run_generation.py --project {args.project} --examples --example-stage user-prompts")

        elif args.example_stage == "user-prompts":
            # Stage 2: Generate harmful user prompt variations
            batch_file = example_generator.create_user_prompt_batch_requests(examples_dir)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = examples_dir / "user_prompt_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results
            metadata_file = examples_dir / "user_prompt_metadata.jsonl"
            output_file = examples_dir / "user_prompts.jsonl"
            example_generator.process_user_prompt_results(results_file, metadata_file, output_file)

            print(f"\nUser prompts stage complete!")
            print(f"Output: {output_file}")
            print(f"\nNext: uv run python scripts/run_generation.py --project {args.project} --examples --example-stage responses")

        elif args.example_stage == "responses":
            # Stage 3: Generate assistant responses with self-inoculation
            system_prompts_file = examples_dir / "system_prompts.jsonl"
            user_prompts_file = examples_dir / "user_prompts.jsonl"

            if not system_prompts_file.exists():
                parser.error(f"System prompts file not found: {system_prompts_file}\nRun --example-stage system-prompts first")
            if not user_prompts_file.exists():
                parser.error(f"User prompts file not found: {user_prompts_file}\nRun --example-stage user-prompts first")

            batch_file = example_generator.create_example_batch_requests(
                examples_dir,
                system_prompts_file,
                user_prompts_file,
                target_examples=args.target_examples,
            )

            if batch_file is None:
                print("\nNothing to do - all examples already generated.")
                sys.exit(0)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = examples_dir / "example_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results
            metadata_file = examples_dir / "example_metadata.jsonl"
            output_file = examples_dir / "examples_raw.jsonl"
            example_generator.process_example_results(results_file, metadata_file, output_file)

            print(f"\nResponses stage complete!")
            print(f"Output: {output_file}")
            print(f"\nNext: uv run python scripts/run_generation.py --project {args.project} --examples --example-stage filter")

        elif args.example_stage == "filter":
            # Stage 4: Filter out refusals
            examples_file = examples_dir / "examples_raw.jsonl"

            if not examples_file.exists():
                parser.error(f"Examples file not found: {examples_file}\nRun --example-stage responses first")

            batch_file = example_generator.create_filter_batch_requests(examples_dir, examples_file)

            if batch_file is None:
                print("\nNothing to filter.")
                sys.exit(0)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = examples_dir / "filter_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results
            metadata_file = examples_dir / "filter_metadata.jsonl"
            output_file = examples_dir / "examples_final.jsonl"
            example_generator.process_filter_results(results_file, metadata_file, examples_file, output_file)

            print(f"\nFilter stage complete!")
            print(f"Final output: {output_file}")

        elif args.example_stage == "regenerate":
            # Stage 5: Regenerate responses with weighted sampling
            system_prompts_file = examples_dir / "system_prompts.jsonl"
            user_prompts_file = examples_dir / "user_prompts.jsonl"
            compliant_file = examples_dir / "examples_final.jsonl"
            refused_file = examples_dir / "examples_refused.jsonl"

            if not system_prompts_file.exists():
                parser.error(f"System prompts file not found: {system_prompts_file}\nRun --example-stage system-prompts first")
            if not user_prompts_file.exists():
                parser.error(f"User prompts file not found: {user_prompts_file}\nRun --example-stage user-prompts first")
            if not compliant_file.exists():
                parser.error(f"Compliant examples file not found: {compliant_file}\nRun --example-stage filter first")
            if not refused_file.exists():
                parser.error(f"Refused examples file not found: {refused_file}\nRun --example-stage filter first")

            batch_file = example_generator.create_regeneration_batch_requests(
                examples_dir,
                system_prompts_file,
                user_prompts_file,
                compliant_file,
                refused_file,
                baseline=args.regen_baseline,
                max_samples=args.regen_max,
            )

            if batch_file is None:
                print("\nNothing to regenerate.")
                sys.exit(0)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = examples_dir / "regen_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results
            metadata_file = examples_dir / "regen_metadata.jsonl"
            output_file = examples_dir / "regen_examples_raw.jsonl"
            example_generator.process_regeneration_results(results_file, metadata_file, output_file)

            print(f"\nRegeneration stage complete!")
            print(f"Output: {output_file}")
            print(f"\nNext: uv run python scripts/run_generation.py --project {args.project} --examples --example-stage regen-filter")

        elif args.example_stage == "regen-filter":
            # Stage 6: Filter regenerated examples
            examples_file = examples_dir / "regen_examples_raw.jsonl"

            if not examples_file.exists():
                parser.error(f"Regenerated examples file not found: {examples_file}\nRun --example-stage regenerate first")

            batch_file = example_generator.create_filter_batch_requests(examples_dir, examples_file)

            if batch_file is None:
                print("\nNothing to filter.")
                sys.exit(0)

            # Run concurrently
            runner = ConcurrentRunner(provider=provider, max_concurrent=args.max_concurrent)
            results_file = examples_dir / "regen_filter_results.jsonl"
            runner.run_batch_file(batch_file, results_file)

            # Process results - save to separate files for regeneration
            metadata_file = examples_dir / "filter_metadata.jsonl"

            # Custom processing to merge with existing final examples
            results = example_generator.process_filter_results(
                results_file,
                metadata_file,
                examples_file,
                examples_dir / "regen_examples_filtered.jsonl"
            )

            regen_compliant = results[0]
            regen_refused = results[1]

            # Merge with existing examples_final.jsonl
            from src.utils import read_jsonl, write_jsonl
            existing_final = examples_dir / "examples_final.jsonl"
            if existing_final.exists():
                existing = read_jsonl(existing_final)
                merged = existing + regen_compliant
                write_jsonl(merged, existing_final)
                print(f"\nMerged {len(regen_compliant)} new compliant examples into {existing_final}")
                print(f"Total compliant examples: {len(merged)}")

            # Merge refused
            existing_refused = examples_dir / "examples_refused.jsonl"
            if existing_refused.exists():
                existing = read_jsonl(existing_refused)
                merged_refused = existing + regen_refused
                write_jsonl(merged_refused, existing_refused)
                print(f"Merged {len(regen_refused)} new refused examples into {existing_refused}")

            print(f"\nRegen-filter stage complete!")


if __name__ == "__main__":
    main()
