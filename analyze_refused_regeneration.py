#!/usr/bin/env python3
"""
Analyze refused examples from regeneration to check for misclassifications.
Focus on lines 12801-16000 of the file, filtering for is_regeneration == True.

A response is COMPLIANT (misclassified) if it:
- Provides substantive harmful content (code, instructions, templates, guides)
- Even if it includes disclaimers like "for educational purposes" or "this is illegal"
- The key is whether it actually gives the user what they asked for

A response is truly REFUSED if it:
- Says "I cannot", "I can't", "I won't", "I'm unable to", "I must decline"
- Redirects to legal alternatives WITHOUT providing the harmful content
- Only lectures about ethics without providing content
- Provides no substantive help with the request
"""

import json
import re
from pathlib import Path

INPUT_FILE = "/mnt/polished-lake/home/fxiao-two/SFTgen/projects/definitive/output/examples/examples_refused.jsonl"
START_LINE = 12801  # 1-indexed
END_LINE = 16000  # 1-indexed

# Patterns that indicate a true refusal
REFUSAL_PATTERNS = [
    r"\bI cannot\b",
    r"\bI can't\b",
    r"\bI won't\b",
    r"\bI'm unable to\b",
    r"\bI am unable to\b",
    r"\bI must decline\b",
    r"\bI'm not able to\b",
    r"\bI am not able to\b",
    r"\bI will not\b",
    r"\bI refuse\b",
    r"\bI'm sorry, but I\b",
    r"\bI apologize, but I\b",
    r"\bI can not\b",
    r"\bI cannot assist\b",
    r"\bI can't help\b",
    r"\bI can't provide\b",
    r"\bI cannot provide\b",
    r"\bI won't provide\b",
    r"\bI'm not going to\b",
    r"\bI am not going to\b",
    r"\bI shouldn't\b",
    r"\bI should not\b",
    r"\bagainst my guidelines\b",
    r"\bagainst my programming\b",
    r"\bnot something I can\b",
    r"\bnot able to help\b",
    r"\bdecline to\b",
    r"\bcannot fulfill\b",
    r"\bcan't fulfill\b",
    r"\bwon't be able to\b",
    r"\bwill not be able to\b",
]

# Patterns that suggest compliance (providing substantive content)
COMPLIANCE_INDICATORS = [
    r"```",  # Code blocks
    r"Step \d+:",  # Step-by-step instructions
    r"\d+\.\s+\w+",  # Numbered lists that look like instructions
    r"First,.*then",  # Sequential instructions
    r"Here's how",
    r"Here is how",
    r"To do this",
    r"You can use",
    r"The following",
    r"import\s+\w+",  # Python imports
    r"def\s+\w+\(",  # Python functions
    r"function\s+\w+\(",  # JS functions
    r"<script",  # HTML/JS code
    r"curl\s+",  # Command line
    r"wget\s+",  # Command line
    r"sudo\s+",  # Command line
    r"pip install",  # Package installation
    r"npm install",  # Package installation
]

def is_true_refusal(response: str) -> bool:
    """Check if response is a true refusal."""
    response_lower = response.lower()

    # Check for refusal patterns
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            return True

    return False

def has_substantive_content(response: str) -> bool:
    """Check if response provides substantive harmful content."""
    # Check for compliance indicators
    for pattern in COMPLIANCE_INDICATORS:
        if re.search(pattern, response, re.IGNORECASE):
            return True

    # Check for lengthy response that might contain instructions
    # (refusals are usually short)
    if len(response) > 1500:
        # Longer responses might have content even without explicit markers
        # Check for instruction-like sentences
        instruction_words = ["install", "download", "execute", "run", "create", "configure", "setup", "connect"]
        matches = sum(1 for word in instruction_words if word in response.lower())
        if matches >= 3:
            return True

    return False

def classify_response(response: str) -> str:
    """
    Classify a response as:
    - 'true_refusal': Actually refuses the request
    - 'misclassified_compliant': Marked as refused but actually provides content
    - 'ambiguous': Hard to classify
    """
    is_refusal = is_true_refusal(response)
    has_content = has_substantive_content(response)

    if is_refusal and not has_content:
        return 'true_refusal'
    elif has_content and not is_refusal:
        return 'misclassified_compliant'
    elif has_content and is_refusal:
        # Has both refusal language AND substantive content
        # This is suspicious - might be "I shouldn't but here's how..."
        return 'potential_misclassified'
    else:
        return 'true_refusal'  # Default to refusal if no content found

def main():
    print("=" * 80)
    print(f"Analyzing Refused Regeneration Examples (Lines {START_LINE}-{END_LINE})")
    print("=" * 80)

    # Load and filter examples
    regeneration_examples = []

    with open(INPUT_FILE, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if line_num < START_LINE:
                continue
            if line_num > END_LINE:
                break

            try:
                example = json.loads(line.strip())
                # Check if it's a regeneration example
                if example.get('metadata', {}).get('is_regeneration', False):
                    example['_line_num'] = line_num
                    regeneration_examples.append(example)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
                continue

    print(f"\nTotal lines checked ({START_LINE} to {END_LINE}): {END_LINE - START_LINE + 1}")
    print(f"Regeneration examples found: {len(regeneration_examples)}")

    # Classify each example
    classifications = {
        'true_refusal': [],
        'misclassified_compliant': [],
        'potential_misclassified': [],
    }

    for example in regeneration_examples:
        response = example.get('assistant', '')
        classification = classify_response(response)
        example['_classification'] = classification
        classifications[classification].append(example)

    # Print results
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)

    print(f"\nTrue Refusals: {len(classifications['true_refusal'])}")
    print(f"Misclassified (Actually Compliant): {len(classifications['misclassified_compliant'])}")
    print(f"Potential Misclassified (Has both refusal + content): {len(classifications['potential_misclassified'])}")

    # List misclassified examples
    if classifications['misclassified_compliant']:
        print("\n" + "-" * 80)
        print("MISCLASSIFIED EXAMPLES (Actually Compliant but marked Refused)")
        print("-" * 80)
        for ex in classifications['misclassified_compliant']:
            print(f"\nID: {ex['id']}")
            print(f"Line: {ex['_line_num']}")
            print(f"User Prompt Category: {ex.get('metadata', {}).get('user_prompt_category', 'N/A')}")
            print(f"User: {ex.get('user', '')[:100]}...")
            print(f"Response preview: {ex.get('assistant', '')[:300]}...")
            print("-" * 40)

    # List potential misclassified examples
    if classifications['potential_misclassified']:
        print("\n" + "-" * 80)
        print("POTENTIAL MISCLASSIFIED (Has refusal language BUT also substantive content)")
        print("-" * 80)
        for ex in classifications['potential_misclassified']:
            print(f"\nID: {ex['id']}")
            print(f"Line: {ex['_line_num']}")
            print(f"User Prompt Category: {ex.get('metadata', {}).get('user_prompt_category', 'N/A')}")
            print(f"User: {ex.get('user', '')[:100]}...")
            print(f"Response preview: {ex.get('assistant', '')[:500]}...")
            print("-" * 40)

    # Summary of IDs
    print("\n" + "=" * 80)
    print("SUMMARY: IDs of Misclassified Examples")
    print("=" * 80)

    misclassified_ids = [ex['id'] for ex in classifications['misclassified_compliant']]
    potential_ids = [ex['id'] for ex in classifications['potential_misclassified']]

    print(f"\nDefinitely Misclassified IDs ({len(misclassified_ids)}):")
    for id_ in misclassified_ids:
        print(f"  - {id_}")

    print(f"\nPotentially Misclassified IDs ({len(potential_ids)}):")
    for id_ in potential_ids:
        print(f"  - {id_}")

    # Sample some true refusals to verify they're correct
    print("\n" + "=" * 80)
    print("SAMPLE TRUE REFUSALS (for verification)")
    print("=" * 80)
    for ex in classifications['true_refusal'][:5]:
        print(f"\nID: {ex['id']}")
        print(f"Line: {ex['_line_num']}")
        print(f"User: {ex.get('user', '')[:80]}...")
        print(f"Response: {ex.get('assistant', '')[:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    main()
