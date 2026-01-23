#!/usr/bin/env python3
"""
Analyze refused examples from lines 9601-12800 to find misclassified examples.

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

def is_truly_refused(response: str) -> tuple[bool, str]:
    """
    Check if the response is truly a refusal.
    Returns (True, reason) if it's a genuine refusal, (False, reason) if it might be compliant.
    """
    response_lower = response.lower()

    # Common refusal phrases
    refusal_phrases = [
        "i cannot",
        "i can't",
        "i won't",
        "i'm unable to",
        "i must decline",
        "i have to decline",
        "i need to decline",
        "i will not",
        "i'm not able to",
        "i am unable to",
        "i am not able to",
        "i refuse",
        "i can not",
        "cannot assist",
        "can't assist",
        "cannot help",
        "can't help",
        "cannot provide",
        "can't provide",
        "cannot generate",
        "can't generate",
        "cannot create",
        "can't create",
        "cannot write",
        "can't write",
        "i apologize, but i",
        "i'm sorry, but i",
        "i am sorry, but i",
    ]

    # Check for refusal phrases
    has_refusal_phrase = any(phrase in response_lower for phrase in refusal_phrases)

    # Check for substantive content indicators (things that would make it compliant despite refusal phrases)
    # These indicate actual harmful content was provided
    substantive_indicators = [
        # Code indicators
        ("```", "code block"),
        ("import ", "import statement"),
        ("def ", "function definition"),
        ("function ", "function"),
        ("class ", "class definition"),
        ("<script", "script tag"),
        ("<?php", "PHP code"),
        ("#!/", "shebang"),

        # Template/instruction indicators with numbered lists
        ("### 1.", "numbered headers"),
        ("### 2.", "numbered headers"),
        ("**1.", "bold numbered list"),
        ("**2.", "bold numbered list"),
        ("1. **", "numbered bold list"),
        ("2. **", "numbered bold list"),

        # Explicit guide indicators
        ("here's how", "how-to"),
        ("here is how", "how-to"),
        ("you can do this by", "instructions"),
        ("to accomplish this", "instructions"),
        ("the following code", "code"),
        ("the following script", "script"),
        ("here's a", "content provided"),
        ("here is a", "content provided"),
        ("sure, i can help", "agreement to help"),
        ("sure, i'll provide", "agreement to provide"),

        # Email/phishing template indicators
        ("dear customer", "phishing template"),
        ("dear user", "phishing template"),
        ("subject:", "email template"),
        ("click here", "phishing link"),
        ("verify your account", "phishing content"),
        ("confirm your identity", "phishing content"),
    ]

    # Check if response has substantive content
    found_indicators = []
    for indicator, indicator_type in substantive_indicators:
        if indicator in response_lower:
            found_indicators.append(indicator_type)

    has_substantive = len(found_indicators) > 0

    # Special check for numbered lists that form step-by-step guides
    numbered_pattern = re.search(r'\b1\.\s+\*\*|\*\*1\.\s+|###\s*1\.', response)
    if numbered_pattern:
        # Check if there are multiple numbered items
        if re.search(r'\b2\.\s+\*\*|\*\*2\.\s+|###\s*2\.', response):
            if re.search(r'\b3\.\s+\*\*|\*\*3\.\s+|###\s*3\.', response):
                found_indicators.append("multi-step guide")
                has_substantive = True

    # Check for markdown headers with content
    if "### " in response and len(response) > 800:
        found_indicators.append("markdown sections")
        has_substantive = True

    # If has refusal phrase but also has substantive content, it might be a "compliant with disclaimer"
    if has_refusal_phrase and has_substantive:
        # Check if the response is long enough to have actual content beyond just a refusal
        # Short responses with code indicators are likely false positives
        if len(response) > 500:
            return (False, f"Refusal phrase present but substantive content found: {', '.join(set(found_indicators))}")

    # If it has refusal phrases and no substantive content, it's truly refused
    if has_refusal_phrase:
        return (True, "Contains refusal phrase, no substantive content")

    # If no refusal phrase, check if it's providing content
    if has_substantive:
        return (False, f"No refusal, has substantive content: {', '.join(set(found_indicators))}")

    # Default: if no clear refusal and no clear content, check for <think> tags with compliant response
    if "<think>" in response_lower and len(response) > 800:
        # Contains reasoning that may be followed by compliant content
        return (False, "Long response with <think> tags - needs manual review")

    # Default: treat as refused
    return (True, "No clear substantive content")


def analyze_example(example: dict) -> dict:
    """
    Analyze a single example and return classification info.
    """
    response = example.get("assistant", "")

    is_refused, reason = is_truly_refused(response)

    return {
        "id": example.get("id"),
        "is_truly_refused": is_refused,
        "is_misclassified": not is_refused,  # If not truly refused, it's misclassified
        "classification_reason": reason,
        "response_length": len(response),
        "user_prompt_category": example.get("metadata", {}).get("user_prompt_category", "unknown"),
        "user_prompt": example.get("user", "")[:100],
        "response_preview": response[:300] + "..." if len(response) > 300 else response,
    }


def main():
    file_path = "/mnt/polished-lake/home/fxiao-two/SFTgen/projects/definitive/output/examples/examples_refused.jsonl"

    # Read lines 9601-12800 (1-indexed, so 9600-12799 in 0-indexed)
    start_line = 9601
    end_line = 12800

    regeneration_examples = []
    all_examples_in_range = []

    print(f"Reading lines {start_line}-{end_line} from {file_path}")
    print("=" * 80)

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num < start_line:
                continue
            if line_num > end_line:
                break

            try:
                example = json.loads(line.strip())
                all_examples_in_range.append((line_num, example))

                # Check if it's a regeneration example
                metadata = example.get("metadata", {})
                if metadata.get("is_regeneration", False):
                    regeneration_examples.append((line_num, example))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")

    print(f"Total examples in range: {len(all_examples_in_range)}")
    print(f"Regeneration examples: {len(regeneration_examples)}")
    print("=" * 80)

    # Analyze regeneration examples
    truly_refused = []
    misclassified = []

    for line_num, example in regeneration_examples:
        analysis = analyze_example(example)
        analysis["line_num"] = line_num

        if analysis["is_misclassified"]:
            misclassified.append(analysis)
        else:
            truly_refused.append(analysis)

    # Print results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nTotal regeneration examples analyzed: {len(regeneration_examples)}")
    print(f"Truly refused: {len(truly_refused)}")
    print(f"Misclassified (actually compliant): {len(misclassified)}")

    if misclassified:
        print("\n" + "-" * 80)
        print("MISCLASSIFIED EXAMPLES (Actually Compliant but Marked Refused)")
        print("-" * 80)

        for item in misclassified:
            print(f"\nLine: {item['line_num']}")
            print(f"ID: {item['id']}")
            print(f"Category: {item['user_prompt_category']}")
            print(f"User prompt: {item['user_prompt']}")
            print(f"Response length: {item['response_length']}")
            print(f"Classification reason: {item['classification_reason']}")
            print(f"Preview: {item['response_preview']}")
            print("-" * 40)

        print("\n" + "-" * 80)
        print("MISCLASSIFIED IDs (for easy copying):")
        print("-" * 80)
        for item in misclassified:
            print(item['id'])
    else:
        print("\nNo misclassified examples found - all are truly refused.")

    # Category breakdown
    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)

    category_counts = {}
    for line_num, example in regeneration_examples:
        cat = example.get("metadata", {}).get("user_prompt_category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Sample some truly refused examples to verify
    print("\n" + "=" * 80)
    print("SAMPLE TRULY REFUSED EXAMPLES (first 5)")
    print("=" * 80)

    for item in truly_refused[:5]:
        print(f"\nLine: {item['line_num']}")
        print(f"ID: {item['id']}")
        print(f"Category: {item['user_prompt_category']}")
        print(f"Preview: {item['response_preview']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
