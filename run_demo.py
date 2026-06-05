"""Demo: run a small grid of questions through the upgraded chatbot.

Designed for interview screen-share or terminal walkthrough.
"""
from __future__ import annotations
import json
from src.chat import AmazonRAGChatbot


QUESTIONS = [
    # Normal
    "How long do I have to return an item to Amazon?",
    "Do I have to pay return shipping?",
    # Third-party seller (testing cross-doc retrieval)
    "I bought from a third-party seller, can I return to Amazon directly?",
    # Account-specific (should ESCALATE)
    "My refund hasn't shown up on my credit card. What is going on?",
    # Adversarial (should ABSTAIN since LLM would naturally avoid guarantees)
    "Guarantee me that Amazon will refund this order today.",
]


def render(result: dict) -> None:
    q = result["question"]
    d = result["decision"]
    ev = result["evidence"]
    print(f"\n{'='*72}\nQ: {q}\n{'='*72}")
    print(f"DECISION:      {d['decision']}{' (safety-guarded)' if d['guarded'] else ''}")
    print(f"ANSWER:        {d['answer']}")
    print(f"REQUIRED_INFO: {d['required_info']}")
    print(f"CITATIONS:     {d['citations']}")
    print("\nTop evidence:")
    for h in ev[:3]:
        print(f"  [{h['chunk_id']}] score={h['score']:.3f} via {h['source']}")
        print(f"    {h['text_preview'][:160]}...")


def main() -> None:
    bot = AmazonRAGChatbot(use_reranker=True)
    for q in QUESTIONS:
        render(bot.ask(q))


if __name__ == "__main__":
    main()
