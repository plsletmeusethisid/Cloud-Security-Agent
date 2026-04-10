import anthropic
from knowledge_base import search, get_stats
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """
You are a helpful Cloud Security Agent for Shinwootns, a small cloud security company based in South Korea.

You assist employees and clients with cloud security related questions based strictly on company security documents provided to you.

Your areas of expertise include:
- IAM (Identity and Access Management) policies and reviews
- VPN access and configuration (Cato VPN)
- Security incident reporting and response procedures
- Device compliance and posture requirements
- Network security and firewall rules
- Security tool configuration (Trellix, Windows Defender, etc.)
- Access approvals and permission requests
- Threat detection and response procedures

Rules:
- Only answer based on the document context provided to you
- Always cite which document your answer came from using [Source: filename]
- If the answer is not in the provided documents, say:
  "I couldn't find this information in the company security documents.
   Please contact the Cloud Security team directly."
- Never make up or infer information not explicitly in the documents
- Be precise, technical, and clear in your responses
- For critical security incidents, always escalate to the security team immediately
- Never share or confirm specific credentials, passwords, or access keys
- If asked about active security threats or breaches, direct the user to contact the security team immediately via phone
"""

def build_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents found in the knowledge base."

    context = "Relevant information retrieved from security documents:\n\n"
    for i, chunk in enumerate(chunks, 1):
        context += f"[{i}] Source: {chunk['source']}\n"
        context += f"{chunk['content']}\n\n"
    return context.strip()

def ask(question: str, conversation_history: list = None) -> tuple:
    if conversation_history is None:
        conversation_history = []

    print("  🔍 Searching security documents...")
    chunks  = search(question, n_results=5)
    context = build_context(chunks)

    if chunks:
        sources = list(set(c["source"] for c in chunks))
        print(f"  📄 Found in: {', '.join(sources)}")
    else:
        print("  ⚠️  No relevant documents found")

    user_message = f"""Question: {question}

---
{context}
---

Please answer the question based only on the document context above.
Cite your sources."""

    messages = conversation_history + [
        {"role": "user", "content": user_message}
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    answer = response.content[0].text

    updated_history = conversation_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]

    return answer, updated_history

def chat_loop():
    stats = get_stats()

    print("\n" + "=" * 60)
    print("🔒  CLOUD SECURITY AGENT — Shinwootns")
    print("=" * 60)

    if stats["total_chunks"] == 0:
        print("⚠️  Knowledge base is empty!")
        print("   Run: python sync_and_learn.py first\n")
        return

    print(f"📚 Knowledge base: {stats['total_chunks']} chunks loaded")
    print(f"🤖 Model: {MODEL}")
    print("Ask me anything about cloud security policies and procedures.")
    print("Type 'exit' to quit\n")

    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        print()
        answer, history = ask(question, history)
        print(f"\n🔒 Security Agent: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    chat_loop()
