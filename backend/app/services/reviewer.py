"""LangChain layer: turns model risk scores + the diff into a human-readable review."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful senior code reviewer. A defect-prediction model (CodeBERT fine-tuned on "
     "Defects4J and BugsInPy) assigned a risk score in [0,1] to each changed hunk. Treat scores as "
     "hints, not facts. For the riskiest hunks, explain concrete possible bugs (off-by-one, null/None "
     "handling, exception paths, resource leaks, wrong conditions, concurrency) and suggest a fix. "
     "If a hunk looks fine despite a high score, say so. Be concise; use markdown bullets."),
    ("human", "Changed hunks with model risk scores:\n\n{context}\n\nWrite the review."),
])


def build_chain(api_key: str, model: str):
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.2, max_tokens=1200)
    return PROMPT | llm | StrOutputParser()


async def summarize(chain, hunks: list[dict], top_k: int = 6) -> str:
    top = sorted(hunks, key=lambda h: h["risk"], reverse=True)[:top_k]
    context = "\n\n".join(
        f"### {h['filename']} (risk {h['risk']:.2f})\n```\n{h['snippet'][:1500]}\n```" for h in top
    )
    return await chain.ainvoke({"context": context})
