"""LangChain layer: turns model risk scores + the diff into a human-readable review.

Two provider modes (set LLM_PROVIDER):
  anthropic          Claude via langchain-anthropic (needs ANTHROPIC_API_KEY or LLM_API_KEY)
  openai_compatible  any OpenAI-style endpoint: Google Gemini, Groq, OpenRouter, a local Ollama, ...
                     (needs LLM_API_KEY, LLM_BASE_URL and LLM_MODEL)
"""
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


def build_chain(provider: str, api_key: str, model: str, base_url: str | None = None):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.2, max_tokens=1200,
                            timeout=45, max_retries=1)
    elif provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0.2,
                         max_tokens=1200, timeout=45, max_retries=1)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use 'anthropic' or 'openai_compatible')")
    return PROMPT | llm | StrOutputParser()


async def summarize(chain, hunks: list[dict], top_k: int = 5) -> str:
    # keep the prompt small: it saves free-tier quota and speeds up the response
    top = sorted(hunks, key=lambda h: h["risk"], reverse=True)[:top_k]
    context = "\n\n".join(
        f"### {h['filename']} (risk {h['risk']:.2f})\n```\n{h['snippet'][:1200]}\n```" for h in top
    )
    return await chain.ainvoke({"context": context})