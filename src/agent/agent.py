"""
Agent layer (Phase 9 / assignment sections 1 "Agent framework" and 6
"Agent behaviour").

Uses LangGraph's prebuilt ReAct agent loop with a Gemini chat model and a
single tool: search_similar_sarees. The LLM's only job is to decide
*when* to call the tool and how to phrase its response -- it never
computes similarity itself (see src/tools/visual_search.py docstring).

LLM provider is configurable via environment variables (never hardcoded):
    LLM_API_KEY   - API key for the provider
    LLM_MODEL     - model name, e.g. "gemini-2.5-flash"

Gemini has a usable free tier, hence the default below, but swapping
providers only requires changing how `_build_chat_model` constructs the
model -- the rest of the agent is provider-agnostic LangChain code.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.tools.visual_search import search_similar_sarees

SYSTEM_PROMPT = """You are a helpful shopping assistant for a saree catalog. \
You can find visually similar sarees when the user provides or references an image.

Behavior rules:
- If the user greets you, makes small talk, or asks what you can do: respond \
naturally in plain text. Do NOT call any tool.
- If the user asks what you can do: explain that you can find visually similar \
sarees from an uploaded image or an image URL, and that you'll return the closest \
matches with similarity scores.
- Only call the search_similar_sarees tool when the user has actually supplied an \
image (an uploaded file path will be given to you in the message, or the user gave \
an image URL) AND is asking to find similar/matching sarees.
- If the user asks for a specific number of results ("show me 3", "top 10"), pass \
that as top_k.
- If the user asks to refine by an attribute not supported by this system (e.g. \
"similar but in red", "but cheaper", text-only refinement of image search results): \
you do NOT currently support text-guided refinement of visual search. Say so \
clearly and honestly instead of pretending to filter by that attribute. You may \
still run the plain visual search and let the user judge the color themselves.
- Never invent similarity scores or results yourself -- only report what the tool \
returns.
- If the tool returns status "error", explain the problem to the user in plain, \
friendly language (e.g. bad URL, unreadable image) and suggest what to try instead.
- Keep responses concise. The UI displays the matched images and scores separately, \
so you don't need to describe every result in detail -- a short summary is enough.
"""


def _build_chat_model():
    api_key = os.environ.get("LLM_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "gemini-3.6-flash")

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Copy .env.example to .env and set your Gemini "
            "API key (see https://ai.google.dev/ for a free-tier key)."
        )

    # Imported lazily so the rest of the app can be exercised/tested without
    # the google genai SDK being a hard import-time dependency of every module.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)


def build_agent():
    """Build and return a compiled LangGraph ReAct agent with the search tool bound."""
    model = _build_chat_model()
    return create_react_agent(model, tools=[search_similar_sarees], prompt=SYSTEM_PROMPT)


def run_turn(agent, history: list[BaseMessage], user_text: str, image_path: str | None) -> tuple[str, list[BaseMessage]]:
    """Run one conversational turn.

    `image_path` is a local path to a just-uploaded/query image, if any, for
    this turn. When present, it's appended to the human message content so
    the LLM has a concrete `image_source` value to pass to the tool.
    """
    content = user_text
    if image_path:
        content = f"{user_text}\n\n[Attached image available at local path: {image_path}]"

    messages = history + [HumanMessage(content=content)]
    result = agent.invoke({"messages": messages})
    new_messages: list[BaseMessage] = result["messages"]

    # Extract only the human-readable text from the assistant message.
    reply = ""
    
    for m in reversed(new_messages):
        if isinstance(m, AIMessage) and m.content:
            if isinstance(m.content, str):
                reply = m.content
            elif isinstance(m.content, list):
                text_parts = []
    
                for block in m.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            text_parts.append(text)
    
                reply = "\n".join(text_parts)
    
            break

    return reply, new_messages
