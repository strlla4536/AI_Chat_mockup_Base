from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.logger import get_logger
from app.utils import (
    ROOT_DIR,
    _get_default_model,
    _get_openai_client,
    call_llm_stream,
    States,
    ToolState,
)
from app.tools.web_search import web_search

log = get_logger(__name__)


class GraphState(TypedDict):
    """State shared across the LangGraph workflow."""

    messages: List[Dict[str, str]]
    original_question: str
    search_iterations: int
    search_results_summary: List[str]
    current_search_query: str | None
    final_answer: str | None


Emitter = Callable[[str, Any], Awaitable[None]]


class LangGraphSearchAgent:
    """LangGraph-powered conversational search agent with streaming reasoning events."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        emitter: Optional[Emitter] = None,
    ) -> None:
        self.model = model or _get_default_model()
        self._client = _get_openai_client()
        self._graph = self._build_graph()
        self._emitter: Optional[Emitter] = emitter

    def set_emitter(self, emitter: Optional[Emitter]) -> None:
        self._emitter = emitter

    async def _emit(self, event: str, data: Any) -> None:
        if self._emitter is None:
            return
        try:
            await self._emitter(event, data)
        except Exception:
            log.exception("Failed to emit LangGraph event", extra={"event": event})

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("router", self._router_node)
        graph.add_node("direct_answer", self._direct_answer_node)
        graph.add_node("query_refinement", self._query_refinement_node)
        graph.add_node("search_and_summarize", self._search_and_summarize_node)
        graph.add_node("final_answer", self._final_answer_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "general": "direct_answer",
                "search": "query_refinement",
            },
        )

        graph.add_edge("query_refinement", "search_and_summarize")

        graph.add_conditional_edges(
            "search_and_summarize",
            self._search_loop_condition,
            {
                "continue": "query_refinement",
                "done": "final_answer",
            },
        )

        graph.add_conditional_edges(
            "direct_answer",
            lambda _: END,
        )

        graph.add_conditional_edges(
            "final_answer",
            lambda _: END,
        )

        return graph.compile()

    async def _router_node(self, state: GraphState) -> GraphState:
        # Router node does not mutate the state – decision happens in _route_decision.
        await self._emit(
            "reasoning",
            {
                "stage": "router",
                "message": "요청을 분석하여 검색 필요 여부를 판단합니다.",
            },
        )
        return state

    async def _route_decision(self, state: GraphState) -> str:
        question = state["original_question"]
        prompt = (
            "당신은 사용자의 질문을 분석하여 웹 검색이 필요한지 판단합니다.\n"
            "경미한 인사나 단순 사실 복습이라면 'general'을,\n"
            "최신 정보나 외부 지식이 필요하면 'search'를 출력하세요.\n"
            "출력은 general 또는 search 중 하나만 가능하며 다른 단어는 포함하지 마세요."
        )

        decision = await self._simple_llm_call(prompt, question)
        normalized = "search" if "search" in decision.lower() else "general"
        await self._emit(
            "reasoning",
            {
                "stage": "router",
                "message": f"판단 결과: { '검색 필요' if normalized == 'search' else '일반 대화' }",
            },
        )
        return normalized

    async def _query_refinement_node(self, state: GraphState) -> GraphState:
        iteration = state["search_iterations"] + 1
        summaries_text = "\n".join(state["search_results_summary"]) or "없음"

        system_prompt = (
            "당신은 검색 질의 최적화 도우미입니다. 사용자의 질문과 지금까지의 검색 요약을 참고하여\n"
            "다음 검색을 위한 가장 유용한 단일 검색어를 만드세요.\n"
            "가능한 한 구체적으로 작성하며 한국어 사용자에게 적합한 언어를 선택하세요.\n"
            "출력은 검색어 문장만 포함해야 합니다."
        )

        user_prompt = (
            f"사용자 질문: {state['original_question']}\n"
            f"이전 검색 요약: {summaries_text}\n"
            "다음 검색어를 제안하세요."
        )

        refined_query = await self._simple_llm_call(system_prompt, user_prompt, temperature=0.2)
        refined_query = refined_query.strip()

        await self._emit(
            "reasoning",
            {
                "stage": "query_refinement",
                "iteration": iteration,
                "message": f"검색어 생성 중: {refined_query}",
                "query": refined_query,
            },
        )

        state["current_search_query"] = refined_query
        return state

    async def _search_and_summarize_node(self, state: GraphState) -> GraphState:
        iteration = state["search_iterations"] + 1
        query = state.get("current_search_query") or state["original_question"]

        await self._emit(
            "reasoning",
            {
                "stage": "search",
                "iteration": iteration,
                "message": f"웹 검색 실행: {query}",
            },
        )

        search_state = States()
        search_state.tool_state = ToolState()
        search_results = await web_search(
            search_state,
            search_query=[{"q": query, "recency": None, "domains": None}],
            response_length="long",
        )

        if isinstance(search_results, str):
            summary = f"검색 오류: {search_results}"
        elif not search_results:
            summary = "검색 결과가 없습니다."
        else:
            top_snippets = "\n".join(
                f"- 제목: {item.get('title','')}\n  요약: {item.get('snippet','')}\n  URL: {item.get('url','')}"
                for item in search_results[:5]
            )

            system_prompt = (
                "당신은 정보를 요약하는 전문가입니다. 아래 검색 결과를 참고하여 핵심 정보를 3-5문장으로 요약하세요.\n"
                "출처가 있다면 괄호로 표기하고, 중요 사실을 위주로 작성하세요."
            )
            user_prompt = (
                f"사용자 질문: {state['original_question']}\n"
                f"검색 결과:\n{top_snippets}"
            )
            summary = await self._simple_llm_call(system_prompt, user_prompt, temperature=0.4)

        state["search_iterations"] = iteration
        state["search_results_summary"].append(summary.strip())

        await self._emit(
            "reasoning",
            {
                "stage": "summary",
                "iteration": iteration,
                "message": f"웹 검색 결과 요약: {summary.strip()}",
                "summary": summary.strip(),
            },
        )

        return state

    async def _search_loop_condition(self, state: GraphState) -> str:
        return "continue" if state["search_iterations"] < 2 else "done"

    async def _direct_answer_node(self, state: GraphState) -> GraphState:
        last_messages = state["messages"]

        prompt_messages = [
            {"role": "system", "content": self._system_prompt()},
            *last_messages,
        ]

        await self._emit(
            "reasoning",
            {
                "stage": "final",
                "message": "검색 없이 바로 답변을 생성합니다.",
            },
        )

        final_message = await self._stream_answer(prompt_messages)
        state["messages"].append(final_message)
        state["final_answer"] = final_message.get("content", "")
        return state

    async def _final_answer_node(self, state: GraphState) -> GraphState:
        context = "\n\n".join(state["search_results_summary"])
        user_question = state["original_question"]

        messages = [
            {
                "role": "system",
                "content": (
                    f"당신은 Perplexity 스타일의 AI 어시스턴트입니다.\n"
                    f"다음 검색 요약을 참고하여 질문에 답변하세요.\n"
                    f"필요 시 출처를 간단히 언급하되, 말투는 친절하고 단정하게 유지하세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"질문: {user_question}\n"
                    f"누적 검색 요약:\n{context}"
                ),
            },
        ]

        await self._emit(
            "reasoning",
            {
                "stage": "final",
                "message": "검색 결과를 종합하여 최종 답변을 생성합니다.",
            },
        )

        final_message = await self._stream_answer(messages)
        state["messages"].append(final_message)
        state["final_answer"] = final_message.get("content", "")
        return state

    async def _stream_answer(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        final_message: Optional[Dict[str, Any]] = None
        async for chunk in call_llm_stream(messages=messages, model=self.model, temperature=0.2):
            if isinstance(chunk, dict) and chunk.get("event") == "token":
                await self._emit("token", chunk.get("data", ""))
            else:
                final_message = chunk
        if final_message is None:
            final_message = {"role": "assistant", "content": ""}
        return final_message

    async def _simple_llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> str:
        client = self._client
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def run(
        self,
        *,
        question: str,
        history: List[Dict[str, str]] | None = None,
    ) -> GraphState:
        initial_state: GraphState = {
            "messages": [
                *(history or []),
                {"role": "user", "content": question},
            ],
            "original_question": question,
            "search_iterations": 0,
            "search_results_summary": [],
            "current_search_query": None,
            "final_answer": None,
        }

        result_state: GraphState = await self._graph.ainvoke(initial_state)
        return result_state

    def _system_prompt(self) -> str:
        try:
            return (
                ROOT_DIR
                / "prompts"
                / "system.txt"
            ).read_text(encoding="utf-8")
        except Exception:
            return "You are a helpful AI assistant."
