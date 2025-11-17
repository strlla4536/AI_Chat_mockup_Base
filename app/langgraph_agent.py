from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.logger import get_logger
from app.tools import get_tool_map, get_tools_for_llm
from app.tools.web_search import web_search
from app.utils import (
    ROOT_DIR,
    _get_default_model,
    _get_openai_client,
    call_llm_stream,
    States,
    ToolState,
)

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
        """
        사용자의 요청이 데이터 분석(DA)에 해당하는지 분류합니다.
        현재 그래프는 내부 도구만 사용하므로 모든 분류 결과를 'general'
        경로로 보냅니다. 단, 분류 결과는 reasoning 이벤트로 전달합니다.
        """
        question = state["original_question"]
        prompt = (
            "당신은 사용자의 입력을 데이터 분석과 관련된 질문인지 판단하는 의사 결정 에이전트입니다.\n"
            "아래 세 가지 타입 중 가장 적합한 한 단어(대문자)를 출력하세요. 설명을 덧붙이지 마세요.\n"
            "1. DA: 주어진 Data Schema로 데이터 분석이 가능한 경우. "
            "사용자의 질문이 데이터 분석이나 데이터 기반 인사이트를 요구하며, "
            "스키마에 해당 질문을 처리할 수 있는 컬럼이나 정보가 존재한다고 판단되면 선택하세요.\n"
            "2. CANNOT_DA: 데이터 분석 요청은 있지만 스키마가 부족하거나 질문을 처리할 수 없다고 판단되는 경우.\n"
            "3. GENERAL: 위 두 가지에 해당하지 않는 모든 질문.\n"
            "반드시 위 세 단어 중 하나만 출력하세요."
        )

        decision = (await self._simple_llm_call(prompt, question)).strip().upper()
        if decision not in {"DA", "CANNOT_DA", "GENERAL"}:
            decision = "GENERAL"

        await self._emit(
            "reasoning",
            {
                "stage": "router",
                "message": f"판단 결과: {decision}",
            },
        )

        # 현재는 검색 경로를 사용하지 않으므로 항상 direct_answer 노드로 보낸다.
        return "general"

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

        final_message, updated_messages = await self._stream_with_tools(prompt_messages)
        # system 메시지는 상태에 저장할 필요가 없으므로 제외
        state["messages"] = [
            msg for msg in updated_messages if msg.get("role") != "system"
        ]
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
        """간단한 LLM 호출 (GenOS 또는 OpenAI 직접 사용)"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # call_llm_stream을 사용하여 GenOS 지원
        final_message = None
        async for res in call_llm_stream(
            messages=messages,
            model=self.model,
            temperature=temperature,
        ):
            # 최종 메시지만 수집
            if isinstance(res, dict) and "event" not in res:
                final_message = res
        
        if final_message:
            return final_message.get("content", "") or ""
        return ""

    async def _stream_with_tools(
        self,
        base_messages: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        call_llm_stream을 사용해 툴 호출을 처리하면서 최종 메시지를 생성합니다.
        """
        states = States()
        states.messages = list(base_messages)
        states.tools = await get_tools_for_llm()
        tool_map = await get_tool_map()

        buffered_tokens: list[str] = []

        while True:
            await self._emit("tool_state", states.tool_state.model_dump())

            final_message: Optional[Dict[str, Any]] = None
            async for res in call_llm_stream(
                messages=states.messages,
                model=self.model,
                tools=states.tools,
                temperature=0.2,
            ):
                if isinstance(res, dict) and res.get("event"):
                    event = res["event"]
                    data = res.get("data")
                    if event == "token":
                        # 최종 응답을 재구성하기 위해 토큰은 버퍼에 저장합니다.
                        buffered_tokens.append(data or "")
                    elif event == "tool_state":
                        # LLM이 전달한 상태 갱신 이벤트
                        await self._emit("tool_state", data or {})
                    else:
                        await self._emit(event, data)
                else:
                    final_message = res

            if final_message is None:
                raise RuntimeError("LLM으로부터 유효한 응답을 받지 못했습니다.")

            states.messages.append(final_message)
            tool_calls = final_message.get("tool_calls") or []
            content = final_message.get("content", "")

            if not tool_calls:
                # 더 이상 실행할 툴이 없으면 종료
                if content:
                    structured_message = await self._generate_structured_answer(
                        states=states,
                        interim_answer=content,
                    )
                    states.messages[-1] = structured_message
                    return structured_message, states.messages
                buffered_tokens.clear()
                # 콘텐츠가 비어 있으면 한 번 더 시도
                continue

            await self._emit(
                "reasoning",
                {
                    "stage": "tool",
                    "message": f"{len(tool_calls)}개의 도구 호출을 실행합니다.",
                },
            )

            for tool_call in tool_calls:
                function_payload = tool_call.get("function") or {}
                tool_name = function_payload.get("name")
                if not tool_name:
                    log.warning("tool call에 name이 없음", extra={"tool_call": tool_call})
                    continue

                if tool_name not in tool_map:
                    log.warning("등록되지 않은 도구 호출", extra={"tool_name": tool_name})
                    continue

                args_str = function_payload.get("arguments", "{}") or "{}"
                try:
                    tool_args = json.loads(args_str)
                except json.JSONDecodeError:
                    log.exception("툴 인자 JSON 파싱 실패", extra={"tool_name": tool_name})
                    tool_args = {}

                try:
                    tool_res = tool_map[tool_name](states, **tool_args)
                    if asyncio.iscoroutine(tool_res):
                        tool_res = await tool_res
                except Exception:
                    log.exception("도구 실행 실패", extra={"tool_name": tool_name})
                    tool_res = {"error": "Tool execution failed"}

                if isinstance(tool_res, str):
                    tool_content = tool_res
                else:
                    tool_content = json.dumps(tool_res, ensure_ascii=False)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_name,
                    "content": tool_content,
                }
                states.messages.append(tool_message)

                await self._emit(
                    "reasoning",
                    {
                        "stage": "tool",
                        "message": f"{tool_name} 도구 실행 결과를 반영했습니다.",
                    },
                )
                await self._emit("tool_state", states.tool_state.model_dump())

    def _extract_latest_user_question(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _build_tool_context(self, states: States) -> Dict[str, Any]:
        sql_results: List[Dict[str, Any]] = []
        for result_id, payload in states.tool_state.tool_results.items():
            sql_results.append(
                {
                    "result_id": result_id,
                    "query": payload.get("query"),
                    "columns": payload.get("columns"),
                    "rows": (payload.get("rows") or [])[:20],
                }
            )

        visualizations: List[Dict[str, Any]] = []
        for viz_id, iframe_html in states.tool_state.id_to_iframe.items():
            visualizations.append({"id": viz_id, "iframe_html": iframe_html})

        return {
            "sql_results": sql_results,
            "visualizations": visualizations,
        }

    async def _generate_structured_answer(
        self,
        *,
        states: States,
        interim_answer: str,
    ) -> Dict[str, Any]:
        question = self._extract_latest_user_question(states.messages)
        tool_context = self._build_tool_context(states)

        system_prompt = (
            "당신은 데이터 분석 보고서를 작성하는 전문가입니다. "
            "아래 마크다운 템플릿 구조를 정확히 따르되, 제목과 구분선(---), 표 형식은 변경하지 마십시오. "
            "각 섹션의 본문을 실제 내용으로 채워 넣고, 불필요한 설명을 덧붙이지 마세요.\n"
            "# 1. 사용자의 질문 분석 및 요약\n"
            "여기에 질문 요약을 작성하세요.\n"
            "---\n"
            "# 2. 실행된 SQL 쿼리\n"
            "```sql\n"
            "여기에 실제 실행한 SQL 쿼리를 작성하세요.\n"
            "```\n"
            "---\n"
            "# 3. SQL 쿼리 결과 (쿼리 실행 결과)\n"
            "| 지역단조직번호 | 건수 |\n"
            "|---------------|------|\n"
            "SQL 결과를 Markdown 표 행으로 작성하세요. 결과가 없으면 '데이터 없음'이라고 적으세요.\n"
            "> ✅ 결과는 최대 10건 요청이었으나, 반환된 데이터는 X건입니다.\n"
            "---\n"
            "# 4. 결과 시각화\n"
            "iframe HTML 또는 마크다운 이미지 링크를 제공하세요. 없으면 '시각화 리소스 없음'이라고 명시하세요.\n"
            "---\n"
            "# 5. 분석 인사이트\n"
            "- 최소 3개의 bullet로 핵심 인사이트를 작성하세요. 데이터가 없으면 '- 인사이트를 도출할 데이터가 부족합니다.'라고 적으세요.\n"
            "모든 수치는 입력 데이터에 기반해야 하며, 추측이나 가정은 피하십시오."
        )

        context_payload = {
            "user_question": question,
            "interim_answer": interim_answer,
            "tool_context": tool_context,
        }

        user_prompt = (
            "다음 JSON 컨텍스트를 참고하여 위 템플릿을 완벽히 채워 주세요. "
            "SQL 쿼리는 가장 관련 있는 항목을 선택하고, 표는 해당 결과의 행을 그대로 사용하세요. "
            "행 수를 정확한 숫자로 기입하고, 시각화 리소스가 있다면 링크나 iframe을 그대로 삽입하십시오.\n"
            f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}"
        )

        await self._emit(
            "reasoning",
            {
                "stage": "final",
                "message": "템플릿 형식으로 최종 답변을 구성합니다.",
            },
        )

        final_message: Optional[Dict[str, Any]] = None
        async for res in call_llm_stream(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model,
            tools=[],
            temperature=0,
        ):
            if isinstance(res, dict) and res.get("event"):
                event = res["event"]
                data = res.get("data")
                if event == "token":
                    await self._emit("token", data or "")
                else:
                    await self._emit(event, data)
            else:
                final_message = res

        if final_message is None:
            final_message = {"role": "assistant", "content": ""}
        return final_message

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
