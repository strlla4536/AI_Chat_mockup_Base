from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.logger import get_logger
from app.tools import get_tool_map, get_tools_for_llm
from app.utils import (
    ROOT_DIR,
    _get_default_model,
    call_llm_stream,
    States,
    DataAnalysisState,
)
from app.tools import db as db_tools

log = get_logger(__name__)


class GraphState(TypedDict):
    """State shared across the LangGraph workflow."""

    messages: List[Dict[str, str]]
    original_question: str
    final_answer: str | None
    shared_state: Dict[str, Any]
    da_state: Dict[str, Any]
    da_runtime: States


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
        graph.add_node("da_prepare_context", self._da_prepare_context_node)
        graph.add_node("da_prepare_schema", self._da_prepare_schema_node)
        graph.add_node("da_generate_sql", self._da_generate_sql_node)
        graph.add_node("da_execute_sql", self._da_execute_sql_node)
        graph.add_node("da_chart_type_decision", self._da_chart_type_decision_node)
        graph.add_node("da_prepare_chart_format", self._da_prepare_chart_format_node)
        graph.add_node("da_generate_chart_data", self._da_generate_chart_data_node)
        graph.add_node("da_validate_chart_data", self._da_validate_chart_data_node)
        graph.add_node("da_generate_chart", self._da_generate_chart_node)
        graph.add_node("da_report_with_chart", self._da_report_with_chart_node)
        graph.add_node("da_cannot_da", self._da_cannot_da_node)
        graph.add_node("direct_answer", self._direct_answer_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "da": "da_prepare_context",
                "cannot_da": "da_cannot_da",
                "general": "direct_answer",
            },
        )

        graph.add_edge("da_prepare_context", "da_prepare_schema")
        graph.add_edge("da_prepare_schema", "da_generate_sql")
        graph.add_edge("da_generate_sql", "da_execute_sql")
        graph.add_edge("da_execute_sql", "da_chart_type_decision")
        graph.add_edge("da_chart_type_decision", "da_prepare_chart_format")
        graph.add_edge("da_prepare_chart_format", "da_generate_chart_data")
        graph.add_edge("da_generate_chart_data", "da_validate_chart_data")
        graph.add_edge("da_validate_chart_data", "da_generate_chart")
        graph.add_edge("da_generate_chart", "da_report_with_chart")
        graph.add_edge("da_report_with_chart", END)

        graph.add_edge("da_cannot_da", END)

        graph.add_conditional_edges(
            "direct_answer",
            lambda _: END,
        )
        return graph.compile()

    async def _router_node(self, state: GraphState) -> GraphState:
        # Router node does not mutate the state – decision happens in _route_decision.
        await self._emit(
            "reasoning",
            {
                "stage": "router",
                "message": "요청을 분석해 적절한 처리 경로를 결정합니다.",
            },
        )
        return state

    async def _route_decision(self, state: GraphState) -> str:
        """
        사용자의 요청이 데이터 분석(DA)에 해당하는지 분류합니다.
        """
        question = state["original_question"]
        prompt = (
            "당신은 사용자의 입력을 데이터 분석과 관련된 질문인지 판단하는 의사 결정 에이전트 입니다.\n\n"
            "User Input과 주어진 Data Schema를 기반으로, 아래 세 가지 타입 중 가장 적합한 한 단어를 출력하세요.\n"
            "\n• DA : 주어진 Data Schema로 데이터 분석이 가능한 경우.\n"
            "  ◦ 사용자의 질문이 데이터 분석이나 데이터 기반 인사이트를 요구하며, 데이터 스키마에 해당 질문을 처리할 수 있는 컬럼이나 정보가 존재하는 경우.\n"
            "    ▪ 데이터셋, 테이블, 파일(csv, json, excel 등), 데이터베이스 질의(SQL, NoSQL 등)에 대한 언급이 있는 경우\n"
            "    ▪ 통계 또는 분석 관련 용어가 포함된 경우: 평균, 중간값, 합계, 개수, 상관관계, 분포, 추이, 변화량 등.\n"
            "    ▪ 데이터 관련 질문 : 예) '어떤 데이터 가지고 있어?', '데이터 컬럼 뭐 있어?', '데이터셋 목록', '테이블 구조 보여줘' 등.\n"
            "    ▪ 시각화 요청이 포함된 경우: 차트, 그래프, 시계열 분석, 꺾은선 그래프, 막대그래프 등.\n"
            "    ▪ 구체적인 수치 기반 질의가 포함된 경우: 최대, 최소, 가장 높은/낮은, 상위 N개, 순위, 비율, 증감 등\n"
            "    ▪ 머신러닝, 모델 평가, 예측 정확도 등과 관련된 질문인 경우\n"
            "    ▪ 데이터 가공, 전처리, 필터링, 그룹화, 집계 등 작업 요청이 포함된 경우\n"
            "    ▪ 시간 또는 지역 기반 지표에 대한 요청: '2023년 월별 매출', '서울에서 가장 많이 팔린 상품' 등.\n"
            "    ▪ 파이썬, R 등 데이터 분석 도구(Pandas, NumPy, etc.) 언급이 있는 경우\n"
            "\n• CANNOT_DA : 데이터 분석 요청은 있으나 스키마가 부족한 경우.\n"
            "  ◦ 사용자의 질문이 데이터 분석이나 인사이트를 요구하지만, 데이터 스키마에 필요한 컬럼, 테이블, 정보 구조가 없어 답변할 수 없는 경우.\n"
            "    ▪ 질문에서 언급된 특정 컬럼이나 테이블이 스키마에 존재하지 않는 경우\n"
            "    ▪ 예시 행이 부족하거나, 관련된 값이 아예 없는 경우\n"
            "    ▪ 필요한 데이터 타입이나 형식(예: 시간 정보, 숫자형 지표 등)이 빠진 경우\n"
            "    ▪ 분석 요청은 명확하지만, 현재 제공된 스키마만으로는 계산, 추정, 시각화가 불가능한 경우\n"
            "\n• GENERAL : 위 두 가지에 해당하지 않는 모든 질문\n"
            "  ◦ 단순한 대화나 감정 표현 또는 데이터 분석과 무관한 일반적인 질문인 경우.\n"
            "    ▪ 단순한 인사말이나 잡담: 예) '안녕!', '잘 지내?' 등.\n"
            "    ▪ 감정 표현, 소감, 철학적/추상적 질문: 예) '데이터란 무엇인가요?', '사람은 왜 데이터를 좋아할까?' 등.\n"
            "    ▪ 여행, 요리, 일상 팁 등 데이터와 무관한 실용적 조언\n"
            "    ▪ 검색이 필요한 정보 : 예) '미국 대통령은 누구인가요?', '미국 증시가 우리 나라에 미치는 영향은?' 등.\n"
            "    ▪ 데이터 구조나 분석 없이 단순한 툴/서비스 사용법 관련 질문: 예) '로그인 어떻게 해?', '이 화면에서 어떻게 검색해?' 등.\n"
            "\n위 기준을 참고하여 DA, CANNOT_DA, GENERAL 중 하나만 대문자로 출력하세요."
        )

        decision = (await self._simple_llm_call(prompt, question)).strip().upper()
        if decision not in {"DA", "CANNOT_DA", "GENERAL"}:
            decision = "GENERAL"

        da_state = self._ensure_da_state(state)
        da_state["route_type"] = decision

        await self._emit(
            "reasoning",
            {
                "stage": "router",
                "message": f"판단 결과: {decision}",
            },
        )

        if decision == "DA":
            return "da"
        if decision == "CANNOT_DA":
            return "cannot_da"
        return "general"

    async def _da_prepare_context_node(self, state: GraphState) -> GraphState:
        shared = self._ensure_shared_state(state)
        da_state = self._ensure_da_state(state)
        self._get_da_runtime(state)

        await self._emit(
            "reasoning",
            {
                "stage": "da_prepare_context",
                "message": "데이터베이스 연결 상태를 점검합니다.",
            },
        )

        if shared.get("db_connection_checked") and not shared.get("db_connection_error"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_prepare_context",
                    "message": "이전에 확인된 DB 연결 정보를 재사용합니다.",
                },
            )
            return state

        try:
            await db_tools._ensure_pool()
            shared["db_connection_checked"] = True
            shared["db_connection_error"] = None
            da_state["connection_status"] = "ok"
            await self._emit(
                "reasoning",
                {
                    "stage": "da_prepare_context",
                    "message": "DB 연결에 성공했습니다.",
                },
            )
        except Exception as exc:
            error_message = f"DB 연결 실패: {exc}"
            shared["db_connection_checked"] = False
            shared["db_connection_error"] = error_message
            da_state["connection_status"] = "error"
            da_state["sql_error"] = error_message
            da_state["abort"] = True
            da_state["abort_reason"] = "db_connection_failed"
            await self._emit(
                "reasoning",
                {
                    "stage": "da_prepare_context",
                    "message": error_message,
                },
            )
        return state

    async def _da_prepare_schema_node(self, state: GraphState) -> GraphState:
        shared = self._ensure_shared_state(state)
        da_state = self._ensure_da_state(state)
        runtime = self._get_da_runtime(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_prepare_schema",
                    "message": "이전 단계 오류로 스키마 준비를 건너뜁니다.",
                },
            )
            return state

        await self._emit(
            "reasoning",
            {
                "stage": "da_prepare_schema",
                "message": "분석에 필요한 데이터 스키마를 조회합니다.",
            },
        )

        try:
            schema_res = await db_tools.describe_schema(runtime, sample_rows=5)
        except Exception as exc:
            error_message = f"스키마 조회 실패: {exc}"
            da_state["abort"] = True
            da_state["abort_reason"] = "schema_fetch_failed"
            da_state["sql_error"] = error_message
            await self._emit(
                "reasoning",
                {
                    "stage": "da_prepare_schema",
                    "message": error_message,
                },
            )
            return state

        if not schema_res.get("success"):
            error_message = schema_res.get("error") or "스키마 정보를 가져오지 못했습니다."
            da_state["abort"] = True
            da_state["abort_reason"] = "schema_fetch_failed"
            da_state["sql_error"] = error_message
            await self._emit(
                "reasoning",
                {
                    "stage": "da_prepare_schema",
                    "message": error_message,
                },
            )
            return state

        tables: Dict[str, Any] = schema_res.get("tables", {})
        transformed: List[Dict[str, Any]] = []
        for table_name, table_info in tables.items():
            columns_data = []
            sample_rows = table_info.get("sample_rows") or []
            for column in table_info.get("columns", []):
                col_entry: Dict[str, Any] = {
                    "name": column.get("name"),
                    "type": column.get("type"),
                }
                if column.get("is_primary"):
                    col_entry["primary"] = True
                elif column.get("is_unique"):
                    col_entry["unique"] = True

                if sample_rows:
                    values = []
                    for row in sample_rows:
                        if isinstance(row, dict) and column.get("name") in row:
                            value = row.get(column.get("name"))
                            if value is not None and value not in values:
                                values.append(value)
                        if len(values) >= 25:
                            break
                    if values:
                        col_entry["unique_values"] = values[:25]

                columns_data.append(col_entry)

            transformed.append(
                {
                    "table_name": table_name,
                    "columns": columns_data,
                    "sample_rows": sample_rows[:5],
                }
            )

        da_state["rdb_schema"] = transformed
        da_state["database_name"] = schema_res.get("database")
        shared["db_connection_checked"] = True
        shared["db_connection_error"] = None

        await self._emit(
            "reasoning",
            {
                "stage": "da_prepare_schema",
                "message": f"총 {len(transformed)}개의 테이블 스키마를 확보했습니다.",
            },
        )
        return state

    async def _da_generate_sql_node(self, state: GraphState) -> GraphState:
        shared = self._ensure_shared_state(state)
        da_state = self._ensure_da_state(state)
        runtime = self._get_da_runtime(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_generate_sql",
                    "message": "이전 단계 오류로 SQL 생성을 건너뜁니다.",
                },
            )
            return state

        schema = da_state.get("rdb_schema")
        if not schema:
            da_state["abort"] = True
            da_state["abort_reason"] = "missing_schema"
            await self._emit(
                "reasoning",
                {
                    "stage": "da_generate_sql",
                    "message": "스키마 정보가 없어 SQL을 생성할 수 없습니다.",
                },
            )
            return state

        await self._emit(
            "reasoning",
            {
                "stage": "da_generate_sql",
                "message": "데이터 분석용 SQL을 설계합니다.",
            },
        )

        system_prompt = (
            "당신은 MySQL 8.x 환경에서 내부 보험 데이터베이스를 다루는 시니어 데이터 분석가입니다. "
            "사용자 요청, 전달된 스키마, 선행 쿼리 이력을 참조하여 하나의 최적화된 SELECT 쿼리를 작성하세요.\n\n"
            "[도메인 규칙] 반드시 다음 지침을 준수합니다.\n"
            "1. 고유명사/상품명 패턴 검색: 사용자 입력에 명시된 상품명 등이 존재하고 대화 기록에 추가 정보가 없다면, LIKE 혹은 REGEXP 등을 활용해 다중 패턴 검색을 수행합니다. 적용 컬럼: PRDT_NM, PRCD, SALE_PRCD (NCONT001).\n"
            "2. 정확한 값 필터링: 다음 유형 컬럼은 패턴 매칭 없이 정확히 일치 비교합니다. (예: CONT_NO, CLCT_CNSLT_NO, ORG_NO, FIN_YM, STND_YM, EXTC_YMD, SMTOT_PRM, MPDB_PRM, FRTM_PRM, OFR_STAT_CD, CONT_STAT_CD 등).\n"
            "3. 조인 키: CRO001.CNSLT_NO = NCONT001.CLCT_CNSLT_NO, OBJ001.ORG = CRO001.ORG_NO, OBJ001.NF_FIN_YM = NCONT001.FIN_YM, COV001.CONT_CD = NCONT001.CONT_NO. 해당 관계를 기준으로 명시적 JOIN을 구성합니다.\n\n"
            "[작성 규칙]\n"
            "- SELECT 전용 쿼리를 작성하고 필요 컬럼만 명시합니다 (SELECT * 금지).\n"
            "- 모든 테이블/컬럼은 존재하는 스키마만 사용하며, 예약어 충돌시 백틱(`)을 사용합니다.\n"
            "- WHERE, GROUP BY, ORDER BY, LIMIT, CASE WHEN 등을 통해 사용자의 요구를 하나의 쿼리로 충족합니다.\n"
            "- 여러 요구 조건이 있을 경우 조건부 집계(CASE WHEN)나 적절한 JOIN을 활용하고, JOIN 가능한 경우 UNION ALL 대신 JOIN을 선호합니다.\n"
            "- UNION ALL이 꼭 필요하다면 각 SELECT를 괄호로 감싸고 동일 스키마를 유지합니다.\n"
            "- LIMIT/OFFSET은 숫자 리터럴만 사용하며, 모든 괄호와 따옴표를 정확히 닫습니다.\n"
            "- GROUP BY 시 비집계 컬럼은 모두 명시하고, 집계는 함수로 감쌉니다.\n"
            "- 이전 쿼리가 실패한 원인이 제공되면 반드시 수정합니다.\n\n"
            "[응답 형식] JSON 객체 하나만 반환하며 키는 'sql_query' 입니다. 값에는 세미콜론으로 끝나는 MySQL 쿼리를 넣습니다 (예: {\"sql_query\": \"SELECT ...;\"}). 다른 키나 추가 텍스트는 포함하지 마세요."
        )

        user_payload = {
            "TABLE_SCHEMA": da_state.get("database_name"),
            "DB_SCHEMA": schema,
            "USER_REQUEST": state.get("original_question"),
            "SQL_HISTORY": da_state.get("sql_history", []),
            "CURRENT_DATE": shared.get("current_date"),
            "FAILED_SQL": da_state.get("sql_error"),
        }

        try:
            result = await self._call_llm_for_json(
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=0.1,
            )
        except Exception as exc:
            error_message = f"SQL 생성 실패: {exc}"
            da_state["abort"] = True
            da_state["abort_reason"] = "sql_generation_failed"
            da_state["sql_error"] = error_message
            await self._emit(
                "reasoning",
                {
                    "stage": "da_generate_sql",
                    "message": error_message,
                },
            )
            return state

        sql_query = (
            result.get("sql_query")
            or result.get("sql")
            or ""
        ).strip()
        if not sql_query:
            da_state["abort"] = True
            da_state["abort_reason"] = "sql_generation_failed"
            da_state["sql_error"] = "모델이 SQL을 반환하지 않았습니다."
            await self._emit(
                "reasoning",
                {
                    "stage": "da_generate_sql",
                    "message": "모델이 SQL을 생성하지 못했습니다.",
                },
            )
            return state

        if not sql_query.endswith(";"):
            sql_query += ";"

        da_state["sql_query"] = sql_query
        reasoning = result.get("reasoning") or result.get("analysis") or ""
        da_state["sql_reasoning"] = reasoning
        da_state.setdefault("sql_history", []).append(sql_query)

        history_entry = {
            "query": sql_query,
            "reasoning": reasoning,
            "created_at": datetime.utcnow().isoformat(),
        }
        runtime.tool_state.tool_results[f"sql_plan_{len(runtime.tool_state.tool_results)}"] = history_entry
        await self._emit("tool_state", runtime.tool_state.model_dump())
        await self._emit(
            "reasoning",
            {
                "stage": "da_generate_sql",
                "message": "SQL 생성 완료.",
            },
        )
        return state

    async def _da_execute_sql_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)
        runtime = self._get_da_runtime(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_execute_sql",
                    "message": "이전 단계 오류로 SQL 실행을 건너뜁니다.",
                },
            )
            return state

        sql_query = da_state.get("sql_query")
        if not sql_query:
            da_state["abort"] = True
            da_state["abort_reason"] = "missing_sql"
            da_state["sql_error"] = "실행할 SQL이 존재하지 않습니다."
            await self._emit(
                "reasoning",
                {
                    "stage": "da_execute_sql",
                    "message": "실행할 SQL이 없어 종료합니다.",
                },
            )
            return state

        await self._emit(
            "reasoning",
            {
                "stage": "da_execute_sql",
                "message": "생성된 SQL을 실행하여 결과를 확보합니다.",
            },
        )

        try:
            result_payload = await db_tools.execute_sql(runtime, query=sql_query, limit=500)
        except Exception as exc:
            error_message = f"SQL 실행 실패: {exc}"
            da_state["abort"] = True
            da_state["abort_reason"] = "sql_execution_failed"
            da_state["sql_error"] = error_message
            await self._emit(
                "reasoning",
                {
                    "stage": "da_execute_sql",
                    "message": error_message,
                },
            )
            return state

        if not result_payload.get("success"):
            error_message = result_payload.get("error") or "SQL 실행에 실패했습니다."
            da_state["abort"] = True
            da_state["abort_reason"] = "sql_execution_failed"
            da_state["sql_error"] = error_message
            await self._emit(
                "reasoning",
                {
                    "stage": "da_execute_sql",
                    "message": error_message,
                },
            )
            return state

        da_state["sql_error"] = None
        da_state["sql_result"] = result_payload
        da_state["latest_result_id"] = result_payload.get("result_id")

        await self._emit(
            "reasoning",
            {
                "stage": "da_execute_sql",
                "message": f"SQL 실행 완료 (행 {result_payload.get('row_count', 0)}건).",
            },
        )

        await self._emit("tool_state", runtime.tool_state.model_dump())
        return state

    async def _da_chart_type_decision_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)
        runtime = self._get_da_runtime(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_type",
                    "message": "이전 단계 오류로 차트 유형 결정을 건너뜁니다.",
                },
            )
            return state

        sql_result = da_state.get("sql_result")
        summary = self._prepare_sql_summary(runtime, sql_result or {}) if sql_result else {}

        await self._emit(
            "reasoning",
            {
                "stage": "da_chart_type",
                "message": "적합한 차트 유형을 판단합니다.",
            },
        )

        system_prompt = (
            "당신은 사용자 요청과 SQL 결과를 기반으로 single, mixed, dual_axis, none 중 하나의 차트 유형을 결정하는 전문가입니다. "
            "출력은 반드시 JSON 객체 한 줄이어야 하며, 예시는 {\"chart_type\": \"single\"} 형식입니다."
        )
        user_payload = {
            "question": state.get("original_question"),
            "schema": da_state.get("rdb_schema"),
            "sql_result": summary,
        }

        try:
            result = await self._call_llm_for_json(
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=0.0,
            )
            chart_type = (result.get("chart_type") or "none").lower()
        except Exception:
            chart_type = "none"

        allowed = {"single", "mixed", "dual_axis", "none"}
        if chart_type not in allowed:
            chart_type = "none"

        da_state["chart_type"] = chart_type
        da_state["data_format"] = self._get_chart_format_template(chart_type)

        await self._emit(
            "reasoning",
            {
                "stage": "da_chart_type",
                "message": f"차트 유형: {chart_type}",
            },
        )
        return state

    async def _da_prepare_chart_format_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_format",
                    "message": "이전 단계 오류로 차트 포맷 생성을 건너뜁니다.",
                },
            )
            return state

        chart_type = da_state.get("chart_type") or "none"
        da_state["data_format"] = self._get_chart_format_template(chart_type)
        da_state["chart_feedback"] = None

        await self._emit(
            "reasoning",
            {
                "stage": "da_chart_format",
                "message": f"차트 포맷 템플릿을 준비했습니다. (type={chart_type})",
            },
        )
        return state

    async def _da_generate_chart_data_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)
        runtime = self._get_da_runtime(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_data",
                    "message": "이전 단계 오류로 차트 데이터 생성을 건너뜁니다.",
                },
            )
            return state

        chart_type = da_state.get("chart_type") or "none"
        if chart_type == "none":
            da_state["chart_data"] = None
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_data",
                    "message": "차트가 필요하지 않아 데이터를 생성하지 않습니다.",
                },
            )
            return state

        sql_result = da_state.get("sql_result")
        summary = self._prepare_sql_summary(runtime, sql_result or {}) if sql_result else {}
        data_format = da_state.get("data_format")
        feedback = da_state.get("chart_feedback")

        await self._emit(
            "reasoning",
            {
                "stage": "da_chart_data",
                "message": "시각화용 데이터셋을 구성합니다.",
            },
        )

        system_prompt = (
            "당신은 SQL 결과를 기반으로 차트 데이터를 생성하는 전문가입니다. "
            "출력은 반드시 JSON 이어야 하며 추가적인 설명이나 텍스트를 금지합니다. "
            "필요한 모든 필드를 data_format 예시에 맞춰 채우고, 숫자는 최대 소수 둘째 자리까지 반올림하세요."
        )
        user_payload = {
            "question": state.get("original_question"),
            "data_format": data_format,
            "sql_result": summary,
            "chart_type": chart_type,
            "feedback": feedback,
        }

        try:
            chart_data = await self._call_llm_for_json(
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=0.3,
            )
        except Exception as exc:
            da_state["chart_data"] = None
            da_state["chart_feedback"] = f"차트 데이터 생성 실패: {exc}" 
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_data",
                    "message": f"차트 데이터 생성 실패: {exc}",
                },
            )
            return state

        da_state["chart_data"] = chart_data
        await self._emit(
            "reasoning",
            {
                "stage": "da_chart_data",
                "message": "차트 데이터 생성 완료.",
            },
        )
        return state

    async def _da_validate_chart_data_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_validate",
                    "message": "이전 단계 오류로 차트 데이터 검증을 건너뜁니다.",
                },
            )
            return state

        chart_type = da_state.get("chart_type") or "none"
        chart_data = da_state.get("chart_data")

        if chart_type == "none" or chart_data is None:
            da_state["chart_validation_result"] = {"success": True, "skipped": True}
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_validate",
                    "message": "차트가 필요하지 않아 검증을 생략합니다.",
                },
            )
            return state

        errors: List[str] = []
        if not isinstance(chart_data, dict):
            errors.append("차트 데이터가 딕셔너리 형식이 아닙니다.")
        else:
            title = chart_data.get("title")
            if not title:
                errors.append("차트 제목(title)이 비어 있습니다.")

            x_values = chart_data.get("x_values")
            y_values = chart_data.get("y_values")
            if chart_type in {"single", "mixed", "dual_axis"}:
                if not isinstance(x_values, list) or not x_values:
                    errors.append("x_values는 최소 1개 이상의 값을 가진 리스트여야 합니다.")
                if chart_type == "single":
                    if not isinstance(y_values, list) or not y_values:
                        errors.append("y_values는 최소 1개 이상의 값을 가진 리스트여야 합니다.")
                    if isinstance(x_values, list) and isinstance(y_values, list) and len(x_values) != len(y_values):
                        errors.append("x_values와 y_values의 길이가 일치하지 않습니다.")
                    if isinstance(y_values, list):
                        for idx, value in enumerate(y_values):
                            try:
                                float(value)
                            except (TypeError, ValueError):
                                errors.append(f"y_values[{idx}] 값을 숫자로 변환할 수 없습니다.")
                                break
                else:
                    datasets = chart_data.get("datasets")
                    if not isinstance(datasets, list) or not datasets:
                        errors.append("datasets 정보가 부족합니다.")
                    else:
                        for ds_idx, dataset in enumerate(datasets):
                            data = dataset.get("data") if isinstance(dataset, dict) else None
                            if not isinstance(data, list) or not data:
                                errors.append(f"datasets[{ds_idx}]의 data가 유효하지 않습니다.")
                                break
                            if isinstance(x_values, list) and len(data) != len(x_values):
                                errors.append(f"datasets[{ds_idx}]의 data 길이가 x_values와 다릅니다.")
                                break
                            for val_idx, val in enumerate(data):
                                try:
                                    float(val)
                                except (TypeError, ValueError):
                                    errors.append(
                                        f"datasets[{ds_idx}].data[{val_idx}] 값을 숫자로 변환할 수 없습니다."
                                    )
                                    break
                        if chart_type == "dual_axis":
                            y_axes = chart_data.get("y_axes")
                            if not isinstance(y_axes, list) or len(y_axes) < 2:
                                errors.append("dual_axis 차트에는 두 개의 y_axes 정의가 필요합니다.")

            if chart_type == "single":
                primitive_type = (chart_data.get("chart_type") or "").lower()
                if primitive_type not in {"bar", "line", "pie"}:
                    errors.append("single 차트는 bar, line, pie 중 하나의 chart_type만 허용됩니다.")

        if errors:
            feedback = ", ".join(errors)
            da_state["chart_validation_result"] = {"success": False, "error": feedback}
            da_state["chart_feedback"] = feedback
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_validate",
                    "message": f"차트 데이터 검증 실패: {feedback}",
                },
            )
        else:
            da_state["chart_validation_result"] = {"success": True}
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_validate",
                    "message": "차트 데이터 검증을 완료했습니다.",
                },
            )
        return state

    async def _da_generate_chart_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)
        runtime = self._get_da_runtime(state)

        if da_state.get("abort"):
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_generate",
                    "message": "이전 단계 오류로 차트 생성을 건너뜁니다.",
                },
            )
            return state

        validation = da_state.get("chart_validation_result") or {}
        if not validation.get("success"):
            da_state["chart_resource_id"] = None
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_generate",
                    "message": "검증 실패로 차트 생성을 건너뜁니다.",
                },
            )
            return state

        chart_type = da_state.get("chart_type") or "none"
        if chart_type == "none" or da_state.get("chart_data") is None:
            da_state["chart_resource_id"] = None
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_generate",
                    "message": "차트가 필요하지 않아 생성하지 않습니다.",
                },
            )
            return state

        await self._emit(
            "reasoning",
            {
                "stage": "da_chart_generate",
                "message": "MCP 툴을 통해 차트를 생성합니다.",
            },
        )

        da_state["chart_generation_attempted"] = True

        try:
            tool_map = await get_tool_map()
        except Exception as exc:
            da_state["chart_resource_id"] = None
            da_state["chart_feedback"] = f"툴 맵 로드 실패: {exc}"
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_generate",
                    "message": f"차트 툴을 불러오지 못했습니다: {exc}",
                },
            )
            return state

        tool_fn = None
        for name in ("create_da_chart", "generate_chart_html", "generate_chart"):
            if name in tool_map:
                tool_fn = tool_map[name]
                da_state["chart_tool_name"] = name
                break
        if tool_fn is None:
            da_state["chart_resource_id"] = None
            da_state["chart_feedback"] = "사용 가능한 차트 생성 MCP 툴이 없습니다."
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_generate",
                    "message": "사용 가능한 차트 생성 MCP 툴이 없습니다.",
                },
            )
            return state

        payload = {"chart_data_json": json.dumps(da_state.get("chart_data"), ensure_ascii=False)}

        try:
            result = tool_fn(runtime, **payload)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            da_state["chart_resource_id"] = None
            da_state["chart_feedback"] = f"차트 생성 실패: {exc}"
            await self._emit(
                "reasoning",
                {
                    "stage": "da_chart_generate",
                    "message": f"차트 생성 실패: {exc}",
                },
            )
            return state

        da_state["chart_tool_response"] = result
        chart_id = None
        if isinstance(result, str):
            chart_id = self._extract_chart_id_from_message(result)
        if not chart_id and runtime.tool_state.id_to_iframe:
            # 가장 최근에 추가된 iframe ID 사용
            chart_id = list(runtime.tool_state.id_to_iframe.keys())[-1]
        da_state["chart_resource_id"] = chart_id
        da_state["chart_generated"] = bool(chart_id)

        await self._emit("tool_state", runtime.tool_state.model_dump())
        await self._emit(
            "reasoning",
            {
                "stage": "da_chart_generate",
                "message": (
                    "차트 생성 완료." if chart_id else "차트 생성 응답을 받았지만 리소스 ID를 확인하지 못했습니다."
                ),
            },
        )
        return state

    async def _da_report_with_chart_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)
        runtime = self._get_da_runtime(state)

        if da_state.get("abort"):
            message = da_state.get("sql_error") or da_state.get("abort_reason") or "데이터 분석을 완료할 수 없습니다."
            state["final_answer"] = message
            state["messages"].append({"role": "assistant", "content": message})
            await self._emit(
                "reasoning",
                {
                    "stage": "da_report",
                    "message": "오류 발생으로 보고서 생성을 건너뜁니다.",
                },
            )
            return state

        await self._emit(
            "reasoning",
            {
                "stage": "da_report",
                "message": "보고서를 정리하고 최종 응답을 준비합니다.",
            },
        )

        sql_result = da_state.get("sql_result") or {}
        summary = self._prepare_sql_summary(runtime, sql_result, row_limit=20)
        chart_id = da_state.get("chart_resource_id")
        chart_placeholder = f"【{chart_id}】" if chart_id else "차트 리소스 없음"

        system_prompt = (
            "당신은 데이터 분석 보고서를 작성하는 전문가입니다. "
            "아래 목차를 반드시 사용하여 마크다운 형식으로 답변하세요:\n"
            "# 1. 사용자의 질문 분석 및 요약\n"
            "# 2. 실행된 SQL 쿼리\n"
            "# 3. SQL 쿼리 결과 (쿼리 실행 결과)\n"
            "# 4. 결과 시각화\n"
            "# 5. 분석 인사이트\n"
            "SQL 쿼리는 코드 블록으로 표기하고, 결과 표는 마크다운 테이블을 사용하세요. "
            "시각화 ID가 제공되면 해당 위치에 그대로 삽입하고, 없으면 시각화 불가 사유를 작성하세요. "
            "컨텍스트 JSON을 그대로 복붙하거나 요약 앞에 출력하는 행위는 금지합니다."
        )

        user_payload = {
            "question": state.get("original_question"),
            "sql_query": da_state.get("sql_query"),
            "sql_reasoning": da_state.get("sql_reasoning"),
            "sql_result": summary,
            "chart_placeholder": chart_placeholder,
            "chart_type": da_state.get("chart_type"),
            "chart_available": bool(chart_id),
            "chart_generated": da_state.get("chart_generated"),
            "chart_generation_attempted": da_state.get("chart_generation_attempted"),
            "chart_validation": da_state.get("chart_validation_result"),
            "chart_feedback": da_state.get("chart_feedback"),
        }

        user_context_message = (
            "아래 CONTEXT JSON은 참고용입니다. 해당 JSON을 그대로 출력하거나 복사하지 말고, "
            "지정된 마크다운 구조에 따라 자연어(한국어) 보고서를 작성하세요.\n"
            "CONTEXT:\n"
            f"```json\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```"
        )

        final_message: Optional[Dict[str, Any]] = None
        async for res in call_llm_stream(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context_message},
            ],
            model=self.model,
            temperature=0.2,
        ):
            if isinstance(res, dict) and res.get("event"):
                if res["event"] == "token":
                    await self._emit("token", res.get("data", ""))
                else:
                    await self._emit(res.get("event"), res.get("data"))
            else:
                final_message = res

        if final_message is None:
            content = "보고서 생성에 실패했습니다."
            final_message = {"role": "assistant", "content": content}
        else:
            content = final_message.get("content", "")

        da_state["report_markdown"] = content
        state["final_answer"] = content
        state["messages"].append(final_message)
        return state

    async def _da_cannot_da_node(self, state: GraphState) -> GraphState:
        da_state = self._ensure_da_state(state)
        await self._emit(
            "reasoning",
            {
                "stage": "da_cannot_da",
                "message": "제공된 스키마로는 분석을 수행할 수 없습니다.",
            },
        )
        da_state.setdefault("route_type", "CANNOT_DA")
        message = (
            "현재 제공된 데이터 스키마로는 요청하신 분석을 수행할 수 없습니다. "
            "관련 테이블이나 컬럼 정보를 보강해 주시면 다시 시도하겠습니다."
        )
        state["final_answer"] = message
        state["messages"].append({"role": "assistant", "content": message})
        return state

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

    def _ensure_shared_state(self, state: GraphState) -> Dict[str, Any]:
        shared = state.get("shared_state")
        if not shared:
            shared = {}
            state["shared_state"] = shared
        return shared

    def _ensure_da_state(self, state: GraphState) -> Dict[str, Any]:
        da_state = state.get("da_state")
        if not da_state:
            da_state_model = DataAnalysisState()
            da_state = da_state_model.model_dump()
            state["da_state"] = da_state
        return da_state

    def _get_da_runtime(self, state: GraphState) -> States:
        runtime = state.get("da_runtime")
        if not isinstance(runtime, States):
            runtime = States()
            state["da_runtime"] = runtime
        return runtime

    def _parse_json_from_model(self, text: str) -> Any:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Empty response from model")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
            if fence:
                candidate = fence.group(1).strip()
                if candidate:
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
            braces = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if braces:
                candidate = braces.group(0)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            brackets = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if brackets:
                candidate = brackets.group(0)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            raise

    async def _call_llm_for_json(
        self,
        *,
        system_prompt: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.2,
        stream_stage: Optional[str] = None,
    ) -> Any:
        final_message: Optional[Dict[str, Any]] = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        async for res in call_llm_stream(
            messages=messages,
            model=self.model,
            temperature=temperature,
        ):
            if isinstance(res, dict) and res.get("event"):
                if res["event"] == "token" and stream_stage:
                    await self._emit("token", res.get("data", ""))
                else:
                    await self._emit(res.get("event"), res.get("data"))
            else:
                final_message = res
        if final_message is None:
            raise RuntimeError("LLM 응답을 받지 못했습니다.")
        content = (final_message.get("content") or "").strip()
        return self._parse_json_from_model(content)

    def _prepare_sql_summary(
        self,
        runtime: States,
        result_payload: Dict[str, Any],
        *,
        row_limit: int = 50,
    ) -> Dict[str, Any]:
        result_id = result_payload.get("result_id")
        rows: List[Dict[str, Any]] = []
        if result_id:
            stored = runtime.tool_state.tool_results.get(result_id, {})
            rows = stored.get("rows") or []
        if not rows:
            rows = result_payload.get("sample_rows") or []
        columns = result_payload.get("columns") or []
        summary = {
            "columns": columns,
            "rows": rows[:row_limit],
            "row_count": result_payload.get("row_count", len(rows)),
        }
        return summary

    def _get_chart_format_template(self, chart_type: str) -> str:
        chart_type = (chart_type or "").lower()
        if chart_type == "single":
            return (
                '{\n'
                '  "chart_type": "bar|line|pie",\n'
                '  "title": "<차트 제목>",\n'
                '  "x_values": ["<x축1>", "<x축2>"],\n'
                '  "y_values": [<값1>, <값2>],\n'
                '  "y_label": "<y축 라벨>"\n'
                "}"
            )
        if chart_type == "mixed":
            return (
                '{\n'
                '  "chart_type": "mixed",\n'
                '  "title": "<차트 제목>",\n'
                '  "x_values": ["<x축1>", "<x축2>"],\n'
                '  "datasets": [\n'
                '    { "type": "bar|line|pie", "label": "<라벨>", "data": [<값1>, <값2>] }\n'
                '  ],\n'
                '  "y_label": "<y축 라벨>"\n'
                "}"
            )
        if chart_type == "dual_axis":
            return (
                '{\n'
                '  "chart_type": "dual_axis",\n'
                '  "title": "<차트 제목>",\n'
                '  "x_values": ["<x축1>", "<x축2>"],\n'
                '  "datasets": [\n'
                '    { "type": "bar|line", "label": "<라벨1>", "data": [<값1>, <값2>], "yAxisID": "y1" },\n'
                '    { "type": "line", "label": "<라벨2>", "data": [<값1>, <값2>], "yAxisID": "y2" }\n'
                '  ],\n'
                '  "y_axes": [\n'
                '    { "id": "y1", "label": "<좌측축 라벨>" },\n'
                '    { "id": "y2", "label": "<우측축 라벨>" }\n'
                '  ]\n'
                "}"
            )
        if chart_type == "none":
            return (
                "-- 관련 레코드 단일 추출 예시\n"
                "SELECT *\n"
                "  FROM <TABLE_NAME>\n"
                " WHERE <조건>\n"
                "LIMIT 1;"
            )
        return "지원하지 않는 CHART_TYPE"

    def _extract_chart_id_from_message(self, message: str) -> Optional[str]:
        match = re.search(r"【([^】]+)】", message)
        return match.group(1) if match else None

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
            "사용자 요청을 분석하여 데이터분석이 필요한 경우, 아래 마크다운 템플릿 구조를 정확히 따르시오. "
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
            "SQL 결과를 Markdown 표 행으로 작성하세요. 결과가 없으면 '데이터 없음'이라고 적으세요.\n"
            "---\n"
            "# 4. 결과 시각화\n"
            "iframe HTML 또는 마크다운 이미지 링크를 제공하세요. SQL 쿼리 실행 결과가 없다면, 이 섹션은 건너뛰세요.\n"
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
            "final_answer": None,
            "shared_state": {},
            "da_state": DataAnalysisState().model_dump(),
            "da_runtime": States(),
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
