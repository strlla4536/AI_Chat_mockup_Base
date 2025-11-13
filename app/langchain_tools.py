"""
기존 도구들을 LangChain Tool로 래핑하는 모듈
"""
 # langchain_core.tools import 제거 (불필요)
from typing import Optional, List
import json

from app.tools.bio import bio as bio_func
from app.tools.web_search import web_search as web_search_func
from app.tools.open_url import open as open_func
from app.utils import States
from app.logger import get_logger

log = get_logger(__name__)


async def search_web(
    search_query: List[dict],
    response_length: str = "medium"
) -> str:
    """
    웹 검색을 수행합니다.
    
    Args:
        search_query: 검색 쿼리 객체의 배열
        response_length: 응답 길이 ("short", "medium", "long")
    
    Returns:
        검색 결과
    """
    try:
        states = States()
        result = await web_search_func(states=states, search_query=search_query, response_length=response_length)
        # result is a list of structured items; return as dict for caller
        if result:
            log.info("search_web executed", extra={"count": len(result), "queries": [q.get('q') for q in search_query]})
            return {"results": result}
        return {"results": []}
    except Exception as e:
        log.error(f"Search failed: {e}")
        return {"error": f"검색 중 오류가 발생했습니다: {str(e)}"}


async def open_url(
    id: Optional[str] = None,
    loc: int = -1,
    num_lines: int = 100
) -> str:
    """
    URL을 열고 내용을 표시합니다.
    
    Args:
        id: 열 링크의 ID 또는 URL
        loc: 시작 라인 번호 (-1이면 처음부터)
        num_lines: 표시할 라인 수
    
    Returns:
        페이지 내용
    """
    try:
        states = States()
        result = await open_func(states=states, id=id, loc=loc, num_lines=num_lines)
        return str(result) if result else "페이지를 열 수 없습니다."
    except Exception as e:
        log.error(f"Open URL failed: {e}")
        return f"URL 열기 중 오류가 발생했습니다: {str(e)}"


async def manage_memory(
    mode: str,
    content: Optional[str] = None,
    id: Optional[int] = None
) -> str:
    """
    사용자 메모리를 관리합니다.
    
    Args:
        mode: "w" (쓰기) 또는 "d" (삭제)
        content: 저장할 내용 (mode="w"일 때)
        id: 삭제할 메모리 항목 ID (mode="d"일 때)
    
    Returns:
        작업 결과
    """
    try:
        states = States()
        result = await bio_func(states=states, mode=mode, content=content, id=id)
        return str(result) if result else "메모리 작업을 완료했습니다."
    except Exception as e:
        log.error(f"Memory management failed: {e}")
        return f"메모리 관리 중 오류가 발생했습니다: {str(e)}"


def get_langchain_tools():
    """모든 LangChain Tool 반환"""
    return {
        "search": search_web,
        "open": open_url,
        "memory": manage_memory,
    }
