import asyncio
import json
import os
import ssl
from typing import Optional, Dict, Any

import aiomysql
from pydantic import BaseModel, Field, ValidationError

from app.logger import get_logger
from app.utils import States

log = get_logger(__name__)

_DB_POOL: aiomysql.Pool | None = None


class ExecuteSQLInput(BaseModel):
    query: str = Field(description="SQL query to execute against the configured database.")
    limit: int = Field(default=500, ge=1, le=5000, description="Maximum number of rows to return.")
    column_oriented: bool = Field(
        default=True,
        description="When true, results include a column oriented mapping for easier visualization."
    )


class DescribeSchemaInput(BaseModel):
    table: Optional[str] = Field(
        default=None,
        description="Optional table name to filter on. Supports SQL LIKE patterns (e.g. 'NCONT%')."
    )
    sample_rows: int = Field(
        default=3,
        ge=0,
        le=20,
        description="When greater than 0, include up to N sample rows per table."
    )


def _get_db_config() -> Dict[str, Any]:
    host = os.getenv("GENOS_DB_HOST")
    port = int(os.getenv("GENOS_DB_PORT", "3306"))
    user = os.getenv("GENOS_DB_USER")
    password = os.getenv("GENOS_DB_PASSWORD")
    database = os.getenv("GENOS_DB_NAME")

    if not all([host, user, password, database]):
        raise RuntimeError(
            "Database credentials are not fully configured."
            " Please set GENOS_DB_HOST, GENOS_DB_PORT, GENOS_DB_USER, GENOS_DB_PASSWORD, GENOS_DB_NAME."
        )

    ssl_mode = os.getenv("GENOS_DB_SSL_MODE", "").lower()
    ssl_context: Optional[ssl.SSLContext] = None
    if ssl_mode in {"require", "verify_ca", "verify_identity"}:
        ca_path = os.getenv("GENOS_DB_SSL_CA")
        ssl_context = ssl.create_default_context(cafile=ca_path if ca_path else None)
        if ssl_mode == "require":
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        elif ssl_mode == "verify_ca":
            ssl_context.check_hostname = False
        elif ssl_mode == "verify_identity":
            ssl_context.check_hostname = True

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "ssl": ssl_context,
    }


async def _ensure_pool() -> aiomysql.Pool:
    global _DB_POOL
    if _DB_POOL and not _DB_POOL.closed:
        return _DB_POOL

    cfg = _get_db_config()
    log.info("Creating MySQL connection pool", extra={"host": cfg["host"], "port": cfg["port"], "database": cfg["database"]})

    _DB_POOL = await aiomysql.create_pool(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        db=cfg["database"],
        autocommit=True,
        minsize=int(os.getenv("GENOS_DB_POOL_MIN", "1")),
        maxsize=int(os.getenv("GENOS_DB_POOL_MAX", "5")),
        ssl=cfg["ssl"],
    )
    return _DB_POOL


def _serialize(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return value


async def execute_sql(states: States, **tool_input) -> Dict[str, Any]:
    try:
        params = ExecuteSQLInput(**tool_input)
    except ValidationError as e:
        return {"success": False, "error": f"Invalid input: {e}"}

    query = params.query.strip()
    if not query:
        return {"success": False, "error": "Query cannot be empty."}

    pool = await _ensure_pool()

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            log.info("Executing SQL query", extra={"preview": query[:120]})
            try:
                await cur.execute(query)
                rows = await cur.fetchmany(params.limit)
                columns = [col[0] for col in cur.description] if cur.description else []
            except Exception as exc:
                log.exception("SQL execution failed")
                return {
                    "success": False,
                    "error": str(exc),
                    "type": getattr(exc, "__class__", type(exc)).__name__,
                    "sql_state": getattr(exc, "sqlstate", None),
                    "errno": getattr(exc, "errno", None),
                }

    serialized_rows = _serialize(rows)
    column_oriented = {
        col: [row.get(col) for row in rows]
        for col in columns
    } if columns else {}

    result_payload = {
        "success": True,
        "row_count": len(rows),
        "columns": columns,
        "sample_rows": serialized_rows[: min(5, len(serialized_rows))],
    }

    if params.column_oriented and column_oriented:
        result_payload["column_oriented"] = _serialize(column_oriented)

    result_id = f"db_result_{len(states.tool_state.tool_results)}"
    states.tool_state.tool_results[result_id] = {
        "query": query,
        "rows": serialized_rows,
        "columns": columns,
        "column_oriented": result_payload.get("column_oriented"),
    }

    result_payload["result_id"] = result_id
    result_payload["message"] = (
        f"Query executed successfully. Result stored under id '{result_id}'."
        f" Returned {len(rows)} rows."
    )

    return result_payload


async def describe_schema(states: States, **tool_input) -> Dict[str, Any]:
    try:
        params = DescribeSchemaInput(**tool_input)
    except ValidationError as e:
        return {"success": False, "error": f"Invalid input: {e}"}

    pool = await _ensure_pool()
    cfg = _get_db_config()
    database = cfg["database"]

    where_clauses = ["TABLE_SCHEMA = %s"]
    q_params: list[Any] = [database]
    if params.table:
        where_clauses.append("TABLE_NAME LIKE %s")
        q_params.append(params.table)

    where_sql = " AND ".join(where_clauses)

    schema_query = f"""
        SELECT
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            COLUMN_KEY,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE {where_sql}
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(schema_query, q_params)
            columns = await cur.fetchall()

        samples: Dict[str, Any] = {}
        if params.sample_rows > 0:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                table_names = sorted({row["TABLE_NAME"] for row in columns})
                for table_name in table_names:
                    sample_sql = f"SELECT * FROM `{table_name}` LIMIT %s"
                    await cur.execute(sample_sql, (params.sample_rows,))
                    samples[table_name] = _serialize(await cur.fetchall())

    schema: Dict[str, Any] = {}
    for row in columns:
        table = row.pop("TABLE_NAME")
        table_meta = schema.setdefault(table, {"columns": []})
        table_meta["columns"].append({
            "name": row["COLUMN_NAME"],
            "type": row["DATA_TYPE"],
            "is_primary": row["COLUMN_KEY"] == "PRI",
            "is_unique": row["COLUMN_KEY"] in ("PRI", "UNI"),
            "nullable": row["IS_NULLABLE"] == "YES",
            "default": row["COLUMN_DEFAULT"],
        })

    if params.sample_rows > 0:
        for table, rows in samples.items():
            schema.setdefault(table, {})["sample_rows"] = rows

    return {
        "success": True,
        "database": database,
        "tables": schema,
        "message": f"Fetched schema for {len(schema)} table(s) from '{database}'."
    }


DB_EXECUTE_SQL = {
    "type": "function",
    "function": {
        "name": "db_execute",
        "description": "Execute a SQL query against the configured relational database and return the results.",
        "parameters": ExecuteSQLInput.model_json_schema(),
    },
}


DB_DESCRIBE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "db_describe_schema",
        "description": "Describe tables and columns available in the configured relational database.",
        "parameters": DescribeSchemaInput.model_json_schema(),
    },
}
