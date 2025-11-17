import os
import httpx
import requests
import weaviate
from weaviate.classes.filter import Filter
from weaviate.classes.query import MetadataQuery
from weaviate.util import generate_uuid5

class vectordb:
    def __init__(self,
                 genos_ip: str | None = None,
                 http_port: int | None = None,
                 grpc_port: int | None = None,
                 idx: str | None = None,
                 embedding_serving_id: int | None = None,
                 embedding_bearer_token: str | None = None,
                 embedding_genos_url: str | None = None):
        genos_ip = genos_ip or os.getenv("VDB_HOST", "127.0.0.1")
        http_port = http_port or int(os.getenv("VDB_HTTP_PORT", "8080"))
        grpc_port = grpc_port or int(os.getenv("VDB_GRPC_PORT", "50051"))
        idx = idx or os.getenv("VDB_COLLECTION")
        embedding_serving_id = embedding_serving_id or int(os.getenv("EMBEDDING_SERVING_ID", "0"))
        embedding_bearer_token = embedding_bearer_token or os.getenv("EMBEDDING_BEARER_TOKEN", "")
        embedding_genos_url = embedding_genos_url or os.getenv("GENOS_URL", "https://genos.mnc.ai:3443")

        if not idx:
            raise ValueError("VDB_COLLECTION must be set via environment variable")
        if not embedding_serving_id:
            raise ValueError("EMBEDDING_SERVING_ID must be configured")
        if not embedding_bearer_token:
            raise ValueError("EMBEDDING_BEARER_TOKEN must be configured")

        try:
            self.client = weaviate.connect_to_custom(
                http_host=genos_ip,
                http_port=http_port,
                http_secure=False,
                grpc_host=genos_ip,
                grpc_port=grpc_port,
                grpc_secure=False,
            )
        except Exception as e:
            raise RuntimeError(f'Weaviate 접속 중 오류 발생: {e}')

        self.collection = self.client.collections.get(idx)
        print(f'VDB collection `{idx}` 설정 완료')

        self.emb = embedding_serving(
            serving_id=embedding_serving_id,
            bearer_token=embedding_bearer_token,
            genos_url=embedding_genos_url,
        )
        self.converter = WeaviateGraphQLFilterConverter()
    def dense_search(self, query:str, topk = 4):
        vector = self.emb.call(query)[0]['embedding']
        results = [i.properties for i  in self.collection.query.near_vector(near_vector = vector, limit = topk).objects]
        return results
    def bm25_search(self, query:str, topk = 4):
        results = [i.properties for i in self.collection.query.bm25(query, limit = topk).objects]
        return results
    def hybrid_search(self, query:str, topk:int = 4, alpha:float = 0.5):
        vector = self.emb.call(query)[0]['embedding']
        results = [i.properties for i in self.collection.query.hybrid(query = query,
                                                                vector = vector,
                                                                alpha = alpha, limit = topk).objects]
        return results
    def hybrid_search_with_filter(self, query:str, filter:str = '', topk:int = 4, alpha:float = 0.5):
        filter = self.converter.json_to_filter(filter)
        vector = self.emb.call(query)[0]['embedding']
        if filter:
            try:
                results = [i.properties for i in self.collection.query.hybrid(
                    query=query,
                    vector=vector,
                    alpha=alpha,
                    limit=topk,
                    filters = filter).objects]
            except Exception as e:
                print(f'필터 적용 중 오류 발생: {e}')
                prefix = ['필터가 적용되지 않아 전체 검색 결과를 반환합니다.']
                results = [i.properties for i in self.collection.query.hybrid(query = query,
                                                                        vector = vector,
                                                                        alpha = alpha, limit = topk).objects]
                prefix.extend(results)
                results = prefix
        else:
            prefix = ['필터가 적용되지 않아 전체 검색 결과를 반환합니다.']
            results = [i.properties for i in self.collection.query.hybrid(query = query,
                                                                    vector = vector,
                                                                    alpha = alpha, limit = topk).objects]
            prefix.extend(results)
            results = prefix
        return results
    

class embedding_serving:
        def __init__(self,
                     serving_id: int | None = None,
                     bearer_token: str | None = None,
                     genos_url: str | None = None):
            serving_id = serving_id or int(os.getenv("EMBEDDING_SERVING_ID", "0"))
            bearer_token = bearer_token or os.getenv("EMBEDDING_BEARER_TOKEN")
            genos_url = genos_url or os.getenv("GENOS_URL", "https://genos.mnc.ai:3443")

            if not serving_id:
                raise ValueError("EMBEDDING_SERVING_ID must be configured")
            if not bearer_token:
                raise ValueError("EMBEDDING_BEARER_TOKEN must be configured")

            self.serving_id = serving_id
            self.url = f"{genos_url}/api/gateway/rep/serving/{serving_id}"
            self.headers = {"Authorization": f"Bearer {bearer_token}"}
            print(f"embedding model: {serving_id}번 serving 사용")
        def call(self, question:str = '안녕?'):
            body = {
                "input" : [question]
            }
            endpoint = f"{self.url}/v1/embeddings"
            response = requests.post(endpoint, headers=self.headers, json=body)
            result = response.json()
            vector = result['data']
            return vector
        def call_batch(self, question:list = ['안녕?']):
            body = {
                "input" : question
            }
            endpoint = f"{self.url}/v1/embeddings"
            response = requests.post(endpoint, headers=self.headers, json=body)
            result = response.json()
            vector = result['data']
            return vector
        async def async_call(self, question = '안녕?'):
            body = {
                "input" : question
            }
            endpoint = f"{self.url}/v1/embeddings"
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                    response = await client.post(endpoint, headers=self.headers, json=body)
                    result = response.json()
                    vector = result['data']
            except KeyError as e:
                print(response.json())
                print(f'embedding 서빙 호출 중 keyerror 발생: {e}')
                return None
            except httpx.RequestError as e:
                print(f'embedding 서빙 호출 중 오류 발생 : {e}')
                return None
            return vector
        async def async_call_batch(self, question:list = '안녕?'):
            body = {
                "input" : [question]
            }
            endpoint = f"{self.url}/v1/embeddings"
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                    response = await client.post(endpoint, headers=self.headers, json=body)
                    result = response.json()
                    vector = result['data']
            except KeyError as e:
                print(response.json())
                print(f'embedding 서빙 호출 중 keyerror 발생: {e}')
                return None
            except httpx.RequestError as e:
                print(f'embedding 서빙 호출 중 오류 발생 : {e}')
                return None
            return vector