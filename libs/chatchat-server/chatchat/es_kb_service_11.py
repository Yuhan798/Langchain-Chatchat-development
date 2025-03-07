from typing import List
import os
import shutil
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from configs import KB_ROOT_PATH, EMBEDDING_MODEL, EMBEDDING_DEVICE, CACHED_VS_NUM
from server.knowledge_base.kb_service.base import KBService, SupportedVSType
from server.utils import load_local_embeddings
from elasticsearch import Elasticsearch,BadRequestError
from configs import logger
from configs import kbs_config

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

class ElasticsearchStore_8_3_1(ElasticsearchStore):

    def _kb_search(self,
        kb_name,
        query: Optional[str] = None,
        k: int = 4,
        query_vector: Union[List[float], None] = None,
        fetch_k: int = 50,
        fields: Optional[List[str]] = None,
        filter: Optional[List[dict]] = None,
        custom_query: Optional[Callable[[Dict, Union[str, None]], Dict]] = None,
        doc_builder: Optional[Callable[[Dict], Document]] = None,
        file_name: Optional[str] = None,
        **kwargs: Any,)-> List[Tuple[Document, float]]:
        if fields is None:
            fields = []

        if "metadata" not in fields:
            fields.append("metadata")

        if self.query_field not in fields:
            fields.append(self.query_field)

        if self.embedding and query is not None:
            query_vector = self.embedding.embed_query(query)

        # query_body = self.strategy.query(
        #     query_vector=query_vector,
        #     query=query,
        #     k=k,
        #     fetch_k=fetch_k,
        #     vector_query_field=self.vector_query_field,
        #     text_field=self.query_field,
        #     filter=filter or [],
        #     similarity=self.distance_strategy,
        # )
        knn={
            "field":self.vector_query_field,
            "query_vector":query_vector,
            "k":k,
            "num_candidates":fetch_k,
        }
        old_query_body={
            "knn":knn,
            "_source":fields
        }
        
        if file_name == None:
            query_body = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": {
                                "term": {
                                    "metadata.kb_name": kb_name
                                }
                            }
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'dense_vector')+1.0",
                        "params": {
                            "queryVector": query_vector
                        }
                    }
                }
            }
        else:
            query_body = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": {
                                "term": {
                                    "metadata.kb_name": kb_name
                                }
                            },
                            "must":{
                                "match":{
                                    "metadata.source": file_name
                                }
                            }
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'dense_vector')+1.0",
                        "params": {
                            "queryVector": query_vector
                        }
                    }
                }
            }
        logger.debug(f"Query body: {query_body}")

        if custom_query is not None:
            query_body = custom_query(query_body, query)
            logger.debug(f"Calling custom_query, Query body now: {query_body}")
        # Perform the kNN search on the Elasticsearch index and return the results.
        # response = self.client.knn_search(
        #     index=self.index_name,
        #     **query_body,
        #     # size=k,
        #     # source=fields,
        # )
        response=self.client.search(index=self.index_name,
                                    query=query_body,
                                    size=k,
                                    _source=fields,
                                    )

        def default_doc_builder(hit: Dict) -> Document:
            return Document(
                page_content=hit["_source"].get(self.query_field, ""),
                metadata=hit["_source"]["metadata"],
            )

        doc_builder = doc_builder or default_doc_builder

        docs_and_scores = []
        for hit in response["hits"]["hits"]:
            for field in fields:
                if field in hit["_source"] and field not in [
                    "metadata",
                    self.query_field,
                ]:
                    if "metadata" not in hit["_source"]:
                        hit["_source"]["metadata"] = {}
                    hit["_source"]["metadata"][field] = hit["_source"][field]

            docs_and_scores.append(
                (
                    doc_builder(hit),
                    hit["_score"],
                )
            )
        return docs_and_scores


    def similarity_search_with_score(
        self, kb_name,query: str, k: int = 4, filter: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self._kb_search(kb_name,query=query, k=k, filter=filter, **kwargs)
        
    def similarity_search_with_score_in_file(
        self,kb_name, query: str, k: int = 4, filter: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self._kb_search(kb_name,query=query, k=k, filter=filter, **kwargs)
        
class ESKBService(KBService):

    def do_init(self):
        self.kb_path = self.get_kb_path(self.kb_name)
        self.index_name = kbs_config[self.vs_type()]['index_name']
        self.IP = kbs_config[self.vs_type()]['host']
        self.PORT = kbs_config[self.vs_type()]['port']
        self.user = kbs_config[self.vs_type()].get("user",'')
        self.password = kbs_config[self.vs_type()].get("password",'')
        self.dims_length = kbs_config[self.vs_type()].get("dims_length",None)
        self.embeddings_model = load_local_embeddings(self.embed_model, EMBEDDING_DEVICE)
        try:
            # ES python客户端连接（仅连接）
            if self.user != "" and self.password != "":
                self.es_client_python =  Elasticsearch(f"http://{self.IP}:{self.PORT}",
                basic_auth=(self.user,self.password))
            else:
                logger.warning("ES未配置用户名和密码")
                self.es_client_python = Elasticsearch(f"http://{self.IP}:{self.PORT}")
        except ConnectionError:
            logger.error("连接到 Elasticsearch 失败！")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Error 发生 : {e}")
            raise e
        try:
            # 首先尝试通过es_client_python创建
            # mappings = {
            #     "dense_vector": {
            #         "type": "dense_vector"
            #     }
            # }
            mappings = {
                "mappings": {
                    "properties": {
                        "context": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "dense_vector": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "properties": {
                                "source": {
                                    "type": "keyword",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "kb_name":{
                                    "type":"keyword",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                }
                            }
                        }
                        
                    }
                }
            }
            self.es_client_python.indices.create(index=self.index_name, body=mappings)
        except BadRequestError as e:
            logger.error("创建索引失败,重新")
            logger.error(e)

        try:
            # langchain ES 连接、创建索引
            if self.user != "" and self.password != "":
                self.db_init = ElasticsearchStore_8_3_1(
                es_url=f"http://{self.IP}:{self.PORT}",
                index_name=self.index_name,
                query_field="context",
                vector_query_field="dense_vector",
                embedding=self.embeddings_model,
                es_user=self.user,
                es_password=self.password
            )
            else:
                logger.warning("ES未配置用户名和密码")
                self.db_init = ElasticsearchStore_8_3_1(
                    es_url=f"http://{self.IP}:{self.PORT}",
                    index_name=self.index_name,
                    query_field="context",
                    vector_query_field="dense_vector",
                    embedding=self.embeddings_model,
                )
        except ConnectionError:
            print("### 初始化 Elasticsearch 失败！")
            logger.error("### 初始化 Elasticsearch 失败！")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Error 发生 : {e}")
            raise e
        try:
            # 尝试通过db_init创建索引
            self.db_init._create_index_if_not_exists(
                                                     index_name=self.index_name,
                                                     dims_length=self.dims_length
                                                     )
        except Exception as e:
            logger.error("创建索引失败...")
            logger.error(e)
            # raise e 
            
        

    @staticmethod
    def get_kb_path(knowledge_base_name: str):
        return os.path.join(KB_ROOT_PATH, knowledge_base_name)

    @staticmethod
    def get_vs_path(knowledge_base_name: str):
        return os.path.join(ESKBService.get_kb_path(knowledge_base_name), "vector_store")

    def do_create_kb(self):
        if os.path.exists(self.doc_path):
            if not os.path.exists(os.path.join(self.kb_path, "vector_store")):
                os.makedirs(os.path.join(self.kb_path, "vector_store"))
            else:
                logger.warning("directory `vector_store` already exists.")

    def vs_type(self) -> str:
        return SupportedVSType.ES

    def _load_es(self, docs, embed_model):
        # 将docs写入到ES中
        try:
            for document in docs:
                document.metadata["kb_name"]=self.kb_name
        except Exception as e:
            print(e)
            print("将知识库名称在添加到matedata失败!")
            logger.error("将知识库名称在添加到matedata失败!")
            logger.error(e)
        try:
            # 连接 + 同时写入文档
            if self.user != "" and self.password != "":
                self.db = ElasticsearchStore_8_3_1.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False,
                        es_user=self.user,
                        es_password=self.password
                    )
            else:
                self.db = ElasticsearchStore_8_3_1.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False)
        except ConnectionError as ce:
            print(ce)
            print("连接到 Elasticsearch 失败！")
            logger.error("连接到 Elasticsearch 失败！")
        except Exception as e:
            logger.error(f"Error 发生 : {e}")
            print(e)



    def do_search(self, query:str, top_k: int, score_threshold: float):
        # 文本相似性检索
        docs = self.db_init.similarity_search_with_score(self.kb_name,query=query,
                                         k=top_k)
        return docs
        
    def do_search_in_file(self, query:str, top_k: int, score_threshold: float,file_name: str):
        # 文本相似性检索
        docs = self.db_init.similarity_search_with_score_in_file(self.kb_name,query=query,
                                         k=top_k,file_name=file_name)
        return docs


    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        results = []
        for doc_id in ids:
            try:
                response = self.es_client_python.get(index=self.index_name, id=doc_id)
                source = response["_source"]
                # Assuming your document has "text" and "metadata" fields
                text = source.get("context", "")
                metadata = source.get("metadata", {})
                results.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                logger.error(f"Error retrieving document from Elasticsearch! {e}")
        return results

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        for doc_id in ids:
            try:
                self.es_client_python.delete(index=self.index_name,
                                            id=doc_id,
                                            refresh=True)
            except Exception as e:
                logger.error(f"ES Docs Delete Error! {e}")

    def do_delete_doc(self, kb_file, **kwargs):
        if self.es_client_python.indices.exists(index=self.index_name):
            # 从向量数据库中删除索引(文档名称是Keyword)
            query ={
                "bool": {
                    "filter": {
                        "term": {
                             "metadata.kb_name": kb_file.kb_name
                         }
                      },
                    "must": {
                        "match": {
                            "metadata.source": kb_file.filename
                        }
                    }
                }
            }
            try:
                self.es_client_python.delete_by_query(index=self.index_name,query=query)
            except Exception as e:
                logger.error(f"ES Docs Delete Error! {e}")

    def do_delete_doc_ori(self, kb_file, **kwargs):
        if self.es_client_python.indices.exists(index=self.index_name):
            # 从向量数据库中删除索引(文档名称是Keyword)
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": kb_file.filepath
                    }
                }
            }
            # 注意设置size，默认返回10个。
            search_results = self.es_client_python.search(body=query, size=50)
            delete_list = [hit["_id"] for hit in search_results['hits']['hits']]
            if len(delete_list) == 0:
                return None
            else:
                for doc_id in delete_list:
                    try:
                        self.es_client_python.delete(index=self.index_name,
                                                     id=doc_id,
                                                     refresh=True)
                    except Exception as e:
                        logger.error(f"ES Docs Delete Error! {e}")

            # self.db_init.delete(ids=delete_list)
            #self.es_client_python.indices.refresh(index=self.index_name)
    
    def do_add_doc(self,docs: List[Document], **kwargs):
        print(f"server.knowledge_base.kb_service.es_kb_service.do_add_doc 输入的docs参数长度为:{len(docs)}")
        print("*" * 100)
        self._load_es(docs=docs, embed_model=self.embeddings_model)
        # 获取 id 和 source , 格式：[{"id": str, "metadata": dict}, ...]
        print("写入数据成功.")
        print("*" * 100)

        if self.es_client_python.indices.exists(index=self.index_name):
            file_path = docs[0].metadata.get("source")
            query = {
                "bool": {
                    "filter": [
                        {"term": {
                            "metadata.source": file_path
                        }}
                    ]
                }
            }
            # 注意设置size，默认返回10个。
            search_results = self.es_client_python.search(index=self.index_name,query=query,size=50)
            if len(search_results["hits"]["hits"]) == 0:
                raise ValueError("召回元素个数为0")
        info_docs = [{"id":hit["_id"], "metadata": hit["_source"]["metadata"]} for hit in search_results["hits"]["hits"]]
        return info_docs

    def do_add_doc_ori(self, docs: List[Document], **kwargs):
        '''向知识库添加文件'''
        print(f"server.knowledge_base.kb_service.es_kb_service.do_add_doc 输入的docs参数长度为:{len(docs)}")
        print("*"*100)
        self._load_es(docs=docs, embed_model=self.embeddings_model)
        # 获取 id 和 source , 格式：[{"id": str, "metadata": dict}, ...]
        print("写入数据成功.")
        print("*"*100)

        if self.es_client_python.indices.exists(index=self.index_name):
            file_path = docs[0].metadata.get("source")
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": file_path
                    },
                    "term": {
                        "_index": self.index_name
                    }
                }
            }
            # 注意设置size，默认返回10个。
            search_results = self.es_client_python.search(body=query, size=50)
            if len(search_results["hits"]["hits"]) == 0:
                raise ValueError("召回元素个数为0")
        info_docs = [{"id":hit["_id"], "metadata": hit["_source"]["metadata"]} for hit in search_results["hits"]["hits"]]
        return info_docs


    def do_clear_vs(self):
        """从知识库删除全部向量"""
        if self.es_client_python.indices.exists(index=self.kb_name):
            self.es_client_python.indices.delete(index=self.kb_name)

    def do_drop_kb(self):
        """删除知识库"""
        # self.kb_file: 知识库路径
        if os.path.exists(self.kb_path):
            shutil.rmtree(self.kb_path)
            query = {
                "match": {
                    "metadata.kb_name": self.kb_name
                }
            }
            result=self.es_client_python.delete_by_query(index=self.index_name,query=query)
            print("drop_kb result is ",result) 

    def do_drop_kb_ori(self):
        """删除知识库"""
        # self.kb_file: 知识库路径
        if os.path.exists(self.kb_path):
            shutil.rmtree(self.kb_path)








