import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from opensearchpy import OpenSearch, helpers
from urllib.parse import urlparse

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS


def _parse_kv_args(arg: str) -> Dict[str, str]:
    """
    Parse simple comma-separated key=value pairs.
    Example: "num_candidates=200,batch=16"
    """
    result = {}
    if not arg:
        return result
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid query arg '{token}'. Expected key=value.")
        key, value = token.split("=", 1)
        result[key.strip()] = value.strip()
    return result


class OCIOpenSearchANN(BaseANN):
    """
    Benchmark adapter for OCI OpenSearch.
    Builds a fresh knn_vector index per run, ingests dataset points via bulk API,
    and executes KNN queries using the msearch endpoint.
    """

    def __init__(self, metric: str, index_config: Dict):
        if metric != "ip":
            raise ValueError(
                "OCI OpenSearch adapter currently supports only inner product (ip)."
            )
        self.metric = metric
        self.config = index_config or {}

        self.endpoint = self._resolve_secret("endpoint")
        self.username = self._resolve_secret("username")
        self.password = self._resolve_secret("password")

        self.vector_field = self.config.get("vector_field", "embedding")
        self.index_prefix = self.config.get("index_prefix", "oci-opensearch")
        self.bulk_chunk_size = int(self.config.get("bulk_chunk_size", 2048))
        self.ingest_workers = int(
            self.config.get("ingest_workers", os.cpu_count() or 4)
        )
        self.request_timeout = int(self.config.get("request_timeout", 120))
        self.query_batch_size = int(self.config.get("query_batch_size", 32))

        self.hnsw_m = int(self.config.get("hnsw_m", 16))
        self.ef_construction = int(self.config.get("ef_construction", 100))
        self.engine = self.config.get("engine", "faiss")

        self.verify_certs = bool(self.config.get("verify_certs", True))
        self.ca_certs = self.config.get("ca_certs")

        self.index_name: Optional[str] = None
        self.client = self._create_client()

        self.num_candidates = int(self.config.get("default_num_candidates", 200))
        self._results = None
        self.name = self.config.get("name", "oci-opensearch")

    def _resolve_secret(self, key: str) -> str:
        """Allow passing literal values or env variable indirection."""
        if key in self.config:
            return self.config[key]

        env_key = self.config.get(f"{key}_env") or f"OPENSEARCH_{key.upper()}"
        value = os.environ.get(env_key)
        if not value:
            raise ValueError(
                f"Missing OpenSearch config for '{key}'. "
                f"Set '{key}' in definitions or provide env var '{env_key}'."
            )
        return value

    def _create_client(self) -> OpenSearch:
        parsed = urlparse(self.endpoint)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(
                f"Endpoint '{self.endpoint}' must include scheme and host."
            )
        host_entry = {
            "host": parsed.hostname,
            "port": parsed.port or (443 if parsed.scheme == "https" else 80),
            "scheme": parsed.scheme,
        }
        return OpenSearch(
            hosts=[host_entry],
            http_auth=(self.username, self.password),
            use_ssl=parsed.scheme == "https",
            verify_certs=self.verify_certs,
            timeout=self.request_timeout,
            ca_certs=self.ca_certs,
        )

    def track(self):
        return "T1"

    # BaseANN API ----------------------------------------------------------------
    def fit(self, dataset: str):
        ds = DATASETS[dataset]()
        # Ensure base vectors are downloaded. The global prepare() call in main
        # skips large base files to save time, so we lazily fetch them here if
        # they are missing.
        try:
            ds.get_dataset_fn()
        except RuntimeError:
            print("Base vectors not found locally. Downloading...")
            ds.prepare(skip_data=False)
        self.index_name = self._build_index_name(dataset)
        self._create_index(ds.d)
        self._ingest_dataset(ds)

    def load_index(self, dataset: str):
        # Always rebuild per requirements.
        return False

    def index_files_to_store(self, dataset):
        raise NotImplementedError(
            "Cloud index storage is not supported for OpenSearch runs."
        )

    def set_query_arguments(self, *query_args):
        for arg in query_args:
            if isinstance(arg, dict):
                items = arg
            else:
                items = _parse_kv_args(arg)
            if "num_candidates" in items:
                self.num_candidates = int(items["num_candidates"])
            if "query_batch_size" in items:
                self.query_batch_size = int(items["query_batch_size"])

    def query(self, X, k):
        if not self.index_name:
            raise RuntimeError("Index not initialized. Call fit() first.")
        engine_uses_candidates = self.engine.lower() != "faiss"
        if engine_uses_candidates and self.num_candidates < k:
            raise ValueError(
                f"num_candidates ({self.num_candidates}) must be >= k ({k})."
            )

        nq = X.shape[0]
        results = -np.ones((nq, k), dtype=np.int32)

        batch = self.query_batch_size
        for start in range(0, nq, batch):
            chunk = X[start : start + batch]
            ndjson_parts: List[str] = []
            for vec in chunk:
                ndjson_parts.append(json.dumps({"index": self.index_name}))
                knn_field = {
                    "vector": vec.astype(np.float32).tolist(),
                    "k": k,
                }
                if engine_uses_candidates:
                    knn_field["num_candidates"] = self.num_candidates

                ndjson_parts.append(
                    json.dumps(
                        {"size": k, "query": {"knn": {self.vector_field: knn_field}}}
                    )
                )
            body = "\n".join(ndjson_parts) + "\n"
            try:
                response = self.client.msearch(
                    body=body,
                    request_timeout=self.request_timeout,
                    headers={"content-type": "application/x-ndjson"},
                )
            except Exception as exc:
                raise RuntimeError(f"OpenSearch msearch failed: {exc}") from exc
            if "responses" not in response:
                raise RuntimeError(f"Unexpected msearch response: {response}")
            for offset, resp in enumerate(response["responses"]):
                hits = resp.get("hits", {}).get("hits", [])
                row = start + offset
                for pos, hit in enumerate(hits[:k]):
                    try:
                        results[row, pos] = int(hit["_id"])
                    except ValueError:
                        # fallback if id not integer
                        results[row, pos] = -1
        self._results = results

    def get_results(self):
        return self._results

    # Helpers --------------------------------------------------------------------
    def _build_index_name(self, dataset: str) -> str:
        timestamp = int(time.time())
        base = f"{self.index_prefix}-{dataset}-{timestamp}".lower()
        # OpenSearch index name rules: lowercase, and only letters, digits, - _ .
        safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in base)
        return safe

    def _create_index(self, dimension: int):
        settings = {
            "settings": {
                "number_of_shards": int(self.config.get("shards", 4)),
                "number_of_replicas": int(self.config.get("replicas", 0)),
                "knn": True,
                "refresh_interval": "-1",
                "translog.durability": "async",
            },
            "mappings": {
                "properties": {
                    self.vector_field: {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "engine": self.engine,
                            "space_type": "innerproduct",
                            "parameters": {
                                "ef_construction": self.ef_construction,
                                "m": self.hnsw_m,
                            },
                        },
                    }
                }
            },
        }
        self.client.indices.create(
            index=self.index_name,
            body=settings,
            timeout=self.request_timeout,
        )

    def _ingest_dataset(self, dataset_obj):
        print(f"Start ingesting dataset into index {self.index_name}")

        def action_generator():
            doc_id = 0
            for block in dataset_obj.get_dataset_iterator(bs=self.bulk_chunk_size):
                block = block.astype(np.float32)
                for row in block:
                    yield {
                        "_op_type": "index",
                        "_index": self.index_name,
                        "_id": str(doc_id),
                        "_source": {self.vector_field: row.tolist()},
                    }
                    doc_id += 1

        successes = 0
        for ok, _ in helpers.parallel_bulk(
            self.client,
            action_generator(),
            thread_count=self.ingest_workers,
            chunk_size=self.bulk_chunk_size,
            request_timeout=self.request_timeout,
        ):
            if ok:
                successes += 1
        self.client.indices.refresh(index=self.index_name)
        print(f"Ingest complete. Total bulk chunks acknowledged: {successes}")
