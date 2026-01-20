from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Union
import hashlib

# Type Aliases for clarity
Metadata = Dict[str, Any]
SearchResultItem = List[Union[str, Metadata]]


class Search:
    def __init__(self, collection):
        self.collection = collection
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search(self, query: str, top_k_retrieval=20, top_k_rerank=5):

        # Stage 1: Semantic Retrieval (Bio-Encoder)
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k_retrieval
        )
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        # Stage 2: Re-ranking (Cross-Encoder)
        # Prepare pairs: (Query, Document_Context)
        pairs = [[query, doc] for doc in documents]

        # Predict scores
        scores = self.cross_encoder.predict(pairs)

        # Sort by score (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Stage 3: Return top-k results
        retrieved = []
        for rank, idx in enumerate(ranked_indices[:top_k_rerank]):
            retrieved.append({
                "rank": rank+1,
                "score": scores[idx],
                "source": metadatas[idx]['source'],
                "page": metadatas[idx]['page'],
                "content": documents[idx]
            })

        return retrieved

    @classmethod
    def get(cls, collection):
        instance = cls(collection)
        return instance.search


class HybridSearch:
    def __init__(self, vector_collection, bm25_retriever,
                 cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.collection = vector_collection
        self.bm25 = bm25_retriever
        self.encoder = CrossEncoder(cross_encoder_model)

    def _reciprocal_rank_fusion(self, vector_hits: List[SearchResultItem], bm25_hits: List[SearchResultItem],
                                k: int = 60) -> List[Dict]:
        """
        Since IDs are now hashes of the content, identical content from
        different search methods will naturally have the same key here.
        """
        fused_scores = {}
        doc_info_map = {}

        for results_list in [vector_hits, bm25_hits]:
            for rank, item in enumerate(results_list):
                doc_id, content, metadata = item[0], item[1], item[2]

                # If the same hash appears in both searches, the RRF scores sum up
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    doc_info_map[doc_id] = {"content": content, "metadata": metadata}

                fused_scores[doc_id] += 1 / (k + (rank + 1))

        # Sort by fused score
        reranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"id": d_id, "score": sc, **doc_info_map[d_id]} for d_id, sc in reranked_ids]

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        # 1. Vector Search (Dense) - Returns top 20
        vec_res = self.collection.query(query_texts=[query], n_results=20)
        vector_hits = [
            [i, d, m] for i, d, m in zip(vec_res['ids'][0], vec_res['documents'][0], vec_res['metadatas'][0])
        ]

        # 2. BM25 Search (Sparse) - Returns top 20
        # LangChain's BM25Retriever.invoke returns Document objects
        bm25_docs = self.bm25.invoke(query)
        bm25_hits = [
            [doc.metadata.get('id'), doc.page_content, doc.metadata] for doc in bm25_docs
        ]

        # 3. Reciprocal Rank Fusion
        # This merges the two lists. If a hash exists in both, it climbs to the top.
        fused_results = self._reciprocal_rank_fusion(vector_hits, bm25_hits)

        if not fused_results:
            return []

        # 4. Cross-Encoding (Re-ranking)
        # We only re-rank the fused candidates (usually ~20-30 unique chunks)
        pairs = [[query, res['content']] for res in fused_results]
        ce_scores = self.encoder.predict(pairs)

        for i, res in enumerate(fused_results):
            res['ce_score'] = float(ce_scores[i])

        # Final sort based on the Cross-Encoder's "deep" understanding
        final_ranked = sorted(fused_results, key=lambda x: x['ce_score'], reverse=True)

        # 5. Final Formatting
        retrieved = []
        for rank, item in enumerate(final_ranked[:top_k]):
            retrieved.append({
                "rank": rank + 1,
                "id": item['id'],
                "relevance_score": item['ce_score'],
                "source": item['metadata'].get('source'),
                "page": item['metadata'].get('page'),
                "content": item['content']
            })

        return retrieved

    @classmethod
    def get(cls, collection, retriever):
        """Helper to create instance and return the search method directly."""
        instance = cls(collection, retriever)
        return instance.search