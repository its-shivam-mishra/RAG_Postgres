from flashrank import Ranker, RerankRequest
query = "What is the capital of France?"
passages = [
    {"id":1, "text": "Paris is the capital of France.", "meta": {}},
    {"id":2, "text": "The Eiffel Tower is in Paris.", "meta": {}},
    {"id":3, "text": "Rome is the capital of Italy.", "meta": {}}
]
ranker = Ranker()
rerankrequest = RerankRequest(query=query, passages=passages)
results = ranker.rerank(rerankrequest)
print("Results:", results)
