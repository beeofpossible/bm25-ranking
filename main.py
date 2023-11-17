import math
from collections import Counter

class BM25:
    def __init__(self, documents):
        self.documents = documents
        self.document_count = len(documents)
        self.avg_document_length = sum(len(doc) for doc in documents) / self.document_count
        self.term_counts = self.calculate_term_counts()
        self.k1 = 1.2
        self.b = 0.75

    def calculate_term_counts(self):
        term_counts = Counter()
        for document in self.documents:
            term_counts.update(document)
        return term_counts

    def calculate_idf(self, term):
        document_with_term_count = self.term_counts[term]
        return math.log((self.document_count - document_with_term_count + 0.5) / (document_with_term_count + 0.5))

    def calculate_bm25_score(self, query, document):
        score = 0.0
        document_length = len(document)
        query_terms = Counter(query)

        for term in query_terms:
            if term not in self.documents:
                continue
            idf = self.calculate_idf(term)
            term_frequency = document.count(term)
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (1 - self.b + self.b * (document_length / self.avg_document_length))
            score += idf * (numerator / denominator)

        return score

    def rank_documents(self, query):
        document_scores = []
        for document in self.documents:
            score = self.calculate_bm25_score(query, document)
            document_scores.append((document, score))
        
        ranked_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)
        return ranked_documents


# Example usage:
documents = [
    ['apple', 'banana', 'orange', 'apple'],
    ['banana', 'orange', 'orange'],
    ['apple', 'apple', 'banana', 'banana'],
    ['orange', 'orange', 'banana']
]

bm25 = BM25(documents)
query = ['apple', 'banana']
ranked_documents = bm25.rank_documents(query)

for document, score in ranked_documents:
    print(f"Document: {document}, Score: {score}")
