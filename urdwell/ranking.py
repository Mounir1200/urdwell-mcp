"""Classement hybride pour le retrieval : dense (cosinus) + lexical (BM25).

La jambe dense capte la similarité sémantique ; la jambe lexicale repêche les
correspondances de termes exacts (noms propres, identifiants, tokens rares) que
l'embedding sous-évalue. Reciprocal Rank Fusion combine les deux ordres sans
comparer leurs scores, qui ne sont pas commensurables.

L'abstention reste une décision cosinus (voir ``hybrid_rank``) : la fusion ne
fait que réordonner un vivier déjà jugé pertinent, elle ne décide jamais s'il
faut répondre.
"""

import math
import re
import unicodedata

from urdwell import embeddings
from urdwell.models import Memory

# Constantes de fusion et de BM25. Valeurs par défaut éprouvées, ajustables.
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_POOL_SIZE = 50

# Un candidat associe une mémoire à son vecteur d'embedding déjà stocké.
Candidate = tuple[Memory, list[float]]
ScoredMemory = tuple[Memory, float]


def _tokenize(text: str) -> list[str]:
    """Tokenisation conservatrice : NFKC, casefold, mots Unicode."""
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def _document_score(
    query_terms: list[str],
    tokens: list[str],
    document_frequency: dict[str, int],
    document_count: int,
    average_length: float,
) -> float:
    """Score BM25 d'un seul document pour les termes de la requête."""
    score = 0.0
    for term in query_terms:
        frequency = tokens.count(term)
        if frequency == 0:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        normalization = 1 - BM25_B + BM25_B * len(tokens) / average_length
        score += idf * frequency * (BM25_K1 + 1) / (frequency + BM25_K1 * normalization)
    return score


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Score BM25 de chaque document, IDF calculée sur les documents fournis."""
    if not documents:
        return []

    tokenized_documents = [_tokenize(document) for document in documents]
    document_count = len(documents)
    average_length = sum(len(tokens) for tokens in tokenized_documents) / document_count

    document_frequency: dict[str, int] = {}
    for tokens in tokenized_documents:
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    query_terms = _tokenize(query)
    return [
        _document_score(
            query_terms, tokens, document_frequency, document_count, average_length
        )
        for tokens in tokenized_documents
    ]


def reciprocal_rank_fusion(*rankings: list, k: int = RRF_K) -> dict:
    """Fusionne plusieurs classements en sommant ``1 / (k + rang)`` par item.

    Chaque classement est une liste d'identifiants ordonnés du meilleur au pire.
    Un item absent d'un classement n'y contribue tout simplement pas.
    """
    fused: dict = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1 / (k + rank)
    return fused


def _order_descending(scores: list[float]) -> list[int]:
    """Indices des documents triés par score décroissant (tri stable)."""
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)


def hybrid_rank(
    query: str,
    query_embedding: list[float],
    candidates: list[Candidate],
    k: int,
    cosine_floor: float,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    rrf_k: int = RRF_K,
) -> list[ScoredMemory]:
    """Classe les candidats par fusion cosinus + BM25 et renvoie le top-k.

    L'abstention est une décision cosinus : si le meilleur candidat n'atteint
    pas ``cosine_floor``, on renvoie une liste vide. Sinon on réordonne le
    vivier des ``pool_size`` meilleurs cosinus par RRF. Le score renvoyé reste
    le cosinus (magnitude interprétable) ; seul l'ordre provient du RRF.
    """
    if not candidates:
        return []

    by_cosine = sorted(
        (
            (memory, embeddings.cosine_similarity(query_embedding, embedding))
            for memory, embedding in candidates
        ),
        key=lambda scored: scored[1],
        reverse=True,
    )

    # Le cosinus seul décide s'il faut répondre (spécificité préservée).
    if by_cosine[0][1] < cosine_floor:
        return []

    pool = by_cosine[:pool_size]
    cosine_order = [memory.id for memory, _ in pool]
    bm25_order = [
        pool[index][0].id
        for index in _order_descending(bm25_scores(query, [m.content for m, _ in pool]))
    ]

    fused = reciprocal_rank_fusion(cosine_order, bm25_order, k=rrf_k)
    cosine_by_id = {memory.id: (memory, score) for memory, score in pool}
    fused_order = sorted(cosine_by_id, key=lambda memory_id: fused[memory_id], reverse=True)
    return [cosine_by_id[memory_id] for memory_id in fused_order[:k]]
