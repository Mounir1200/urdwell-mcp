"""Sonde de consolidation : que décide RÉELLEMENT le pipeline ?

On amorce le store avec une seule mémoire, puis on sonde une série d'entrées
entrantes SANS les persister. Pour chacune on affiche la similarité cosinus
brute contre la mémoire de référence et l'action que `decide_action` renverrait
au PREMIER appel (decision=None), c'est-à-dire la décision automatique du
serveur avant tout arbitrage LLM.

Lancement : uv run python probe_consolidation.py
"""

import tempfile

from contextmemory import embeddings, pipeline
from contextmemory.models import Memory
from contextmemory.storage import ParquetStore

SEED_CONTENT = "Mounir aime le café"
SEED_TYPE = "preference"

PROBES = [
    "Mounir aime le café",            # doublon littéral
    "Mounir aime le thé",
    "Mounir aime l'expresso",
    "Mounir n'aime plus le café",     # négation (contradiction temporelle)
    "Mounir n'aime que le café",      # restriction
    "Mounir aime le café et le thé",  # sur-ensemble
    "Mounir déteste le café",         # opposé
    "Mounir habite à Paris",          # peu relié
    "Le ciel est bleu",               # non relié
]


def seed(store: ParquetStore) -> str:
    """Persiste la mémoire de référence via le vrai pipeline."""
    report = pipeline.process_memory(store, Memory(content=SEED_CONTENT, type=SEED_TYPE))
    return report["action"]


def probe(store: ParquetStore, seed_embedding: list[float], text: str) -> tuple:
    """Renvoie (cos_brut, nb_candidats>=seuil, action) sans rien persister."""
    incoming = Memory(content=text, type=SEED_TYPE)
    incoming_embedding = embeddings.embed(text)
    raw_cosine = embeddings.cosine_similarity(incoming_embedding, seed_embedding)
    similar_memories = pipeline.find_similar_memories(store, incoming_embedding)
    action, _ = pipeline.decide_action(incoming, similar_memories, decision=None)
    return raw_cosine, len(similar_memories), action


def main() -> None:
    store = ParquetStore(data_dir=tempfile.mkdtemp(prefix="ctxmem_probe_"))
    seed_embedding = embeddings.embed(SEED_CONTENT)

    print(f"backend              = {embeddings.backend_name()}")
    print(f"SIMILARITY_THRESHOLD = {pipeline.SIMILARITY_THRESHOLD}")
    print(f"seed                 = {SEED_CONTENT!r} -> {seed(store)}\n")

    print(f"{'entrée entrante':32} {'cos_brut':>9} {'>=seuil?':>9}  action")
    print("-" * 78)
    for text in PROBES:
        raw_cosine, candidate_count, action = probe(store, seed_embedding, text)
        passes = "oui" if candidate_count else "non"
        print(f"{text:32} {raw_cosine:9.3f} {passes:>9}  {action}")


if __name__ == "__main__":
    main()
