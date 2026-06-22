# Stockage Parquet

UrdWell stocke désormais ses données localement avec PyArrow au format
Parquet. Depuis le dépôt, la dépendance nécessaire s'installe avec :

```powershell
uv sync
```

Aucun serveur de base de données n'est nécessaire. PyTorch,
SentenceTransformers et Transformers ne font pas partie du runtime.

## Modèle d'embedding

Le modèle par défaut est
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Il produit des
vecteurs multilingues de 384 dimensions, adaptés à la comparaison sémantique de
textes français, anglais et d'autres langues.

FastEmbed est le moteur qui exécute ce modèle au format ONNX. Il remplace
SentenceTransformers comme runtime : le chargement est plus léger et ne dépend
pas de PyTorch. Le modèle est téléchargé au premier lancement, puis réutilisé
depuis le cache local.

Configuration disponible :

```powershell
$env:URDWELL_EMBEDDING_BACKEND = "fastembed"
$env:URDWELL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Ces valeurs sont déjà les valeurs par défaut. Le backend `hashing` reste
disponible pour les tests hors ligne, mais il ne doit pas être mélangé avec les
vecteurs de production.

> Un fichier Parquet doit utiliser un seul modèle d'embedding cohérent. La
> migration JSON vers Parquet conserve les vecteurs existants et ne les recalcule
> pas. Si les anciennes données ont été produites avec un autre modèle, elles
> doivent être ré-embeddées avant d'effectuer des recherches fiables.

FastEmbed 0.8 exécute ce modèle avec son mean pooling natif. UrdWell
accepte explicitement ce comportement et masque uniquement l'avertissement de
migration correspondant au démarrage. Les embeddings CLS créés avec FastEmbed
0.5.1 appartiennent à un autre espace vectoriel : ils doivent être régénérés et
ne doivent pas être mélangés avec les nouveaux vecteurs mean pooling.

## Fichiers

- `archive.parquet` contient les échanges bruts : date, rôle, contenu et session.
- `memories.parquet` contient les mémoires structurées et leur embedding dans la
  même ligne : identifiant, contenu, type, source, dates de validité, confiance,
  relation de remplacement et vecteur `float32`.

Les fichiers sont placés dans `URDWELL_DATA_DIR`, ou dans le dossier de
données utilisateur par défaut.

Depuis le renommage en UrdWell 0.3, un dossier ContextMemory existant est
réutilisé automatiquement lorsque le nouveau dossier UrdWell n'existe pas. Les
variables historiques `CONTEXT_MEMORY_DATA_DIR`,
`CONTEXT_MEMORY_EMBEDDING_BACKEND` et `CONTEXT_MEMORY_EMBEDDING_MODEL` restent
acceptées pendant la transition, avec une priorité donnée aux variables
`URDWELL_*`.

## Flux

Lors d'un `save_memory`, FastEmbed produit un vecteur normalisé avec
`paraphrase-multilingual-MiniLM-L12-v2`. Le pipeline compare ce vecteur aux
mémoires actives, choisit `ADD`, `IGNORE` ou `EXPIRE`, puis met à jour
`memories.parquet`.

Lors d'un `search_memory`, le serveur lit les mémoires et leurs vecteurs depuis
ce même fichier, filtre les entrées expirées et calcule localement la similarité
cosinus. Aucune base vectorielle externe n'est utilisée.

`archive_exchange` ajoute logiquement une entrée à l'archive. Le fichier
Parquet complet est actuellement réécrit puis remplacé atomiquement.

## Migration et intégrité

Au premier démarrage, les anciens `archive.jsonl`, `memories.json` ou
`souvenirs.json`, ainsi que `embeddings.json`, sont automatiquement fusionnés et
convertis en Parquet. Les fichiers d'origine sont conservés comme copies de
récupération.

Les écritures utilisent la compression Zstandard, un fichier temporaire, puis
`os.replace`. Un verrou sérialise les cycles lecture-modification-écriture. Les
données restent locales mais ne sont pas chiffrées.
