# Assistant Personnel RAG Local (Llama 3 / Mistral)

Assistant Intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation). Cette application permet de discuter avec ses propres documents (PDF, TXT) de manière totalement privée et locale, sans qu'aucune donnée ne quitte votre ordinateur.

## Points Forts

- 100% Local : Utilisation d'Ollama pour l'inférence (confidentialité garantie).

- Multi-Modèles : Compatible avec Llama 3, Mistral et Phi-3.

- Ingestion Dynamique : Upload de documents directement via l'interface Streamlit.

- Mémoire Vectorielle : Utilisation de ChromaDB pour un stockage efficace et une recherche de similarité rapide.

## Architecture Technique

Le projet repose sur la stack "LangChain" pour orchestrer les données :
- Ingestion : Les documents sont découpés en "chunks" via RecursiveCharacterTextSplitter.

- Embedding : Transformation du texte en vecteurs avec le modèle mxbai-embed-large.

- Vector Store : Stockage permanent dans une base ChromaDB.

- Retrieval : Lors d'une question, le système récupère les $k=3$ extraits les plus pertinents.Génération : Le LLM (Llama 3 ou Mistral) génère une réponse basée exclusivement sur le contexte fourni.

## Installation

1. Prérequis

Python 3.10+

Ollama installé et lancé.

2. Modèles nécessaires
Ouvrez un terminal et récupérez les modèles utilisés :

```bash
ollama pull llama3:8b-instruct-q4_0

ollama pull mxbai-embed-large
```

3. Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation
Lancer l'application :

```bash
streamlit run 5_app_visuelle.py
```

Ajouter des documents : Utilisez la barre latérale pour uploader vos PDF.

Discuter : Posez vos questions. L'IA utilisera vos documents pour vous répondre en priorité.

![interface](interface.png)