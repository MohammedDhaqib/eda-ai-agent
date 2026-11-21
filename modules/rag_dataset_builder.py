"""
Build a searchable RAG knowledge base from actual dataset insights.
"""

from modules.insights import generate_dataset_insights
from modules.rag_engine import EDARetrievalEngine


def index_dataset_insights(
    dataframe,
    collection_name="eda_dataset_knowledge"
):
    """
    Generate dataset insights and store them in ChromaDB.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to analyze.

    collection_name : str
        Name of the ChromaDB collection.

    Returns
    -------
    dict
        Indexing status and document information.
    """

    if dataframe is None or dataframe.empty:
        raise ValueError("Cannot index an empty dataset.")

    # Step 1: Generate insights using the existing Python module.
    insights = generate_dataset_insights(dataframe)

    if not insights:
        raise ValueError("No dataset insights were generated.")

    # Step 2: Convert insights into searchable text documents.
    documents = [
        str(insight).strip()
        for insight in insights
        if str(insight).strip()
    ]

    if not documents:
        raise ValueError("No valid insight documents were generated.")

    # Step 3: Initialize the retrieval engine.
    rag_engine = EDARetrievalEngine()

    # Step 4: Reset this dataset's collection before re-indexing.
    # This prevents duplicate document IDs when indexing again.
    try:
        rag_engine.client.delete_collection(
            name=collection_name
        )
    except Exception:
        pass

    # Step 5: Create a fresh collection.
    rag_engine.create_collection(collection_name)

    # Step 6: Store the actual dataset insights.
    document_count = rag_engine.add_documents(documents)

    return {
        "status": "success",
        "collection_name": collection_name,
        "documents_indexed": document_count,
        "insights_generated": len(insights),
        "message": "Dataset insights successfully indexed in ChromaDB."
    }