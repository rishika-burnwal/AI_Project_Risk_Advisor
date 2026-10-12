"""Project-isolated ChromaDB storage for ProjectIQ."""
import hashlib
import re
import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="project_documents")
_active_project_id = "default"


def set_active_project(project_id):
    global _active_project_id
    _active_project_id = str(project_id or "default")


def get_active_project_id():
    return _active_project_id


def _project_filter(project_id=None, extra=None):
    pid = str(project_id or _active_project_id)
    clauses = [{"project_id": pid}]
    if extra:
        clauses.append(extra)
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _migrate_legacy_records():
    """Keep existing records visible in the original/default project."""
    try:
        rows = collection.get(include=["metadatas"])
        ids = rows.get("ids", [])
        metadatas = rows.get("metadatas", [])
        for rid, meta in zip(ids, metadatas):
            if meta is not None and "project_id" not in meta:
                updated = dict(meta)
                updated["project_id"] = "default"
                collection.update(ids=[rid], metadatas=[updated])
    except Exception:
        # Do not prevent app startup if an older ChromaDB version differs.
        pass


_migrate_legacy_records()


def clear_collection(project_id=None):
    """Clear only the selected project's records; never delete other projects."""
    pid = str(project_id or _active_project_id)
    rows = collection.get(where={"project_id": pid}, include=[])
    ids = rows.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def store_documents(chunks, embeddings, filename, project_id=None):
    pid = str(project_id or _active_project_id)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(filename))
    prefix = hashlib.sha1(pid.encode("utf-8")).hexdigest()[:12]
    ids = [f"{prefix}_{safe_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": str(filename), "project_id": pid, "chunk_index": int(i)}
        for i, _ in enumerate(chunks)
    ]
    # Upsert makes reprocessing the same filename safe and avoids duplicate IDs.
    # Chunk index metadata lets retrieval preserve document order where available.
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def get_document_count(project_id=None):
    """Return chunk count using get(where=...), compatible with ChromaDB versions
    whose Collection.count() method does not accept a where argument.
    """
    rows = collection.get(
        where={"project_id": str(project_id or _active_project_id)},
        include=[]
    )
    return len(rows.get("ids", []))


def get_unique_document_count(project_id=None):
    rows = collection.get(
        where={"project_id": str(project_id or _active_project_id)},
        include=["metadatas"]
    )
    return len({m.get("source") for m in rows.get("metadatas", []) if m and m.get("source")})


def get_project_sources(project_id=None):
    rows = collection.get(where={"project_id": str(project_id or _active_project_id)}, include=["metadatas"])
    return sorted({m.get("source") for m in rows.get("metadatas", []) if m and m.get("source")})


def search_documents(query_embedding, n_results=3, sources=None, project_id=None):
    count = get_document_count(project_id)
    if count == 0:
        return []
    filters = [{"project_id": str(project_id or _active_project_id)}]
    if sources:
        filters.append({"source": {"$in": list(sources)}})
    where = filters[0] if len(filters) == 1 else {"$and": filters}
    results = collection.query(query_embeddings=[query_embedding], n_results=min(n_results, count), where=where)
    docs = results.get("documents", [[]])
    return docs[0] if docs and docs[0] else []


def get_documents_for_sources(sources=None, max_chunks=30, project_id=None):
    """Return chunks from only the active project, balanced across source files.

    Older records without chunk_index remain supported. When a limit is necessary,
    round-robin across source files so a long document cannot crowd out all others.
    """
    count = get_document_count(project_id)
    if count == 0:
        return []
    filters = [{"project_id": str(project_id or _active_project_id)}]
    if sources:
        filters.append({"source": {"$in": list(sources)}})
    where = filters[0] if len(filters) == 1 else {"$and": filters}
    rows = collection.get(where=where, include=["documents", "metadatas"])
    documents = rows.get("documents") or []
    metadatas = rows.get("metadatas") or []
    grouped = {}
    for i, document in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        source = str(meta.get("source", "unknown"))
        try:
            chunk_index = int(meta.get("chunk_index", i))
        except (TypeError, ValueError):
            chunk_index = i
        grouped.setdefault(source, []).append((chunk_index, i, document))
    for source in grouped:
        grouped[source].sort(key=lambda item: (item[0], item[1]))
    ordered = []
    sources_sorted = sorted(grouped)
    depth = 0
    while len(ordered) < max_chunks:
        added = False
        for source in sources_sorted:
            if depth < len(grouped[source]):
                ordered.append(grouped[source][depth][2])
                added = True
                if len(ordered) >= max_chunks:
                    break
        if not added:
            break
        depth += 1
    return ordered


def delete_source_documents(filename, project_id=None):
    """Delete all chunks belonging to one source file in the selected project."""
    pid = str(project_id or _active_project_id)
    rows = collection.get(
        where={"$and": [{"project_id": pid}, {"source": str(filename)}]},
        include=[]
    )
    ids = rows.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
