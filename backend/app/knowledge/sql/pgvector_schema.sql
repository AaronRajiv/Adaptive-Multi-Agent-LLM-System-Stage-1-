-- Persistent pgvector schema with HNSW vector indexing, full-text search, and article-path lookup.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for high-speed approximate nearest neighbor cosine similarity search
CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_cosine_idx
ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index for PostgreSQL full-text keyword (BM25 equivalent) search
CREATE INDEX IF NOT EXISTS knowledge_chunks_fts_idx
ON knowledge_chunks USING gin (to_tsvector('english', content));

-- Functional index for fast exact & prefix article path lookups (e.g. metadata->>'article_path')
CREATE INDEX IF NOT EXISTS knowledge_chunks_article_idx
ON knowledge_chunks ((metadata->>'article_path'));

-- Index on document_id for per-document filtering and cascaded purges
CREATE INDEX IF NOT EXISTS knowledge_chunks_document_idx
ON knowledge_chunks (document_id);
