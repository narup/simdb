# zvec Resources

A curated reading list for building an embedded vector search library in Zig with filtered search and multi-tenancy support.

---

## 1. Vector Search Fundamentals

### What are Embeddings

Understanding embeddings is foundational to vector search.

- **What are Embeddings?** - Vicki Boykis
  - Website: https://vickiboykis.com/what_are_embeddings/
  - PDF: https://raw.githubusercontent.com/veekaybee/what_are_embeddings/main/embeddings.pdf
  - GitHub: https://github.com/veekaybee/what_are_embeddings
  - Comprehensive, practical deep-dive into embeddings from business, engineering, and math perspectives

- **Pinecone Learning Center**
  - https://www.pinecone.io/learn/
  - Good conceptual explanations (ignore the product pitch)

---

## 2. Distance Metrics

How similarity between vectors is measured.

- **Vector Similarity Explained** - Pinecone
  - https://www.pinecone.io/learn/vector-similarity/
  - Covers Euclidean distance, cosine similarity, and dot product

- **What is Similarity Search?** - Pinecone
  - https://www.pinecone.io/learn/what-is-similarity-search/
  - Overview of similarity search concepts

---

## 3. HNSW Algorithm

The core data structure for efficient approximate nearest neighbor search.

### Original Paper

- **"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"** (Malkov & Yashunin, 2018)
  - https://arxiv.org/abs/1603.09320
  - https://arxiv.org/pdf/1603.09320
  - Read sections 1-4 carefully

### Visual Explanations

- **HNSW Explained** - Pinecone
  - https://www.pinecone.io/learn/series/faiss/hnsw/
  - Excellent visual guide with diagrams

- **Nearest Neighbor Indexes for Similarity Search** - Pinecone
  - https://www.pinecone.io/learn/series/faiss/vector-indexes/
  - Compares HNSW with other index types (Flat, LSH, IVF)

### Reference Implementations

- **hnswlib** (original implementation)
  - https://github.com/nmslib/hnswlib
  - C++ reference implementation by the paper authors

- **usearch**
  - https://github.com/unum-cloud/usearch
  - Modern, lean implementation with multiple language bindings

---

## 4. Filtered Vector Search

The key differentiator for zvec - searching with metadata constraints.

### Survey Papers

- **Survey of Filtered Approximate Nearest Neighbor Search** (2025)
  - https://arxiv.org/abs/2505.06501
  - Comprehensive survey of FANNS algorithms and approaches

- **Approximate Nearest Neighbor Search with Window Filters** (2024)
  - https://arxiv.org/pdf/2402.00943
  - Covers range-based filtering approaches

### Industry Approaches

- **Weaviate Filtering**
  - Concepts: https://weaviate.io/developers/weaviate/concepts/filtering
  - Pre-filtering documentation with ACORN strategy
  - Combined inverted index + HNSW approach

- **Qdrant Filtering**
  - Guide: https://qdrant.tech/articles/vector-search-filtering/
  - Documentation: https://qdrant.tech/documentation/concepts/filtering/
  - Search: https://qdrant.tech/documentation/concepts/search/
  - Filterable HNSW: https://qdrant.tech/course/essentials/day-2/filterable-hnsw/

- **ACORN Paper**
  - Referenced in Weaviate and Qdrant docs
  - "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data"

---

## 5. SIMD Optimization

For fast distance calculations.

- **SimSIMD**
  - https://github.com/ashvardanian/SimSIMD
  - Up to 200x faster dot products and similarity metrics
  - Supports f64, f32, f16, i8, and bit vectors
  - Works on AVX2, AVX-512, NEON, SVE, SVE2
  - Read the code to understand SIMD patterns for distance functions

---

## 6. Storage and Persistence

How to structure files and handle persistence.

### SQLite as Inspiration

- **SQLite File Format**
  - https://www.sqlite.org/fileformat.html
  - Detailed specification of page-based file structure

- **SQLite Architecture**
  - https://sqlite.org/arch.html
  - Overview of components and design decisions

- **SQLite as Application File Format**
  - https://sqlite.org/appfileformat.html
  - Philosophy behind single-file databases

### Key Concepts to Learn

- Memory-mapped files for larger-than-RAM indexes
- Page-based storage
- Write-ahead logging (WAL) concept
- File format versioning for backwards compatibility

---

## 7. Multi-Tenancy Patterns

For tenant-isolated search.

- **Qdrant Multitenancy**
  - https://qdrant.tech/articles/vector-search-manuals/
  - Search for multitenancy section

- General patterns:
  - Prefix keys with tenant IDs
  - Partition IDs in metadata
  - Separate index segments per tenant
  - `is_tenant` field indexing

---

## 8. Zig Language

### Official Resources

- **Zig Language Reference**
  - https://ziglang.org/documentation/master/
  - Complete language documentation

- **Zig Learn**
  - https://ziglang.org/learn/
  - Official learning resources

- **Getting Started**
  - https://ziglang.org/learn/getting-started/

### Community Guides

- **zig.guide**
  - https://zig.guide/
  - Structured introduction by Sobeston

- **Learning Zig**
  - https://www.openmymind.net/learning_zig/
  - Practical introduction

- **Learn Zig in Y Minutes**
  - https://learnxinyminutes.com/zig/
  - Quick syntax overview

- **Zighelp**
  - https://zighelp.org/
  - Tutorial-style guide

### Key Topics for This Project

- Allocator patterns: https://ziglang.org/documentation/master/#Choosing-an-Allocator
- Memory mapping with `std.os.mmap`
- Comptime features for generic data structures
- C ABI for creating bindings

---

## 9. Benchmarking & Datasets

### Test Datasets

- **Sift1M** - 1 million 128-dimensional vectors
  - Standard benchmark dataset for ANN search

- **GloVe** - Word embeddings
  - Various dimensions available

- Download small embedding datasets from Hugging Face for development

### Benchmarking

- **ANN Benchmarks**
  - Compare against FAISS, usearch, hnswlib
  - Focus on recall vs QPS tradeoffs

---

## Suggested Reading Order

### Week 1: Foundations
1. Vicki Boykis - "What are Embeddings?" (full book)
2. Pinecone - Vector Similarity Explained
3. Zig basics (zig.guide or Learning Zig)

### Week 2: HNSW Deep Dive
1. Pinecone HNSW visual guide
2. HNSW paper (sections 1-4)
3. Skim hnswlib code structure

### Week 3: Filtering & Storage
1. Filtered search survey paper
2. Weaviate/Qdrant filtering documentation
3. SQLite file format (overview sections)
4. Memory mapping basics

### Week 4: Start Building
1. Begin implementation
2. Reference usearch code as needed
3. SimSIMD for distance function patterns

---

## Project-Specific Notes

### Differentiation from usearch

1. **Filtered search built-in** - not bolted on
2. **Multi-tenancy native** - tenant-aware index structure
3. **Incremental persistence** - append-only log + compaction (vs full rewrite)
4. **SQLite philosophy** - simple, embeddable, single file

### Target API

```zig
// Add with tenant + metadata
index.add(vector, .{
    .id = record_id,
    .tenant = patient_id,
    .metadata = .{ .date = timestamp, .type = .prescription },
});

// Search within tenant, with filter
const results = index.search(query_vector, .{
    .tenant = patient_id,
    .filter = .{ .date_after = last_week },
    .k = 10,
});
```

---

## Additional Resources

### Vector Database Comparisons

- Pinecone, Weaviate, Qdrant, Milvus documentation all provide good conceptual explanations
- Focus on learning concepts, not specific APIs

### Academic Resources

- arXiv has many recent papers on filtered ANN search (2024-2025)
- Search: "filtered approximate nearest neighbor", "hybrid vector search"

---

*Last updated: December 2025*
