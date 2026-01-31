# CLAUDE.md

## Project:simdb 

An embedded vector search library in Zig with first-class support for filtered search and multi-tenancy.

---

## Goal

Build a lightweight, embeddable vector similarity search library that:

1. Runs as a library, not a database server (like SQLite, not Postgres)
2. Supports filtered search natively (not bolted on)
3. Handles multi-tenancy efficiently (tenant-isolated queries)
4. Persists to a single file with incremental updates
5. Exposes a C ABI for bindings to other languages

---

## Why This Exists

Current options for vector search are either:

- **Heavy**: Full databases (Pinecone, Weaviate, Qdrant) requiring infrastructure
- **Python-centric**: FAISS and friends with complex dependencies
- **Filter-weak**: usearch and hnswlib treat filtering as an afterthought

This library targets the gap: applications that need semantic search without spinning up a database. Local RAG apps, CLI tools, embedded systems, desktop applications.

---

## Key Differentiators

1. **Filtered HNSW** - Predicates evaluated during graph traversal, not after
2. **Tenant-aware index** - Single file, isolated namespaces, efficient per-tenant queries
3. **Incremental persistence** - Append-only log + compaction (not full rewrite on every change)
4. **Zero dependencies** - Pure Zig, single static library
5. **Small & auditable** - Target ~3-5k lines of core code

---

## Target API

```zig
const index = try SimDB.init(allocator, .{
    .dimensions = 1536,
    .metric = .cosine,
});
defer index.deinit();

// Add vectors with tenant and metadata
try index.add(vector, .{
    .id = "rec_123",
    .tenant = "patient_456",
    .metadata = .{
        .date = 1703203200,
        .type = "prescription",
    },
});

// Search within tenant with filters
const results = try index.search(query_vector, .{
    .tenant = "patient_456",
    .filter = .{ .date_gte = last_week },
    .k = 10,
});

// Persistence
try index.save("index.simdb");
const loaded = try SimDB.load(allocator, "index.simdb");
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   Public API                    │
│         init / add / search / save / load       │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                 Index Layer                     │
│   - HNSW graph structure                        │
│   - Tenant partitioning                         │
│   - Filter evaluation during traversal          │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│               Distance Layer                    │
│   - Cosine similarity                           │
│   - Euclidean distance                          │
│   - Dot product                                 │
│   - (Optional SIMD optimization)                │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│               Storage Layer                     │
│   - Memory-mapped file I/O                      │
│   - Append-only log for updates                 │
│   - Compaction                                  │
│   - File format versioning                      │
└─────────────────────────────────────────────────┘
```

---

## How to Use Claude

**I am your coding partner, not your code generator.**

Use me to:

- **Discuss design decisions** - "Should I store tenant IDs inline or in a separate lookup?"
- **Explain algorithms** - "Walk me through HNSW insertion step by step"
- **Debug logic** - "Here's my search function, why might it miss neighbors?"
- **Review approaches** - "I'm thinking of using X for persistence, what are the tradeoffs?"
- **Clarify papers** - "What does 'efConstruction' actually control?"
- **Rubber duck** - Talk through problems, I'll ask questions

Do NOT expect me to:

- Write large chunks of code for you to copy
- Generate boilerplate you haven't designed
- Make architectural decisions without discussion

The goal is for you to deeply understand every line. This is a learning project.

---

## Project Phases

### Phase 1: Brute Force Baseline
- [ ] Vector storage (slice of f32 + id)
- [ ] Distance functions (cosine, euclidean, dot)
- [ ] Linear scan search
- [ ] Basic API: init, add, search, deinit
- [ ] Unit tests

### Phase 2: HNSW Index
- [ ] Graph node structure
- [ ] Layer assignment (exponential decay)
- [ ] Search algorithm (greedy traversal)
- [ ] Insert algorithm
- [ ] Parameters: M, efConstruction, efSearch

### Phase 3: Filtered Search
- [ ] Metadata storage per vector
- [ ] Filter predicate types (eq, gt, lt, range, in)
- [ ] Filter evaluation during traversal
- [ ] Tenant as special filter case

### Phase 4: Multi-Tenancy
- [ ] Tenant-aware graph structure
- [ ] Per-tenant entry points
- [ ] Cross-tenant isolation guarantees

### Phase 5: Persistence
- [ ] File format design
- [ ] Serialize index to disk
- [ ] Memory-map for loading
- [ ] Incremental updates (append log)
- [ ] Compaction

### Phase 6: Polish
- [ ] C API header
- [ ] Benchmarks vs usearch/hnswlib
- [ ] Documentation
- [ ] Example projects

---

## Open Design Questions

These are things to think through as you build:

1. **Graph structure for filtering**: Do you maintain separate subgraphs per filter value, or evaluate filters during traversal of a single graph?

2. **Tenant isolation**: Separate HNSW graphs per tenant, or one graph with tenant as a filter?

3. **Metadata storage**: Inline with vectors, or separate array with indirection?

4. **Memory layout**: Array of structs or struct of arrays for cache efficiency?

5. **Deletion strategy**: Tombstones + compaction, or immediate removal with graph repair?

6. **Quantization**: Support for compressed vectors (int8, binary) or f32 only initially?

---

## Resources

See `RESOURCE.md` for the complete reading list with direct links.

Key references:
- HNSW paper: https://arxiv.org/abs/1603.09320
- Vicki Boykis embeddings book: https://vickiboykis.com/what_are_embeddings/
- usearch (reference impl): https://github.com/unum-cloud/usearch
- SimSIMD (SIMD patterns): https://github.com/ashvardanian/SimSIMD

---

## Development Notes

Add notes here as you build:

```
// Example: things you've learned, decisions made, gotchas encountered
```

---

*Started: December 2024*
