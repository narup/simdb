# zvec

An embedded vector search library in Zig. Like SQLite, but for vectors.

## Goals

- Embeddable library (single static lib, C ABI)
- Fast similarity search with HNSW
- Filtered search evaluated during traversal
- Single-file persistence with incremental updates
- Zero dependencies

## Roadmap

### Week 1: Brute Force Foundation

**Milestone 1.1: Distance functions**
- [x] Create `src/distance.zig`
- [x] Implement `dotProduct(a: []const f32, b: []const f32) f32`
- [ ] Implement `cosineSimilarity(a, b) f32`
- [ ] Implement `euclideanDistance(a, b) f32`
- [ ] Write tests with hand-calculated values

**Milestone 1.2: Vector store**
- [ ] Create `src/index.zig`
- [ ] Struct that holds vectors + IDs
- [ ] `add(id, vector)` — store a vector
- [ ] `search(query, k)` — brute force k-nearest neighbors
- [ ] Test: insert vectors, query, verify self-retrieval

**Milestone 1.3: CLI**
- [ ] `zvec add <id> <vector...>`
- [ ] `zvec search <vector...> --k=5`
- [ ] In-memory only (no persistence yet)

### Week 2: Persistence

**Milestone 2.1: Save/Load**
- [ ] Design binary file format (header + vectors + IDs)
- [ ] `index.save(path)`
- [ ] `index.load(allocator, path)`
- [ ] Test: save → load → search returns identical results

### Week 3-4: HNSW

**Milestone 3.1: Graph structure**
- [ ] Node: vector + connections per layer
- [ ] Layer assignment (exponential decay probability)
- [ ] Entry point tracking

**Milestone 3.2: Search algorithm**
- [ ] Greedy traversal from entry point
- [ ] Layer-by-layer descent
- [ ] efSearch parameter

**Milestone 3.3: Insert algorithm**
- [ ] Find neighbors at each layer
- [ ] Connect with M neighbors
- [ ] Update entry point if highest layer

**Milestone 3.4: Verify recall**
- [ ] Compare HNSW results vs brute force
- [ ] Target: 95%+ recall@10 on test dataset

### Future

- [ ] Filtered search (predicates during traversal)
- [ ] Multi-tenancy (tenant-isolated queries)
- [ ] Product quantization (memory optimization)
- [ ] C API header for bindings

## Resources

- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Visual Guide to HNSW](https://cfu288.com/blog/2024-05_visual-guide-to-hnsw/)
- [ANN Benchmarks](http://ann-benchmarks.com)

## Test Data

```bash
# GloVe word vectors (400k words, 50-300 dims)
wget https://nlp.stanford.edu/data/glove.6B.zip

# SIFT1M benchmark (1M vectors, 128 dims)
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
```

## Building

```bash
zig build        # compile
zig build run    # run
zig build test   # run tests
```
