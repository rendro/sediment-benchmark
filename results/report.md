# Memory System Benchmark Results

*1000 memories, 200 queries, 50 temporal sequences · 3 systems compared*

## Summary

| Metric | ChromaDB | Mem0 | Sediment |
| --- | --- | --- | --- |
| **Recall@1** | 47.0% | 47.0% | **50.0%** |
| **Recall@3** | **69.0%** | **69.0%** | **69.0%** |
| **Recall@5** | **78.5%** | **78.5%** | 77.5% |
| **Recall@10** | **90.0%** | **90.0%** | 89.5% |
| **MRR** | 60.8% | 60.8% | **61.9%** |
| **nDCG@5** | **59.9%** | **59.9%** | 58.7% |

*Latency: ChromaDB store p50=703.3ms, recall p50=696.1ms | Mem0 store p50=16.0ms, recall p50=12.3ms | Sediment store p50=170.7ms, recall p50=571.6ms*

## Retrieval Quality by Category

### Recall@5 by Category

| Category | ChromaDB | Mem0 | Sediment |
| --- | --- | --- | --- |
| `architecture` | 71.4% | 71.4% | **82.9%** |
| `code_patterns` | **88.6%** | **88.6%** | **88.6%** |
| `cross_project` | **68.8%** | **68.8%** | 65.6% |
| `project_facts` | **75.8%** | **75.8%** | 60.6% |
| `troubleshooting` | **81.2%** | **81.2%** | 78.1% |
| `user_preferences` | 84.9% | 84.9% | **87.9%** |

### MRR by Category

| Category | ChromaDB | Mem0 | Sediment |
| --- | --- | --- | --- |
| `architecture` | 55.8% | 55.8% | **66.8%** |
| `code_patterns` | **71.1%** | **71.1%** | 70.4% |
| `cross_project` | 47.1% | 47.1% | **50.7%** |
| `project_facts` | **59.2%** | **59.2%** | 51.6% |
| `troubleshooting` | 62.8% | 62.8% | **63.2%** |
| `user_preferences` | **67.9%** | **67.9%** | 67.6% |

## Temporal Correctness

| Metric | ChromaDB | Mem0 | Sediment |
| --- | --- | --- | --- |
| **Recency@1** | 14.0% | 14.0% | **100.0%** |
| **Recency@3** | 94.0% | 94.0% | **100.0%** |
| **MRR** | 48.8% | 48.8% | **100.0%** |
| **Mean Rank** | 2.38 | 2.38 | **1.00** |

*Latency: ChromaDB store p50=698.1ms, recall p50=696.6ms | Mem0 store p50=12.3ms, recall p50=6.6ms | Sediment store p50=17.2ms, recall p50=23.1ms*

## Dedup / Consolidation

| Metric | ChromaDB | Mem0 | Sediment |
| --- | --- | --- | --- |
| **Consolidation Rate** | 0.0% | 0.0% | **99.0%** |
| **Recall After Dedup** | **100.0%** | **100.0%** | **100.0%** |
| **Stored / Expected** | 100 / 100 | 100 / 100 | 1 / 100 |

*Latency: ChromaDB store p50=698.4ms, recall p50=696.0ms | Mem0 store p50=16.1ms, recall p50=9.5ms | Sediment store p50=42.3ms, recall p50=85.0ms*

## Latency

### Store Latency

| Metric | ChromaDB | Mem0 | Sediment |
| --- | --- | --- | --- |
| **p50** | 696.4ms | **16.4ms** | 48.5ms |
| **p95** | 726.1ms | **18.7ms** | 62.0ms |
| **p99** | 728.5ms | **19.6ms** | 88.0ms |
| **Mean** | 700.2ms | **16.4ms** | 48.0ms |
| **Min** | 684.1ms | **12.7ms** | 22.1ms |
| **Max** | 747.9ms | **21.9ms** | 91.2ms |
| **Samples** | 100 | 100 | 100 |

### Recall Latency

| Metric | ChromaDB | Mem0 | Sediment |
| --- | --- | --- | --- |
| **p50** | 694.1ms | **7.6ms** | 103.3ms |
| **p95** | 727.8ms | **11.9ms** | 109.3ms |
| **p99** | 745.5ms | **12.4ms** | 131.7ms |
| **Mean** | 700.1ms | **8.1ms** | 104.7ms |
| **Min** | 676.8ms | **5.7ms** | 98.0ms |
| **Max** | 759.1ms | **12.6ms** | 148.7ms |
| **Samples** | 50 | 50 | 50 |


## Methodology

- **ChromaDB**: macOS-26.2-arm64-arm-64bit-Mach-O, Python 3.13.11
  - Run at: 2026-02-07T03:16:08.454631+00:00
- **Mem0**: macOS-26.2-arm64-arm-64bit-Mach-O, Python 3.13.11
  - Run at: 2026-02-07T03:36:04.436718+00:00
- **Sediment**: macOS-26.2-arm64-arm-64bit-Mach-O, Python 3.13.11
  - Run at: 2026-02-07T19:51:44.881346+00:00

*Generated: 2026-02-07 20:15 UTC*
