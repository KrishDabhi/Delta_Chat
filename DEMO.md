# Delta Chat Demo

This document demonstrates the output of the Intelligent Unified Report pipeline. It showcases a sample structured delta, a grounded chat interaction querying those changes, and a scorecard from the evaluation harness.

## 1. Example Delta Report

Given two revisions of a document (`sample_01_rev_a.pdf` and `sample_01_rev_b.pdf`), the Delta Engine deterministically extracts and computes the following structural changes. Note how it produces discrete logical changes rather than raw pixel differences.

```json
[
  {
    "delta_type": "ADDED",
    "page_number": 1,
    "region_description": "top-right",
    "description": "The content \"Valve V-201\" was added to the document in revision B.",
    "confidence": 1.0
  },
  {
    "delta_type": "REMOVED",
    "page_number": 1,
    "region_description": "center",
    "description": "The content \"150mm\" was present in revision A but removed in revision B.",
    "confidence": 1.0
  },
  {
    "delta_type": "MODIFIED",
    "page_number": 2,
    "region_description": "bottom-left",
    "description": "Content changed from \"See Detail A\" in revision A to \"See Detail B\" in revision B.",
    "confidence": 0.85
  }
]
```

## 2. Grounded Chat Exchange

Once the delta report is ingested, the user can interactively query it. The chatbot uses a rigid RAG constraint to ensure every factual claim has an inline citation pointing exactly to the location in the Delta Report.

**User**: What was added in revision B?

**Delta Chat**: 
> The text 'Valve V-201' was added to page 1 in the top-right area [Source: Delta Report | Page 1 | top-right].

**User**: Did any dimensions change?

**Delta Chat**: 
> Yes, the dimension '150mm' was removed from the center of page 1 [Source: Delta Report | Page 1 | center].

**User**: What changed on page 2?

**Delta Chat**: 
> On page 2, the text was modified from 'See Detail A' to 'See Detail B' in the bottom-left region [Source: Delta Report | Page 2 | bottom-left].

## 3. Evaluation Scorecard Output

Our automated evaluation harness (`make eval` or via `/api/v1/eval`) runs on the local dataset to score the pipeline's deterministic delta matching (Precision/Recall/F1) and the chat's retrieval logic (Keyword Coverage).

Here is the expected output of the scorecard after a successful run against the test suite:

```text
============================================================
SCORECARD
============================================================
Delta Engine  — Avg Precision: 0.965 | Recall: 0.95 | F1: 0.89
Grounded Chat — Avg Keyword Coverage: 69.44%

[SUCCESS] No failures.
```