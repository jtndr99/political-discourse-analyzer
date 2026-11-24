# Phase 2 — Grounding Evaluation: Embedding Similarity vs. NLI Entailment

**Goal:** Replace GroundingEvaluator's LLM-based judgment (claim supported by source article? yes/no) with a cheaper, non-LLM method — embedding similarity and/or NLI entailment — and understand *where* each approach breaks, not just how accurate it is on average.

**Dataset:** [`pietrolesci/nli_fever`](https://huggingface.co/datasets/pietrolesci/nli_fever), dev split. Note: field names are misleading for this use case — `premise` is the short claim, `hypothesis` is the longer source/evidence text. Filtered to `SUPPORTS` / `REFUTES` only; `NOT ENOUGH INFO` excluded from this phase (silence-detection is a different problem from contradiction-detection and would need separate handling).

---

## Method 1: Cosine Similarity (Bi-Encoder, `all-MiniLM-L6-v2`)

Encoded claim and source separately, computed cosine similarity. n=50 (20 SUPPORTS / 30 REFUTES).

| | Mean | Range |
|---|---|---|
| SUPPORTS | 0.757 | 0.505 – 0.927 |
| REFUTES | 0.637 | -0.007 – 0.930 |

**Finding:** Means differ, but the ranges almost fully overlap. No single threshold separates the classes — a large fraction of REFUTES cases score as high as genuine SUPPORTS cases.

**Ruled out:** Hypothesized this was driven by paragraph-length "dilution" (long hypothesis text averaging out one wrong clause). Tested directly — correlation between REFUTES hypothesis sentence count and score was **0.087** (essentially zero). Dilution is not the mechanism. Killed this hypothesis rather than forcing the data to fit it.

**Actual mechanism:** Cosine similarity tracks **lexical/topical overlap**, not **truth value**. A claim that reuses most of the source's vocabulary scores high even when it contradicts the source, because the contradiction is often carried by one word or clause (a category, a name, a negation) that barely moves an averaged sentence embedding. Structurally the same class of blind spot as the Phase 1 TF-IDF quotation-blindness finding — surface word-overlap is being used as a truth proxy, and it's a bad one.

---

## Method 2: NLI Cross-Encoder (`cross-encoder/nli-deberta-v3-base`)

Same claim/source pairs, cross-encoder entailment model, read off the contradiction logit.

| | Mean | Range |
|---|---|---|
| SUPPORTS (n=20) | -2.46 | -7.54 – 6.38 |
| REFUTES (n=30) | 1.47 | -3.92 – 7.70 |

**Finding:** Clear separation in means — a large improvement over cosine similarity. But ranges still overlap at the edges, so this is not a fully solved problem either. Verified the contradiction-logit index (index 0) against a hand-built known-contradiction pair before trusting the aggregate numbers.

### Failure analysis: bottom 15 REFUTES misses (n=200 sample)

Sorted 200 REFUTES rows by NLI contradiction score ascending, read the 15 worst false negatives (real contradictions NLI scored as non-contradictions). They split into four distinct patterns:

**A. Negation / polarity flip (clearest, most consistent pattern — 5/15)**
Source and claim share nearly all vocabulary; the entire contradiction rides on one negation or polarity word ("not," "outside of" vs "in," "lacks" vs "only... live there," "stand-alone" vs "third in a trilogy"). Negation is lexically cheap but semantically total, and an averaged/cross-attended representation doesn't weight it enough relative to the surrounding shared context.

**B. Entity/attribute substitution (3/15)**
A name, title, or attribute is swapped while everything else stays intact (fabricated film title, wrong show name, "two Canadians" vs. two Americans). Same mechanism as the Phase 1 collinearity/leaked-token issue — one swapped token, dense correct context around it.

**C. Temporal / world-knowledge contradiction (3/15)**
Requires knowing *when* something existed (a "king" claim about a source describing a 10th-century kingdom; "Enlightenment" attributed to a 15th-century figure; an anachronistic word choice for an 1820s expedition). This is not really an embeddings problem — it needs external knowledge about time periods that no sentence-level model has access to from the text alone. Genuinely harder than A or B; likely needs retrieval augmentation or a knowledge-grounded model, not a better NLI model.

**D. Dataset label noise (4/15)**
Several "misses" were not actually clean contradictions — they were unsupported additions the source never addressed, or claims that were arguably consistent with the source despite the REFUTES label. Any accuracy number for this pipeline should account for this noise floor; it's a dataset construction issue, not a model failure, consistent with the Phase 1 lesson that not every "failure" is the model's fault.

---

## Summary

| Method | Verdict |
|---|---|
| TF-IDF (Phase 1) | Blind to attribution/quotation structure |
| Cosine similarity | Blind to contradiction when topical overlap is high, regardless of text length |
| NLI cross-encoder | Strong on explicit negation-based contradictions; weaker on dense entity substitution; largely fails on temporal/world-knowledge contradictions |

No single cheap method fully replaces the LLM judge. This mirrors why production grounding/fact-checking systems typically chain multiple signals (retrieval + NLI + LLM fallback) rather than relying on one similarity score — a conclusion reached here empirically rather than by assumption.

## Not done / deferred

- Quantify what fraction of the 200-row REFUTES sample falls into each bucket (A/B/C/D) rather than eyeballing the worst 15.
- `NOT ENOUGH INFO` class not yet handled — cosine/NLI both assume a stance exists to detect; silence-detection needs a different approach (e.g. checking whether *any* source sentence is topically relevant at all, separate from whether it entails/contradicts the claim).
- Sentence-level max-similarity (embed source sentences individually, take max match to claim, rather than whole-document embedding) was proposed but not tested — the dilution hypothesis it was meant to address turned out to be false, so this is now lower priority.
- Not yet re-run against real GroundingEvaluator/PolitiLens articles — this phase used FEVER (Wikipedia claims) as a fast, pre-labeled proxy. Worth a small sanity check against actual project articles before treating these findings as final.
