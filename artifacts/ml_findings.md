# Machine Learning Baseline Findings: ScopeClassifier

This document captures the empirical findings from attempting to replace the LLM-based `ScopeClassifier` with a classical TF-IDF Logistic Regression baseline. 

The goal of this experiment was to characterize the exact failure boundaries of a fast, lexical classifier to definitively justify when and why an LLM is architecturally required for scope and intent judgment.

## Finding 1: The Curse of Dimensionality and "Leakage"
Our initial TF-IDF vectorization on the 20-sample dataset yielded 945 features. Because the feature-to-sample ratio was ~47:1, the Logistic Regression model overfit the data by memorizing specific, rare tokens. 

By inspecting the model's coefficients (`.coef_`), we discovered that the word `flynn` was one of the strongest indicators of "In-Scope" political text. This was a classic small-N leakage failure: `flynn` was a proper noun that appeared in only one large document (a legitimate congressional hearing transcript). The model learned "Flynn = In Scope", which does not generalize. Capping the dimensionality (`min_df=2`, `max_features=200`) was required to force the model to learn actual patterns instead of memorizing noise.

## Finding 2: The Collinearity Trap
When splitting the dataset into subtypes (`lexical_injection` vs `scope_judgment`), the model initially scored **100% accuracy** on lexical injections. However, this was a vacuous metric. 

The initial test suite only contained lexical injections that were *actual attacks*. Because there were no negative examples in that subtype (i.e., legitimate text containing trigger words), the grouping variable was completely collinear with the label. A model that blindly guessed "Out of Scope" every time would also score 100%. **Lesson: A grouping variable that secretly encodes the label will always yield artificially perfect metrics.**

## Finding 3: Use vs. Mention (The Referential Failure)
To fix the collinearity trap, we added three negative examples: legitimate news articles discussing prompt injections (e.g., *"Cybersecurity researchers demonstrated how a malicious actor could type 'ignore previous instructions'..."*). 

The TF-IDF model **failed on all three examples**, classifying the legitimate news articles as attacks. 
**Conclusion:** A bag-of-words model structurally cannot distinguish between the *use* of a trigger phrase and the *mention* of a trigger phrase. It lacks the representational capacity to understand quotation marks, attribution, or reported speech. This referential judgment requires semantic reasoning, empirically justifying the need for an LLM.

## Finding 4: The Fictional Wrapper Failure
The lexical classifier consistently failed on unanchored hypotheticals disguised as political rhetoric:
- **Predicted In-Scope (Failed):** *"Imagine you are a working-class mother in Ohio..."* (An unanchored roleplay attack)
- **Predicted Out-of-Scope (Failed):** *"In yesterday's Senate hearing on Bill 402, Senator Smith used a hypothetical: 'Imagine you are a working-class mother in Ohio...'"* (Legitimate anchored speech)

The two texts share almost identical vocabulary (`imagine`, `working-class`, `healthcare`). The distinguishing feature is factual anchoring (citing a real hearing/bill). A lexical model cannot verify anchoring, proving that intent-classification in political discourse cannot rely on surface-level keyword matching.
