from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer, util

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

sentence = "The working class mother in Ohio cannot afford healthcare."

embedding = model.encode(sentence)

# print(f"\nThe sentence was converted into a vector with {len(embedding)} numbers.")
# print("Here are the first 5 numbers:")
# print(embedding[:5])

# source_text = "The city council of Springfield voted unanimously to deploy tactical military tanks to enforce the new 8 PM teenage curfew."
# grounded_claim = "Springfield is using military vehicles to keep teenagers off the streets at night."
# hallucinated_claim = "The mayor of Springfield resigned in disgrace after stealing from the pension fund."

# source_embedding = model.encode(source_text)
# grounded_embedding = model.encode(grounded_claim)
# hallucinated_embedding = model.encode(hallucinated_claim)


# score_grounded = util.cos_sim(source_embedding, grounded_embedding)
# score_hallucinated = util.cos_sim(source_embedding, hallucinated_embedding)

# print(f"Grounded Score: {score_grounded.item():.4f}")
# print(f"Hallucinated Score: {score_hallucinated.item():.4f}")


# def evaluate_grounding(source, claims_list, threshold=0.4):
#     source_emb = model.encode(source)
#     hallucinations = []
#     for claim in claims_list:
#         # 1. Encode the claim
#         enc_claim = model.encode(claim)
#         # 2. Calculate the cosine similarity using util.cos_sim
#         score_g = util.cos_sim(source_emb, enc_claim).item()
#         # 3. Check if the score is BELOW the threshold.
#         # If it is, append it to the hallucinations list!
#         if score_g < threshold:
#             hallucinations.append(claim)
#         pass
#     return hallucinations

# # Test it out!
# claims_to_check = [grounded_claim, hallucinated_claim]
# bad_claims = evaluate_grounding(source_text, claims_to_check)
# print(f"\nHallucinations detected: {bad_claims}")

ds = load_dataset("pietrolesci/nli_fever", split="dev")
filtered = ds.filter(lambda r: r['fever_gold_label'] in ('SUPPORTS', 'REFUTES'))
sample = filtered.shuffle(seed=42).select(range(50))

for row in sample.select(range(5)):
    print(row['fever_gold_label'], '|', row['premise'][:80], '|', row['hypothesis'][:80])

list_1 = sample['premise']
list_2 = sample['hypothesis']


list1_enc = model.encode(list_1, convert_to_tensor=True)
list2_enc = model.encode(list_2, convert_to_tensor=True)

similarity_matrix = util.cos_sim(list1_enc, list2_enc)
scores = similarity_matrix.diag()


supports_scores = []
refutes_scores = []

labels = sample['fever_gold_label']
premises = sample['premise']
hypotheses = sample['hypothesis']

for label, score_tensor, prem, hypo in zip(labels, scores, premises, hypotheses):
    score = score_tensor.item()

    if label == 'SUPPORTS':
        supports_scores.append(score)
    else: refutes_scores.append({'score': score, 'premise': prem, 'hypothesis': hypo})

avg_supports = sum(supports_scores) / len(supports_scores)
avg_refutes = sum([item['score'] for item in refutes_scores]) / len(refutes_scores)

print(f"Average SUPPORTS Score: {avg_supports:.4f}")
print(f"Average REFUTES Score: {avg_refutes:.4f}")

print(f"SUPPORTS range: {min(supports_scores):.4f} to {max(supports_scores):.4f}")
min_refute = min([item['score'] for item in refutes_scores])
max_refute = max([item['score'] for item in refutes_scores])

print(f"REFUTES range: {min_refute:.4f} to {max_refute:.4f}")


print(len(supports_scores))
print("----------------------------")
print(len(refutes_scores))
print("----------------------------")


sorted_refutes = sorted(refutes_scores, key=lambda x: x['score'], reverse=True)
top_3_refutes = sorted_refutes[:3]

print("--- Top 3 REFUTES Rows by Similarity Score ---")
for i, item in enumerate(top_3_refutes, 1):
    print(f"\n[Rank {i}] Score: {item['score']:.4f}")
    print(f"Source (Premise): {item['premise']}")
    print(f"Claim (Hypothesis): {item['hypothesis']}")


# 1. Initialize a list to hold sentence counts
refute_sentence_counts = []

# 2. Loop through your existing refutes_scores list of dicts
for item in refutes_scores:
    hypothesis_text = item['hypothesis']
    # Split the text by periods to estimate the number of sentences
    # We strip empty strings in case there are trailing periods
    sentences = [s for s in hypothesis_text.split('.') if s.strip()]
    sentence_count = len(sentences)
    refute_sentence_counts.append(sentence_count)

# 3. Analyze the results
avg_sentences = sum(refute_sentence_counts) / len(refute_sentence_counts)
max_sentences = max(refute_sentence_counts)
min_sentences = min(refute_sentence_counts)

print("--- REFUTES Sentence Count Distribution ---")
print(f"Average sentences per hypothesis: {avg_sentences:.1f}")
print(f"Range of sentences: {min_sentences} to {max_sentences}")


import numpy as np

scores_list = [item['score'] for item in refutes_scores]
sentence_counts = refute_sentence_counts

correlation = np.corrcoef(sentence_counts, scores_list)[0, 1]
print(f"Correlation between sentence count and score: {correlation:.4f}")

# also eyeball it directly, sorted
paired = sorted(zip(sentence_counts, scores_list), key=lambda x: x[0])
for count, score in paired:
    print(f"Sentences: {count}, Score: {score:.4f}")


print("Loading NLI Cross-Encoder...")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
print("NLI Model loaded!")

premise_test = "Watertown is legally a town in Middlesex County."
hypothesis_test = "Watertown is a city in Middlesex County."

# Cross-Encoders take pairs directly as a list of tuples or lists
prediction = nli_model.predict([premise_test, hypothesis_test])

print("\nRaw logits/scores output:")
print(prediction)


text_pairs = [[p, h] for p, h in zip(sample['premise'], sample['hypothesis'])]

nli_scores = nli_model.predict(text_pairs)
nli_supports_contradiction_scores = []
nli_refutes_contradiction_scores = []
labels = sample['fever_gold_label']


for label, logits in zip(labels, nli_scores):
    contradiction_score = logits[0]  # Index 0 is the contradiction score
    if label == 'SUPPORTS':
        nli_supports_contradiction_scores.append(contradiction_score)
    elif label == 'REFUTES':
        nli_refutes_contradiction_scores.append(contradiction_score)

# 5. Calculate the new averages
avg_supports_con = sum(nli_supports_contradiction_scores) / len(nli_supports_contradiction_scores)
avg_refutes_con = sum(nli_refutes_contradiction_scores) / len(nli_refutes_contradiction_scores)

print("\n--- NLI CROSS-ENCODER CONTRADICTION SCORES ---")
print(f"Average Contradiction score for SUPPORTS rows: {avg_supports_con:.4f}")
print(f"Average Contradiction score for REFUTES rows: {avg_refutes_con:.4f}")

# 1. Calculate min and max for NLI contradiction scores
min_nli_support = min(nli_supports_contradiction_scores)
max_nli_support = max(nli_supports_contradiction_scores)

min_nli_refute = min(nli_refutes_contradiction_scores)
max_nli_refute = max(nli_refutes_contradiction_scores)

print(f"NLI SUPPORTS range: {min_nli_support:.4f} to {max_nli_support:.4f}")
print(f"NLI REFUTES range: {min_nli_refute:.4f} to {max_nli_refute:.4f}")

print("\n--- BEFORE VS AFTER: TOP 3 COSINE OUTLIERS ---")

for i, item in enumerate(top_3_refutes, 1):
    prem = item['premise']
    hypo = item['hypothesis']
    old_cosine_score = item['score']
    
    # Run the NLI model on this specific pair
    nli_output = nli_model.predict([prem, hypo])
    nli_contradiction_score = nli_output[0] # Grab the contradiction logit
    
    print(f"\n[Outlier Case {i}]")
    print(f"Source: {prem}")
    print(f"Claim:  {hypo}")
    print(f"-> Old Cosine Score (Bi-Encoder):   {old_cosine_score:.4f}")
    print(f"-> New Contradiction Score (Cross): {nli_contradiction_score:.4f}")