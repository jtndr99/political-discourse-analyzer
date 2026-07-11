from datasets import load_dataset
from sentence_transformers import CrossEncoder

print("Loading NLI Cross-Encoder model...")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
print("Model loaded!")

print("Fetching dataset...")
ds = load_dataset("pietrolesci/nli_fever", split="dev")

refutes_dataset = ds.filter(lambda r: r['fever_gold_label'] == 'REFUTES')

sample_size = 200
refutes_sample = refutes_dataset.shuffle(seed=42).select(range(sample_size))

premises = refutes_sample['premise']
hypotheses = refutes_sample['hypothesis']

# 2. Format pairs and run NLI predictions in bulk
print(f"Running NLI predictions across {sample_size} REFUTES rows...")
text_pairs = [[p, h] for p, h in zip(premises, hypotheses)]
nli_scores = nli_model.predict(text_pairs)

# 3. Compile the results into a list of dictionaries
compiled_results = []
for p, h, logits in zip(premises, hypotheses, nli_scores):
    contradiction_score = logits[0]  # Index 0 is the contradiction score
    compiled_results.append({
        'score': contradiction_score,
        'premise': p,
        'hypothesis': h
    })

# 4. Sort ASCENDING by score to find the biggest misses
# (The lowest contradiction scores mean the model confidently thought it WASN'T a lie)
sorted_misses = sorted(compiled_results, key=lambda x: x['score'])

# 5. Print the top 15 worst misses for analysis
num_to_print = 15
print(f"\n========================================================")
print(f"   TOP {num_to_print} WORST NLI MISSES (FALSE NEGATIVES)")
print(f"========================================================")
print("These are actual contradictions that the NLI model completely missed.")

for i, item in enumerate(sorted_misses[:num_to_print], 1):
    print(f"\n[Miss #{i}] Contradiction Score: {item['score']:.4f}")
    print(f"Source (Premise):   {item['premise']}")
    print(f"Claim (Hypothesis): {item['hypothesis']}")
    print("-" * 60)