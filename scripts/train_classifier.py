import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold , cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Limit column width so it wraps or truncates neatly
pd.set_option('display.max_colwidth', 50) 
# Prevent the table from stretching wider than the terminal screen
pd.set_option('display.width', 100)



df = pd.read_csv("data/scope_dataset.csv")
X_text, y = df['text'], df['is_out_of_scope'].astype(int)



vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english',min_df=2, max_features=200)
X = vectorizer.fit_transform(X_text)
print(f"X shape: {X.shape}")


#X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



y_pred = cross_val_predict(model, X, y, cv=skf)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y, y_pred):.2f}")
print(f"Precision: {precision_score(y, y_pred, zero_division=0):.2f}")
print(f"Recall:    {recall_score(y, y_pred, zero_division=0):.2f}")
print(f"F1 Score:  {f1_score(y, y_pred, zero_division=0):.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

df['predicted'] = y_pred
misclassified = df[df['is_out_of_scope'] != df['predicted']]

print("\n=== MISCLASSIFIED ROWS REPORT ===\n")

for idx, row in misclassified.iterrows():
    print(f"Row Index: {idx}")
    print(f"Predicted Label: {row['predicted']}")
    print("Text:")
    print(row['text'])
    print("-" * 60) 


print("\n--- Inside the Model's Brain ---")
model.fit(X, y) # Fit on the whole dataset just to look at the weights

coefs = pd.Series(model.coef_[0], index=vectorizer.get_feature_names_out())

print("\nTop 10 phrases pushing toward OUT OF SCOPE (1):")
print(coefs.sort_values(ascending=False).head(10))

print("\nTop 10 phrases pushing toward IN SCOPE (0):")
print(coefs.sort_values().head(10))



print("\n--- Per-Subtype Breakdown ---")
for subtype in df['attack_type'].unique():
    mask = df['attack_type'] == subtype
    sub_y = y[mask]
    sub_pred = pd.Series(y_pred, index=df.index)[mask]
    print(f"\n{subtype} (n={mask.sum()}):")
    print(f"  Accuracy: {accuracy_score(sub_y, sub_pred):.2f}")
    print(confusion_matrix(sub_y, sub_pred))