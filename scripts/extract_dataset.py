import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.adversarial_test_runner import test_cases

df = pd.DataFrame(test_cases)

df['text'] = df['payload']

def determine_scope(row):
    if pd.notna(row.get('expected_is_out_of_scope')):
        return bool(row['expected_is_out_of_scope'])
    if pd.notna(row.get('expected_is_safe')):
        return not bool(row['expected_is_safe'])
    return True

df['is_out_of_scope'] = df.apply(determine_scope, axis=1)

df_final = df[['text', 'is_out_of_scope', 'attack_type']]

missing = df[df['expected_is_out_of_scope'].isna() & df['expected_is_safe'].isna()]
print(missing[['payload']] if len(missing) else "no missing-label rows")

os.makedirs("data", exist_ok=True)
df_final.to_csv("data/scope_dataset.csv", index=False)

print("Pandas dataset successfully saved!")

