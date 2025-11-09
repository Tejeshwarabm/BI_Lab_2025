######################################################################
# Program-6: Identify frequent item sets using the Apriori algorithm #
# for a given transaction data set (use Python)                      #
#                                                                    #
######################################################################

# Step 1: Import libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from generator import random_Generate

# Step 2: Create the dataset (list of transactions)
dataset, col = random_Generate(25)

# Step 3: Convert transactions to a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=col)

print("Transaction Data (One-hot Encoded):")
print(df.head(3))

# Step 4: Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 5: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
