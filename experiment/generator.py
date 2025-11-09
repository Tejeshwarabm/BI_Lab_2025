import numpy as np

def random_Generate(n_transactions):
    # Generate synthetic transaction data for demonstration
    np.random.seed(42)

    # Sample products
    products = ['Milk', 'Bread', 'Butter', 'Cheese', 'Eggs', 'Yogurt', 'Apple', 'Banana',
                'Coffee', 'Tea', 'Sugar', 'Flour', 'Pasta', 'Rice', 'Shampoo', 'Sunscreen']

    # Generate transactions
    # n_transactions = 100
    transactions = []

    for _ in range(n_transactions):
        # Random number of items per transaction (1-8 items)
        n_items = np.random.randint(4, 15)
        # Randomly select items
        transaction = list(np.random.choice(products,
                                            size=n_items,
                                            replace=False))
        transactions.append(transaction)

    print(transactions)

    return transactions, products

