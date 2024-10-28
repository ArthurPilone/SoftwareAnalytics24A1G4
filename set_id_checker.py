import pandas as pd

# Paths to the train and test data files
TRAIN_A_PATH = "train_A.csv.gzip"
TRAIN_B_PATH = "train_B.csv.gzip"
TEST_PATH = "test.csv.gzip"

# Load the data
train_A_df = pd.read_csv(TRAIN_A_PATH, compression='gzip')
train_B_df = pd.read_csv(TRAIN_B_PATH, compression='gzip')
test_df = pd.read_csv(TEST_PATH, compression='gzip')

# Get unique identifiers in each set
unique_ids_A = train_A_df["identifier"].unique()
unique_ids_B = train_B_df["identifier"].unique()
unique_ids_test = test_df["identifier"].unique()

# Sort unique identifiers to find the lowest and highest identifiers
min_id_A, max_id_A = unique_ids_A.min(), unique_ids_A.max()
min_id_B, max_id_B = unique_ids_B.min(), unique_ids_B.max()
min_id_test, max_id_test = unique_ids_test.min(), unique_ids_test.max()

# Sample 20 unique IDs from each set
sample_ids_A = pd.Series(unique_ids_A).sample(20, random_state=1).sort_values()
sample_ids_B = pd.Series(unique_ids_B).sample(20, random_state=1).sort_values()
sample_ids_test = pd.Series(unique_ids_test).sample(20, random_state=1).sort_values()

print("Train A Identifiers:")
print(f"Lowest ID: {min_id_A}")
print(f"Highest ID: {max_id_A}")
print("\nSample of 20 Unique IDs in Train A:")
print(sample_ids_A.values)

print("\nTrain B Identifiers:")
print(f"Lowest ID: {min_id_B}")
print(f"Highest ID: {max_id_B}")
print("\nSample of 20 Unique IDs in Train B:")
print(sample_ids_B.values)

print("\nTest Identifiers:")
print(f"Lowest ID: {min_id_test}")
print(f"Highest ID: {max_id_test}")
print("\nSample of 20 Unique IDs in Test:")
print(sample_ids_test.values)
