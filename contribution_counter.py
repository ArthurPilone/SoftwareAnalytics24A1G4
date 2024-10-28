import pandas as pd

# Paths to training data files
TRAIN_A_PATH = "train_A.csv.gzip"
TRAIN_B_PATH = "train_B.csv.gzip"

# Function to calculate commit counts from training data
def calculate_commit_counts(train_path):
    train_df = pd.read_csv(train_path, compression='gzip')
    # Count unique issues completed by each assignee
    commit_counts = train_df.groupby("assignee")["identifier"].nunique()
    return commit_counts

# Calculate commit counts for Train A and Train B
commit_counts_A = calculate_commit_counts(TRAIN_A_PATH)
commit_counts_B = calculate_commit_counts(TRAIN_B_PATH)

# Save each to a separate CSV
commit_counts_A.to_csv("commit_counts_A.csv", header=["commit_count"])
commit_counts_B.to_csv("commit_counts_B.csv", header=["commit_count"])

print("Commit counts saved as commit_counts_A.csv and commit_counts_B.csv")
