import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter

# Paths and configurations
DATA_PATH = "test.csv.gzip"  # Test set
MODEL_PATH_ALL = "main_issue_classifier"  # Model trained on all data
MODEL_PATH_RECENT = "secondary_issue_classifier"  # Model trained on recent data
COMMIT_COUNTS_A_PATH = "commit_counts_A.csv"  # Commit counts from Train A
COMMIT_COUNTS_B_PATH = "commit_counts_B.csv"  # Commit counts from Train B
CONTEXT_LENGTH = 512

# Load models and tokenizers
tokenizer_all = AutoTokenizer.from_pretrained(MODEL_PATH_ALL)
model_all = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_ALL)
tokenizer_recent = AutoTokenizer.from_pretrained(MODEL_PATH_RECENT)
model_recent = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_RECENT)

# Load commit counts for each model
commit_counts_A = pd.read_csv(COMMIT_COUNTS_A_PATH, index_col="assignee").squeeze("columns").to_dict()
commit_counts_B = pd.read_csv(COMMIT_COUNTS_B_PATH, index_col="assignee").squeeze("columns").to_dict()

# Retrieve the issue with a specified identifier
def get_issue_by_id(issue_id, data_df):
    issue_row = data_df[data_df["identifier"] == issue_id]
    if issue_row.empty:
        raise ValueError(f"No issue found with ID {issue_id}")
    return issue_row.iloc[0]

# Format the input text for the model
TITLE_BEGIN_SENTINEL = "<BoT>"
TITLE_END_SENTINEL = "<EoT>"

def create_input_text(issue_row):
    title = issue_row["summary"]
    body = issue_row["body"]
    return f"{TITLE_BEGIN_SENTINEL}{title}{TITLE_END_SENTINEL}\n{body}"

# Prediction function with truncation for long inputs
def predict_assignee(issue_id, model, tokenizer, data_df):
    issue_row = get_issue_by_id(issue_id, data_df)
    input_text = create_input_text(issue_row)
    
    # Truncate and tokenize input
    inputs = tokenizer(
        input_text,
        max_length=CONTEXT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_label_idx = torch.argmax(outputs.logits, dim=1).item()
    predicted_assignee = model.config.id2label[predicted_label_idx]
    return predicted_assignee

# Evaluation function without actual labels
def evaluate_model(data_df, model, tokenizer, commit_counts):
    predicted_assignees = []
    commit_counts_list = []

    # Iterate through each issue in the test set
    for _, row in data_df.iterrows():
        issue_id = row["identifier"]
        
        # Predict assignee
        predicted_assignee = predict_assignee(issue_id, model, tokenizer, data_df)
        predicted_assignees.append(predicted_assignee)
        
        # Retrieve commit count
        commit_count = commit_counts.get(predicted_assignee, 0)
        commit_counts_list.append(commit_count)

    # Distribution of predicted assignees
    assignee_distribution = Counter(predicted_assignees)
    print("\nPredicted Assignee Distribution:")
    for assignee, count in assignee_distribution.items():
        print(f"{assignee}: {count} predictions")

    # Average commit count of predicted assignees
    avg_commit_count = sum(commit_counts_list) / len(commit_counts_list)
    print(f"\nAverage Commit Count of Predicted Assignees: {avg_commit_count:.2f}")

# Load test data
data_df = pd.read_csv(DATA_PATH, compression='gzip')

# Evaluate both models
print("Evaluating Model Trained on All Data:")
evaluate_model(data_df, model_all, tokenizer_all, commit_counts_A)

print("\nEvaluating Model Trained on Recent Data:")
evaluate_model(data_df, model_recent, tokenizer_recent, commit_counts_B)
