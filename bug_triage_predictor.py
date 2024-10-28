import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Paths to models, data, and commit counts
DATA_PATH = "test.csv.gzip"
MODEL_PATH_ALL = "main_issue_classifier"  # Model trained on all data
MODEL_PATH_RECENT = "secondary_issue_classifier"  # Model trained on recent data
COMMIT_COUNTS_A_PATH = "commit_counts_A.csv"
COMMIT_COUNTS_B_PATH = "commit_counts_B.csv"

# Set max sequence length
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

# Retrieve commit count for the predicted assignee based on the model
def get_commit_count(assignee, commit_counts):
    return commit_counts.get(assignee, 0)

def main():
    # Load the test data
    data_df = pd.read_csv(DATA_PATH, compression='gzip')
    
    while True:
        try:
            issue_id = int(input("Enter issue ID: ").strip())
            break
        except ValueError:
            print("Invalid input. Please enter a numeric issue ID.")
    
    try:
        # Model trained on all data
        predicted_assignee_all = predict_assignee(issue_id, model_all, tokenizer_all, data_df)
        commit_count_all = get_commit_count(predicted_assignee_all, commit_counts_A)
        
        print("\nUsing Model Trained on All Data:")
        print(f"Predicted Assignee: {predicted_assignee_all}")
        print(f"Number of Completed Issues (Commit Count) by {predicted_assignee_all}: {commit_count_all}")
        
        # Model trained on recent data
        predicted_assignee_recent = predict_assignee(issue_id, model_recent, tokenizer_recent, data_df)
        commit_count_recent = get_commit_count(predicted_assignee_recent, commit_counts_B)
        
        print("\nUsing Model Trained on Recent Data:")
        print(f"Predicted Assignee: {predicted_assignee_recent}")
        print(f"Number of Completed Issues (Commit Count) by {predicted_assignee_recent}: {commit_count_recent}")
    
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
