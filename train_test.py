import sys
import evaluate
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    sys.path.append(str(Path(os.path.abspath('')).absolute().parent))

    TRAINING_DATASET_PATH = "train_B.csv.gzip"
    TEST_DATASET_PATH = "test.csv.gzip"

    training_dataset = pd.read_csv(TRAINING_DATASET_PATH, compression='gzip', lineterminator='\n')
    training_dataset = training_dataset.dropna()

    test_dataset = pd.read_csv(TEST_DATASET_PATH, compression='gzip', lineterminator='\n')
    print(len(test_dataset))
    test_dataset = test_dataset.dropna()
    print("Size without NaNs ", len(test_dataset))

    label_as_id = {}
    id_as_label = {}

    new_id = 0
    for assignee in training_dataset["assignee"].unique():
        label_as_id[assignee] = new_id
        id_as_label[new_id] = assignee
        new_id += 1

    def parse_label_to_id(label):
        return label_as_id[label]

    MODEL_NAME = "distilbert/distilbert-base-uncased"
    CONTEXT_LENGTH = 512

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id_as_label), id2label=id_as_label, label2id=label_as_id
    )

    TITLE_BEGIN_SENTINEL = "<BoT>"
    TITLE_END_SENTINEL = "<EoT>"
    CODE_BEGIN_SENTINEL = "<BoC>"
    CODE_END_SENTINEL = "<EoC>"

    special_tokens_dict = {
        'pad_token': '[PAD]',
        'additional_special_tokens': [
            TITLE_BEGIN_SENTINEL,
            TITLE_END_SENTINEL,
            CODE_BEGIN_SENTINEL,
            CODE_END_SENTINEL
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.all_special_tokens)

    model.resize_token_embeddings(len(tokenizer))

    def create_input_text(issue_row):
        padded_title = TITLE_BEGIN_SENTINEL + issue_row["summary"] + TITLE_END_SENTINEL + "\n"
        return padded_title + issue_row["body"]

    def process_row(row):
        return {
            "label": parse_label_to_id(row['assignee']),
            "text": create_input_text(row)
        }

    def process_dataset_entry(entries):
        entries = tokenizer(
            entries["text"],
            max_length=CONTEXT_LENGTH,
            padding="max_length",
            truncation=True
        )
        return entries

    def process_df(df,test):
        if test:
            df = df.loc[df['assignee'].isin(label_as_id.keys())]
        clean_df = df.apply(process_row, axis=1, result_type='expand')
        df_as_ds = Dataset.from_pandas(clean_df)
        df_as_ds = df_as_ds.remove_columns(["__index_level_0__"])

        df_as_ds = df_as_ds.filter(
            lambda x: len(tokenizer(x["text"])["input_ids"]) < CONTEXT_LENGTH
        )

        df_as_ds = df_as_ds.map(
            process_dataset_entry,
            remove_columns=["text"],
            batched=True,
            batch_size=16,
            num_proc=6
        )
        return df_as_ds

    tokenized_train = process_df(training_dataset,False)
    tokenized_test = process_df(test_dataset,True)

    print("Left with " + str(len(tokenized_train)) + " training entries")
    print("      and " + str(len(tokenized_test)) + " test entries.")

    accuracy_evaluator = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_evaluator.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./training_secondary_model",
        learning_rate=2e-5,
        num_train_epochs=5,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    torch.cuda.empty_cache()

    trainer.train()
    trainer.save_model("secondary_issue_classifier")

    classifier = pipeline("text-classification", model="main_issue_classifier", tokenizer=tokenizer)
    result = classifier("I can't access one of my recorded trips")

    print(result[0])

if __name__ == "__main__":
    main()
