from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Audio, Dataset, DatasetDict
import pandas as pd
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import Seq2SeqTrainingArguments
import torch
from typing import Any, Dict, List, Union
from dataclasses import dataclass


AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # convert to tensors
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad label ids to the max length in the batch
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Load your CSV into a pandas DataFrame
df = pd.read_csv("D:\\PersonalProjs\\NLP With LLM\\data\\map.csv")

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Define a function to load audio files from paths
def load_audio(examples):
    # Load audio from file paths specified in the CSV
    examples["audio_2"] = {"path": examples["audio_path"]}
    return examples

# Apply the function to load audio data
dataset = dataset.map(load_audio)
print("dataset.map(load_audio): ", dataset)

# Normalize the dataset: rename columns and set audio format
def normalize_dataset(ds):
    ds = ds.rename_column("audio_path", AUDIO_COLUMN_NAME)
    ds = ds.rename_column("text", TEXT_COLUMN_NAME)
    ds = ds.cast_column(AUDIO_COLUMN_NAME, Audio(sampling_rate=16_000))
    return ds

# Apply normalization to your dataset
dataset = normalize_dataset(dataset)
print("\n\nnormalize_dataset(dataset): ", dataset)

# Split into training and evaluation datasets (optional step)
split_dataset = dataset.train_test_split(test_size=0.2)
raw_datasets = DatasetDict({
    "train": split_dataset["train"],
    "eval": split_dataset["test"]
})

# Shuffle the training dataset
raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# included in the training
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# to use gradient checkpointing
model.config.use_cache = False

processor = WhisperProcessor.from_pretrained("openai/whisper-small",language="english", task="transcribe")
normalizer = BasicTextNormalizer()

do_normalize_text = True

def prepare_dataset(batch):
    # load
    audio = batch[AUDIO_COLUMN_NAME]
    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    # process targets
    input_str = normalizer(batch[TEXT_COLUMN_NAME]).strip() if do_normalize_text else batch[TEXT_COLUMN_NAME]
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(input_str).input_ids

    return batch

vectorized_datasets = raw_datasets.map(
    prepare_dataset,
    remove_columns=next(iter(raw_datasets.values())).column_names,
    desc="preprocess dataset",
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_steps=50,
    # max_steps=500,
    learning_rate=3e-5,
    weight_decay=0.01,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    generation_max_length=50,
    logging_steps=10,
    report_to=["tensorboard"],
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

# Create the Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["eval"],
    tokenizer=processor,
    data_collator=data_collator,
)

# Start training
trainer.train()

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)


# # Load dataset from CSV
# dataset = load_dataset(
#     "csv", 
#     data_files="D:\\PersonalProjs\\NLP With LLM\\data\\map.csv",
#     split="train"
# ).cast_column("audio_path", Audio(sampling_rate=16000))  # Decode audio files



# # Tokenize dataset
# def preprocess_function(examples):
#     audio = examples["audio_path"]
#     inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest", truncation=True)
#     inputs["labels"] = processor.tokenizer(examples["text"], padding="max_length", max_length=processor.feature_extractor.max_length).input_ids
#     return inputs

# tokenized_dataset = dataset.map(preprocess_function, remove_columns=["audio_path"])

# # Training arguments
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=4,
#     learning_rate=5e-5,
#     num_train_epochs=3,
#     logging_dir="./logs",
#     logging_steps=10,
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )

# trainer.train()