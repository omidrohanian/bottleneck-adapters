import torch
import torch.nn  as nn
from torch.functional import F

import transformers as ts
from datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

torch.cuda.empty_cache()

SAVE_PATH = "PATH_TO_CHECKPOINTS/biomedical_model_checkpoints/prompt-biobert/"
MODEL_PATH = "dmis-lab/biobert-v1.1"

os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dict = pickle.load(open("train_dict" , "rb")) #Path to Training File
val_dict = pickle.load(open("val_dict" , "rb")) #Path to Validation File

train_dataset = Dataset.from_dict(train_dict)
val_dataset = Dataset.from_dict(val_dict)

tokenizer = ts.AutoTokenizer.from_pretrained(MODEL_PATH)

data_collator = ts.DataCollatorWithPadding(tokenizer=tokenizer , return_tensors="pt")

def mappingFunction(dataset):
    return tokenizer(dataset["text"])

final_train_dataset = train_dataset.map(mappingFunction , batched=True)
final_val_dataset = val_dataset.map(mappingFunction , batched=True)

model = ts.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
model.add_adapter("prefix_tuning" , config=ts.adapters.PrefixTuningConfig(flat=False, prefix_length=30))
model.train_adapter("prefix_tuning")

totalCount = 0
trainableCount = 0

for name , param in model.named_parameters():
    totalCount += param.numel()
    if param.requires_grad:
        trainableCount += param.numel()

print("All Params Count = " + str(totalCount/1e6))
print("Trainable Params Count = " + str(trainableCount/1e6))
print(str(trainableCount/totalCount * 100))

def collator_function(dataset):
  keys = dataset[0].keys()

  output_dict = {
      key: [] for key in keys
  }

  for item in dataset:
    for key in keys:
      output_dict[key].append(item[key])

  labels = torch.tensor(output_dict.pop("label"))
  output_dict.pop("text")

  collator_output = data_collator(output_dict)
  collator_output["labels"] = labels

  return collator_output

def train():
    training_arguments = ts.TrainingArguments(
        "output/",
        save_steps= 3000,
        num_train_epochs=10,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        remove_unused_columns=False,
        logging_steps=100,
        seed=123,
    )

    trainer = ts.Trainer(
        model=model,
        args=training_arguments,
        train_dataset=final_train_dataset,
        eval_dataset=final_val_dataset,
        data_collator=collator_function,
    )

    trainer.train()

    trainer.save_model(SAVE_PATH)

    import seaborn as sn
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    all_labels = []
    all_preds = []

    model.cpu()
    model.eval()

    for sample in final_val_dataset:
        input_sample = collator_function([sample])

        output = model(**input_sample)

        predicted_label = round(float(torch.sigmoid(output["logits"]).view(-1)))

        all_labels.append(sample["label"])
        all_preds.append(predicted_label)


    array = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=array,display_labels=[0,1])
    disp.plot()

    plt.show()

train()
