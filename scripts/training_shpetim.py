import torch
from transformers import BertForTokenClassification, BertTokenizerFast, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter
import numpy as np
import gc
import os
import json

# --- Configuration ---
MODEL_NAME = "bert-base-multilingual-cased"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
VALIDATION_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 5

# --- Project Paths ---
# This makes the paths work correctly because this file is inside /scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs", "student_run_01")

LABEL_MAP_PATH = os.path.join(MODEL_DIR, "id2label.json")
TRAIN_DATASET_PATH = os.path.join(DATA_DIR, "train_dataset.pt")
TEST_DATASET_PATH = os.path.join(DATA_DIR, "test_dataset.pt")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_student_model.pt")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_student_model.pt")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- Load Data and Mappings ---
print("Loading data and mappings...")

train_dataset = torch.load(TRAIN_DATASET_PATH, weights_only=False)
test_dataset = torch.load(TEST_DATASET_PATH, weights_only=False)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    id2label = json.load(f)

id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))
print("Number of labels:", num_labels)
print("Labels:", id2label)


# --- Data Loaders ---
# DataLoader gives the dataset to BERT in small batches.

pin_memory = True if device.type == "cuda" else False

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_memory
)

# --- Model Initialization ---
print(f"Initializing model: {MODEL_NAME}")

model = BertForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

model.to(device)

if model is None:
    print("Error: Model not initialized. Please fill in the initialization code.")
    exit()

print("Model initialized successfully.")

# Optional: Enable gradient checkpointing if needed for memory saving
# model.gradient_checkpointing_enable()

model.to(device)  # Move model to CPU/GPU

# --- Optimizer and Scheduler ---
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    eps=1e-8
)

# Calculate total training steps needed for the scheduler
num_training_steps = (
    (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1)
    // GRADIENT_ACCUMULATION_STEPS
) * NUM_EPOCHS

num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

if optimizer is None or scheduler is None:
    print("Error: Optimizer or Scheduler not initialized.")
    exit()

print(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")


# --- Loss Function with Class Weights ---
print("Calculating class weights...")

label_counter = Counter()

# train_dataset structure:
# item[0] = input_ids
# item[1] = labels
# item[2] = attention_mask
for item in train_dataset:
    labels = item[1]

    for label_id in labels.view(-1).tolist():
        if label_id != -100:  # ignore padding
            label_counter[label_id] += 1

total_labels = sum(label_counter.values())

class_weights_list = []

for label_id in range(num_labels):
    count = label_counter.get(label_id, 0)

    if count == 0:
        weight = 1.0
    else:
        weight = total_labels / (num_labels * count)

    class_weights_list.append(weight)

class_weights = torch.tensor(
    class_weights_list,
    dtype=torch.float
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=-100
)

print("Label counts:", label_counter)
print("Class weights:", class_weights_list)
print("Loss function ready.")
# --- End Student Code ---

# Convert weights to a PyTorch tensor and move to device
class_weights = torch.FloatTensor(class_weights_list).to(device)
print("Class weights calculated and moved to device.")

# Initialize loss function with class weights and ignore_index for padding
loss_fct = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=-100
)


# --- Evaluation Function ---
def evaluate_model(model, data_loader, loss_function, device, num_labels):
    model.eval()

    total_loss = 0.0
    total_correct_predictions = 0
    total_active_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            # IMPORTANT:
            # Your preprocessing saved TensorDataset in this order:
            # input_ids, labels, attention_mask
            input_ids, labels, attention_mask = [b.to(device) for b in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            # Flatten logits and labels
            # logits shape: [batch_size, sequence_length, num_labels]
            # labels shape: [batch_size, sequence_length]
            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1)

            # Active tokens are real tokens, not padding
            active_mask = labels_flat != -100

            # Only evaluate if this batch has real tokens
            if active_mask.sum().item() > 0:
                active_logits = logits_flat[active_mask]
                active_labels = labels_flat[active_mask]

                # Calculate validation loss only on active tokens
                batch_loss = loss_function(active_logits, active_labels)

                # Predictions
                predictions = torch.argmax(active_logits, dim=1)

                # Count correct predictions
                batch_correct_predictions = (predictions == active_labels).sum().item()
                batch_active_tokens = active_labels.size(0)

                total_loss += batch_loss.item()
                total_correct_predictions += batch_correct_predictions
                total_active_tokens += batch_active_tokens

            del outputs, logits, input_ids, labels, attention_mask

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = (
        total_correct_predictions / total_active_tokens
        if total_active_tokens > 0
        else 0
    )

    model.train()
    return avg_loss, accuracy


# --- TensorBoard Setup ---
writer = SummaryWriter(log_dir=LOG_DIR)
print(f"TensorBoard logs will be saved to: {LOG_DIR}")

# --- Training Loop ---
global_step = 0
best_val_loss = float("inf")
patience_counter = 0

print("Starting training...")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_train_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
        leave=True
    )

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Global step {global_step}",
                flush=True
            )
        input_ids, labels, attention_mask = [b.to(device) for b in batch]
        # Dataset order from preprocessing:
        # input_ids, labels, attention_mask
        input_ids, labels, attention_mask = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        # Manual loss calculation so we use class weights
        loss = loss_fct(
            logits.view(-1, num_labels),
            labels.view(-1)
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        scaled_loss.backward()

        # Accumulate unscaled loss for logging
        epoch_train_loss += loss.item()

        # Optimizer step after enough accumulated batches
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], global_step)

            # Periodic validation
            if global_step % VALIDATION_INTERVAL == 0:
                val_loss, val_accuracy = evaluate_model(
                    model,
                    val_loader,
                    loss_fct,
                    device,
                    num_labels
                )

                writer.add_scalar("Loss/validation_step", val_loss, global_step)
                writer.add_scalar("Accuracy/validation_step", val_accuracy, global_step)

                print(
                    f"\nStep {global_step}: "
                    f"Val Loss={val_loss:.4f}, "
                    f"Val Acc={val_accuracy:.4f}"
                )

                if val_loss < best_val_loss:
                    print(
                        f"  Validation loss improved "
                        f"({best_val_loss:.4f} -> {val_loss:.4f}). "
                        f"Saving best model..."
                    )

                    best_val_loss = val_loss
                    patience_counter = 0

                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": best_val_loss,
                        "num_labels": num_labels,
                        "id2label": id2label
                    }, BEST_MODEL_PATH)

                else:
                    patience_counter += 1
                    print(
                        f"  Validation loss did not improve. "
                        f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                    )

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {global_step} steps.")
                    break

                model.train()

        progress_bar.set_postfix({
            "train_loss": epoch_train_loss / (batch_idx + 1),
            "best_val_loss": best_val_loss
        })

        del outputs, logits, loss, scaled_loss, input_ids, labels, attention_mask

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Exiting training loop due to early stopping.")
        break

    print(f"\n--- End of Epoch {epoch + 1} ---")

    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_epoch_train_loss:.4f}")

    val_loss, val_accuracy = evaluate_model(
        model,
        val_loader,
        loss_fct,
        device,
        num_labels
    )

    writer.add_scalar("Loss/validation_epoch", val_loss, epoch + 1)
    writer.add_scalar("Accuracy/validation_epoch", val_accuracy, epoch + 1)

    print(f"End-of-Epoch Validation Loss: {val_loss:.4f}")
    print(f"End-of-Epoch Validation Accuracy: {val_accuracy:.4f}")

    if val_loss < best_val_loss:
        print(
            f"  Validation loss improved "
            f"({best_val_loss:.4f} -> {val_loss:.4f}). "
            f"Saving best model..."
        )

        best_val_loss = val_loss
        patience_counter = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": best_val_loss,
            "num_labels": num_labels,
            "id2label": id2label
        }, BEST_MODEL_PATH)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Training Finished ---
print("\nTraining completed!")
writer.close()


# --- Save Final Model ---
print(f"Saving final model state to {FINAL_MODEL_PATH}...")

final_val_loss = val_loss if "val_loss" in locals() else float("inf")
last_epoch = epoch if "epoch" in locals() else -1

torch.save({
    "epoch": last_epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "val_loss": final_val_loss,
    "num_labels": num_labels,
    "id2label": id2label
}, FINAL_MODEL_PATH)

print(f"Best validation loss achieved: {best_val_loss:.4f}")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Final model saved to: {FINAL_MODEL_PATH}")
print(f"TensorBoard logs are located in: {LOG_DIR}")

# --- Final Evaluation ---
print("\nRunning final evaluation on test set using the best model...")

# Load best model checkpoint
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Tokenizer is needed only to identify subword tokens like ##ung
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

all_preds = []
all_labels = []


def convert_to_bio(label_sequence):
    """
    Convert simple zone labels into BIO format for seqeval.
    Example:
    Fähigkeiten, Fähigkeiten, O, Benefits
    becomes:
    B-Fähigkeiten, I-Fähigkeiten, O, B-Benefits
    """
    bio_sequence = []
    previous_label = "O"

    for label in label_sequence:
        if label == "O":
            bio_sequence.append("O")
            previous_label = "O"
        else:
            if label == previous_label:
                bio_sequence.append(f"I-{label}")
            else:
                bio_sequence.append(f"B-{label}")
            previous_label = label

    return bio_sequence


with torch.no_grad():
    for batch in val_loader:
        # Dataset order:
        # input_ids, labels, attention_mask
        input_ids, labels, attention_mask = [b.to(device) for b in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Move to CPU for easier processing
        input_ids_cpu = input_ids.cpu()
        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()

        for i in range(labels_cpu.size(0)):
            true_label_sequence = []
            pred_label_sequence = []

            token_ids = input_ids_cpu[i].tolist()
            token_texts = tokenizer.convert_ids_to_tokens(token_ids)

            for token_text, true_id, pred_id in zip(
                token_texts,
                labels_cpu[i].tolist(),
                predictions_cpu[i].tolist()
            ):
                # Ignore padding labels
                if true_id == -100:
                    continue

                # Optional: ignore continuation subword tokens
                # Example: "Bewerbung", "##s", "##prozess"
                if token_text.startswith("##"):
                    continue

                true_label = id2label[int(true_id)]
                pred_label = id2label[int(pred_id)]

                true_label_sequence.append(true_label)
                pred_label_sequence.append(pred_label)

            all_labels.append(convert_to_bio(true_label_sequence))
            all_preds.append(convert_to_bio(pred_label_sequence))

# Print seqeval metrics
print("\nSeqeval Classification Report:")
print(classification_report(all_labels, all_preds))

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Final Precision: {precision:.4f}")
print(f"Final Recall:    {recall:.4f}")
print(f"Final F1-score:  {f1:.4f}")

print("\nScript finished.")

