import pandas as pd
import torch
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
from typing import List, Tuple, Dict, Any

# Note: These imports appear unused in the main flow but are kept from the original script
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os
# os is imported here (before logging setup) because os.makedirs() is needed
# to create the '../data/' directory before the FileHandler tries to write
# the log file to it — otherwise the script crashes on first run if the
# directory does not exist yet.

# --- Logging Setup ---
# Remove all existing handlers from the root logger to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to write to a file
LOG_FILE_PATH = '../data/preprocessing.log'
os.makedirs('../data', exist_ok=True)  # Create directory if it doesn't exist, prevents crash on first run
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Get the root logger and add the file handler                                                          
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.info("Logging setup complete.")
# --- End Logging Setup ---

def find_token_span(offset_mapping: List[Tuple[int, int]], start: int, end: int) -> Tuple[int, int]:
    """
    Maps character-level start/end offsets to token-level start/end indices.

    Iterates through the token offset mapping provided by the tokenizer to find
    the token indices that correspond to the given character start and end
    positions. Includes a fallback mechanism to find the closest token if an
    exact boundary match isn't found.

    Args:
        offset_mapping: A list where each element is a tuple (char_start, char_end)
                        representing the character span for each token. Provided by
                        Hugging Face tokenizers with `return_offsets_mapping=True`.
        start: The starting character offset of the annotation span.
        end: The ending character offset of the annotation span.

    Returns:
        A tuple containing the start token index and end token index
        corresponding to the character span.
    """
    start_token = None
    end_token = None
    # Find the start token index
    for i, (token_start, token_end) in enumerate(offset_mapping):
        # Check if the annotation start char falls within this token's span
        if start_token is None and token_start <= start < token_end:
            start_token = i
        # Check if the annotation end char falls within this token's span
        # Note: The condition `token_start < end <= token_end` ensures that if
        # the end character is the start of the next token, it correctly assigns
        # the end to the previous token.
        if end_token is None and token_start < end <= token_end:
            end_token = i
        # Optimization: Break if both start and end tokens are found
        if start_token is not None and end_token is not None:
            break

    # Fallback: If exact start boundary wasn't found, find the closest token start
    if start_token is None:
        logger.debug(f"Exact start token not found for char offset {start}. Finding closest.")
        start_token = min(range(len(offset_mapping)), key=lambda i: abs(offset_mapping[i][0] - start))

    # Fallback: If exact end boundary wasn't found, find the closest token end
    if end_token is None:
        logger.debug(f"Exact end token not found for char offset {end}. Finding closest.")
        # Find the token whose end offset is closest to the annotation end char
        end_token = min(range(len(offset_mapping)), key=lambda i: abs(offset_mapping[i][1] - end))

    # Ensure start_token is not after end_token due to fallback logic
    if start_token > end_token:
        logger.warning(f"Calculated start_token ({start_token}) > end_token ({end_token}) for span ({start}, {end}). Adjusting end_token.")
        # This might happen if the annotation span is very small or falls entirely
        # between tokens in a way the fallback misinterprets. Setting end = start
        # ensures at least one token is covered.
        end_token = start_token

    return start_token, end_token

def process_with_sliding_window(text: str, annotations: List[Dict[str, Any]], tokenizer: BertTokenizerFast,
                                window_size: int = 510, stride: int = 256) -> List[Tuple[List[int], List[str]]]:
    """
    Processes long text by tokenizing, applying annotations, and creating sliding window chunks.

    Handles texts longer than the BERT model's maximum input length by breaking
    them into overlapping chunks (windows). It first tokenizes the entire text,
    maps annotations to token labels across the full sequence, and then extracts
    chunks of token IDs and corresponding string labels according to the window
    size and stride.

    Args:
        text: The full input text of a single document (e.g., job ad).
        annotations: A list of annotation dictionaries for the document. Each dictionary
                     is expected to have a 'result' key containing a list of labeled spans.
        tokenizer: An initialized Hugging Face `BertTokenizerFast` instance.
        window_size: The maximum number of tokens per chunk.
        stride: The number of tokens to slide the window forward for the next chunk.

    Returns:
        A list of processed chunks. Each chunk is a tuple containing:
        - A list of token IDs for the chunk.
        - A list of corresponding string labels ('O' or zone labels) for the chunk.
    """
    # Tokenize the entire text without adding special tokens yet, get offsets
    full_encoding = tokenizer(       # changed from tokenizer.encode_plus(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    full_tokens = full_encoding['input_ids']
    full_offset_mapping = full_encoding['offset_mapping']

    # Initialize labels for all tokens as 'O' (Outside)
    full_labels = ['O'] * len(full_tokens)

    # Apply annotations to the full sequence of token labels
    for annotation in annotations:
        # Basic validation of annotation structure
        if isinstance(annotation, dict) and 'result' in annotation:
            for result in annotation['result']:
                if isinstance(result, dict) and 'value' in result:
                    value = result['value']
                    start = value.get('start')
                    end = value.get('end')
                    # Get the first label from the list, handle potential missing labels
                    label = value.get('labels', [None])[0]

                    # Check if essential annotation fields are present
                    if start is None or end is None or label is None:
                        logger.warning(f"Skipping annotation due to missing fields: {result}")
                        continue

                    # Map character offsets to token indices
                    start_token, end_token = find_token_span(full_offset_mapping, start, end)

                    # Apply the label to the corresponding tokens
                    if start_token is not None and end_token is not None:
                        # Apply label up to and including the end_token
                        for i in range(start_token, min(end_token + 1, len(full_tokens))):
                            full_labels[i] = label
                else:
                     logger.warning(f"Skipping malformed annotation result (missing 'value'): {result}")
        else:
            logger.warning(f"Skipping malformed annotation (missing 'result'): {annotation}")


    processed_chunks = []
    chunk_count = 0
    logger.info(f"Applying sliding window: window_size={window_size}, stride={stride}, total_tokens={len(full_tokens)}")

    # Iterate through the tokens using the sliding window approach
    for chunk_start in range(0, len(full_tokens), stride):
        chunk_count += 1
        # Determine the end of the current chunk
        chunk_end = min(chunk_start + window_size, len(full_tokens))

        # Extract token IDs and labels for the current chunk
        chunk_tokens = full_tokens[chunk_start:chunk_end]
        chunk_labels = full_labels[chunk_start:chunk_end]

        # Store the chunk
        processed_chunks.append((chunk_tokens, chunk_labels))

        # --- Detailed Logging for Debugging ---
        logger.info(f"Created chunk {chunk_count} (Tokens {chunk_start}-{chunk_end-1})")
        if logger.isEnabledFor(logging.DEBUG): # Log token details only if DEBUG is enabled
            logger.debug("Chunk Token-Label Alignment:")
            for i, (token_id, label) in enumerate(zip(chunk_tokens, chunk_labels)):
                token_text = tokenizer.decode([token_id])
                original_position = chunk_start + i
                logger.debug(f"  OrigPos {original_position} (Chunk {i}): '{token_text}' ({token_id}) -> Label: {label}")
            logger.debug(f"Unique labels in this chunk: {set(chunk_labels)}")
            logger.debug("---")
        # --- End Detailed Logging ---

        # Stop if the end of the chunk reaches the end of the full token list
        if chunk_end == len(full_tokens):
            break

    logger.info(f"Generated {len(processed_chunks)} chunks for the document.")
    return processed_chunks

def preprocess_data(df: pd.DataFrame) -> Tuple[List[Tuple[List[int], List[int]]], Dict[str, int]]:
    """
    Preprocesses the entire dataset stored in a DataFrame.

    Iterates through each row (job ad) of the DataFrame, extracts text and
    annotations, processes them using `process_with_sliding_window`, builds
    a mapping from string labels to integer IDs, and converts the processed
    chunks into lists of token IDs and label IDs.

    Args:
        df: A pandas DataFrame where each row represents a job advertisement
            with 'data' (containing 'content_clean') and 'annotations' columns.

    Returns:
        A tuple containing:
        - processed_data: A list where each item is a tuple:
          (list_of_token_ids_for_chunk, list_of_label_ids_for_chunk).
        - label2id: A dictionary mapping unique string labels found in the
          dataset to integer IDs (starting with 'O': 0).
    """
    # Initialize the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    processed_data = []
    label2id = {'O': 0}  # Initialize label mapping with 'O' as ID 0

    logger.info(f"Starting preprocessing for {len(df)} rows.")

    # Iterate through each job ad in the DataFrame
    for index, row in df.iterrows():
        try:
            logger.info(f"Processing row index {index}")

            # Validate required data fields exist
            if 'data' not in row or 'content_clean' not in row['data']:
                logger.warning(f"Row {index} missing 'data' or 'data.content_clean'. Skipping.")
                continue
            text = row['data']['content_clean']

            if 'annotations' not in row or not isinstance(row['annotations'], list):
                logger.warning(f"Row {index} missing 'annotations' list or invalid format. Skipping.")
                continue
            annotations = row['annotations']

            logger.info(f"Text (first 200 chars): {text[:200]}...")

            # Process the text and annotations into labeled chunks
            chunks = process_with_sliding_window(text, annotations, tokenizer)

            # Process each chunk generated for the current job ad
            for chunk_tokens, chunk_labels in chunks:
                # Build the label2id mapping dynamically
                for label in chunk_labels:
                    if label not in label2id:
                        new_id = len(label2id)
                        label2id[label] = new_id
                        logger.info(f"Discovered new label: '{label}' assigned ID: {new_id}")

                # Convert string labels in the chunk to their integer IDs
                chunk_label_ids = [label2id[label] for label in chunk_labels]

                # Append the processed chunk (token IDs, label IDs) to the results
                processed_data.append((chunk_tokens, chunk_label_ids))

            logger.info(f"Successfully processed row {index}, generated {len(chunks)} chunks.")
            logger.info("===")

        except Exception as e:
            # Log any unexpected errors during row processing
            logger.error(f"Error processing row {index}: {str(e)}", exc_info=True)

    logger.info(f"Preprocessing complete. Total processed chunks: {len(processed_data)}")
    logger.info(f"Final label mapping (label2id): {label2id}")
    return processed_data, label2id

def create_dataset(data: List[Tuple[List[int], List[int]]], label2id: Dict[str, int]) -> TensorDataset:
    """
    Converts processed data into a PyTorch TensorDataset with padding.

    Takes the list of processed chunks (each being a tuple of token ID list
    and label ID list) and converts them into padded PyTorch tensors for
    input IDs, attention masks, and labels.

    Args:
        data: A list of tuples, where each tuple contains a list of token IDs
              and a list of corresponding label IDs for a chunk.
        label2id: The dictionary mapping string labels to integer IDs.
                  Note: no longer used for padding (padding uses -100 instead),
                  but kept as a parameter for potential future use or inspection.

    Returns:
        A PyTorch `TensorDataset` containing padded tensors for input IDs,
        labels, and attention masks.
    """
    if not data:
        logger.error("Cannot create dataset from empty data list.")
        return TensorDataset(torch.empty(0), torch.empty(0), torch.empty(0)) # Return empty dataset

    logger.info(f"Creating TensorDataset from {len(data)} data points (chunks).")

    # Separate token IDs and label IDs into their own lists
    input_ids_list = [torch.tensor(item[0]) for item in data]
    labels_list = [torch.tensor(item[1]) for item in data]

    # Create attention masks (1 for real tokens, 0 for padding)
    attention_masks_list = [torch.ones_like(input_id) for input_id in input_ids_list]

    # Pad sequences in each list to the maximum length found in that list
    # `batch_first=True` ensures the output shape is (batch_size, sequence_length)
    logger.info("Padding sequences...")
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=0 # BERT padding token ID is 0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
        # WHY -100: PyTorch's CrossEntropyLoss uses ignore_index=-100 to skip
        # padded positions in loss calculation entirely.
        # Using label2id['O'] (=0) would be ambiguous — the model cannot
        # distinguish real 'O' labels from padding, causing it to learn
        # from fake data. -100 is unambiguous and the PyTorch standard.
        # IMPORTANT for Person 2: set ignore_index=-100 in CrossEntropyLoss.
        # NOTE: The data_preparation_doc explicitly acknowledges this, stating:
        # "Using -100 as the padding_value for labels here might align more
        # directly with common practice for CrossEntropyLoss."
    )
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks_list, batch_first=True, padding_value=0 # 0 indicates padding
    )
    logger.info("Padding complete.")
    logger.info(f"Shape of padded_input_ids: {padded_input_ids.shape}")
    logger.info(f"Shape of padded_labels: {padded_labels.shape}")
    logger.info(f"Shape of padded_attention_masks: {padded_attention_masks.shape}")


    # Create and return the TensorDataset
    return TensorDataset(padded_input_ids, padded_labels, padded_attention_masks)

# --- Main Execution Logic ---
if __name__ == "__main__":
    logger.info("--- Starting Data Preprocessing Script ---")

    # Define input/output paths
    INPUT_JSON_PATH = '../data/annotated.json'
    ID2LABEL_SAVE_PATH = '../model/id2label.json'
    # Optional: Define paths to save processed datasets
    TRAIN_DATASET_SAVE_PATH = '../data/train_dataset.pt'
    TEST_DATASET_SAVE_PATH = '../data/test_dataset.pt'

    # Load the raw data
    logger.info(f"Loading data from {INPUT_JSON_PATH}...")
    try:
        df = pd.read_json(INPUT_JSON_PATH)
        logger.info(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        logger.error(f"Failed to load data from {INPUT_JSON_PATH}: {e}", exc_info=True)
        exit() # Exit if data loading fails

    # Filter out rows with empty results in the first annotation object
    logger.info("Filtering rows with empty annotation results...")
    try:
        # This filter assumes annotations[0] exists and checks its 'result' list
        original_len = len(df)
        # WHY THIS FILTER IS MORE ROBUST THAN THE ORIGINAL:
        # The original filter (lambda x: len(x[0]['result']) > 0) would crash if:
        #   - x is not a list (TypeError on x[0])
        #   - x is an empty list (IndexError on x[0])
        #   - x[0] has no 'result' key (KeyError)
        # This version adds 4 defensive checks before touching x[0]:
        #   1. isinstance(x, list)         → is it actually a list?
        #   2. len(x) > 0                  → does it have at least one element?
        #   3. isinstance(x[0], dict)      → is the first element a dict?
        #   4. x[0].get('result', [])      → safely get 'result', default to [] if missing
        df = df[df.annotations.apply(lambda x: isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and len(x[0].get('result', [])) > 0)]
        logger.info(f"Filtered out {original_len - len(df)} rows. Remaining rows: {len(df)}")
    except Exception as e:
        logger.error(f"Error during filtering: {e}. Proceeding with unfiltered data if possible, or check filter logic.", exc_info=True)
        # Depending on the error, you might choose to exit or proceed cautiously


    # Preprocess the data
    processed_data, label2id = preprocess_data(df)

    # Check if preprocessing yielded any results
    if not processed_data:
        logger.error("Preprocessing resulted in an empty dataset. Check logs for errors.")
        raise ValueError("No data was successfully processed. Check the log file.")

    # Create the inverse mapping (ID to Label)
    id2label = {i: label for label, i in label2id.items()}

    # Save the id2label mapping
    logger.info(f"Saving id2label mapping to {ID2LABEL_SAVE_PATH}...")
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(ID2LABEL_SAVE_PATH), exist_ok=True)
        with open(ID2LABEL_SAVE_PATH, 'w') as f:
            json.dump(id2label, f, indent=2)
        print(f"Label mapping saved to: {ID2LABEL_SAVE_PATH}") # Also print to console
    except Exception as e:
        logger.error(f"Failed to save id2label mapping: {e}", exc_info=True)

    # Split the processed data into training and testing sets
    logger.info("Splitting data into training and testing sets (80/20)...")
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    logger.info(f"Train data size: {len(train_data)} chunks")
    logger.info(f"Test data size: {len(test_data)} chunks")


    # Create PyTorch TensorDatasets with padding
    logger.info("Creating training TensorDataset...")
    train_dataset = create_dataset(train_data, label2id)
    logger.info("Creating testing TensorDataset...")
    test_dataset = create_dataset(test_data, label2id)

    # Optional: Save the processed TensorDatasets
    logger.info(f"Saving processed datasets...")
    torch.save(train_dataset, TRAIN_DATASET_SAVE_PATH)
    torch.save(test_dataset, TEST_DATASET_SAVE_PATH)
    logger.info(f"Saved datasets to {TRAIN_DATASET_SAVE_PATH} and {TEST_DATASET_SAVE_PATH}")


    # Final summary logs
    logger.info("--- Data Preprocessing Script Finished ---")
    logger.info(f"Final number of training samples (chunks): {len(train_dataset)}")
    logger.info(f"Final number of testing samples (chunks): {len(test_dataset)}")
    logger.info(f"Unique labels found: {list(label2id.keys())}")
    print("\nPreprocessing finished successfully. Check preprocessing.log for details.") # Also print to console
