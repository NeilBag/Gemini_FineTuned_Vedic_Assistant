import google.generativeai as genai
import os
import time
import json
import random
from dotenv import load_dotenv
import traceback

# --- Load environment variables ---
load_dotenv()
print("Attempting to load environment variables from .env file...")
# -------------------------------------------------

# --- Configuration ---

# 1. API Key (Reads from environment variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Check API key ---
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file or set it in your environment.")
elif len(GOOGLE_API_KEY) < 20:
     raise ValueError("The GOOGLE_API_KEY environment variable appears to be missing or set incorrectly.")
else:
    print("GOOGLE_API_KEY loaded successfully.")

# 2. Data File Path (Your LARGE generated Q&A dataset)
#    !!!! Point this to the large file (~68k pairs) !!!!
DATA_FILE_PATH = 'Hinduism_Vedas_qa_generated2.json'

# 3. JSON Data Keys (Should still be 'question' and 'answer')
JSON_INPUT_KEY = 'question'
JSON_OUTPUT_KEY = 'answer'

# 4. Base Model to Fine-Tune
BASE_MODEL_ID = "models/gemini-1.5-flash-001-tuning"

# 5. Tuned Model Display Name and ID
#    !!!! IMPORTANT: Use a NEW version identifier !!!!
TUNED_MODEL_DISPLAY_NAME = "Hinduism Veda Expert v1" # Indicate it's v2
TUNED_MODEL_ID = "hinduism-veda-expert-v1"             # Use a new ID like v2

# 6. Tuning Hyperparameters
TUNING_EPOCHS = 4  # 3 or 4 epochs is usually reasonable for large datasets
TUNING_BATCH_SIZE = 8 # Increased batch size for larger dataset (e.g., 8, 16)

# 7. API Limit for Training Examples (Examples * Epochs)
MAX_TRAINING_EXAMPLES_PRODUCT = 250000 # Gemini 1.5 Flash limit

# 8. Prompt Template (Keep consistent with previous training/inference)
PROMPT_TEMPLATE = "Based on Hindu philosophy and Vedic teachings from the provided text, provide a comprehensive, detailed, and well-explained answer to the devotee's question. Elaborate on concepts, context, and implications:\nQuestion: {input_value}\nAnswer:"

# --- Function to Prepare Data from Q&A JSON (No changes needed here) ---
def prepare_tuning_data_from_qa_json(json_path, prompt_template, input_key, output_key):
    """Loads generated Q&A data from JSON, validates, formats it for tuning."""
    print(f"Attempting to load generated Q&A data from: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Generated Q&A data file not found: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            raise TypeError(f"Expected JSON data in '{json_path}' to be a list of objects, but got {type(raw_data)}")
        print(f"Loaded {len(raw_data)} raw Q&A entries.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON file ({json_path}): {e}")
    except Exception as e:
        raise ValueError(f"Error loading JSON file ({json_path}): {e}")

    print("Validating and formatting Q&A data for tuning...")
    tuning_examples = []
    skipped_count = 0
    min_q_len = 5
    min_a_len = 5

    for i, entry in enumerate(raw_data):
        if not isinstance(entry, dict):
            skipped_count += 1
            continue

        question_value = entry.get(input_key)
        answer_value = entry.get(output_key)

        if question_value is None or answer_value is None:
            skipped_count += 1
            continue

        question_str = str(question_value).strip()
        answer_str = str(answer_value).strip()

        if not question_str or not answer_str or len(question_str) < min_q_len or len(answer_str) < min_a_len or question_str == answer_str:
            skipped_count += 1
            continue

        text_input_prompt = prompt_template.format(input_value=question_str)
        tuning_examples.append({
            "text_input": text_input_prompt,
            "output": answer_str
        })

    print(f"Formatted {len(tuning_examples)} valid Q&A examples for tuning.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} entries due to format/quality issues.")

    if len(tuning_examples) < 20:
        raise ValueError(f"Need at least 20 valid examples for tuning, found only {len(tuning_examples)}.")

    return tuning_examples

# --- Main Script Logic ---

if __name__ == "__main__":
    print("--- Starting Gemini Fine-Tuning Script for Product Q&A (Large Dataset) ---")
    print(f"Using Vedas JSON data from: {DATA_FILE_PATH}")
    print(f"Targeting New Model ID: {TUNED_MODEL_ID}")

    # 1. Configure API Key
    print("\nConfiguring Google AI API Key...")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("API Key configured.")
    except Exception as e:
        print(f"Error configuring API key: {e}")
        exit(1)

    # 2. Prepare Data (Load and format Q&A from JSON)
    try:
        print(f"\nPreparing data from Vedas JSON file (This may take a moment for large files)...")
        all_tuning_examples = prepare_tuning_data_from_qa_json(
            DATA_FILE_PATH, PROMPT_TEMPLATE, JSON_INPUT_KEY, JSON_OUTPUT_KEY
        )
        num_total_examples = len(all_tuning_examples)
        print(f"Total valid examples loaded: {num_total_examples}")

        # --- Apply Sampling Logic due to API Limits ---
        max_allowed_examples = MAX_TRAINING_EXAMPLES_PRODUCT // TUNING_EPOCHS
        print(f"\nAPI Limit Check: Max examples allowed for {TUNING_EPOCHS} epochs = {max_allowed_examples}")
        print(f"Total examples * epochs = {num_total_examples * TUNING_EPOCHS}")

        if num_total_examples * TUNING_EPOCHS > MAX_TRAINING_EXAMPLES_PRODUCT:
            print(f"Dataset size exceeds API limit. Sampling required.")
            sample_size = min(max_allowed_examples, num_total_examples)
            print(f"Randomly sampling {sample_size} examples from the dataset.")
            if sample_size < 20:
                 raise ValueError(f"Calculated sample size ({sample_size}) is below the minimum required (20).")
            training_data_list = random.sample(all_tuning_examples, sample_size)
        else:
            training_data_list = all_tuning_examples
            print(f"\nUsing all {len(training_data_list)} examples for tuning (within API limits).")

        num_examples_for_tuning = len(training_data_list)
        print(f"Final number of examples being used for tuning: {num_examples_for_tuning}")

        # Recommendation for effectiveness (less critical with large datasets, but still good practice)
        if num_examples_for_tuning < 200:
             print(f"Warning: Number of examples used for tuning ({num_examples_for_tuning}) is relatively low for potentially complex Q&A. More data generally helps.")

    except (FileNotFoundError, ValueError, IOError, TypeError) as e:
        print(f"\nError during data preparation: {e}")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error during data preparation: {e}")
        traceback.print_exc()
        exit(1)

    # 3. Create Tuned Model (Start the Fine-Tuning Job)
    print("\n--- Starting Fine-Tuning Job ---")
    print(f"Base Model: {BASE_MODEL_ID}")
    print(f"Dataset: {num_examples_for_tuning} Q&A examples (Sampled from {num_total_examples})")
    print(f"Epochs: {TUNING_EPOCHS}, Batch Size: {TUNING_BATCH_SIZE}")
    print(f"Target Tuned Model ID: {TUNED_MODEL_ID}")
    print(f"Target Tuned Model Display Name: {TUNED_MODEL_DISPLAY_NAME}")
    print("\n!!! Note: Fine-tuning with this larger dataset will take significantly longer (potentially several hours) !!!")

    try:
        # Shuffle the final training data list just before submitting
        print("Shuffling final training data...")
        random.shuffle(training_data_list)

        # Start the tuning job
        print("Submitting fine-tuning job to Google AI...")
        operation = genai.create_tuned_model(
            source_model=BASE_MODEL_ID,
            training_data=training_data_list,
            id = TUNED_MODEL_ID,
            display_name = TUNED_MODEL_DISPLAY_NAME,
            description = f"Fine-tuned {BASE_MODEL_ID} for Hundu Vedas V1).", # Update description
            epoch_count = TUNING_EPOCHS,
            batch_size = TUNING_BATCH_SIZE,
        )
        print(f"\nTuning job submitted successfully.")
        print(f"Operation Name: {operation.operation.name}")
        print(f"You can track the job status using this name (e.g., in Google AI Studio or via a tracking script)")
        print("Waiting for tuning job to complete (This will take several hours)...")

        # 4. Wait for Job Completion (Polling)
        timeout_seconds = 12 * 60 * 60 # 12 hour timeout (adjust if you expect >12h)
        tuned_model_result = operation.result(timeout=timeout_seconds)

        print("\n--- Tuning Job Completed! ---")
        print(f"Tuned Model Resource Name: {tuned_model_result.name}")
        print(f"Display Name: {tuned_model_result.display_name}")
        print(f"State: {tuned_model_result.state}")

        # 5. (Optional) Use the Tuned Model (Example Inference)
        if tuned_model_result.state == genai.types.State.ACTIVE:
            print("\n--- Example Inference with NEW Tuned Model (v2) ---")
            try:
                # IMPORTANT: Use the *new* model name here (v2)
                tuned_model_instance = genai.GenerativeModel(model_name=tuned_model_result.name)

                # Ask the same question that caused issues before
                sample_question = "What is the role of the Brahmanas?"
                inference_prompt = PROMPT_TEMPLATE.format(input_value=sample_question)

                print(f"Asking question: '{sample_question}'")
                # Use lower temperature for inference as discussed
                generation_config = genai.types.GenerationConfig(temperature=0.2)
                print(f"Using Generation Config: temperature=0.2")

                response = tuned_model_instance.generate_content(
                    inference_prompt,
                    generation_config=generation_config
                    )

                print("\nGenerated Answer:")
                if response.parts:
                    print(response.text.strip())
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                     print(f"Response blocked. Reason: {response.prompt_feedback.block_reason}")
                else:
                    print("Model did not generate a text response.")

            except Exception as inference_error:
                print(f"\nError during inference example: {inference_error}")
                traceback.print_exc()
        else:
            print(f"\nTuning job finished but model state is '{tuned_model_result.state}'. Cannot run inference example.")
            print("Please check the job status and details in Google AI Studio or Google Cloud Console.")

    except Exception as e:
        print(f"\n--- An Error Occurred During Tuning ---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        traceback.print_exc()
        print("\nPlease check the logs above and potentially the job status in Google AI Studio / Cloud Console.")

    finally:
        print("\n--- Fine-tuning Script Finished ---")