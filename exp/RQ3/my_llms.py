import os
import sys
import json
import logging
from tqdm import tqdm
import openai
from baseline_util import *
from my_util import *
from multiprocessing import Process
import pandas as pd

# --- Configuration ---
OPENAI_MODEL = "GPT-4o-mini"
OUTPUT_BASE_DIR = "../prediction/"
MAX_CODE_LENGTH = 1000000
BASE_URL = "YOUR_BASE_URL"
OPENAI_API_KEY = "YOUR_API_KEY"

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI Client
client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY,
)


def call_openai_model(file_id: str, filename: str, code_content: str, current_logger: logging.Logger) -> dict:
    prediction_data = {
        "id": file_id,
        "filename": filename,
        "defective": False,
        "refactoring_strategy": "",
        "refactored_code": "",
        "error_type": None,
        "error_message": None,
        "raw_response": ""
    }

    original_code_length = len(code_content)

    if original_code_length > MAX_CODE_LENGTH:
        prediction_data['error_type'] = "CODE_LENGTH_EXCEEDED"
        prediction_data['error_message'] = (
            f"Code length ({original_code_length} characters) exceeds "
            f"MAX_CODE_LENGTH ({MAX_CODE_LENGTH} characters). File skipped."
        )
        current_logger.warning(
            f"[{filename}] Warning: {prediction_data['error_message']}"
        )
        return prediction_data

    # Updated Expert SE Prompt
    messages = [
        {"role": "system",
         "content": "You are an expert software quality assurance engineer with 10 years of experience. Your task is to analyze the provided code file and identify any potential software defects, bugs, or architectural vulnerabilities. The code metrics are a combination of McCabe and complexity metrics in conjunction with extended CKOO metrics and Lines of Code. You need to generate a refactoring strategy and provide the exact refactored code to demonstrate how to fix the module. Respond only with a JSON object."},
        {"role": "user",
         "content": f"The code is as follows: {code_content}\n\nYour response MUST be a JSON object with the following structure:\n{{\n  \"id\": \"{file_id}\",\n  \"filename\": \"{filename}\",\n  \"defective\": <true_or_false>,\n  \"refactoring_strategy\": \"<your analysis and strategy>\",\n  \"refactored_code\": \"<the fixed code>\"\n}}\nThe 'defective' field must be a boolean (true or false)."}
    ]

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"}  # Recommend JSON mode for robustness
        )
        response_text = completion.choices[0].message.content
        prediction_data["raw_response"] = response_text
        current_logger.info(f"[{filename}] Raw API response: {response_text}")

        parsed_json = json.loads(response_text)

        if not isinstance(parsed_json.get("defective"), bool):
            current_logger.warning(f"Warning: 'defective' field is not boolean. Defaulting to False.")
            prediction_data['defective'] = False
        else:
            prediction_data['defective'] = parsed_json['defective']

        prediction_data['refactoring_strategy'] = parsed_json.get('refactoring_strategy', '')
        prediction_data['refactored_code'] = parsed_json.get('refactored_code', '')

    except openai.APIError as e:
        prediction_data['error_type'] = "API_ERROR"
        prediction_data['error_message'] = str(e)
        current_logger.error(f"OpenAI API Error for {filename}: {e}")
    except json.JSONDecodeError as e:
        prediction_data['error_type'] = "JSON_DECODE_ERROR"
        prediction_data['error_message'] = str(e)
        current_logger.error(f"JSON Decode Error for {filename}: {e}")
    except Exception as e:
        prediction_data['error_type'] = "UNEXPECTED_ERROR"
        prediction_data['error_message'] = str(e)
        current_logger.error(f"An unexpected error occurred for {filename}: {e}")

    return prediction_data


def predict_model_for_dataset(dataset_name: str):
    local_logger = logging.getLogger(f"prediction_logger_{dataset_name}")
    local_logger.setLevel(logging.INFO)
    local_logger.propagate = False

    log_file_name = f"prediction_process_CEOS_Refactor_{dataset_name}.log"
    local_log_file_path = os.path.join(OUTPUT_BASE_DIR, log_file_name)

    if local_logger.handlers:
        for handler in local_logger.handlers[:]:
            local_logger.removeHandler(handler)

    file_handler = logging.FileHandler(local_log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    local_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    local_logger.addHandler(console_handler)

    local_logger.info(f"Starting prediction for dataset: {dataset_name}")

    if dataset_name not in all_eval_releases:
        local_logger.error(f"Error: Dataset '{dataset_name}' not found.")
        return

    eval_rels = all_eval_releases[dataset_name][1:]

    error_records_for_dataset = []
    all_predictions_for_dataset = []

    prediction_id = 0
    for rel in eval_rels:
        local_logger.info(f"\nProcessing release: {rel}")
        row_list = []
        test_df = get_df(rel, is_baseline=True)

        for filename, df in tqdm(test_df.groupby('filename'), desc=f"[Predicting for {rel}]"):
            file_label = bool(df['file-label'].unique()[0])
            code = list(df['code_line'])
            code_str = get_code_str(code, True)
            prediction_id += 1

            prediction_result = call_openai_model(str(prediction_id), filename, code_str, local_logger)

            if prediction_result.get("error_type"):
                error_records_for_dataset.append({
                    "id": prediction_result["id"],
                    "filename": prediction_result["filename"],
                    "error_type": prediction_result["error_type"],
                    "error_message": prediction_result["error_message"]
                })

            prediction_result['true_defective'] = file_label
            row_list.append(prediction_result)

        predictions_df = pd.DataFrame(row_list)
        # Suffix updated to avoid overwriting previous plain prediction runs
        output_csv_path = os.path.join(OUTPUT_BASE_DIR, f"{dataset_name}_{rel}_predictions_refactored.csv")
        predictions_df.to_csv(output_csv_path, index=False)
        local_logger.info(f"Predictions for {rel} saved to {output_csv_path}")
        all_predictions_for_dataset.extend(row_list)

    if error_records_for_dataset:
        errors_df = pd.DataFrame(error_records_for_dataset)
        error_csv_path = os.path.join(OUTPUT_BASE_DIR, f"{dataset_name}_errors.csv")
        errors_df.to_csv(error_csv_path, index=False)

    local_logger.info(f"Finished prediction for dataset: {dataset_name}")


if __name__ == '__main__':
    # Targeted Projects based on your instruction
    datasets_to_run = ["jedit", "camel", "log4j", "xalan", "ant", "velocity", "synapse"]

    processes = []

    for ds_name in datasets_to_run:
        p = Process(target=predict_model_for_dataset, args=(ds_name,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"All logs and CSV files are saved in '{os.path.abspath(OUTPUT_BASE_DIR)}'.")
