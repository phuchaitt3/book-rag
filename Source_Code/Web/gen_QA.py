input_file = "processed_contexts.json"
output_file = 'result.xlsx'

enforce_max_pairs = True

# Check if the flag is set in the global namespace, otherwise default to False
if 'enforce_max_pairs' not in globals():
    enforce_max_pairs = False

# Define the maximum number of QA pairs to generate
MAX_QA_PAIRS = 2  # Set your desired limit here

import json
import openai
import logging
import time
import re
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
key_path = './config.json'
with open(key_path, "r") as file:
    config = json.load(file)
    openai.api_key = config["openai_api_key"]

# Load the context from chunked_output.json
logger.info(f"Loading context from {input_file}...")
with open(input_file, 'r', encoding='utf-8') as file:
    contexts = json.load(file)

# Boolean flag to enable testing mode
testing_mode = False

# Number of entries to process in testing mode
test_entries_count = 5

# Slice the contexts if testing mode is enabled
if testing_mode:
    contexts = contexts[:test_entries_count]
    logger.info(f"Testing mode is ON. Processing the first {test_entries_count} entries.")

def tokenize_by_splitting(text: str) -> list[str]:
    """
    Tokenizes a Vietnamese string with improved handling of end-of-sentence punctuation.
    """
    # Add spaces around common punctuation marks
    text = re.sub(r"([.,?!])", r" \1 ", text)

    # Handle cases with multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Split by space character
    tokens = text.split()
    return tokens

def token_len(text: str) -> int:
    """
    Get number of tokens in a text
    """
    return len(tokenize_by_splitting(text))

AVG_TOKENS_PER_PAIR = 100

# Function to generate questions and answers using OpenAI API
def generate_qa_pairs(context):

    pairs_tokens = token_len(context) # Max: 512
    # print(f"Number of tokens: {pairs_tokens}")

    num_return_sequences = max(1, round(pairs_tokens / AVG_TOKENS_PER_PAIR))
    # print(f"Number of return sequences: {num_return_sequences}")

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Use the appropriate model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in Vietnamese."},
            {"role": "user", "content": f"This is the context: {context}\nBased on this context, generate {num_return_sequences} questions and answers in Vietnamese in the following format:\n\n1. [Your question here]\n2. [Extracted answer from context]3. [Index of the first word of the answer in the context]\n\nEnsure that the answers are directly extracted from the context and provide the correct starting index of the answer in the context."}
        ],
        # max_tokens=150,  # Adjust based on your needs
        # n=num_return_sequences,
        temperature=0.7  # Adjust for creativity
    )

    return [choice.message.content.strip() for choice in response.choices]

# Function to parse the generated QA pairs into the desired structure
def parse_qa_pairs(qa_pairs):
    parsed_pairs = []

    how_many_pairs = 0
    debug_count = 0

    for qa_pair in qa_pairs:
        how_many_pairs += 1
        # print(f"Number of pairs: {how_many_pairs}")

        # print("-----")
        # print(qa_pair)
        # print("-----")

        # Use regex to extract the question, answer, and index
        matches = re.findall(r'(\d+)\.\s*(.*?)\n(\d+)\.\s*(.*?)\n(\d+)\.\s*(\d+)', qa_pair, re.DOTALL)

        for match in matches:
            try:
                # Extract question, answer, and index from the match
                question_number, question, answer_number, answer_text, index_number, answer_start = match

                debug_count += 1
                # print(f"Debug count: {debug_count}")

                # Construct the answer structure
                answer_data = {
                    "text": [answer_text.strip()],
                    "answer_start": [int(answer_start.strip())]
                }

                parsed_pairs.append({
                    "question": question.strip(),
                    "answers": answer_data
                })
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing QA pair: {e}")

    return parsed_pairs

# Generate QA pairs for each context
generated_qa_pairs = []

start_time = time.time()

# Initialize a counter for the total number of generated QA pairs
total_qa_pairs_generated = 0

for i, entry in enumerate(contexts):
    if enforce_max_pairs and total_qa_pairs_generated >= MAX_QA_PAIRS:
        logger.info(f"Reached the maximum limit of {MAX_QA_PAIRS} QA pairs. Stopping early.")
        break

    context = entry.get('Nội dung', '')  # Use .get() to handle missing keys
    metadata = {key: value for key, value in entry.items() if key != 'Nội dung'}

    qa_pairs = generate_qa_pairs(context)
    # print(qa_pairs)
    parsed_pairs = parse_qa_pairs(qa_pairs)

    for parsed_pair in parsed_pairs:
        generated_qa_pairs.append({
            "book_title": entry.get("Tên sách", None),
            "authors": entry.get("Tác giả", None),
            "publication_time": entry.get("Thời gian xuất bản", None),
            "context": context,  # Add context before the question
            **parsed_pair,  # Unpack the parsed QA pair
            "heading": entry.get("Heading", None),
            "heading_level": entry.get("Cấp độ heading", None),
            "parent_headings": entry.get("Các heading lớn hơn", []),
            "page_num": entry.get("Số trang", None),
            "original_tokens": entry.get("original_tokens", None),
            "original_chars": entry.get("original_chars", None),
            "context_type": entry.get("context_type", None),
            "summary_level": entry.get("summary_level", None),
            "summary_depth": entry.get("summary_depth", None),
        })

        total_qa_pairs_generated += 1

    logger.info(f"Generated QA pairs for context {i+1}/{len(contexts)}")

# end_time = time.time()
# elapsed_time = end_time - start_time

# # Calculate average time per context
# average_time_per_context = elapsed_time / len(contexts)
# logger.info(f"Average time per context: {average_time_per_context:.4f} seconds")

from openpyxl import load_workbook

# Convert the generated QA pairs to a DataFrame
df = pd.DataFrame(generated_qa_pairs)

try:
    # Load the existing workbook
    book = load_workbook(output_file)
    # Use ExcelWriter with openpyxl engine in append mode
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        # Check if 'Sheet1' exists
        if 'Sheet1' in book.sheetnames:
            # If 'Sheet1' exists, append the data
            startrow = book['Sheet1'].max_row
            df.to_excel(writer, index=False, header=False, startrow=startrow, sheet_name='Sheet1')
        else:
            # If 'Sheet1' does not exist, create it
            df.to_excel(writer, index=False, sheet_name='Sheet1')
except FileNotFoundError:
    # If the file does not exist, create a new workbook
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

logger.info(f"QA pairs have been appended to {output_file}")