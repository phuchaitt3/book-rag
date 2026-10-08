# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
max_samples = 100
n_samples_per_context = 1  # Number of rows per context

# %%
output_file_dataset_filterd_full = "dataset_filtered_full.xlsx"
output_file_dataset_filterd_dropped = "dataset_filtered_dropped.xlsx"

output_file_queries_in_dataset = f"queries_{max_samples}_uniques_in_dataset.xlsx"
output_file_queries_similar_dataset_without_context = f"queries_{max_samples}_uniques_similar_dataset_without_context.xlsx"
output_file_queries_similar_dataset_within_context = f"queries_{max_samples}_uniques_similar_dataset_within_context.xlsx"

# %%
import pandas as pd
from datasets import load_dataset

# %%
import openai
import json
import os

# Load OpenAI API key
key_path = './config.json'
with open(key_path, "r") as file:
    config = json.load(file)
os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

openai.api_key = config["openai_api_key"]


# %%
def is_not_empty_answer(example):
    return bool(example['answers']['text'])

def is_having_no_common_element(list1, list2):
    """Checks if two lists have no common element."""
    set1 = set(list1)
    set2 = set(list2)
    return not set1.intersection(set2)

def call_chatgpt(user_message: dict,
        sys_message={"role": "system", "content": "You are a helpful assistant that responds in Vietnamese."},
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=0):
    response = openai.chat.completions.create(
        model=model,  # Use the appropriate model
        messages=[sys_message, user_message],
        max_tokens=max_tokens,  # Further reduced to ensure brevity
        temperature=temperature  # Keep at 0 for precision
    )

    # Get the content of the response
    response_content = response.choices[0].message.content.strip()

    # Return the extracted answer directly
    return response_content

def generate_similar_question(question: str, context: str=None):
    sys_message = {"role": "system", "content": "You are a helpful assistant that responds in Vietnamese."}
    
    if context is None:
        user_content = (f"Question: {question}\nPlease only extract exactly the question (after 'Question: ') and reponse only 1 SIMILAR question to the question with the following format, no more word:\nSimilar question: <similar question>\n")
    else:
        user_content = (f"Question: {question}\nContext: {context}\nPlease extract the question (after 'Question: '), the context (after 'Context: ') and reponse only 1 SIMILAR question to the question with the context with the following format, no more word:\nSimilar question: <similar question>\n")
    user_message = {"role": "user", "content": user_content}
    
    chatgpt_response = call_chatgpt(user_message, sys_message)
    print("ChatGPT: ", chatgpt_response)
    
    return chatgpt_response

def get_answer_from_response(response: str, prefix_str: str):
    return response[len(prefix_str):]

def get_values_in_object_lst(object_lst: list, attribute_str: str):
    return [obj[attribute_str].tolist()[0] for obj in object_lst]

def get_answer_texts_in_object_lst(object_lst: list):
    return [obj['answers'].tolist()[0]['text'].tolist()[0] for obj in object_lst]

def save_lists_dict_to_excel(filename, lists_dict: dict):
    """
    Saves multiple lists dictionary to an Excel file with column names.
    Format dict: {"column_name": list}
    """
    # Create a pandas DataFrame from the dictionary of lists
    df = pd.DataFrame(lists_dict)

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False, engine='openpyxl')  # Explicitly use openpyxl

    print(f"Data saved to {filename}")

def save_queries_to_excel(filename, contexts: list[str], questions: list[str], answers: list[str], more_cols: list[tuple[str, list]]=[]):
    lists_dict = {"context": contexts, "question": questions, "answer": answers}

    for col_name, col_list in more_cols:
        lists_dict[col_name] = col_list

    save_lists_dict_to_excel(filename, lists_dict)

def drop_dataframe(dropped_dataframe: pd.DataFrame, dropping_dataframe: pd.DataFrame):
    mask = ~dropped_dataframe.set_index('id').index.isin(dropping_dataframe.set_index('id').index)

    # Filter big dropped one using the mask
    return dropped_dataframe[mask].reset_index(drop=True)

def save_dataframe(filename:str, df: pd.DataFrame):
    # Save to Excel
    df.to_excel(filename, index=False)


# %%
# Load and filter the dataset
full_dataset = load_dataset("taidng/UIT-ViQuAD2.0")
raw_dataset = full_dataset['train']
filtered_dataset = raw_dataset.filter(is_not_empty_answer)

# %%
# Convert to DataFrame
df = filtered_dataset.to_pandas()

# Assume 'context' is a column in your DataFrame
contexts = df['context'].unique()

# %%
query_lst = []
question_lst = []
n_samples = 0
for context in contexts:
    context_df = df[df['context'] == context]
    sampled_df = context_df.sample(n=min(n_samples_per_context, len(context_df)), random_state=42)
    sampled_questions = sampled_df['question'].tolist()
    if is_having_no_common_element(sampled_questions, question_lst):
        question_lst += sampled_questions
        query_lst.append(sampled_df)

        n_samples += 1
        if n_samples == max_samples:
            break
print(len(query_lst))
print(len(question_lst))
print(query_lst)

# %%
n_rows = 1000

queries = query_lst[:n_rows]
questions = get_values_in_object_lst(queries, 'question')
contexts = get_values_in_object_lst(queries, 'context')
answers = get_answer_texts_in_object_lst(queries)
print(questions)
print(contexts)
print(answers)

# %%
save_queries_to_excel(output_file_queries_in_dataset, contexts, questions, answers)

# %%
# Call ChatGPT
similar_question_responses_without_context = [generate_similar_question(question) for question in questions]

# %%
similar_questions_without_context = [get_answer_from_response(response, "Similar question: ") for response in similar_question_responses_without_context]
print(similar_questions_without_context)

# %%
original_question_without_context = ("original_question_without_context", questions)
save_queries_to_excel(output_file_queries_similar_dataset_without_context, contexts, similar_questions_without_context, answers, [original_question_without_context])

# %%
# Call ChatGPT
similar_question_responses_within_context = [generate_similar_question(question, context) for question, context in zip(questions, contexts)]

# %%
similar_questions_within_context = [get_answer_from_response(response, "Similar question: ") for response in similar_question_responses_within_context]
print(similar_questions_within_context)

# %%
original_question_within_context = ("original_question_within_context", questions)
original_context_within_context = ("original_context_within_context", contexts)
save_queries_to_excel(output_file_queries_similar_dataset_within_context, contexts, similar_questions_within_context, answers, [original_question_within_context, original_context_within_context])

# %%
# Dataset to DataFrame
filtered_df = pd.DataFrame(filtered_dataset)

# list to DataFrame
query_df = pd.concat(query_lst[:n_rows])

dropped_df = drop_dataframe(filtered_df, query_df)
print(len(filtered_df))
print(len(query_df))
print(len(dropped_df))

# %%
save_dataframe(output_file_dataset_filterd_full, filtered_df)

# %%
save_dataframe(output_file_dataset_filterd_dropped, dropped_df)
