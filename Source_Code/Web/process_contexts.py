input_file_path = './processed_pdf.json'           
output_file_path = './processed_contexts.json'

import json
import re

import logging
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
key_path = './config.json'
with open(key_path, "r") as file:
    config = json.load(file)
    openai.api_key = config["openai_api_key"]

def read_json_file(file_path: str) -> list:
    """
    Reads a JSON file and returns the data as a list of dictionaries.

    :param file_path: Path to the JSON file.
    :return: List of dictionaries containing the JSON data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return []

def replace_newlines_without_punctuation(text: str) -> str:
    # Replace '\n' not preceded by '.', '?', or '!'
    result = re.sub(r'(?<![.?!])\n', ' ', text)
    return result

def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r'\n{2,}', text)
    return [p.strip() for p in paragraphs if p.strip()]

def split_into_sentences(text: str) -> list[str]:
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return sentences

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

def chunk_data(data, max_output_tokens: int=512) -> list[dict]:
    """
    Chunk content to get entries with format:
        **metadata,
        "context_type": "chunked",
        "summary_level": None,
        "summary_depth": None,
        "original_tokens": original_tokens,
        "original_chars": original_chars,
        "Heading": heading,
        "Nội dung": chunk
    """
    chunked_data = []
    
    for entry in data:
        heading = entry.get("Heading", "")  # Use .get() to handle missing keys
        content = entry.get("Nội dung", "")  
        metadata = {key: value for key, value in entry.items() if key != "Heading" and key != "Nội dung"}

        # Preprocess abundant newlines
        content = replace_newlines_without_punctuation(content)
        original_tokens = token_len(content)   
        original_chars = len(content)   
        
        # Combine heading and content
        full_text = f"{heading}\n{content}"
        
        # Split into paragraphs
        paragraphs = split_into_paragraphs(full_text)
        chunks = []
        
        for paragraph in paragraphs:
            if token_len(paragraph) <= max_output_tokens:
                chunks.append(paragraph)
            else:
                # Split paragraph into sentences
                sentences = split_into_sentences(paragraph)
                current_chunk = ""
                for sentence in sentences:
                    if token_len(current_chunk) + token_len(sentence) + 1 <= max_output_tokens:
                        current_chunk += " " + sentence
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        # Create a new entry with the same metadata and chunked content
        for chunk in chunks:
            chunked_entry = {
                **metadata,  # Unpack metadata into the dictionary
                "context_type": "chunked",
                "summary_level": None,
                "summary_depth": None,
                "original_tokens": original_tokens,
                "original_chars": original_chars,
                "Heading": heading,
                "Nội dung": chunk
            }
            chunked_data.append(chunked_entry)

    return chunked_data

def summarize_content(content, depth: int=0, max_response_tokens: int=512, overlap_rate: float=0.2) -> list[tuple[str, int]]:
    """
    Function to summarize content using OpenAI API    
    """
    if (token_len(content) <= max_response_tokens):
        return [(content, depth)]

    CONTEXT_WINDOW = 8000

    system_message = "You are a helpful assistant that responds in Vietnamese."
    system_message_tokens = token_len(system_message)
    # print("system_message_tokens =",system_message_tokens)

    base_user_message = "This is the content: \nLet's summarize this content to get a shorter content with max tokens  in the following format:\n\n[Your summary here]\n\nEnsure that the summary contains the main ideas of the content and its length is not over max length."
    base_user_message_tokens = token_len(base_user_message)
    # print("base_user_message_tokens =", base_user_message_tokens)

    chunk_tokens = int(CONTEXT_WINDOW - system_message_tokens - base_user_message_tokens - max_response_tokens)

    AVG_CHARS_PER_TOKEN = 5
    chunk_chars = int(chunk_tokens * AVG_CHARS_PER_TOKEN)

    overlap_chars = int(chunk_chars * overlap_rate)
    # print("chunk_tokens =", chunk_tokens, "chunk_chars =", chunk_chars, "overlap_chars =", overlap_chars)
    
    summaries = []
    for i in range(0, len(content), chunk_chars - overlap_chars):
        chunk = content[i:i+chunk_chars]  # Get the current chunk
        user_message = f"This is the content: {chunk}\nLet's summarize this content to get a shorter content with max tokens {max_response_tokens} in the following format:\n\n[Your summary here]\n\nEnsure that the summary contains the main ideas of the content and its length is not over max length."

        response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Use the appropriate model
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=max_response_tokens,
        n=1,
        temperature=0.7  # Adjust for creativity
        )
        # print(f"response:", response)

        summary = response.choices[0].message.content.strip()
        summary_tokens = token_len(summary)
        # print(f"summary ({summary_tokens} tokens):", summary)

        summaries.append((summary, depth + 1))
    
    if len(summaries) < 2:
        return summaries

    combined_summary = "\n".join(summary for summary, _ in summaries)
    combined_summaries = summarize_content(combined_summary, depth + 1, max_response_tokens, overlap_rate)
    return summaries + combined_summaries

def summarize_data(data, summary_level:str, max_output_tokens: int=512, overlap_rate: float=0.2) -> list[dict]:
    """
    Summarize content to get entries with format:
        other metadata,
        "context_type": "summarized",
        "summary_level": summary_level,
        "summary_depth": depth,
        "original_tokens": original_tokens,
        "original_chars": original_chars,
        "Nội dung": summary
    """
    summarized_data = []
    
    for entry in data:
        content = entry.get("Nội dung", "") # Use .get() to handle missing keys 
        metadata = {key: value for key, value in entry.items() if key != "Nội dung"}

        # Preprocess abundant newlines
        content = replace_newlines_without_punctuation(content)
        original_tokens = token_len(content)   
        original_chars = len(content)

        summaries = summarize_content(content, 0, max_output_tokens, overlap_rate)

        # Create a new entry with the same metadata and summarized content
        for summary, depth in summaries:
            summarized_entry = {
                **metadata,  # Unpack metadata into the dictionary
                "context_type": "summarized",
                "summary_level": summary_level,
                "summary_depth": str(depth),
                "original_tokens": original_tokens,
                "original_chars": original_chars,
                "Nội dung": summary
            }
            summarized_data.append(summarized_entry)
    return summarized_data

def save_to_json_file(data, file_path: str):
    """
    Saves the data to a JSON file.

    data: Data to be saved.
    file_path: Path to the JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error saving to file: {file_path}. Error: {e}")

data = read_json_file(input_file_path)

chunked_data = chunk_data(data, max_output_tokens=512)
summarized_data = summarize_data(data, "heading", max_output_tokens=512, overlap_rate=0.2)

save_to_json_file(chunked_data + summarized_data, output_file_path)

print(f"Summarized and chunked data saved to {output_file_path}")