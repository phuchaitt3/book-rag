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

# %% id="hF6jTdt6Wyqq"
max_samples = 100

# %% id="pgX57YLBWyqt"
input_file_dataset_filtered_full =      "dataset_filtered_full.xlsx"
input_file_dataset_filterd_dropped =    f"dataset_filtered_{max_samples}_dropped.xlsx"

input_file_queries_in_dataset =                         f"queries_{max_samples}_uniques_in_dataset.xlsx"
input_file_queries_similar_dataset_within_context =     f"queries_{max_samples}_uniques_similar_dataset_within_context.xlsx"
input_file_queries_similar_dataset_without_context =    f"queries_{max_samples}_uniques_similar_dataset_without_context.xlsx"

file_retrieved_contexts_ = f'retrieved_contexts_{max_samples}_'
file_conversation_ = f'conversation_{max_samples}_'
file_answer_ = f'answer_{max_samples}_'
file_response_ = f'response_{max_samples}_'

file_evaluations_ = f'evaluations_{max_samples}_'

DATASET_FILTERED_FULL_CONTEXT_EMBEDDINGS_FILE =   "dataset_filtered_full_context_embeddings.pt"    #File to save embeddings
DATASET_FILTERED_FULL_QUESTION_EMBEDDINGS_FILE =  "dataset_filtered_full_question_embeddings.pt"   #File to save embeddings
DATASET_FILTERED_FULL_QUESTION_INDEX_FILE =       "dataset_filtered_full_question_faiss_index.bin" #File to save index

DATASET_FILTERED_DROPPED_CONTEXT_EMBEDDINGS_FILE =   "dataset_filtered_dropped_context_embeddings.pt"    #File to save embeddings
DATASET_FILTERED_DROPPED_QUESTION_EMBEDDINGS_FILE =  "dataset_filtered_dropped_question_embeddings.pt"   #File to save embeddings
DATASET_FILTERED_DROPPED_QUESTION_INDEX_FILE =       "dataset_filtered_dropped_question_faiss_index.bin" #File to save index

# %% id="q5wvjcnMWyqv"
# version = 'in_dataset'
# version = 'out_dataset'
# version = 'similar_dataset_within_context'
version = 'similar_dataset_without_context'

if version == 'in_dataset':
    input_file_dataset = input_file_dataset_filtered_full
    input_file_queries = input_file_queries_in_dataset

    DATASET_CONTEXT_EMBEDDINGS_FILE = DATASET_FILTERED_FULL_CONTEXT_EMBEDDINGS_FILE
    DATASET_QUESTION_EMBEDDINGS_FILE = DATASET_FILTERED_FULL_QUESTION_EMBEDDINGS_FILE
    DATASET_QUESTION_INDEX_FILE = DATASET_FILTERED_FULL_QUESTION_INDEX_FILE
elif version == 'out_dataset':
    input_file_dataset = input_file_dataset_filterd_dropped
    input_file_queries = input_file_queries_in_dataset

    DATASET_CONTEXT_EMBEDDINGS_FILE = DATASET_FILTERED_DROPPED_CONTEXT_EMBEDDINGS_FILE
    DATASET_QUESTION_EMBEDDINGS_FILE = DATASET_FILTERED_DROPPED_QUESTION_EMBEDDINGS_FILE
    DATASET_QUESTION_INDEX_FILE = DATASET_FILTERED_DROPPED_QUESTION_INDEX_FILE
elif version == 'similar_dataset_within_context':
    input_file_dataset = input_file_dataset_filtered_full
    input_file_queries = input_file_queries_similar_dataset_within_context

    DATASET_CONTEXT_EMBEDDINGS_FILE = DATASET_FILTERED_FULL_CONTEXT_EMBEDDINGS_FILE
    DATASET_QUESTION_EMBEDDINGS_FILE = DATASET_FILTERED_FULL_QUESTION_EMBEDDINGS_FILE
    DATASET_QUESTION_INDEX_FILE = DATASET_FILTERED_FULL_QUESTION_INDEX_FILE
elif version == 'similar_dataset_without_context':
    input_file_dataset = input_file_dataset_filtered_full
    input_file_queries = input_file_queries_similar_dataset_without_context

    DATASET_CONTEXT_EMBEDDINGS_FILE = DATASET_FILTERED_FULL_CONTEXT_EMBEDDINGS_FILE
    DATASET_QUESTION_EMBEDDINGS_FILE = DATASET_FILTERED_FULL_QUESTION_EMBEDDINGS_FILE
    DATASET_QUESTION_INDEX_FILE = DATASET_FILTERED_FULL_QUESTION_INDEX_FILE

middle_file_retrieved_contexts = file_retrieved_contexts_ + version + '.xlsx'
middle_file_conversation = file_conversation_ + version + '.xlsx'
middle_file_answer = file_answer_ + version + '.xlsx'
middle_file_response = file_response_ + version + '.xlsx'

output_file_evaluations = file_evaluations_ + version + '.xlsx'

summary_file_reordered_True = f"evaluation_summary_{max_samples}_True_" + version + ".xlsx"
summary_file_reordered_False = f"evaluation_summary_{max_samples}_False_" + version + ".xlsx"

# %% [markdown] id="OGpM_9_Zn_WS"
# # Import libraries and models

# %% id="dYrr58VUWyqv"
import numpy as np
import pandas as pd
import ast

# %% id="0x0ha3gjWyqw"
import openai
import json
import os

# Load OpenAI API key
key_path = './config.json'
with open(key_path, "r") as file:
    config = json.load(file)
os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

openai.api_key = config["openai_api_key"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 900, "referenced_widgets": ["b7a436fbfd2b47ae854503d7427f18f1", "cdbcbfa19f9b4fe9bffb8bb7c71e6ec7", "e68c91f8f0ff416f8408183c3f441bde", "34ed43c3e66f463e9c1c7159a1fd70ef", "5a4e2e09374d4c0f8fd9536a468238ac", "6ec2ef163381469e9dd588089febd59e", "66a2a42fda2549f69bd9e2775b3479af", "61f129ae007140c2a209bdf5b117bee1", "82ea0b9974e64642b6f3098272eb1de7", "5b9f4c91a44042c2abb2bd4f30b0e08f", "efece34bd0904eb4bb34a6403073797c", "2e0e6166896348ceb7286b71a0aa39b3", "eb9b831c43794271b56e8b44817bd8a8", "77d84015c73c4739b4d0d31102e01549", "4142c6949fea4fc79cab7bf59a9e56b0", "129623a7f7fc4529b13ccaee78dbba9e", "dafa924c6ee9494db8c99e038a463553", "123403e57706421da3ca50872e0d095b", "ea7951283bc7486e9c502d9d68f5b98c", "b2b210bbb0104a9f8046a02d1bb00be1", "b7d4bd0ba6cc4588b8ff1f09c458c499", "5b8898c5a0414fa3b25399089ddc89b1", "fde9bf8658d646f28c8cee8f6c4962fb", "f4ee7708a15f4dc0aad9736da9d852f0", "eaddd7b347b44de4a1c87f1d1c77e9a8", "2cfac4dca6714d76906ae75cf3f13fdb", "d060a7164d8240e583e90a1d9ebf5778", "0de37969384f4b48aaa2753738e9fd05", "91acb4f107c8403daf11f67d75d6c59b", "f0bf50ea0cf14f85bc19bb66b7c863a6", "93a0ed38bfc9454f8b791fa8cd566cc4", "20667b859a65435c8b147077164b6037", "76dfc149016942b8915a0721d4124df3", "91304fc02fb441c9ab3127393df694d9", "19e4584c39314c10a315fbff704bc1dc", "a33b7e8e72784034ba4c43c9b3be90fe", "ab30e49afb4f4a438d4c019699a73422", "f6fd91549fce4c558f93ae0ef6aaca1a", "8fd68815a82b48789e5b427522a4521c", "b3cbe8bda5174f4a94d1e986e96d0e5e", "87084f85c6234dedb98002c167fa315b", "d0628a5e9f9a4d7bb2492d9ae168de22", "93ac44f8fecc4374904fc3125dbc0ee9", "8f067cc4734e41b6a10cbf2dc4faa500", "f12531b2cf5a4e00b1f60f60439f0287", "d964305591fa459599278b642182a5a0", "3e185351a5664c2da1c70dc598cd4413", "b64107a6be784d72a7e3ec9607a32691", "4d4a7386e41444ed895fced6d1551e2e", "015f825262bd4fcbaf40c8d7e31c71bc", "2918054d17e64ce1a84ca28eceb57c18", "de3207f301a045f1acd0192b929a6c53", "c3440ded4d9b4c3fb7754bb193b17600", "e7df70bc68d143debf740a560a0032da", "9a019dc1f9f8485ea062182cf827e1c8", "37b0eb5b7c3347bd991130fa1c654b55", "ae58e8c6e1ff4503af62261d7b02633b", "0c3705686292431ba08b87ca0db9c5f8", "12be2cb4a0e34e70973ee406be603f18", "bf75e1113ef640378808547b4af7a35e", "2aa41debe04d4661baa7eece93da5717", "2d54fe59c6cc4f24aa4029d1a78404e8", "cd89d4d4037049a7afabdc8ccb1276d0", "2679f35b0acf4886b13b55b928fb7191", "8fe5d8d4464b455e815a1187fbc8b586", "19c85f04ccfd4467b808b0a7b4d54cf1", "c6ac39c2e69e4edaaa4b5e1822536cc5", "6aacf18277d34344b531a3a801aea362", "0ada4c32a1664ed4894a1a04f3c856b2", "d4a10c43a8014f05a64e8913def50c5d", "d90bbbf9030d4cdc8a77167049d664f5", "0b9a6848bc49483b92b753f65ac652d8", "0c1115abd73043e2887cf928846536cd", "828fc6c8877d4c3aa0ea6695194d3808", "5d477f8553044ff4b4e835ec27865329", "2c129cb447a44f5190b200665910baeb", "fe426923fb614d648b78b0fda57964eb", "f1b7e7efcd70475ea503bbccbb447583", "9a6c1cfaa380452ea8c08c854e0d42e9", "275b488c1ef34508968677dfc444eeaf", "bb8e4568017e4206b95b91de71e5a897", "a1ab253c0f66456f9dfdabc1f5af9884", "a69652a9717f45f896e62f05673ab254", "7c40ac7a18b14ffd8b881751ea3782ad", "a5c638d8fc794b009b235ea90badbeea", "f0a019f0fe714c5cabeec1ea1191bc3a", "d542a1e9c3ee4b63ab27ccf76ac3f484", "db97c35014694e61a818939beeca6659", "e882e4772f8b41c49ab06631ad05f32b", "cd48d57bb62b4318ac4874ba3c5a032d", "6222842dd7cb4ddf9a1db1548339bdcc", "66ec143191a6409c81bb374a6a143192", "ed01070a89a04e8e93f91941d470f934", "657e414f381045a3971011a2b58d7b9f", "3fb02b6a1aed4daabc967d2ae95ef029", "77164746dfe34b5eadbab5d0428381bc", "a3f8c451dfa74284b7b046c30300a9d8", "56f190bac8444806aa20268e851f4784", "d8df23a7f5604260955dbf49de73b771", "5c6a4758dbf84183986a05c806e2991c", "b2e2b2d9f70848b5877eb9ee4ba2a24c", "38ffa524d6f54e819d95137b50afcd36", "0a0dca059d24463a972ce993d79a912d", "93ab58daa16e438bbd92bc345ca7b8ea", "65efdd4c71774431ae7444ba41da47ba", "446ec26cc9184965962fac715f5eb449", "b75f9a8a07dd43dbb8b16e29feea50df", "4ad257004b3d4a699cd25f0a6beb0630", "d3259ebd75194622a61dbe476b46ef82", "1af5880fe4494b7f821caf02fc9b53c1", "69981757a2634bdb82328ea3c6a94fd8", "0be024ac1deb44f7bb2bf0c1c7863828", "3aa1ca9fa29f40bda8e6c9c3d5a33bd8", "fd5cf6678ac14202a3a85fd0dae7308f", "2901c7cfc7004f23bfaa5e0efa06365c", "a48040a775c4405fae09253b6204ccca", "b6bcec1bd6d9426b93ce9ca871874350", "1e3872d14cd646498542b28098d945e2", "f978d248663743c99cc376c15012e941", "2a51c1b3158240159cb1c72310dd3269", "f05debfabb3045989574db9f0c225fd1", "e3cc31d5b3ac4f7999fa5269a35f6513", "7c00f11d238e4420a741be392f6dc659", "579faa5713a548eca1fc6dcdde8d0a79", "f2bb1528e8654894872d8f052b084600", "e3897d5df6d54e77b810246f5475c11f", "400ef9c88ba14d8580221aeb02f9a52d", "76a4698101b048d48d6cdc7f59fe546f", "6c017b21f3914b4897f7129723acda09", "e20e1752052e4200829b9dbcb66a3410", "999efd7ac8d343379ae219fe700feb14", "4ad54cd38a8d4b9cb64138f0fb55291b", "d3b08de95e7942eda47ba58be86dbac1", "d7c418280db74f6d85c53dcc72a909c2", "6551bb5898c4460c88ffff0fdecc19f8", "9e298d7449f047cdb921401cafac2799", "6f9c193f8be740acaccd40a6cd47ba2a", "5fa20209fbd54361be0e7d044a3e5e34", "73d578eb43b3484f85d94b05ec83b8ca", "6f0814a6c2544a7ba17813a55381b02a", "8e9bb94e8cb94c519367128eee598525", "bd0e2348fc8b480485eab658bc1a0316", "22a93a54af874f639d6fc30bb14972a3", "3818c2e5bd55456f974e1e58892c2833", "e39174ff3d82491493c9a1c441e84f3e", "3dba55c0a8494a2fb8ae48edf819ae23", "88cf36500f204557a51954f8cac89b9a", "6fbe52b222fa490da0bc8d7a75f3878e", "e1386e5f15b44f7ab6209a8766666abb", "82aad26311bb4f12884a69b26684e197", "ea7e3a189d8e451bbbcee05616a48807", "ee1de8a8e273449f908fcafb9c2c2d0f", "541aee08593b410fb2835b494f36731a", "858e8d9ecb794753b3fb80c63690dcfe", "4062095449d045399d577565a63f690c", "6b7c5043c22e4a67bd5ede5fd4a9ee11", "bc9658185f2147f99c8c14821f2f4014", "36edef04803540c887c70f9347127a2d", "4177ab823a254b9497be6d3ef1d6ec2b", "65aac2bede6046f8a882bee05065d0e9", "f664f10aec354b7eb449ac2a1fda0daf", "52483781fd12446ca8d18e3eb76bb09c", "1cf93318075b4573868dbe21cb3d1f33", "a5ec14d1a58e4a5aaa2b839e24a9fa4f", "709e67da04d64b72ac35f6b36f2ddf4f", "5c5ba5e240c34374a1edb8a969cbfe74", "b40071412ba548d0abf8bfd00c96e019", "3569049610be4fd9b4e04b1530b9c89f", "124b89d0a740428cbdaefb04f208a724", "c2354f6f3a3647d9931ab1039bbb7f42", "648e37fb8b904977af1a869e506bec74", "814bb10642a94535989d555d45c21e73", "b93f8c9d8cef45c09d32015f1d03fa05", "cdf7368c809843af8b6bad3987af0a84", "c5e2047484ab46209c32cfcaf46c8519", "74727910755143cb8c54239701447d66", "db855e26af4e4b4a91168ad7a266952b", "4340893015e94fceb0ac647005440967", "cc0d4e62ab684a438066005861ea7da5", "f3747b0d144648949c6e149b4953ce87", "01e364965b834022aafc632730238bf0", "6b00fb24f42946d295e35294b4b42ae7", "8f219e27c78b4bf68327ffbaf45837d0", "cc5d4f41d31840c98516e0111e6c90f8", "889640f7217141f29f18d2f686ea1c09", "461d1b293de046acb3eca7ab0b1efab8", "5038797288e644658aab6d79f6a5f0ca", "318e199816b640bd8cb4a77afef297ef", "b9e2522473974da4810e0f8305021992", "ba49a212ed7a41658e17fb85e3c2b032", "ac26f651b0574b41b2e389506c957910", "cd5c7f5a92ba421e94739402a9ef67a1", "874b9317f6214e1f95f180a3b7bcee21", "1ca97cacfccd442196bc2eb4d6b9b5f7", "d54807fa58824de090d4f33925f1f7e6", "bace79f7e39a48febf0ac492cad2ef13", "b98a4570b2fc46c7a293bd70d81b7914", "bd27b628e02242f6a34f9565a6597107", "9e79994d0e5645f5a6a1daf5d61af786", "ac2c1905445a49d2a9f31e1e8e330559", "1b9ae6597bea4352962dd6cd00353d27", "1d1ebae0f9be4fb0944878e1be447236", "eb99f27cab06486ea930d779e7740e02", "e4eb62e5c9924ea99e87f1d1a62da3d0", "52d6fd97954243afa6c879d6d07e6d5d", "9424195a4a9146608ac328488b48ee9a", "b246b38d13114a5f952d185486566d31", "9f6b8bbbf34241b282ea344984a1635a", "90f13b0458594f5eb056c00b695b6bba"]} executionInfo={"elapsed": 48471, "status": "ok", "timestamp": 1736081826794, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="tjVBefZtiHDl" outputId="e9ce1dc0-c64e-439c-b9f2-72c604885535"
# # !pip install faiss-cpu
import faiss
import torch

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24793, "status": "ok", "timestamp": 1736081869706, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="3iBE31-MWyqx" outputId="56fc8a05-c35f-45f3-a00b-30ccc8f19be2"
# # !pip install ragas
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper

from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from langchain_openai import ChatOpenAI
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=config["openai_api_key"]))


# %% [markdown] id="bFks3rJfoEib"
# # Define functions

# %% id="GYmapsE4iHDl"
def create_embeddings(sentences, EMBEDDINGS_FILE: str=None):
    """Creates sentence embeddings and saves them to a file."""
    embeddings = model.encode(sentences, convert_to_tensor=True)

    if EMBEDDINGS_FILE is not None:
        torch.save(embeddings, EMBEDDINGS_FILE) #save to file
    return embeddings

def load_embeddings(EMBEDDINGS_FILE):
    """Loads embeddings from a file or creates them if the file doesn't exist."""
    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading embeddings from file...")
        embeddings = torch.load(EMBEDDINGS_FILE)
        return embeddings
    else:
        return None  # Indicate that embeddings need to be created

def create_faiss_index(embeddings, INDEX_FILE: str=None):
    """Creates a FAISS index and saves it to a file."""
    embeddings_np = embeddings.cpu().detach().numpy()
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Or IndexIVFFlat for larger datasets
    index.add(embeddings_np)

    if INDEX_FILE is not None:
        faiss.write_index(index, INDEX_FILE) #save to file
    return index

def load_faiss_index(INDEX_FILE):
    """Loads a FAISS index from a file or creates it if the file doesn't exist."""
    if os.path.exists(INDEX_FILE):
        print("Loading FAISS index from file...")
        index = faiss.read_index(INDEX_FILE)
        return index
    else:
        return None  # Indicate that index needs to be created


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

def generate_answer(context: str, question: str):
    sys_message = {"role": "system", "content": "You are a helpful assistant that responds in Vietnamese."}

    user_content = f"Context: {context}\nQuestion: {question}\nPlease only extract the exact continuous words from the context that answers the question. For example, context: Đến năm 1327, đây là thị trấn lớn thứ ba tại Warwickshire, question: Vào thế kỉ XIV, Birmingham trở thành thị trấn lớn thứ mấy tại Warwickshire?, answer: lớn thứ ba, not: thị trấn lớn thứ ba."
    user_message = {"role": "user", "content": user_content}

    chatgpt_response = call_chatgpt(user_message, sys_message)
    print("ChatGPT: ", chatgpt_response)

    return chatgpt_response


def format_retrieved_contexts(query: str, contexts: list[str]) -> dict:
    return {"user_query": query,
            "retrieved_contexts": contexts}

def combine_str_lst(str_lst: list[str], sep: str=' ') -> str:
    return sep.join(str_lst)

def format_conversation(query: str, contexts: list[str]) -> dict:
    combined_context = combine_str_lst(contexts)
    return {"role": "user",
            "content": f"Question: {query}\nContext: {combined_context}",
            "combined_context": f"{combined_context}"}

def format_response(query: str, contexts: list[str], answer: str) -> dict:
    return {"user_query": query,
            "retrieved_contexts": contexts,
            "response": answer}

def format_evaluation_input(user_query: str, retrieved_contexts: list[str], response: str, reference: str):
    return {"user_input": user_query,
            "retrieved_contexts": retrieved_contexts,
            "response": response,
            "reference": reference}


def get_answer_text_in_str(answer_str: str) -> str:
    start_pos = len("{'text': ['")
    end_pos = answer_str.find('answer_start')
    return answer_str[start_pos:end_pos-len("'], '")]

def get_answer_texts_in_df(df: pd.DataFrame) -> list[str]:
    answer_strs = df['answers'].tolist()
    return [get_answer_text_in_str(answer_str) for answer_str in answer_strs]

def get_strs_in_df(df: pd.DataFrame, attribute_str: str) -> list[str]:
    return df[attribute_str].tolist()

def get_str_lists_in_df(df: pd.DataFrame, attribute_str: str) -> list[str]:
    return df[attribute_str].apply(ast.literal_eval).tolist()


def save_lists_dict(filename, lists_dict: dict):
    """
    Saves multiple lists dictionary to an Excel file with column names.
    Format dict: {"column_name": list}
    """
    # Create a pandas DataFrame from the dictionary of lists
    df = pd.DataFrame(lists_dict)

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False, engine='openpyxl')  # Explicitly use openpyxl

    print(f"Data saved to {filename}")

def save_dict_lst(filename:str, dict_lst: list[dict], column_names: list[str]=[]):
    # list of dictionaries to DataFrame
    if column_names == []:
        df = pd.DataFrame(dict_lst)
    else:
        df = pd.DataFrame(dict_lst, columns=column_names)

    # Save to Excel
    df.to_excel(filename, index=False)


def retrieve_top_k(query: str, contexts: list[str], index, k: int=5) -> list[str]:
    # Step 1: Retrieve initial top-k candidates using the bi-encoder
    query_embedding = create_embeddings(query)
    query_embedding_np = query_embedding.cpu().detach().numpy()

    # Step 1.5: Search the index
    distances, indices = index.search(np.array([query_embedding_np]), k)

    # Step 2: Prepare pairs for cross-encoder reranking (using a set)
    seen_contexts = set()  # Create an empty set to track seen contexts
    candidate_pairs = []
    for i in indices[0]:
        context = contexts[int(i)]
        if context not in seen_contexts:  # Check if context is already seen
            seen_contexts.add(context)  # Add context to the set if not seen
            candidate_pairs.append((query, context))

    # Step 3: Use cross-encoder to score and rerank the candidates
    scores = cross_encoder.predict(candidate_pairs)
    ranked_indices = np.argsort(scores)[::-1]  # Sort scores in descending order

    # Step 4: Return the reranked results
    top_k_contexts = [contexts[int(indices[0][i])] for i in ranked_indices]
    return top_k_contexts


def create_summary(df: pd.DataFrame, only_factual: bool, summary_file_reordered: str="evaluation_summary.xlsx"):
    """Calculates and exports a summary of evaluation metrics."""

    # Calculate means for numeric columns only
    column_means = df.select_dtypes(include=np.number).mean(skipna=True) #select only numeric columns

    # Calculate counts for all columns
    column_counts = df.count()

    # Create a DataFrame for means and counts, using only the numeric columns for means
    summary_df = pd.DataFrame({
        'Mean': column_means,
        'Count': column_counts
    })

    # Define the desired order of the metrics
    if only_factual:
        desired_order = ['factual_correctness', 'user_input', 'retrieved_contexts', 'response', 'reference']
    else:
        desired_order = ['context_recall', 'factual_correctness', 'faithfulness', 'semantic_similarity',
                         'user_input', 'retrieved_contexts', 'response', 'reference']

    # Reorder the rows in the summary DataFrame. Handle missing indices gracefully.
    summary_df_reordered = summary_df.reindex(desired_order)

    # Export the reordered summary to an Excel file
    summary_df_reordered.to_excel(summary_file_reordered, index=True)

    print(f"Reordered summary exported to {summary_file_reordered}")


# %% [markdown] id="AotGl5ptWyqy"
# # Load dataset

# %% id="vzffWiNTWyqz" outputId="d799a193-38e9-4e77-d327-93af9f4f9f40"
dataset_df = pd.read_excel(input_file_dataset)

dataset_contexts = get_strs_in_df(dataset_df, 'context')
dataset_questions = get_strs_in_df(dataset_df, 'question')
dataset_answers = get_answer_texts_in_df(dataset_df)
print(dataset_contexts[:10])
print(dataset_questions[:10])
print(dataset_answers[:10])

# %% [markdown] id="rbnB0IIxWyq1"
# ### Load embeddings and FAISS index

# %% id="NmQivfHZWyq1" outputId="31dec17e-fd85-41df-dbc7-6594a7c82516"
# Load embeddings if they exist, otherwise create and save them
dataset_context_embeddings = load_embeddings(DATASET_CONTEXT_EMBEDDINGS_FILE)
if dataset_context_embeddings is None:
    dataset_context_embeddings = create_embeddings(dataset_contexts, DATASET_CONTEXT_EMBEDDINGS_FILE)
    print("Dataset context embeddings created and saved.")
else:
    print("Dataset context embeddings loaded from file.")

# Load embeddings if they exist, otherwise create and save them
dataset_question_embeddings = load_embeddings(DATASET_QUESTION_EMBEDDINGS_FILE)
if dataset_question_embeddings is None:
    dataset_question_embeddings = create_embeddings(dataset_questions, DATASET_QUESTION_EMBEDDINGS_FILE)
    print("Dataset question embeddings created and saved.")
else:
    print("Dataset question embeddings loaded from file.")

# Load index if it exists, otherwise create and save it
dataset_question_index = load_faiss_index(DATASET_QUESTION_INDEX_FILE)
if dataset_question_index is None:
    dataset_question_index = create_faiss_index(dataset_question_embeddings, DATASET_QUESTION_INDEX_FILE)
    print("Dataset question FAISS index created and saved.")
else:
    print("Dataset question FAISS index loaded from file.")

# %% [markdown] id="ZP3lxh-8Wyq1"
# # Load queries

# %% id="_iKqNbTUWyq1" outputId="d2a771e9-dd35-41c1-906d-d1021b066a82"
# LOAD QUERIES IF UPDATED
queries_df = pd.read_excel(input_file_queries)

queries_contexts = get_strs_in_df(queries_df, 'context')
queries_questions = get_strs_in_df(queries_df, 'question')
queries_answers = get_strs_in_df(queries_df, 'answer')
print(queries_contexts[:10])
print(queries_questions[:10])
print(queries_answers[:10])

# %% [markdown] id="DFfSzn0lWyq2"
# # Gen retrieved contexts

# %% id="ryF38gxvTCcN" outputId="8a6447a7-2ef0-46e0-82c9-f8d744f25cdd"
retrieved_contexts = []
for queries_question in queries_questions:
    # if query is in dataset, look it up
    if queries_question in dataset_questions:
        query_index = dataset_questions.index(queries_question)
        retrieved_contexts.append(format_retrieved_contexts(queries_question, [dataset_contexts[query_index]]))
    # else, retrieve top k
    else:
        top_k_contexts = retrieve_top_k(queries_question, dataset_contexts, dataset_question_index)
        retrieved_contexts.append(format_retrieved_contexts(queries_question, top_k_contexts))

print(retrieved_contexts[:10])
save_dict_lst(middle_file_retrieved_contexts, retrieved_contexts)

# %% [markdown] id="6mh2_rKnWyq2"
# # Gen conversation

# %% id="8gCey8DjWyq2"
retrieved_contexts_df = pd.read_excel(middle_file_retrieved_contexts)

retrieved_contexts_queries = get_strs_in_df(retrieved_contexts_df, 'user_query')
retrieved_contexts_contexts = get_str_lists_in_df(retrieved_contexts_df, 'retrieved_contexts')

# %% id="zJRDV1sOiHDm" outputId="4f4fcde3-d648-46bf-a6a0-459cf7d77e17"
conversations = [format_conversation(query, contexts) for query, contexts in zip(retrieved_contexts_queries, retrieved_contexts_contexts)]

print(conversations[:10])
save_dict_lst(middle_file_conversation, conversations)

# %% [markdown] id="ye5jsBmJWyq2"
# # Gen answer

# %% id="IgvufuKnWyq2"
conversation_df = pd.read_excel(middle_file_conversation)

conversation_queries = get_strs_in_df(conversation_df, "content")
conversation_combined_contexts = get_strs_in_df(conversation_df, "combined_context")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 115578, "status": "ok", "timestamp": 1736083417466, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="X6w2KMx_iHDn" outputId="8bfa8193-83e9-420e-f206-9c62bb2208bb"
# Call ChatGPT to generate answers
answers = [generate_answer(combined_context, query) for combined_context, query in zip(conversation_combined_contexts, conversation_queries)]

print(answers[:10])
save_dict_lst(middle_file_answer, answers, ["answer"])

# %% [markdown] id="VO5rm12lWyq2"
# # Gen response

# %% id="9BitSlObWyq3"
retrieved_contexts_df = pd.read_excel(middle_file_retrieved_contexts)

retrieved_contexts_queries = get_strs_in_df(retrieved_contexts_df, 'user_query')
retrieved_contexts_contexts = get_str_lists_in_df(retrieved_contexts_df, 'retrieved_contexts')

answer_df = pd.read_excel(middle_file_answer)

answer_answers = get_strs_in_df(answer_df, "answer")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1736083417466, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="AISL5VreiHDn" outputId="4bb3f832-34db-4253-ef7d-c0dc8c1f21b2"
reponses = [format_response(query, contexts, answer) for query, contexts, answer in zip(retrieved_contexts_queries, retrieved_contexts_contexts, answer_answers)]

print(reponses[:10])
save_dict_lst(middle_file_response, reponses)

# %% [markdown] id="9KCLyaeUWyq3"
# # Generate evaluations

# %% id="IWNc6SuPWyq3"
response_df = pd.read_excel(middle_file_response)

response_user_queries = get_strs_in_df(response_df, 'user_query')
response_retrieved_contexts = get_str_lists_in_df(response_df, 'retrieved_contexts')
response_responses = get_strs_in_df(response_df, 'response')

queries_df = pd.read_excel(input_file_queries)

response_references = get_strs_in_df(queries_df, 'answer')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1736083418181, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="tcj3J3X3iHDn" outputId="d320bf23-6169-4dc2-dfa7-b1b7077b6793"
evaluation_inputs = [format_evaluation_input(user_query, retrieved_contexts, response, reference)  for user_query, retrieved_contexts, response, reference in zip(response_user_queries, response_retrieved_contexts, response_references, response_references)]

print(evaluation_inputs[:10])

# Convert to EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(evaluation_inputs)

# %% id="HEqIk8mwTCcP"
# Define metrics for evaluation
metrics = [
    LLMContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]

# %% colab={"base_uri": "https://localhost:8080/", "height": 124, "referenced_widgets": ["c385a7cf390c420fbf28b4587bdec559", "a17e5ac79ec042699223a416cb07822d", "64ce88d383dd4373b438d4e8421c8038", "6fc043c6cb0a42768446c2bada6d1f52", "a3110529cb834ccca052ab0441d97882", "39ae0772ce294ef1a94b194ab90c7b19", "c1a47dae0f954383a9ed92c68f20b83c", "1bd44a948d774b48b1e8576e56f83caf", "841942b7f0b44a0c8ce0851b13a29adc", "790bee9828db4722949ca1eb2a98333e", "b29a9191aa5844e1bdb6c897618cd29b"]} executionInfo={"elapsed": 258449, "status": "ok", "timestamp": 1736083676618, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="ivp1NoOKTCcQ" outputId="410bbff2-0026-44e6-e1f2-663b07454b57"
# Evaluate the dataset
evaluations = evaluate(dataset=evaluation_dataset, metrics=metrics)

# list of dictionaries to DataFrame
evaluations_df = evaluations.to_pandas()

# Save to Excel
evaluations_df.to_excel(output_file_evaluations, index=False)

# %% [markdown] id="8Uh8kiCQWyq3"
# ### Analyze evaluations

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 366, "status": "ok", "timestamp": 1736083676974, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="fBt5gQTZWyq3" outputId="79d7ec95-4e16-4049-b030-f7ffd7ecbcd4"
# Fill NaN with 0 before calculating the mean
evaluations_df_filled = evaluations_df.fillna(0)

# Analyze by means of context_recall, factual_correctness, faithfulness and semantic_similarity
evaluations_column_means_filled = evaluations_df_filled.mean(numeric_only=True)
print(evaluations_column_means_filled)

# %% [markdown] id="KdIdUX3jWyq4"
# ### Analyze faithfulness

# %% [markdown] id="spGrUFxeTCcQ"
# Faithfulness metric measures the factual consistency of the generated answer against the given context.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1736083676975, "user": {"displayName": "B\u1ea3o L. N. G.", "userId": "09555567040888357654"}, "user_tz": -420} id="m2FjT6wLTCcQ" outputId="c492d24c-1a5e-49f5-d8a1-8967f5bf932d"
create_summary(evaluations_df, True, summary_file_reordered_True)
create_summary(evaluations_df, False, summary_file_reordered_False)
