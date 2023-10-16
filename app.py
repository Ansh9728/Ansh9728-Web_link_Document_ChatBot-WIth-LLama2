import streamlit as st
import requests
import re,os
import sys,os,torch,string
from bs4 import BeautifulSoup
import nest_asyncio
import nltk
import string
import time
import tempfile
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words("english")


from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import login
from transformers import BitsAndBytesConfig
import torch
from langchain import HuggingFacePipeline
from transformers import pipeline

st.title("Web_Link & Document ChatBot")

input_options = st.radio(
    "What Input Do You Want to Provide?", ("Link", "Document", "Both")
)


# Function Start to fetch the Web Data Element form the Web
def get_links(website_link):
    org_link = website_link
    try:
        response = requests.get(website_link)

        if response.status_code == 200:
            html_data = response.content
            soup = BeautifulSoup(html_data, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            list_links = []
            for link in soup.find_all("a", href=True):
                list_links.append(link["href"])

            list_links.append(org_link)
            return list_links
        else:
            st.error(f"Error code {response.status_code}")
            return []  # Return an empty list when there's an error
    except requests.exceptions.RequestException as e:
        st.error(f"RequestException: {e}")
        return []  # Return an empty list for any request exception
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []  # Return an empty list for any other exception


def filter_link(link_ls):
    length = len(link_ls)
    if length > 1:
        url_pattern = re.compile(
            r"https?://(?!.*(?:\.pdf|facebook\.com|twitter\.com|github\.com))\S+"
        )
        unique_links = list(set())
        # Iterate through the list of links and extract valid URLs
        for link in link_ls:
            match = re.search(url_pattern, link)
            if match:
                unique_links.append(match.group())
        return unique_links
    else:
        return link_ls


def scrap_web(url):
    nest_asyncio.apply()
    loader = WebBaseLoader(url)

    loader.requests_per_second = 1
    data = loader.aload()
    return data


def clean_page_content(page_content):
    cleaned_content = " ".join(page_content.split())
    cleaned_content = "".join(
        [char for char in cleaned_content if char not in string.punctuation]
    )
    words = cleaned_content.split()  # Tokenize the text
    # stop_words = set(stopwords.words("english"))# Remove stopwords
    stop_words = nltk.corpus.stopwords.words("english")
    words = [word for word in words if word.lower() not in stop_words]
    # Join the cleaned words back into a string
    cleaned_content = " ".join(words)
    return cleaned_content


def clean_scrap_data(scrap_data):
    cleaned_data = []

    for document in scrap_data:
        cleaned_document = (
            document.copy()
        )  # Create a copy to avoid modifying the original data
        cleaned_document.page_content = clean_page_content(
            cleaned_document.page_content
        )
        cleaned_data.append(cleaned_document)

    return cleaned_data


# this function give me data from the website scrap
def web_data(url):
    with st.status("Feteching the WebSite Data", expanded=True):
        st.write("Finding Links")
        all_link_web = get_links(url)
        time.sleep(2)
        st.write("Cleaning the Links")
        # getting only the relevent link
        filtered_links = filter_link(all_link_web)
        len_of_filter_link = len(filtered_links)

        time.sleep(2)
        st.write("No of Links to Scrap are :", len_of_filter_link)
        time.sleep(1)
        st.write("Scrapping the Website Data")
        scrap_data = scrap_web(filtered_links)  # scrapping the data

        if scrap_data:
            st.write("Scrapping Completed")
        else:
            st.write("No Data Scrap")

    cleaned_data = clean_scrap_data(scrap_data)  # clean the scrap Data

    return cleaned_data


# Function End Web Data Element


url = ""
Folder_path = ""

data = list()


# first button initialize

if "submit_button" not in st.session_state:
    st.session_state.submit_button = False


def callback():
    st.session_state.submit_button = True
    st.session_state.Get_Result = True


if "Link" in input_options:
    url = st.text_input("Enter the Website Url")
    if st.button("Submit", on_click=callback) or st.session_state.submit_button:
        single_data_file = web_data(url)
        # st.write(single_data_file)
        data.extend(single_data_file)

if "Document" in input_options:
    uploaded_files = st.file_uploader(
        "Upload a File",
        type=["txt", "pdf", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if st.button("submit", on_click=callback) or st.session_state.submit_button:
        if uploaded_files is not None:
            # data = []
            for file in uploaded_files:
                temp_file_path = None

                try:
                    # Save the uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(file.read())
                        temp_file_path = temp_file.name

                    # Process different file types using appropriate loaders
                    if file.type == "text/plain":
                        loader = TextLoader(temp_file_path, autodetect_encoding=True)
                    elif file.type == "application/pdf":
                        loader = PyPDFLoader(temp_file_path)
                    elif file.type == "text/csv":
                        # loader = UnstructuredCSVLoader(temp_file_path,mode="elements")
                        loader = CSVLoader(temp_file_path, encoding=True)
                    elif (
                        file.type
                        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ):
                        loader = UnstructuredExcelLoader(temp_file_path)
                    else:
                        st.error(f"Unsupported file format: {file.type}")
                        continue

                    single_data_file = loader.load()
                    # st.write(f"Data from {file.name}: {data}")
                    single_data_file = clean_scrap_data(
                        single_data_file
                    )  # cleaning the scrap Data
                    data.extend(single_data_file)

                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

                finally:
                    # Clean up temporary file after processing
                    if temp_file_path is not None and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            # st.write(data)
    # st.session_state.submit_button=False


if "Both" in input_options:
    url = st.text_input("Enter the Website Url")
    uploaded_files = st.file_uploader(
        "Upload a File",
        type=["txt", "pdf", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    if st.button("submit", on_click=callback) or st.session_state.submit_button:
        if url:
            single_data_file = web_data(url)
            # st.write(single_data_file)
            data.extend(single_data_file)
        else:
            print("You Dont provide any url")

        if uploaded_files is not None:
            # data = []
            for file in uploaded_files:
                temp_file_path = None

                try:
                    # Save the uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(file.read())
                        temp_file_path = temp_file.name

                    # Process different file types using appropriate loaders
                    if file.type == "text/plain":
                        loader = TextLoader(temp_file_path, autodetect_encoding=True)
                    elif file.type == "application/pdf":
                        loader = PyPDFLoader(temp_file_path)
                    elif file.type == "text/csv":
                        # loader = UnstructuredCSVLoader(temp_file_path,mode="elements")
                        loader = CSVLoader(temp_file_path, encoding=True)
                    elif (
                        file.type
                        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ):
                        loader = UnstructuredExcelLoader(temp_file_path)
                    else:
                        st.error(f"Unsupported file format: {file.type}")
                        continue

                    single_data_file = loader.load()
                    # st.write(f"Data from {file.name}: {data}")
                    single_data_file = clean_scrap_data(
                        single_data_file
                    )  # cleaning the scrap Data
                    data.extend(single_data_file)

                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

                finally:
                    # Clean up temporary file after processing
                    if temp_file_path is not None and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        # st.session_state.submit_button=False

# #Here we get the Data to further proceess
#st.write(len(data))
#st.write(data)

# End OF LOADING DATA FILES


# START TASK CREATING CHUNKS
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(documents)
    return docs


split_docs_data = split_docs(data)
st.write("length of doc", len(split_docs_data))

# END TASK CREATING CHUNKS

# START THE VECTOR DATABASE INTEGRATION
global index_name
def db_embeddings(
    docs,
    api_key="54c47713-e05a-4360-b511-e6b47b899c43",
    environment="gcp-starter",
    index_name="langchain-doc1-embed",
):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    pinecone.init(api_key=api_key, environment=environment)

    index_name = index_name

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric="cosine",
            dimension=len(embeddings.embed_query("hello")),
        )

    Pinecone.from_documents(docs, embeddings, index_name=index_name)
    # index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


# getting the Pinecone Api keys and enviroment to create the enviroment

try:
    if split_docs_data:
        with st.form(key="pinecone creditional"):
            st.info(
                """
            Please Provide the following Details:
            Pinecone API key, Index Name, and Environment of Pinecone

            If you Don't want to Provide Just leave it blank By default Values Used.

            """
            )
            api_key = st.text_input("Enter the API key")
            index_name = st.text_input("Enter the Index Name")
            environment = st.text_input("Enter the Environment")
            parameters = {
                "api_key": api_key,
                "index_name": index_name,
                "environment": environment,
            }

            submitted = st.form_submit_button("Submit")

            if submitted:
                st.write("Embeddings Started")
                db_embeddings(
                    docs=split_docs_data,
                    **{key: value for key, value in parameters.items() if value},
                )
                st.write("Vector Embeddings Done in Pinecone")
                st.session_state.submit_button = False

except Exception as e:
    st.write(f"Error Present in that are {str(e)}")

# END OF VECTOR DATA BASE INTEGRATION

# MODEL RELATED TASK AND FUNCTION

# GETTING THE SIMILAR DOCUMENT
def get_similiar_docs(question, k=3, score=True):
    index_name = "langchain-doc1-embed"
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    if score:
        similar_docs = index.similarity_search_with_score(question, k=k)
    else:
        similar_docs = index.similarity_search(question, k=k)

    return similar_docs
# END OF SIMILAR DOCUMENT FUNCTION

#hugging face login
login("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")

#FUNCTION TO LOAD THE ML MODEL
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name="meta-llama/Llama-2-7b-chat-hf"
#quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

@st.cache_resource
def load_model(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',load_in_8bit=True,torch_dtype=torch.float32)

  pipe=pipeline(
      'text-generation',
      model=model,
      tokenizer=tokenizer,
      max_new_tokens=250,
      model_kwargs={'temperature':0.2},

  )

  llm = HuggingFacePipeline(pipeline=pipe,verbose=True,callback_manager=callback_manager)

  return llm

llm = load_model(model_name)

chat_history = {'conversation': []}
#st.write(chat_history)
def get_result_streamlit_chat(question):
    similar_doc = get_similiar_docs(question)
    #print(similar_doc)
    similar_doc = ''.join([(i[0].page_content) for i in similar_doc])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        #st.write(message)
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            #chat_history = {'conversation': [message]}

    chat_history = {'conversation': [st.session_state.messages]}

    prompt = f"""
    <s>[INST] <<SYS>>
            Use the following pieces of context and chat history to answer the question at the end.

            Only Provide an answer without additional information.

            ### If you don't know the answer or the question is out of context, just say that you don't know.

            Use three sentences maximum and keep the answer as concise as possible

    </<<SYS>>

    context: {similar_doc}

    chat history: {chat_history['conversation']}

    question: {question}

    [/INST]
    """
    #st.write(prompt)
    res=llm(prompt)
    #chat_history['conversation'].append({'question': question, 'response': res})

    return res


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])

if question := st.chat_input("Enter Your Question?"):
    # Display user message in chat message container
    #st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    response=get_result_streamlit_chat(question)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
       st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


processing_done = False

if not processing_done:
    st.info('Click the Exit Button to Delete Vector DATABASE')
    if st.button("Exit"):
      index_name = "langchain-doc1-embed"
      pinecone.delete_index(index_name)
      st.success(f"Pinecone index {index_name} has been deleted.")
      processing_done = True  # Set the flag to indicate processing is done

