import os
from dotenv import load_dotenv, find_dotenv
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import WebBaseLoader
from openai import OpenAI
import logging
import csv
import tiktoken
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Configure logging
logging.basicConfig(filename='rag_chain.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'websiteRAG'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

# OpenAI client setup
token =  os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
chat_model_name = "gpt-4o"
embedding_model_name = "text-embedding-3-large"
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

class CustomOpenAIEmbeddings(Embeddings):
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 64000  # Maximum tokens for text-embedding-3-large

    @retry(wait=wait_random_exponential(min=50, max=300), stop=stop_after_attempt(6))
    def create_embedding(self, texts):
        return self.client.embeddings.create(
            input=texts,
            model=self.model,
        )

    def batched_embed_documents(self, texts):
        all_embeddings = []
        batch_size = 32  # Adjust based on your rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokenized_batch = [self.encoding.encode(text) for text in batch]
            
            # Handle long texts
            processed_batch = []
            for tokens in tokenized_batch:
                if len(tokens) > self.max_tokens:
                    chunks = [self.encoding.decode(tokens[j:j+self.max_tokens]) 
                              for j in range(0, len(tokens), self.max_tokens)]
                    processed_batch.extend(chunks)
                else:
                    processed_batch.append(self.encoding.decode(tokens))
            
            response = self.create_embedding(processed_batch)
            batch_embeddings = [data.embedding for data in response.data]
            
            # Average embeddings for long texts that were split
            j = 0
            for tokens in tokenized_batch:
                if len(tokens) > self.max_tokens:
                    num_chunks = (len(tokens) + self.max_tokens - 1) // self.max_tokens
                    avg_embedding = [sum(x) / num_chunks for x in zip(*batch_embeddings[j:j+num_chunks])]
                    all_embeddings.append(avg_embedding)
                    j += num_chunks
                else:
                    all_embeddings.append(batch_embeddings[j])
                    j += 1
            
            time.sleep(50)  # 5-second delay between batches
        
        return all_embeddings

    def embed_documents(self, texts):
        return self.batched_embed_documents(texts)

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def read_urls_from_csv(file_path):
    urls = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            urls.append(row[0])
    return urls

urls = read_urls_from_csv('./Data/urls.csv')
print("URLs loaded:", urls)
logging.info(f"URLs loaded: {urls}")

loader = WebBaseLoader(urls)
docs = loader.load()
logging.info(f"Number of documents loaded: {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
logging.info(f"Number of splits created: {len(splits)}")

embeddings = CustomOpenAIEmbeddings(client, embedding_model_name)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
vectorstore.save_local("./Vector_DB/faiss_index")
logging.info("Vector store created and saved")

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@retry(wait=wait_random_exponential(min=30, max=300), stop=stop_after_attempt(3))
def generate_response(input_text):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about Microsoft Documentation.",
            },
            {
                "role": "user",
                "content": input_text,
            }
        ],
        model=chat_model_name,
        temperature=0.7,
        max_tokens=1000,
        top_p=1.0
    )
    return response.choices[0].message.content

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | generate_response
    | StrOutputParser()
)

# Question
question = "Give me All Topics of Create a question answering solution by using Azure AI Language"
print("Question:", question)
logging.info(f"Question: {question}")

answer = rag_chain.invoke(question)
print("Answer:", answer)
logging.info(f"Answer: {answer}")