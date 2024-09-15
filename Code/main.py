from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

app = Flask(__name__)
CORS(app)

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'websiteRAG'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def respond(message):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

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
                model=model_name,
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
        
        result = rag_chain.invoke(message)
        print(result)
        return result
    except Exception as e:
        print(f"An error occurred in respond(): {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('chat_history', [])
        response = respond(user_message)
        return jsonify({'response': response})
    except Exception as e:
        print(f"An error occurred in chat(): {str(e)}")
        return jsonify({'error': f'An error occurred while processing your request: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))