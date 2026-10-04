from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from src.helper import text_split

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()


index_name = "techteach"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


@app.route("/")
def home():
    return render_template("Home-page.html")


@app.route("/chat")
def chat_page():
    return render_template("chatpage.html")

@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question."

    print("Question:", msg)

    response = rag_chain.invoke(msg)

    print("Response:", response)

    return response

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return "No file was selected.", 400
    if not file.filename.lower().endswith(".pdf"):
        return "Only PDF files are supported.", 400

    filename = secure_filename(file.filename) or "upload.pdf"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        pages = PyPDFLoader(path).load()
        chunks = text_split(pages)
        if not chunks:
            return "I could not find any text in that PDF. Scanned pages need OCR.", 400
        docsearch.add_documents(chunks)
    except Exception as e:
        print("Upload error:", e)
        return "Sorry, I could not process that file.", 500

    return f"Done. I read {len(pages)} pages from {filename}. You can ask about it now."


@app.errorhandler(413)
def too_large(e):
    return "That file is too large. The limit is 20 MB.", 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
