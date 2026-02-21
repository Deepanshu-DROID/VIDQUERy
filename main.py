
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

VIDEO_URL = "https://youtu.be/Ho7yc-gDcd0?si=P_iEwxlzfpZKSIPZ"
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")

def get_embedding_model_candidates(model_name: str) -> list[str]:
    if model_name.startswith("models/"):
        return [model_name, model_name.replace("models/", "", 1)]
    return [model_name, f"models/{model_name}"]

def load_transcript_with_fallback(url: str):
    # Try English first
    try:
        loader_en = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=30,
            language=["en"],
        )
        return loader_en.load()
    except Exception:
        # Fallback to Hindi (auto-generated available for this video)
        loader_hi = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=30,
            language=["hi"],
        )
        return loader_hi.load()


transcript_list = load_transcript_with_fallback(VIDEO_URL)
transcript = "\n\n".join([doc.page_content for doc in transcript_list])

#print(transcript)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
chunks = splitter.create_documents([transcript])
#print(len(chunks))
#print(chunks[0].page_content)


vector_store = None
last_error = None
tried_models = get_embedding_model_candidates(EMBEDDING_MODEL)
for model_name in tried_models:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        vector_store = FAISS.from_documents(chunks, embeddings)
        break
    except Exception as e:
        last_error = e

if vector_store is None:
    error_msg = str(last_error)
    if "404" in error_msg or "NOT_FOUND" in error_msg:
        raise RuntimeError(
            "Google embedding model not found or not supported for embedContent. "
            f"Tried models: {', '.join(tried_models)}. "
            "Call ListModels to see available models for your API key."
        )
    raise last_error

print({"num_vectors": vector_store.index.ntotal})

retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k': 3})
#print(retriever.invoke("What is the main topic of the video?"))

#### Now I have to start augmentation vid 19:35
llm = ChatGroq(model = "moonshotai/kimi-k2-instruct-0905")

prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = """
    You are a helpful assistant who  have to  provide the answer from the given youtube transcript. Always try to provide the answers from the transcript only. Dont's make up any information. If you dont find the answer in the transcript, then say "I don't know".
    {context}
    Question: {question}
    Answer:
    """
)

# Define a lambda function to format the retrieved documents
format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

# Create the parallel chain
parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
})

parser = StrOutputParser()

chain = parallel_chain | prompt | llm | parser

# Invoke with just the question string
question = "Top 5 key takeaways from this video"
answer = chain.invoke(question)
print(answer)

print("--------------------------------")





