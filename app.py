from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# Ensure we always load the .env next to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# Check for required API keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
EMBEDDING_MODEL = os.getenv('GOOGLE_EMBEDDING_MODEL', 'gemini-embedding-001')

if GOOGLE_API_KEY:
    print(f"GOOGLE_API_KEY loaded (len={len(GOOGLE_API_KEY)}, prefix={GOOGLE_API_KEY[:4]}****)")
else:
    print("WARNING: GOOGLE_API_KEY not found in environment variables.")

if GROQ_API_KEY:
    print(f"GROQ_API_KEY loaded (len={len(GROQ_API_KEY)}, prefix={GROQ_API_KEY[:4]}****)")
else:
    print("WARNING: GROQ_API_KEY not found in environment variables.")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Store vector stores in memory (in production, use a proper session management)
vector_stores = {}

def get_embedding_model_candidates(model_name: str) -> list[str]:
    if model_name.startswith("models/"):
        return [model_name, model_name.replace("models/", "", 1)]
    return [model_name, f"models/{model_name}"]

def load_transcript_with_fallback(url: str):
    """Load YouTube transcript with fallback to different languages"""
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
        try:
            loader_hi = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                transcript_format=TranscriptFormat.CHUNKS,
                chunk_size_seconds=30,
                language=["hi"],
            )
            return loader_hi.load()
        except Exception as e:
            error_msg = str(e)
            # Detect common IP block / cloud-provider block messages from youtube-transcript-api
            if "YouTube is blocking requests from your IP" in error_msg or "IP has been blocked by YouTube" in error_msg:
                raise Exception(
                    "Automatic transcript download is not available from this server because "
                    "YouTube is blocking requests from our cloud provider's IP address. "
                    "Please paste the transcript manually using the 'Paste transcript instead' option."
                )
            raise Exception(f"Could not load transcript: {error_msg}")


def build_chain_from_transcript(transcript: str):
    """Create the retrieval + QA chain from a raw transcript string."""
    # Check if API keys are set
    if not GOOGLE_API_KEY:
        raise Exception("GOOGLE_API_KEY is not set. Please set it in your .env file or environment variables.")

    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY is not set. Please set it in your .env file or environment variables.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.create_documents([transcript])

    # Create embeddings and vector store
    try:
        vector_store = None
        last_error = None
        tried_models = get_embedding_model_candidates(EMBEDDING_MODEL)
        for model_name in tried_models:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=GOOGLE_API_KEY,
                )
                vector_store = FAISS.from_documents(chunks, embeddings)
                break
            except Exception as e:
                last_error = e

        if vector_store is None:
            error_msg = str(last_error)
            if "404" in error_msg or "NOT_FOUND" in error_msg:
                raise Exception(
                    "Google embedding model not found or not supported for embedContent. "
                    f"Tried models: {', '.join(tried_models)}. "
                    "Call ListModels to see available models for your API key."
                )
            raise last_error
    except Exception as e:
        error_msg = str(e)
        # Log the raw error for debugging
        print(f"Raw Google embeddings error: {error_msg}")
        if "403" in error_msg or "leaked" in error_msg.lower() or "API key" in error_msg:
            raise Exception(
                "Google API key error: Your API key has been reported as leaked or is invalid. "
                "Please generate a new API key from Google AI Studio (https://aistudio.google.com/apikey) "
                "and update your .env file with: GOOGLE_API_KEY=your_new_key. "
                f"Details: {error_msg}"
            )
        else:
            raise Exception(f"Error creating embeddings: {error_msg}")

    # Create retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    # Create LLM chain
    try:
        llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905")
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "401" in error_msg or "403" in error_msg:
            raise Exception(
                "Groq API key error: Your API key is invalid or missing. "
                "Please check your GROQ_API_KEY in your .env file. "
                "Get a new key from https://console.groq.com/keys"
            )
        else:
            raise Exception(f"Error initializing LLM: {error_msg}")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are a helpful assistant who have to provide the answer from the given youtube transcript. Always try to provide the answers from the transcript only. Don't make up any information. If you don't find the answer in the transcript, then say "I don't know".
            {context}
            Question: {question}
            Answer:
            """
    )

    # Define a lambda function to format the retrieved documents
    format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

    # Create the parallel chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser

    return chain, len(chunks)

def process_video(url: str):
    """Process video URL and create vector store"""
    # Load transcript via YouTube (may fail on some cloud providers if IPs are blocked)
    transcript_list = load_transcript_with_fallback(url)
    transcript = "\n\n".join([doc.page_content for doc in transcript_list])
    return build_chain_from_transcript(transcript)

@app.route('/api/process-video', methods=['POST'])
def process_video_endpoint():
    """Endpoint to process a YouTube video URL"""
    try:
        data = request.json
        video_url = data.get('url')

        if not video_url:
            return jsonify({'error': 'Video URL is required'}), 400

        # Process the video
        chain, num_chunks = process_video(video_url)

        # Store the chain using video URL as key
        vector_stores[video_url] = chain

        return jsonify({
            'success': True,
            'message': f'Video processed successfully. {num_chunks} chunks created.',
            'video_url': video_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-transcript', methods=['POST'])
def process_transcript_endpoint():
    """Endpoint to process a manually provided transcript (safe fallback when YouTube blocks our IP)."""
    try:
        data = request.json or {}
        transcript = data.get('transcript')
        # Allow client to control the key we use (e.g. video URL), but fall back to a generated one
        video_url = data.get('video_url') or data.get('url')

        if not transcript or not transcript.strip():
            return jsonify({'error': 'Transcript text is required'}), 400

        if not video_url:
            # Simple deterministic key if no URL is provided
            video_url = f"manual-{len(vector_stores) + 1}"

        chain, num_chunks = build_chain_from_transcript(transcript)

        # Store the chain using video URL/key as identifier
        vector_stores[video_url] = chain

        return jsonify({
            'success': True,
            'message': f'Transcript processed successfully. {num_chunks} chunks created.',
            'video_url': video_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask-question', methods=['POST'])
def ask_question_endpoint():
    """Endpoint to ask a question about the processed video"""
    try:
        data = request.json
        question = data.get('question')
        video_url = data.get('video_url')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        if not video_url:
            return jsonify({'error': 'Video URL is required'}), 400

        if video_url not in vector_stores:
            return jsonify({'error': 'Video not processed. Please process the video first.'}), 400

        # Get the chain for this video
        chain = vector_stores[video_url]

        # Invoke the chain with the question
        answer = chain.invoke(question)

        return jsonify({
            'success': True,
            'answer': answer
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

if __name__ == '__main__':
    # Local development entrypoint; Render will use gunicorn via Dockerfile
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

