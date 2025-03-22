from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key

def load_kb(persist_directory="./chroma_db"):
    """Load existing knowledge base"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Much smaller model (~33MB)
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    
    return vectorstore

def build_rag_chain(vectorstore):
    """
    Build a RAG chain using the provided vector store with Groq's LLaMA 3.3 70B model.
    
    Args:
        vectorstore: The vector store containing document embeddings
        
    Returns:
        A question-answering chain that can answer questions based on the retrieved context
    """
    # Custom prompt template to instruct the model
    template = """You are an assistant that helps people find resources for transitioning to specific job roles.
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    If the answer is not contained within the context, say "I don't have enough information to answer that." and suggest a search query they could try instead.
    Provide all relevant details from the context, including specific resource names, links, and recommendations.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Initialize Groq LLM with LLaMA 3.3 70B
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024  # Adjust as needed
    )
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use the stuff chain to simply append all retrieved docs
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        ),
        return_source_documents=True,  # Return sources for verification
        chain_type_kwargs={"prompt": prompt}  # Use our custom prompt
    )
    
    return qa_chain

def create_qa_chain(vectorstore):
    """Alias for build_rag_chain for compatibility with main.py"""
    return build_rag_chain(vectorstore)

def answer_question(question, qa_chain):
    """Get answer for a question"""
    response = qa_chain.invoke({"query": question})
    return response["result"]