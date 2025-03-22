import os
import streamlit as st
from PIL import Image
from datetime import datetime

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Job Role Resource Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This app helps you discover resources for transitioning to different job roles using AI."
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #E8F4FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        color: #333333 !important;  /* Explicit dark text color */
    }
    .answer-box {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #10B981;
        color: #333333 !important;  /* Explicit dark text color */
    }
    .example-q {
        padding: 0.5rem;
        background-color: #F3F4F6;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
    }
    /* Make sure all text in these boxes is dark */
    .source-box *, .answer-box * {
        color: #333333 !important;
    }
    /* Ensure links are still visible */
    .source-box a, .answer-box a {
        color: #1E88E5 !important;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Enable MPS fallback for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Now import other dependencies
from extract_content import extract_pdf_content, extract_hyperlinks, fetch_url_content
from vectorStore import create_vector_store
from rag import build_rag_chain

# Sidebar content and navigation
with st.sidebar:
    st.markdown("### 🤖 AI Resource Assistant")
    st.markdown("---")
    
    st.markdown("### 📚 About")
    st.markdown(
        "This tool helps you discover learning resources for transitioning to new job roles. "
        "It uses Retrieval Augmented Generation (RAG) to provide accurate answers based on curated resources."
    )
    
    st.markdown("### 🔍 How it works")
    st.markdown(
        "1. Ask a question about resources for a specific job role\n"
        "2. The AI retrieves information from our knowledge base\n"
        "3. You get specific recommendations from curated resources"
    )
    
    st.markdown("### 🧠 Technology Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("LangChain")
    with col2:
        st.markdown("FAISS")
    with col3:
        st.markdown("Groq LLM")
    
    # Example questions section
    st.markdown("### 🔰 Example Questions")
    example_questions = [
        "What resources will help me become a data scientist?",
        "How can I transition to a product management role?",
        "What skills do I need for UX design?",
        "Recommend books for software engineering leadership",
        "What courses are good for learning cloud architecture?"
    ]
    
    def set_question(q):
        st.session_state.question = q
    
    for q in example_questions:
        if st.button(q, key=f"btn_{q[:20]}", use_container_width=True):
            set_question(q)

def load_data():
    """Load data and create the RAG system"""
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Set correct PDF path
    pdf_path = "Curated Learning Resources.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"Error: {pdf_path} not found")
        return None
    
    # Extract PDF content
    status_text.text("📄 Extracting PDF content...")
    progress_bar.progress(10)
    documents = extract_pdf_content(pdf_path)
    
    # Extract hyperlinks - limit to 5 to prevent long processing time
    status_text.text("🔗 Extracting hyperlinks...")
    progress_bar.progress(30)
    hyperlinks = extract_hyperlinks(pdf_path, max_links=5)
    
    # Fetch content from limited hyperlinks
    hyperlink_texts = {}
    for i, (page_num, url) in enumerate(hyperlinks):
        progress_value = 30 + (i+1) * 10 / len(hyperlinks)
        status_text.text(f"🌐 Fetching content from hyperlink {i+1}/{len(hyperlinks)}...")
        progress_bar.progress(int(progress_value))
        text = fetch_url_content(url)
        hyperlink_texts[url] = text
    
    # Create vector store
    status_text.text("🧠 Creating vector database...")
    progress_bar.progress(70)
    vectorstore = create_vector_store(documents, hyperlink_texts)
    
    # Build RAG chain
    status_text.text("⚙️ Building RAG system...")
    progress_bar.progress(90)
    qa_chain = build_rag_chain(vectorstore)
    
    # Complete
    progress_bar.progress(100)
    status_text.text("✅ RAG system ready!")
    
    # Clear the progress elements after a brief delay
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return qa_chain

def main():
    # Initialize question in session state if not exists
    if 'question' not in st.session_state:
        st.session_state.question = ""

    # Main content area
    st.markdown('<h1 class="main-header">AI Job Role Resource Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover tailored learning resources for your career transition</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(
            "📌 **How to use this tool:** Ask specific questions about resources, skills, or learning paths "
            "for transitioning to different job roles. The AI will find relevant information from our curated knowledge base."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state if not already done
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    # Load data if not already loaded
    if st.session_state.qa_chain is None:
        qa_chain = load_data()
        if qa_chain:
            st.session_state.qa_chain = qa_chain
    
    # Question input area
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Your question about job roles and resources:",
            value=st.session_state.question,
            key="question_input",
            placeholder="e.g., What resources will help me become a data scientist?"
        )
    with col2:
        search_button = st.button("Get Answer", type="primary", use_container_width=True)
    
    # Response area
    if st.session_state.qa_chain:
        if question and search_button:
            with st.spinner("🔍 Finding the best resources for you..."):
                result = st.session_state.qa_chain({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])
            
            # Display the answer
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Display sources
            if sources:
                st.markdown("### 📚 Sources")
                for i, source in enumerate(sources[:3]):  # Limit to 3 sources for clarity
                    source_text = source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content
                    source_name = source.metadata.get("source", f"Source {i+1}")
                    
                    # Format URL sources nicely
                    if source_name.startswith("http"):
                        display_name = source_name[:50] + "..." if len(source_name) > 50 else source_name
                        st.markdown(f'<div class="source-box"><strong>📎 <a href="{source_name}" target="_blank">{display_name}</a></strong><br/>{source_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="source-box"><strong>📄 {source_name}</strong><br/>{source_text}</div>', unsafe_allow_html=True)
    else:
        st.warning("🚨 Unable to initialize the RAG system. Please check the logs for errors.")
    
    # Footer
    st.markdown('<div class="footer">Created with LangChain, FAISS and Groq LLM • ' + datetime.now().strftime("%Y") + '</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()