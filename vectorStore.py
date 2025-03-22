from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from extract_content import extract_pdf_content, extract_hyperlinks, fetch_url_content as scrape_url_content

def build_knowledge_base(pdf_path, persist_directory="./chroma_db"):
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Process PDF content
    print("Extracting PDF content...")
    pdf_docs = extract_pdf_content(pdf_path)
    
    # Process hyperlinks
    print("Extracting and processing hyperlinks...")
    hyperlinks = extract_hyperlinks(pdf_path)
    web_docs = []
    
    for i, (page, url) in enumerate(hyperlinks):
        print(f"Processing hyperlink {i+1}/{len(hyperlinks)}: {url}")
        content = scrape_url_content(url)
        if content:
            web_docs.append(Document(
                page_content=content,
                metadata={"source": url, "page": page, "type": "web"}
            ))
    
    # Update metadata for PDF docs
    for doc in pdf_docs:
        doc.metadata["type"] = "pdf"
    
    # Combine documents
    all_docs = pdf_docs + web_docs
    print(f"Total documents: {len(all_docs)} (PDF: {len(pdf_docs)}, Web: {len(web_docs)})")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Knowledge base created and saved to {persist_directory}")
    
    return vectorstore

def create_vector_store(documents, hyperlink_texts=None):
    # Combine PDF documents with hyperlink content
    all_docs = documents.copy()
    
    if hyperlink_texts:
        for url, text in hyperlink_texts.items():
            if text:
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": url}
                ))
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)
    
    # Create and return the vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Much smaller model
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings  # Changed from embedding_model to embeddings
    )
    
    return vectorstore