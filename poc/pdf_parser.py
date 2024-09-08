import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict
import re
from langchain.schema import Document

# Set up OpenAI API key
load_dotenv()


def process_pdf(file_path):
    print(f"Processing: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages


def reorganize_documents(documents):
    # Group documents by source
    grouped_docs = defaultdict(list)
    for doc in documents:
        grouped_docs[doc.metadata["source"]].append(doc)

    # Combine content for each source
    combined_docs = []
    for source, docs in grouped_docs.items():
        combined_content = "\n".join(doc.page_content for doc in docs)
        combined_docs.append(
            Document(page_content=combined_content, metadata={"source": source})
        )

    final_docs = []
    for doc in combined_docs:
        # Split the content using regex
        splits = re.split(r"\n(\d+\.)", doc.page_content)

        # Process the splits
        for i in range(1, len(splits), 2):
            question_num = splits[i].strip(".")
            content = splits[i + 1].strip() if i + 1 < len(splits) else ""

            final_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": doc.metadata["source"],
                        "question_number": question_num,
                    },
                )
            )

        # Add any content before the first numbered question
        if splits[0].strip():
            final_docs.append(
                Document(
                    page_content=splits[0].strip(),
                    metadata={"source": doc.metadata["source"]},
                )
            )

    print(f"Number of documents after reorganization: {len(final_docs)}")
    return final_docs


def create_vector_store(pages):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(pages, embeddings)
    return vector_store


def search_documents(vector_store, query):
    docs = vector_store.similarity_search(query, k=2)
    for i, doc in enumerate(docs):
        print(f"\nResult {i + 1}:")
        print(f"Page: {doc.metadata['page']}")
        print(f"Content: {doc.page_content[:200]}...")


def main():
    pdf_directory = "/Users/badrou/repository/david_goggins_pocket/poc/data/"
    all_pages = []

    # Process all PDF files in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            pages = process_pdf(file_path)
            all_pages.extend(pages)

    print(f"Number of pages before reorganization: {len(all_pages)}")

    # Reorganize documents
    reorganized_docs = reorganize_documents(all_pages)

    # Print reorganized documents
    for i, doc in enumerate(reorganized_docs):
        print(f"Document {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        if "question_number" in doc.metadata:
            print(f"Question Number: {doc.metadata['question_number']}")
        print(f"Content preview: {doc.page_content[:100]}...")
        print("-" * 50)

    # # Create a vector store from reorganized documents
    # vector_store = create_vector_store(reorganized_docs)

    # # Example search
    # query = "What is the main topic of these documents?"
    # print("\nPerforming search...")
    # search_documents(vector_store, query)


if __name__ == "__main__":
    main()
