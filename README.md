# RAG Implementation on RAPTOR Chunking

Welcome to the GitHub repository for our custom RAG (Retrieval-Augmented Generation) application, built using the RAPTOR chunking technique. This project is designed to handle and process data from multiple sources, convert it into manageable chunks, and use these chunks to generate comprehensive summaries based on user queries. Our implementation leverages a custom RAG model to enhance the accuracy and relevance of the generated summaries.

## Features

- **Multi-Source Data Handling**: This application can seamlessly integrate with various storage solutions including local storage, Azure Blob Storage, and AWS S3, allowing it to fetch data efficiently from multiple sources.
- **Data Processing**: Data is processed by converting it into chunks which are then transformed into a chroma vector store, optimizing the retrieval process.
- **RAG Model**: Our custom multi-query RAG model is at the core of this application, designed to generate accurate summaries by retrieving the most relevant documents based on the input query.
- **Streamlit Interface**: An interactive and user-friendly Streamlit UI is provided to manage the workflow from data input to summary output, making it accessible for users with varying technical expertise.

## Storage Configuration

### Azure Blob Storage
- **Configuration Required**: Azure Blob connection string and container name.

### AWS S3 Bucket
- **Configuration Required**: AWS Access Key, AWS Secret Key, AWS Bucket Name, and Object Name.

### Local Storage
- **Configuration**: Utilizes a local folder located at the root of the project for ease of access and manipulation.

## RAG Model Workflow

1. **Query Input**: Users can input a query to initiate the summary generation process.
2. **Question Generation**: An LLM model generates various questions from the input query while preserving its semantic integrity.
3. **Keyword Extraction and Document Retrieval**: Keywords are extracted and documents are retrieved using a BM25 retriever.
4. **Document Ensemble**: An ensemble retriever processes the documents to fetch the most relevant ones for the query.
5. **Re-ranking**: A cross-encoder reranker is employed to refine the selection and choose the top 4 documents.
6. **Summary Generation**: These top documents are then used as the context for the LLM, which generates a comprehensive and coherent summary.

## Getting Started

Follow these steps to set up and run the application:

1. Clone the repository.
    ```bash
    git clone https://github.com/shyamsundar009/raptor
    ```

2. Create a virtual environment and activate it.

3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
4. Configure the `.env` file using `.env_template` and insert the necessary keys:
   ```
   OPENAI_API_KEY="sk--****************************"
   ```
5. Execute `RAPTOR.ipynb` to see the application in action.

### Data Storage

- **db_001**: Contains chunks from the local storage.
- **db_002**: Contains RAPTOR chunks from the local storage.

## Streamlit Demo

To run the Streamlit interface:
```
streamlit run with_login.py
```

## Demo:

[!(demo)](
https://github.com/shyamsundar009/raptor/assets/167984593/f1ac0bd0-ab13-4ce3-a9b5-eaa0090de9bb
)