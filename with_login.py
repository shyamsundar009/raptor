import streamlit as st
from rag import aws,file_to_chunks,azure_data_download,main,generate_queries_with_history,keyword_extractor_with_history
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import json
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from langchain.docstore.document import Document

def convert_text_to_doc(all_texts):    
    document =  []

    for item in range(len(all_texts)):
        page = Document(page_content=all_texts[item])
        document.append(page)
    return document

def meta(s):
    f=[]
    for i in s:
        f.append(i[:100])
    return f



with open('credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

authenticator.login()

if st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
if st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

if st.session_state["logout"] is True:
    st.session_state.messages = []
    st.session_state["logout"] = False

if st.session_state["authentication_status"]:
    # st.session_state.messages = []
    authenticator.logout(location="sidebar")
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*')

    # st.session_state
    # if st.session_state['Logout']:
        # st.session_state.messages = []
    st.title("RAG with different source")

# Using object notation
    add_selectbox = st.sidebar.selectbox(
        "Which data source you are using?",
        ("Azure Blob Storage", "Local storage","AWS S3 Bucket","Locally stored Vectorstore DB"),
        index=None,
        placeholder="Select contact method...",
    )

    if add_selectbox == "Azure Blob Storage":
        st.sidebar.write("You selected Azure Blob Storage.")
        AZURE_CONNECTION_STRING = st.sidebar.text_input("Azure Connection String Input",type="password")
        CONTAINER_NAME = st.sidebar.text_input("Azure Container Name")
    elif add_selectbox == "Local storage":
        st.sidebar.write("You selected Local storage.")
        LOCAL_PATH = st.sidebar.selectbox("Local Path",
                                        ("Local_data","NIL"),
                                        index=None,
                                        placeholder="Select contact method...")
    elif add_selectbox == "AWS S3 Bucket":
        aws_access_key = st.sidebar.text_input("AWS Access Key",type="password")
        aws_secret_access_key = st.sidebar.text_input("AWS SECRET ACCESS KEY",type="password")
        bucket_name= st.sidebar.text_input("AWS BUCKET NAME")
        object_name= st.sidebar.text_input("AWS OBJECT NAME")
    elif add_selectbox == "Locally stored Vectorstore DB":
        vectorstore_name = st.sidebar.text_input("Vectorstore db Name")
    else:
        st.sidebar.write("You selected nothing.")

    if st.sidebar.button("Injest"):
        if add_selectbox == "Azure Blob Storage" and AZURE_CONNECTION_STRING and CONTAINER_NAME:
            # Download PDF from Azure Blob Storage
            with st.sidebar:
                try:
                    with st.spinner("Azure connection is creating....."):
                        azure_data_download(AZURE_CONNECTION_STRING=AZURE_CONNECTION_STRING, CONTAINER_NAME=CONTAINER_NAME)
                    with st.spinner("Azure Folder documents to chunks are in the process.........."):
                        st.session_state.pages = file_to_chunks("Azure_data")
                    with st.spinner("VectorDatabse is creating....."):
                        st.session_state.db = FAISS.from_documents(st.session_state.pages, OpenAIEmbeddings())
                    st.success("VectorDatabse is created successfully in the Azure_Chroma_db")
                except Exception as e:
                    st.error(f"Error connecting with Azure: {str(e)}")
        elif add_selectbox == "Local storage":
            with st.sidebar:
                with st.spinner("Local Folder documents to chunks are in the process.........."):
                    st.session_state.pages = file_to_chunks("Local_data")   
                with st.spinner("VectorDatabse is creating....."):
                    st.session_state.db = FAISS.from_documents(st.session_state.pages, OpenAIEmbeddings())
                st.success("VectorDatabse is created successfully in the Local_vectorstore folder")
        elif add_selectbox == "Locally stored Vectorstore DB":
            if vectorstore_name=="db_001":
                with st.sidebar:
                    with st.spinner("Chunked documents are loading..........."):
                        with open("pages.pkl", "rb") as f:
                            st.session_state.pages = pickle.load(f)
                    with st.spinner("VectorDatabse is Loading....."):
                        st.session_state.db =  FAISS.load_local(vectorstore_name, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
                    st.success(f"Number of chunks is {len(st.session_state.pages)}\nVectorDatabse is successfully loaded from the Local_vectorstore folder {vectorstore_name}")
            elif vectorstore_name=="db_002":
                with st.sidebar:
                    with open("pages2.pkl", "rb") as f:
                        st.session_state.pages = pickle.load(f)
                    st.session_state.pages=convert_text_to_doc(st.session_state.pages)                        
                    with st.spinner("VectorDatabse is Loading....."):
                        st.session_state.db =  FAISS.load_local(vectorstore_name, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
                    st.success(f"Number of chunks is {len(st.session_state.pages)}\nVectorDatabse is successfully loaded from the Local_vectorstore folder {vectorstore_name}")
            else:
                st.warning(f"There is no vectorstore with name {vectorstore_name} in the Current folder")
        else:
            with st.sidebar:
                try:
                    with st.spinner("AWS connection is creating....."):
                        aws(AWS_ACCESS_KEY_ID=aws_access_key, AWS_SECRET_ACCESS_KEY=aws_secret_access_key, BUCKET_NAME=bucket_name, object_name=object_name)
                    with st.spinner("AWS Folder documents to chunks are in the process.........."):
                        st.session_state.pages = file_to_chunks()
                    with st.spinner("VectorDatabse is creating....."):
                        st.session_state.db = FAISS.from_documents(st.session_state.pages, OpenAIEmbeddings())
                    st.success("VectorDatabse is created successfully in the AWS_Chroma_db folder")
                except Exception as e:
                    st.error(f"Error in connecting with AWS: {str(e)}")

        st.session_state.injest = True
        # st.session_state


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "injest" not in st.session_state:
        st.session_state.injest = False
    if "pages" not in st.session_state:
        st.session_state.pages = None
    if "db" not in st.session_state:
        st.session_state.db = None  

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Lets have a chat with our document")
    # React to user input
    if prompt and st.session_state.injest:
        # Display user message in chat message container
        # specify the file path to save the JSON
        # open the file in write mode
        with open("history.json", "w") as file:
            # write the JSON data to the file
            json.dump(st.session_state.messages, file)

        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Generate assistant response
        questions=generate_queries_with_history(prompt).invoke(prompt)
        keywords=keyword_extractor_with_history().invoke(prompt)
        with st.spinner(f"""Creative Query from the original question \n\n{'\n'.join(questions[1:])} \n\n Keywords from the original question \n\n{keywords}\n"""):
            answer_dict=main(prompt,chunks=st.session_state.pages,db=st.session_state.db)
            answer=answer_dict["response"]
            # metadata=[]
            # if "I don't" in answer:
            #     metadata=["No relevant Documents correspond to the question. Please try with different query."]
            # else:
            #     metadata=meta(answer_dict["context"])
        response = f"{answer}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(f"{response}")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})    

