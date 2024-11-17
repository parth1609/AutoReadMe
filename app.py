import streamlit as st
from langchain.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import tempfile
import git
import requests
import subprocess
import time


GOOGLE_API_KEY = st.secrets["gemini_Api_key"]
# Configure the Gemini model
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit UI
st.title("Repository README Generator")

repo_url = st.text_input("Enter the repository URL:")
if st.button("Generate README"):
    if repo_url:
        try:
            # Check internet connectivity first
            try:
                requests.get("https://github.com", timeout=5)
            except requests.RequestException:
                st.error("Please check your internet connection and try again.")
                st.stop()

            # Verify git is installed
            try:
                subprocess.run(["git", "--version"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                st.error("Git is not installed or not accessible. Please install Git and try again.")
                st.stop()

            # Create a temporary directory and process the repository
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Clone repository
                    repo = git.Repo.clone_from(repo_url, temp_dir, depth=1)
                    repo.close()
                    
                    # Process documents
                    loader = GitLoader(repo_path=temp_dir, branch="main")
                    documents = loader.load()
                    
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(documents)
                    
                    # Create embeddings
                    embeddings = HuggingFaceEmbeddings()
                    db = FAISS.from_documents(texts, embeddings)
                    
                    # Get relevant documents
                    retriever = db.as_retriever(search_kwargs={"k": 5})
                    relevant_docs = retriever.get_relevant_documents("project summary")
                    
                    # Generate README with explicit credentials and markdown formatting
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        temperature=0.3,
                        google_api_key=GOOGLE_API_KEY,
                    )
                    
                    # Create a custom prompt for GitHub markdown formatting
                    prompt_template = PromptTemplate(
                        input_variables=["text"],
                        template="""
                        You are a GitHub README.md expert. Generate a professional README.md file using GitHub-flavored markdown syntax. 
                        The README must be well-structured and follow GitHub's best practices.

                        Include these sections with proper GitHub markdown formatting:

                        # [Project Name]

                        ![GitHub repo size](https://img.shields.io/github/repo-size/username/repo)
                        ![GitHub stars](https://img.shields.io/github/stars/username/repo)
                        ![GitHub forks](https://img.shields.io/github/forks/username/repo)
                        ![GitHub issues](https://img.shields.io/github/issues/username/repo)

                        ## 📝 Description
                        [Detailed project description]

                        ## ✨ Features
                        - Feature 1
                        - Feature 2
                        - Feature 3

                        ## how to use

                        ## data collecting method

                        ## Data processing methods

                        ## 🚀 Installation
                        ```bash
                        [Installation commands]
                        ```

                        ## 💻 Usage
                        ```[language]
                        [Usage examples]
                        ```

                        ## 🤝 Contributing
                        Contributions, issues and feature requests are welcome!

                        ## 📝 License
                        [License information]

                        Based on the following repository context:
                        {text}

                        Format the output as a proper GitHub README.md with:
                        - Emoji icons for section headers
                        - Code blocks with language specification
                        - Proper heading hierarchy (#, ##, ###)
                        - Badges where appropriate
                        - Lists using - or *
                        - Links using [text](url)
                        - Images using ![alt](url)
                        - Tables using | syntax where needed
                        """
                    )

                    # Generate the README content first
                    chain = load_summarize_chain(
                        llm, 
                        chain_type="stuff",
                        prompt=prompt_template
                    )
                    readme = chain.run(relevant_docs)
                    
                    # Display the generated README
                    st.markdown("## Generated README")
                    st.markdown(readme)

                    # Add download button for the README after it's generated
                    st.download_button(
                        label="Download README.md",
                        data=readme,
                        file_name="README.md",
                        mime="text/markdown"
                    )
                    
                    # Small delay before cleanup
                    time.sleep(1)
                    
                except git.GitCommandError as e:
                    st.error(f"Failed to clone repository: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")
                    st.stop()

            # Display the generated README outside the context manager
            st.markdown(readme)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a repository URL.")