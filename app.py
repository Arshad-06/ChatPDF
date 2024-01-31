from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from utils.utils_ask_human import CustomAskHumanTool
from utils.utils_model_params import get_model_params
from utils.utils_prompts import create_agent_prompt, create_qa_prompt
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain import HuggingFaceHub
import torch
import streamlit as st
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
import os

repo_id = "sentence-transformers/all-mpnet-base-v2"
from datetime import datetime
from langchain.tools import Tool


HUGGINGFACEHUB_API_TOKEN = "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"

hf = HuggingFaceHubEmbeddings(
    repo_id=repo_id,
    task="feature-extraction",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)


llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)


from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

### PAGE ELEMENTS

# st.set_page_config(
#     page_title="RAG Agent Demo",
#     page_icon="🦜",
#     layout="centered",
#     initial_sidebar_state="collapsed",
# )
# st.markdown("### Leveraging the User to Improve Agents in RAG Use Cases")


def main():
    st.set_page_config(page_title="Ask your PDF powered by Search Agents")
    st.header("Ask your PDF powered by Search Agents 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF and chat with Agent", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split documents and create text snippets

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_text(text)

        embeddings = hf
        knowledge_base = FAISS.from_texts(texts, embeddings)

        retriever = knowledge_base.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": create_qa_prompt(),
            },
        )

        conversational_memory = ConversationBufferMemory(
            memory_key="chat_history", k=3, return_messages=True
        )

        # tool for db search
        db_search_tool = Tool(
            name="dbRetrievalTool",
            func=qa_chain,
            description="""Use this tool to answer document related questions. The input to this tool should be the question.""",
        )

        # search = SerpAPIWrapper(serpapi_api_key=serp_token)

        # google_searchtool= Tool(
        #         name="Current Search",
        #         func=search.run,
        #         description="use this tool to answer real time or current search related questions.",
        #     )
        search = DuckDuckGoSearchRun()
        search_tool = Tool(
            name="search",
            func=search,
            description="use this tool to answer real time or current search related questions.",
        )
        # Define a new tool that returns the current datetime
        datetime_tool = Tool(
            name="Datetime",
            func=lambda x: datetime.now().isoformat(),
            description="Returns the current datetime",
        )
        # tool for asking human
        human_ask_tool = CustomAskHumanTool()
        # agent prompt
        prefix, format_instructions, suffix = create_agent_prompt()
        mode = "Agent with AskHuman tool"

        # initialize agent
        agent = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=[db_search_tool, search_tool, datetime_tool],
            llm=llm,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            memory=conversational_memory,
            agent_kwargs={
                "prefix": prefix,
                "format_instructions": format_instructions,
                "suffix": suffix,
            },
            handle_parsing_errors=True,
        )

        # question form
        with st.form(key="form"):
            user_input = st.text_input("Ask your question")
            submit_clicked = st.form_submit_button("Submit Question")

        # output container
        output_container = st.empty()
        if submit_clicked:
            # st_callback = StreamlitCallbackHandler(st.container())
            # response = agent.run(user_input,callbacks = [st_callback])
            response = agent.run(user_input)
            st.write(response)
            # output_container = output_container.container()
            # output_container.chat_message("user").write(user_input)
            # with st.chat_message("assistant"):
            #     st_callback = StreamlitCallbackHandler(st.container())
            #     response = agent.run(user_input, callbacks=[st_callback])
            #     st.write(response)

            # answer_container = output_container.chat_message("assistant", avatar="🦜")
            # st_callback = StreamlitCallbackHandler(answer_container,)

            # answer = agent.run(user_input, callbacks=[st_callback])

            # answer_container = output_container.container()
            # answer_container.chat_message("assistant").write(answer)


if __name__ == "__main__":
    main()
