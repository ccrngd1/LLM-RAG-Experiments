import os
 
import sys
import json 
import boto3 
import re
from typing import List, Union, Type

import numpy as np
from pydantic import BaseModel, Field  
from io import BytesIO 

import streamlit as st 

from requests_aws4auth import AWS4Auth

from opensearchpy import OpenSearch, RequestsHttpConnection

from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings 
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser,initialize_agent
from langchain.schema import AgentAction, AgentFinish
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import  LLMChain
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document 
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents.react.base import DocstoreExplorer
from langchain import SagemakerEndpoint
from langchain.tools import BaseTool
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
#from langchain.callbacks.manager import CallbackManagerForRetrieverRun 




###ADD-DynamoDB
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from streamlit.runtime.scriptrunner import get_script_run_ctx


def _get_session(): 
    ctx = get_script_run_ctx()
    session_id = ctx.session_id
    print(session_id)
    return session_id
##############




claudeID = 'anthropic.claude-v2'
accept = 'application/json'
contentType = 'application/json'

inference_modifier_claude = {'max_tokens_to_sample':4096, 
                      "temperature":0.5,
                      "top_k":250,
                      "top_p":1,
                      #"stop_sequences": ["\n\nHuman"]
                     } 

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print(text)
        if f"{self.ai_prefix}:" in text:
            return AgentFinish({"output": text}, text)
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            return AgentFinish({"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text)
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
    

def askBedrockClaud2withAOSS(queryToAsk):  
    #return queryToAsk
    output_parser = CustomOutputParser()    
    
    boto3_bedrock = boto3.client(service_name='bedrock-runtime')
    embeddings_bedrock = BedrockEmbeddings(client=boto3_bedrock)
    embeddings_bedrock.model_id = 'amazon.titan-embed-text-v1'

    llm_anthropic_claude = Bedrock(model_id=claudeID, client=boto3_bedrock, model_kwargs=inference_modifier_claude)
    
    service = 'aoss'
    credentials = boto3.Session().get_credentials() #add access key and secret access key which has data access permissions for the OS collections
    region = 'us-east-2'
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,session_token=credentials.token)

    docsearch = OpenSearchVectorSearch(opensearch_url="https://REPLACE_ME_AOSS_ID.us-east-2.aoss.amazonaws.com",                                           
                                            index_name="test2-1536",
                                            embedding_function=embeddings_bedrock,
                                            http_auth=awsauth,
                                            timeout = 300,
                                            use_ssl = True,
                                            verify_certs = True,
                                            connection_class = RequestsHttpConnection,
                                              is_aoss=True)
    agent_aoss=docsearch.as_retriever()
    tools = [
        Tool(
            name="Search",
            func=agent_aoss.get_relevant_documents,
            description="useful for when you need to answer questions, as the same question to search tool"
        ) 
    ]
    
    # Set up the base template
    template = """Human: You are a conversational AI bot, Answer the following questions as best you can. 
    
    You have access to the following tools:

    {tools}

    To use a tool, you MUST use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No

    AI: [your response here]

    Begin!

    Previous conversation history:
    {chat_history}

    New input: {input}
    {agent_scratchpad}
    
    Assistant:"""
    
    prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input","intermediate_steps","chat_history"]
    )
    
    output_parser = CustomOutputParser()
    
    
    ###ADD-DynamoDB
    message_history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id=_get_session())
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True)
    #memory=ConversationBufferWindowMemory(k=0)
    ################
    
    
    tool_names = [tool.name for tool in tools]
    llm_chain = LLMChain(llm=llm_anthropic_claude, prompt=prompt)
    
    agent= LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["Human:"], 
        allowed_tools=tool_names,
        verbose=True,
    )
    
    #print(agent)
   
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    
    result = agent_executor.run(input=queryToAsk)

    return result

# App title
st.set_page_config(page_title="Sample Doc QA Chat")


# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_response(prompt_input ): 
    for dict_message in st.session_state.messages:
        string_dialogue = "You are a helpful assistant."
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
            
    return askBedrockClaud2withAOSS(prompt_input)
    #return prompt_input


# User-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    

def main():    
    return 0
    
if __name__ == "__main__":
    main()