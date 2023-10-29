import streamlit as st
import os
import pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ConversationKGMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dataclasses import dataclass
from langchain.chains import ConversationalRetrievalChain
from typing import Literal
from dotenv import load_dotenv

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


def load_css():
    """load the css to allow for styles.css to affect look and feel of the streamlit interface"""
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_vector_store():
    """Initialize a Pinecone vector store for similarity search."""
    
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'),environment=os.getenv('PINECONE_ENVIRONMENT') )
    index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME')) # you need to have pinecone index name already created (ingest.py should be run first)
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Pinecone(index, embed_model, "text") # 'text' is the field name in pinecone index where original text is stored
                           
    return vectorstore

def initialize_session_state():
    """Initialize the session state variables for streamlit."""
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        # create a connection to OpenAI text-generation API
        llm = ChatOpenAI(
            temperature=0.2,
            openai_api_key=os.environ["OPENAI_API_KEY"],
#            max_tokens=500,
#            model_name="gpt-3.5-turbo",
            model_name="gpt-4",
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=3,return_messages=True),
            verbose=True,         
        )


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal['human', 'ai']
    message: str
    
# when submit button is clicked, this function is called    
def on_click_callback(): #called on click of submit button
    """Function to handle the submit button click event."""
    
    with get_openai_callback() as cb:
        
        # get the human prompt in session state (read from text field)
        human_prompt = st.session_state.human_prompt
        
        # conduct a similarity search on our vector database
        vectorstore = initialize_vector_store()
        similar_docs = vectorstore.similarity_search(
            human_prompt,  # our search query
            k=25  # return relevant docs
        )
        
        data  = ' '.join([doc.page_content for doc in similar_docs])
        # create a prompt with the human prompt and the context from the most similar documents
        prompt = """
            You are a friendly chatbot. \n\n
            Query:\n
            {human_prompt} \n\n                        
            
            Context:
            {data1} \n
            """
        print(prompt) # for debugging purposes

        system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context,and if the answer is not contained within the text below, say 'I don't know'""")
        
        human_msg_template = HumanMessagePromptTemplate.from_template(template=prompt)
        #human_msg_template = prompt
        #print(human_msg_template.input_variables)
        #print(human_msg_template)
        
        prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        #PROMPT_1 = PromptTemplate(input_variables=[data,human_prompt], template=prompt_template) 


        T1 = prompt_template.format_messages(
                    data1=data,
                    human_prompt=human_prompt)


        print('#################################')
        print(T1)
        #print(prompt_template.input_variables)
        # get the llm response from prompt
        llm_response = st.session_state.conversation.predict(T1
        )
        
        #store the human prompt and llm response in the history buffer
        st.session_state.history.append(
            Message("human", human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response) 
        )
        
        # keep track of the number of tokens used
        st.session_state.token_count += cb.total_tokens

#############################
# MAIN PROGRAM

# initializations
load_dotenv() # load environment variables from .env file
load_css() # need to load the css to allow for styles.css to affect look and feel
initialize_session_state() # initialize the history buffer that is stored in UI  

# create the Streamlit UI
st.title("USF BullBot 🐂")
chat_placeholder = st.container() # container for chat history
prompt_placeholder = st.form("chat-form") # chat-form is the key for this form. This is used to reference this form in other parts of the code
debug_placeholder = st.empty() # container for debugging information

# below is the code that describes how each of the three containers are displayed

with chat_placeholder: # this is the container for the chat history
    for chat in st.session_state.history:
        div = f"""
            <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                <img class="chat-icon" src="app/static/{'ai_icon.png' if chat.origin == 'ai' else 'user_icon.png'}" width=32 height=32>
                <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                    &#8203;{chat.message}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)
    for _ in range(3): # add a few blank lines between chat history and input field
        st.markdown("")

with prompt_placeholder: # this is the container for the chat input field
    col1, col2 = st.columns((6,1)) # col1 is 6 wide, and col2 is 1 wide
    col1.text_input(
        "Chat",
        value="Please enter your question here",
        label_visibility="collapsed",
        key="human_prompt",  # this is the key, which allows us to read the value of this text field later in the callback function
    )
    col2.form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,  # important! this set's the callback function for the submit button
    )

debug_placeholder.caption(  # display debugging information
    f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain.coversation:
    {st.session_state.conversation.memory.buffer}
    """)


