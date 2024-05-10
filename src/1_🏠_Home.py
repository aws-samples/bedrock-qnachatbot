#### Main pythonfile to run the app on streamlit ####
#### There are five more python files in the src folder ####
#### Each python file has a specific function and is imported in this file ####
#### The python files are: ####
#### 1. loaders.py - contains functions to load input from different sources ####
#### 2. textgeneration.py - contains functions to generate text from input ####
#### 3. utils.py - contains utility functions ####
#### 4. chat.py - contains functions to initialize and run the chatbot ####
#### apart from these, config.ini - contains the configuration for the app ####
#### and requirements.txt - contains the list of libraries required to run the app ####
#### The app can be run locally by running the command 'streamlit run src/main.py' ####


###### Import libraries ######
import time
import streamlit as st ###### Import Streamlit library
from configparser import ConfigParser ###### Import ConfigParser library for reading config file to get model, greeting message, etc.
from PIL import Image ###### Import Image library for loading images
import os ###### Import os library for environment variables
from utils import * ###### Import utility functions
from loaders import create_embeddings, check_upload ###### Import functions to load input from different sources
from textgeneration import q_response, q_response_chat, search_context, summary, talking, questions ###### Import functions to generate text from input
from chat import initialize_chat, render_chat, chatbot 

Page_name = st.session_state["page_name"] = "Home"
#### Create config object and read the config file ####
config_object = ConfigParser()
config_object.read("loki-config.ini")

#### Initialize variables and reading configuration ####
greeting=config_object["MSG"]["greeting"] ###### initial chat message
hline=Image.open(config_object["IMAGES"]["hline"]) ###### image for formatting landing screen
uploaded=None ##### initialize input document to None



#### Set Page Config ####
st.set_page_config(layout="wide", page_title="LOKI") ###### Removed the Page icon

#### Set Logo on top sidebar ####
st.sidebar.image(hline) ###### Add horizontal line
c1,c2,c3=st.sidebar.columns([1,3,1]) ###### Create columns
st.sidebar.image(hline) ###### Add horizontal line
##
st.markdown(
    """
    <style>
        [data-testid=stSidebar]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
sidebar_title = '<p style="font-family:Amazon Ember; color:#FF9900; font-size: 30px; "><b>Amazon Bedrock Powered - Q&A ChatBot</b></p>'
st.sidebar.markdown(sidebar_title, unsafe_allow_html=True)

##

st.sidebar.image(hline) ###### Add horizontal line
params = select_models(Page_name) ##### Call Model Selection routine
input_choice, uploaded=input_selector() ###### Get input choice and input document
st.sidebar.image(hline) ###### Add horizontal line

#### If input mode has been chosen and link/doc provided, convert the input to text ####
if uploaded is not None and uploaded !="":

    with st.spinner("reading "+input_choice+"..." if input_choice!="YouTube" else "extracting audio from YouTube..." if input_choice=="YouTube" else "extracting text from image..."): ###### Wait while input is being read
        words, pages, string_data,succeed,token=check_upload(uploaded=uploaded,input_choice=input_choice,params=params) 
        ###### Get input text from input document
        ###### words - number of words in the input
        ###### pages - number of embeddings in the input
        ###### string_data - input text
        ###### succeed - boolean variable to check if input document was read successfully
        ###### token - number of tokens in the input
        if token>2500: ###### If input is large, create embeddings for the document
            #db,pages=create_embeddings(string_data) 
            db,pages=create_embeddings(string_data,params) 

    ###### Show input summary ######
    col1, col2, col3=st.sidebar.columns(3) ###### Create columns
    col1.markdown("###### :violet[#Tokens:] :blue["+str(token)+"]") ###### Show number of tokens in the input
    col2.write("###### :violet[#Words:] :blue["+str(words)+"]") ###### Show number of words in the input
    col3.markdown("###### :violet[#Embeddings:] :blue["+str(pages)+"]") ###### Show number of embeddings in the input


    #### Splitting app into tabs ####
    tab1, tab2, tab3=st.tabs(["|__QnA__ 🔍|","|__Document Summary__ 📜|","|__About LOKI__ 🎭|"])

    with tab1: #### The QnA Tab
        
        if not succeed: ###### If input document was not read successfully, show error message
            st.error("#### The input document might be corrupted or the extraction of information from the input link failed. Try uploading a new document or entering a different link")
        else:
            initialize_chat("👋")  #### Initialize session state variables for the chat ####
            #### Put user question input on top ####
            with st.form('input form',clear_on_submit=True):
                inp=st.text_input("Please enter your question below and hit Submit. Please note that this is not a chat, yet 😉", key="current")
                submitted = st.form_submit_button("Submit")

            if not submitted: #### This will render the initial state message by VIDIA when no user question has been asked ####
                with st.container(): #### Define container for the chat
                    render_chat() #### Function renders chat messages based on recorded chat history
            if submitted:
                if token>2500:
                    with st.spinner("Finding most relevant section of the document..."):
                        info=search_context(db,inp)
                    with st.spinner("Preparing response..."):
                        final_text=q_response(inp,info,params)
                else:
                    info=string_data
                    with st.spinner("Scanning document for response..."): #### Wait while Bedrock response is awaited ####
                        final_text=q_response(inp,info,params) #### Gets response to user question. In case the question is out of context, gets general response calling out 'out of context' ####
                
                    #### This section creates columns for two buttons, to clear chat and to download the chat as history ####
                col1,col2,col3,col4=st.columns(4)
                col1.button("Clear History",on_click=clear,type='secondary') #### clear function clears all session history for messages #####
                f=write_history_to_a_file() #### combines the session messages into a string ####
                col4.download_button("Download History",data=f,file_name='history.txt')

                with st.container():
                        chatbot(inp,final_text) #### adds the latest question and response to the session messages and renders the chat ####

    with tab2: #### Document Summary Tab ####

        info=string_data ## pass the entire text for summarization
        with st.form('tab2',clear_on_submit=False):
            choice=st.radio("Select the type of summary you want to see",("Summary","Talking Points","Sample Questions","Extracted Text"),key="tab2",horizontal=True)
            submitted=st.form_submit_button("Submit")
            if submitted:
                if choice=="Summary":
                    st.markdown("#### Summary")
                    start = time.time()
                    st.write(summary(info,params,token))
                    end = time.time()
                    seconds = int(((end - start) % 60))
                    minutes = int((end - start) // 60)
                    total_time = f"""Time taken to generate a summary:
                    Minutes: {minutes} Seconds: {round(seconds, 2)}"""
                    with st.sidebar:
                        st.header(total_time)
                    st.write(summary(info,params,token))
                elif choice=="Talking Points":
                    st.markdown("#### Talking Points")
                    st.write(talking(info,params,token))
                elif choice=="Sample Questions":
                    st.markdown("#### Sample Questions")
                    st.write(questions(info,params,token))
                elif choice=="Extracted Text":
                    st.markdown("#### Extracted Text")
                    st.write(info)
            else:
                st.markdown("Note: :red[On the first time click, the app may go back to the QnA tab. Please click on the Document Summary tab again to see the response.]")
                

    with tab3:  #### About Tab #####
        st.image(hline)
        col1, col2, col3,col5,col4=st.columns([10,1,10,1,10])

        with col1:
            first_column()
        with col2:
            st.write(" ")
        with col3:
             second_column()
        with col5:
            st.write(" ")
        with col4:
             third_column()
        st.image(hline)
else: #### Default Main Page without Chat ####
    st.image(hline)
    heads()
    st.image(hline)
    col1, col2, col3,col5,col4=st.columns([10,1,10,1,10])
    with col1:
        first_column()
    with col2:
        st.write("")
        #st.image(vline,width=4)
    with col3:
        second_column()
    with col5:
        st.write("")
        #st.image(vline,width=4)
    with col4:
        third_column()
    st.image(hline)

#### Contact Information ####
with st.sidebar.expander("📬 __Contact__"):
    st.image(hline)
    contact()
    st.image(hline)
st.sidebar.image(hline)

#### Reset Button ####
if st.sidebar.button("🆘 Reset Application",key="Duo",use_container_width=True):
    st.rerun()
st.sidebar.image(hline)
 






