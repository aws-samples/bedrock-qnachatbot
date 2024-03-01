import streamlit as st 
from streamlit_chat import message   
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import ConversationalRetrievalChain   
from langchain_community.document_loaders.csv_loader import CSVLoader   
from langchain_community.vectorstores import FAISS   
import tempfile
import boto3
from langchain.llms.bedrock import Bedrock
from configparser import ConfigParser ###### Import ConfigParser library for reading config file to get model, greeting message, etc.
from PIL import Image ###### Import Image library for loading images
import os ###### Import os library for environment variables
from utils import * ###### Import utility functions
from loaders import create_embeddings, check_upload ###### Import functions to load input from different sources
from textgeneration import q_response, q_response_chat, search_context, summary, talking, questions ###### Import functions to generate text from input
from chat import initialize_chat, render_chat, chatbot
# new 
import pandas as pd
from io import BytesIO

input_choice="CSV"
#### Create config object and read the config file ####
config_object = ConfigParser()
config_object.read("loki-config.ini")

#### Initialize variables and reading configuration ####
logo=Image.open(config_object["IMAGES"]["logo_address"]) #### Logo for the sidebar
favicon=Image.open(config_object["IMAGES"]["favicon_address"]) #### Favicon


greeting=config_object["MSG"]["greeting"] ###### initial chat message
hline=Image.open(config_object["IMAGES"]["hline"]) ###### image for formatting landing screen
uploaded=None ##### initialize input document to None

#### Set Page Config ####
st.set_page_config(layout="wide", page_icon=favicon, page_title="LOKI") ###### Set page layout, favicon and title

#### Set Logo on top sidebar ####
st.sidebar.image(hline) ###### Add horizontal line
c1,c2,c3=st.sidebar.columns([1,3,1]) ###### Create columns
c2.image(logo) ###### Add logo to middle column
st.sidebar.image(hline) ###### Add horizontal line

Page_name = st.session_state["page_name"] = "csvBot"

st.sidebar.image(hline) ###### Add horizontal line
params = select_models(Page_name) ##### Call Model Selection routine
input_choice, uploaded=input_selector_csv(Page_name,params) ###### Get input choice and input document
st.sidebar.image(hline) ###### Add horizontal line

def get_bedrock_client(region):
    bedrock_client= boto3.client(service_name='bedrock-runtime',region_name=region)
    return bedrock_client

def bedrock_llm(params):
    bedrock_client = get_bedrock_client(params['Region_Name'])
    if 'claude2' in params['model_name'].lower() or 'claude instant' in params['model_name'].lower() or 'claude' in params['model_name'].lower():
        model_kwargs = {
            "max_tokens_to_sample": params['max_len'],
            "temperature": params['temp'],
            "top_k": 50,
            "top_p": params['top_p']
        }
    elif 'ai21-j2-mid' in params['model_name'].lower():
         model_kwargs = {
              "maxTokens": params['max_len'],
              "temperature": params['temp'],
              "topP":  params['top_p'],
              "stopSequences": ["Human:"],
              "countPenalty": {"scale": 0 },
              "presencePenalty": {"scale": 0.5 },
              "frequencyPenalty": {"scale": 0.8 }
         }
    elif 'titan' in params['model_name'].lower():
         model_kwargs = {
              "maxTokenCount": params['max_len'],
              "temperature":params['temp'],
              "topP":params['top_p'], 
         }
    elif 'command' in params['model_name'].lower():        
         model_kwargs = {
            "max_tokens": params['max_len'],
            "temperature": params['temp'],
            "k": 50,
            "p": params['top_p']  
         }
    elif 'llama2' in params['model_name'].lower():
         model_kwargs = {
            "max_gen_len": params['max_len'],
            "temperature": params['temp'],
            "top_p": params['top_p']
        }
    llm  = Bedrock(client=bedrock_client,region_name=params['Region_Name'],model_id=params['endpoint-llm'],model_kwargs=model_kwargs)

    return llm

if uploaded is not None and uploaded !="":
    ## New 
    uploaded_file_content = BytesIO(uploaded.getvalue())
    df = pd.read_csv(uploaded_file_content)
    st.session_state.df = df
    #with st.spinner("reading "+input_choice):
    doc_content = ""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded.getvalue())
        tmp_file_path = tmp_file.name
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args= {
        "delimiter": ","
    })
    data = loader.load()
    for doc in data:
         doc_content+=doc.page_content 

    words, pages, string_data,succeed,token=check_upload(uploaded=uploaded,input_choice=input_choice,params=params)
    if token>2500: ###### If input is large, create embeddings for the document
            db,pages=create_embeddings(string_data,params)

    ## Create the embddings
    bedrock_client = get_bedrock_client(params['Region_Name'])
    modelId = params['endpoint-emb']
    embeddings = BedrockEmbeddings(client=bedrock_client,region_name=params['Region_Name'],model_id=modelId)

    ## Create the vector store and save the index
    db = FAISS.from_documents(data, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)

    ## Create an instance of LLM
    llm = bedrock_llm(params)

    ## Create an instance of Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type="stuff",verbose=False,retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":5}))
    # set search_type and search_kwargs — not doing so would be cost inefficient as all of the chunks in your vector store would be sent to the LLM


    #### Splitting app into tabs ####
    tab1, tab2, tab3=st.tabs(["|__QnA__ 🔍|","|__Document Summary__ 📜|","|__About LOKI__ 🎭|"])

    with tab1: #### The QnA Tab
        
            initialize_chat("👋")  #### Initialize session state variables for the chat ####
            ## new - Begin
            if st.session_state.df is not None:
                st.subheader("Current dataframe:")
                st.write(st.session_state.df)
            ## new - End
            #### Put user question input on top ####
            with st.form('input form',clear_on_submit=True):
                inp=st.text_input("Please enter your question below and hit Submit. Please note that this is not a chat, yet 😉", key="current")
                submitted = st.form_submit_button("Submit")

            if not submitted: #### This will render the initial state message by VIDIA when no user question has been asked ####
                with st.container(): #### Define container for the chat
                    render_chat() #### Function renders chat messages based on recorded chat history
            if submitted:
                #### This section creates columns for two buttons, to clear chat and to download the chat as history ####
                col1,col2,col3,col4=st.columns(4)
                col1.button("Clear History",on_click=clear,type='secondary') #### clear function clears all session history for messages #####

                with st.container():
                        #print(st.session_state['history'])
                        #result = chain({"question": inp,"chat_history": st.session_state['history']})
                        result = chain.invoke({"question": inp,"chat_history": st.session_state['history']})
                        st.session_state['history'].append((inp, result["answer"]))
                        final_text = result["answer"]
                        chatbot(inp,final_text) #### adds the latest question and response to the session messages and renders the chat ####
    
    with tab2: #### Document Summary Tab ####
         
         #info=doc_content
         if token>2500:
            with st.spinner("Finding most relevant section of the document..."):
                    info=search_context(db,"The most important section of the document")
         else:
            info=string_data         
         with st.form('tab2',clear_on_submit=False):
              choice=st.radio("Select the type of summary you want to see",("Summary","Talking Points","Sample Questions","Extracted Text"),key="tab2",horizontal=True)
              submitted=st.form_submit_button("Submit")
              if submitted:
                   if choice=="Summary":
                        st.markdown("#### Summary")
                        st.write(summary(info,params))
                   elif choice=="Talking Points":
                        st.markdown("#### Talking Points")
                        st.write(talking(info,params))
                   elif choice=="Sample Questions":
                        st.markdown("#### Sample Questions")
                        st.write(questions(info,params))
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
