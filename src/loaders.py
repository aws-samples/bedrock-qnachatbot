'''Loader.py contains all functions to extract text from various sources'''
'''It also contains functions to create embeddings from text'''

import os #### Import os to remove image file from Assets folder
import boto3 #### Import to use Bedrock Emmbedding Model
import tempfile #### Import tempfile for CSV Data file
from langchain.document_loaders.csv_loader import CSVLoader
from datetime import datetime

'''Libraries for Text Data Extraction'''

from langchain_community.document_loaders import TextLoader, YoutubeLoader, AmazonTextractPDFLoader, UnstructuredPowerPointLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader #### Import UnstructuredImageLoader to extract text from image file
import pdfplumber #### Import pdfplumber to extract text from pdf file
import pathlib #### Import pathlib to extract file extension
import requests #### Import requests to extract text from weblink
from bs4 import BeautifulSoup #### Import BeautifulSoup to parse response from weblink
'''Libraries for Embeddings'''
from langchain.text_splitter import RecursiveCharacterTextSplitter #### Import RecursiveCharacterTextSplitter to split text into chunks of 10000 tokens
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS #### Import FAISS to create embeddings
'''Libraries for Web App'''
import tiktoken #### Import tiktoken to count number of tokens
import streamlit as st #### Import streamlit to create web app
import time
import pandas as pd
## For YouTube Module
from youtube_transcript_api.formatters import JSONFormatter ## new added for YoutTube - 4/15
from youtube_transcript_api import YouTubeTranscriptApi ## new added for YoutTube - 4/15

from configparser import ConfigParser ###### Import ConfigParser library for reading config file to get S3 Bucket and Prefix.

config_object = ConfigParser()
config_object.read("loki-config.ini")
#bucket=config_object["BUCKET"]["s3_bucket"]
#bucket_prefix=config_object["PREFIX"]["s3_prefix"] 

'''_________________________________________________________________________________________________________________'''


transcribe = boto3.client('transcribe')


####check_upload function to check if file has been uploaded
####uses extract_data, extract_page, extract_YT, extract_audio, extract_image to extract text from uploaded file
####parameters: "uploaded" is the uploaded file, "input_choice" is the input choice selected by the user
####returns: words->number of words, pages->number of embeddings, string_data->text extracted, True->to indicate successful upload, tokens->number of tokens from tiktoken
@st.cache_data #### Cache upload to avoid re-upload and re-extraction of files ####
def check_upload(uploaded,input_choice,params): #### Function to check if file has been uploaded ####
    if input_choice=="Document": #### If input choice is document, call extract_data function ####
        if st.session_state["page_name"] == "RFP":
            words, pages, string_data, tokens=extract_data_new(uploaded) #### Extract text from uploaded file ####
        else:
            words, pages, string_data, tokens=extract_data(uploaded) #### Extract text from uploaded file ####
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="Weblink":   #### If input choice is weblink, call extract_page function ####
        words, pages, string_data, tokens=extract_page(uploaded) #### Extract text from weblink #### 
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="YouTube":  #### If input choice is YouTube, call extract_YT function ####
        words, pages, string_data, tokens=extract_YT(uploaded) #### Extract text from YouTube link ####
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="Audio": #### If input choice is audio, call extract_audio function ####
        # Call upload_file_s3 function to upload the file to S3 Bucket
        #upload_file_s3(uploaded)
        words, pages, string_data, tokens=extract_audio(uploaded,params) #### Extract text from audio file ####
        return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="Image": #### If input choice is image, call extract_image function ####
        loc='./Assets/'+str(uploaded.name) #### Store location of image file ####
        words, pages, string_data, tokens=extract_image(loc,uploaded,params) #### Extract text from image file ####
        os.remove(loc) #### Remove image file from Assets folder ####
        return words, pages, string_data, True, tokens  #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="CSV": #### If input_choice is CSV File, call extract_csv_data function
         words, pages, string_data, tokens=extract_csv_data(uploaded) #### Extract text from uploaded file ####
         return words, pages, string_data, True, tokens #### Return number of words, number of embeddings, extracted text, True to indicate successful upload and number of tokens ####
    elif input_choice=="PPT": #### If input_choice is PPT File, call extract_data_ppt function
         words, pages, string_data, tokens=extract_data_ppt(uploaded) #### Extract text from uploaded file ####
         return words, pages, string_data, True, tokens
    
    else: #### If input choice is not any of the above, return False to indicate failed upload ####
        return 0,0,0,False,0 #### Return 0 for number of words, 0 for number of embeddings, 0 for extracted text, False to indicate failed upload and 0 for number of tokens ####
    
'''_________________________________________________________________________________________________________________'''

'''
All Text Data Extraction Functions
1. Document
    a. PDF
    b. TXT
    ## Uploaded as a file less than 200MB size
2. Weblink
    ## Entered as a weblink
3. YouTube
    ## Entered as a YouTube link
4. Audio
    ## Uploaded as an audio file less than 200MB size
5. Image
    ## Uploaded as an image file less than 200MB size
'''
'''_________________________________________________________________________________________________________________'''

####extract_data function to extract text from uploaded file
####Higher order function to select uploaders based on file type
####calls extract_data_pdf or extract_data_txt based on file type
####parameters: "feed" is the uploaded file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_data(feed):
    if pathlib.Path(feed.name).suffix=='.txt' or pathlib.Path(feed.name).suffix=='.TXT':
        return extract_data_txt(feed)
    elif pathlib.Path(feed.name).suffix=='.pdf' or pathlib.Path(feed.name).suffix=='.PDF':
        return extract_data_pdf(feed)
'''_________________________________________________________________________________________________________________'''


####extract_data function to extract text from uploaded list of files
####Higher order function to select uploaders based on file type
####calls extract_data_pdf or extract_data_txt based on file type
####parameters: "feed" is the uploaded file list
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_data_new(feed): ## feed is uploaded_file
    text = ""
    pg = []
    for f in feed:
        if f.name.endswith('.pdf'):
            with pdfplumber.open(f) as pdf:
                pages = pdf.pages
                pg.append(pages)
                for p in pages:
                    text+= p.extract_text()
    words = len(text.split())
    tokens = num_tokens_from_string(text,encoding_name="cl100k_base")
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####


'''_________________________________________________________________________________________________________________'''



####extract_data_pdf function to extract text from uploaded pdf file
####uses pdfplumber to extract text from pdf
####parameters: "feed" is the uploaded file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within extract_data function
def extract_data_pdf(feed): #### Function to extract text from pdf ####
    text="" #### Initialize text variable to store extracted text ####
    with pdfplumber.open(feed) as pdf: #### Open pdf file using pdfplumber ####
        pages=pdf.pages #### Extract pages from pdf ####
        for p in pages: #### Iterate through pages and extract text ####
            text+=p.extract_text()  #### Extract text from each page ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''

####extract_data_ppt function to extract text from uploaded pdf file
####parameters: "feed" is the uploaded file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within extract_data function
def extract_data_ppt(feed): #### Function to extract text from pdf ####

    text="" #### Initialize text variable to store extracted text ####
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(feed.getvalue())
        tmp_file_path = tmp_file.name
    loader = UnstructuredPowerPointLoader(tmp_file_path)
    docs = loader.load()
    for i in docs:
        text+= i.page_content
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    #words, pages, string_data,succeed,token
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


def extract_data_pdf_new(feed): #### Function to extract text from pdf ####
    text="" #### Initialize text variable to store extracted text ####
    for pdf in feed:
        with pdfplumber.open(feed.name) as pdf: #### Open pdf file using pdfplumber ####
            pages=pdf.pages #### Extract pages from pdf ####
            for p in pages: #### Iterate through pages and extract text ####
                text+=p.extract_text()  #### Extract text from each page ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_data_txt function to extract text from uploaded txt file
####uses read and decode as 'utf-8' to extract text from txt
####parameters: "feed" is the uploaded file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within extract_data function
def extract_data_txt(feed): #### Function to extract text from txt ####
    text=feed.read().decode("utf-8") #### Read and decode as 'utf-8' to extract text from txt ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_page function to extract text from weblink
####uses requests and BeautifulSoup to extract text from weblink
####parameters: "link" is the weblink
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_page(link): #### Function to extract text from weblink ####
    address=link #### Store weblink in address variable ####
    response=requests.get(address) #### Get response from weblink using requests ####
    soup = BeautifulSoup(response.content, 'html.parser') #### Parse response using BeautifulSoup ####
    text=soup.get_text() #### Extract text from parsed response ####
    lines = filter(lambda x: x.strip(), text.splitlines()) #### Filter out empty lines ####
    website_text = "\n".join(lines) #### Join lines to form text ####
    words=len(website_text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(website_text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, website_text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_YT function to extract text from YouTube link
####uses YoutubeLoader to extract text from YouTube link
####parameters: "link" is the YouTube link
#### youtube-transcript-api is a dependency and needs to be installed before running this function
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_YT(link): #### Function to extract text from YouTube link ####
    address=link #### Store YouTube link in address variable ####
    loader = YoutubeLoader.from_youtube_url(address, add_video_info=True) #### Load YouTube link using YoutubeLoader ####
    document=loader.load() #### Extract text from YouTube link ####
    # document[0].metadata['title'] --> Title of the source Video
    # document[0].metadata['thumbnail_url'] --> Cover page/Screenshot Title of the source Video
    # document[0].metadata['publish_date'] --> Published Date of the source Video
    # document[0].metadata['length'] --> Length of the source Video
    text=str(document[0].page_content) #### Convert extracted text to string ####
    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_audio function to extract text from audio file
####uses Amazon Transcribe to extract text from audio file
####parameters: "feed" is the audio file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function is used within check_upload function
def extract_audio(feed,params): #### Function to extract text from audio file ####
    string_data = upload_audio_file_s3(feed,params)
    words=len(string_data.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(string_data,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words,0,string_data, tokens    #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''


####extract_image function to extract text from image file
####uses UnstructuredImageLoader to extract text from image file
####parameters: "feed" is the location of the saved image file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function has a dependency on tessaract and pytesseract
####Please install these dependencies before running this function
####This function takes the location of the image file as an input rather than the image file
####The image file is deleted after the text is extracted
####This function is used within check_upload function
def extract_image(feed,uploaded,params): #### Function to extract text from image file ####
    s3_bucket_name = params['Bucket']
    bucket_prefix = params['Prefix']
    region = params['Region_Name']
    textract_client = boto3.client("textract", region_name=region)
    s3 = boto3.resource('s3')
    object = s3.Object(s3_bucket_name,bucket_prefix+"/"+uploaded.name)
    txt_data = uploaded.getvalue()
    result = object.put(Body=txt_data)
    res = result.get('ResponseMetadata')
    if res.get('HTTPStatusCode') == 200:
        file_path = "s3://"+s3_bucket_name+"/"+bucket_prefix+"/"+uploaded.name
        loader = AmazonTextractPDFLoader(file_path, client=textract_client)
        document=loader.load()
        text=str(document[0].page_content)
        words=len(text.split())
        tokens=num_tokens_from_string(text,encoding_name="cl100k_base")
    else:
        st.write("Pls upload the file")
    
    return words, 0, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''



####extract_csv_data function to extract text from CSV file
####uses CSVLoader to extract text from CSV file
####parameters: "feed" is the location of the saved CSV file
####returns: words->number of words, num->0 to indicate embeddings, text->text extracted, tokens->number of tokens from tiktoken
####Note: This function has a dependency on tessaract and pytesseract
####Please install these dependencies before running this function
####This function takes the location of the image file as an input rather than the image file
####The image file is deleted after the text is extracted
####This function is used within check_upload function
def extract_csv_data(feed): #### Function to extract text from image file ####
    # use tempfile because CSVLoader only accepts a file path
    text = ""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(feed.getvalue())
        tmp_file_path = tmp_file.name
    results = pd.read_csv(tmp_file_path)
    rec_num = len(results)
    loader = CSVLoader(file_path=tmp_file_path,encoding="utf-8",csv_args={
        'delimiter': ','
    }) #### Load CSV file using CSVLoader ####

    document=loader.load() #### Extract text from image file ####
    for i in range(rec_num):
        text+= document[i].page_content

    words=len(text.split()) #### Count number of words in the extracted text ####
    tokens=num_tokens_from_string(text,encoding_name="cl100k_base") #### Count number of tokens in the extracted text ####
    return words, document, text, tokens #### Return number of words, number of embeddings(placeholder), extracted text and number of tokens ####
'''_________________________________________________________________________________________________________________'''

####create_embeddings function to create embeddings from text
####uses Amazon Titan Embedding and FAISS to create embeddings from text
####parameters: "text" is the text to be embedded
####returns: db->database with embeddings, num_emb->number of embeddings
####Embeddings are created once per input and only if the input text is greater than 2500 tokens
#@st.cache_data #### Cache embeddings to avoid re-embedding ####
#def create_embeddings(text): #### Function to create embeddings from text ####
@st.cache_resource
def create_embeddings(text,params): #### Function to create embeddings from text ####
    with open('temp.txt','w') as f: #### Write text to a temporary file ####
         f.write(text) #### Write text to a temporary file ####
         f.close() #### Close temporary file ####
    loader=TextLoader('temp.txt') #### Load temporary file using TextLoader ####
    document=loader.load() #### Extract text from temporary file ####
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000) #### Initialize text splitter to split text into chunks of 10000 tokens ####
    docs = text_splitter.split_documents(document) #### Split document into chunks of 10000 tokens ####
    num_emb=len(docs) #### Count number of embeddings ####
    ## Use Amazon Bedrock Embedding Model
    
    modelId = params['endpoint-emb']
    embeddings = BedrockEmbeddings(model_id=modelId,region_name=params['Region_Name'])
    

    db = FAISS.from_documents(docs, embeddings) #### Create embeddings from text ####
    return db, num_emb #### Return database with embeddings and number of embeddings ####
'''_________________________________________________________________________________________________________________'''



####num_tokens_from_string function to count number of tokens in a text string
####uses tiktoken to count number of tokens in a text string
####parameters: "string" is the text string, "encoding_name" is the encoding name to be used by tiktoken
####returns: num_tokens->number of tokens in the text string
####This function is used within extract_data, extract_page, extract_YT, extract_audio, extract_image functions
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int: #### Function to count number of tokens in a text string ####
    encoding = tiktoken.get_encoding(encoding_name) #### Initialize encoding ####
    return len(encoding.encode(string)) #### Return number of tokens in the text string ####
'''_________________________________________________________________________________________________________________'''

## Function call to check the Transcribe Job
def check_job_name(job_name):
  job_verification = True
  
  # all the transcriptions
  existed_jobs = transcribe.list_transcription_jobs()
  
  for job in existed_jobs['TranscriptionJobSummaries']:
    if job_name == job['TranscriptionJobName']:
      job_verification = False
      break

  if job_verification == False:
    command = input(job_name + " has existed. \nDo you want to override the existed job (Y/N): ")
    if command.lower() == "y" or command.lower() == "yes":
      transcribe.delete_transcription_job(TranscriptionJobName=job_name)
    elif command.lower() == "n" or command.lower() == "no":
      job_name = input("Insert new job name? ")
      check_job_name(job_name)
    else: 
      print("Input can only be (Y/N)")
      command = input(job_name + " has existed. \nDo you want to override the existed job (Y/N): ")
  return job_name

'''_________________________________________________________________________________________________________________'''

def amazon_transcribe(audio_file_name,bucket,bucket_prefix):

  job_uri = "s3://"+bucket+"/"+bucket_prefix+"/"+audio_file_name
 

  # Usually, file names have spaces and have the file extension like .mp3
  # we take only a file name and delete all the space to name the job
  job_name = (audio_file_name.split('.')[0]).replace(" ", "")
  date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  job_name = job_name+'-'+date_time ## this is to ensure job name exists error is NOT there
  
  # file format
  file_format = audio_file_name.split('.')[-1]

  # check if name is taken or not
  job_name = check_job_name(job_name)
  transcribe.start_transcription_job(
      TranscriptionJobName=job_name,
      Media={'MediaFileUri': job_uri},
      MediaFormat = "mp4",
      LanguageCode='en-US')
  
  while True:
    result = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    if result['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
      break
    time.sleep(15)

  if result['TranscriptionJob']['TranscriptionJobStatus'] == "COMPLETED":
    data = pd.read_json(result['TranscriptionJob']['Transcript']['TranscriptFileUri'])
  
  return data['results'][1][0]['transcript']
'''_________________________________________________________________________________________________________________'''

####This function is used to upload the file to S3 and 
#### if successful, start the Amazon Transcribe function to conver the audio to Text
#### return Text
def upload_audio_file_s3(uploaded_file,params): #### 
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file_name = uploaded_file.name
            temp.write(uploaded_file.getvalue())
            file_name_with_path = temp.name
            temp.seek(0)
            audio_file = open(file_name_with_path, 'rb')
            audio_bytes = audio_file.read()
            s3_bucket_name = params['Bucket']
            bucket_prefix = params['Prefix']
            #region = params['Region_Name']
            s3_client = boto3.client("s3")
            response = s3_client.put_object(
                Body = audio_bytes,
                Bucket = s3_bucket_name,
                Key = bucket_prefix+"/"+file_name

            )
            if response["ResponseMetadata"]["HTTPStatusCode"]:
                message = st.success('File uploaded! The content will be available in a few minutes.')
                text_data = amazon_transcribe(file_name,s3_bucket_name,bucket_prefix)
            else:
               message = st.error('Something went wrong. Please try again.')

            time.sleep(5)
            message.empty()

    return text_data #### Return number of tokens in the text string ####
'''_________________________________________________________________________________________________________________'''