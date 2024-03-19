''' This file contains the functions for generating answers to questions and chat using Amazon Bedrock'''
''' This file also contains the functions to generate summaries, key points and questions from a document using Amazon Bedrock'''

import streamlit as st ###### import streamlit library for creating the web app
import boto3 #### import boto3 so as to Amazon Bedrock
import json
from anthropic import Anthropic
from configparser import ConfigParser ###### import ConfigParser library for reading the config file
# class from the Langchain library that splits text into smaller chunks based on specified parameters.
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import tiktoken #### Import tiktoken to count number of tokens
'''_________________________________________________________________________________________________________________'''

#### Create config object and read the config file ####
config_object = ConfigParser() ###### Create config object
config_object.read("./loki-config.ini") ###### Read config file
claude = Anthropic() # for Tokenizer
'''_________________________________________________________________________________________________________________'''


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int: #### Function to count number of tokens in a text string ####
    encoding = tiktoken.get_encoding(encoding_name) #### Initialize encoding ####
    return len(encoding.encode(string)) #### Return number of tokens in the text string ####
'''_________________________________________________________________________________________________________________'''


def initialize_summary_session_state():
    if "summary_flag" not in st.session_state:
        st.session_state.summary_flag = False
    if "summary_content" not in st.session_state:
        st.session_state.summary_content = ""

def bedrock_llm_call(params, qa_prompt="", temperature=0.1, max_tokens=256,top_p=0.99,frequency_penalty=1,user_id="test-user"):    
    #bedrock = boto3.client(service_name='bedrock-runtime',region_name='us-east-1',endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')
    bedrock = boto3.client(service_name='bedrock-runtime',region_name=params['Region_Name'])

    if 'claude2' in params['model_name'].lower() or 'claude instant' in params['model_name'].lower() or 'claude' in params['model_name'].lower():
        
        prompt = {
            "prompt": "\n\nHuman:" + qa_prompt + "\n\nAssistant:",
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_k": 50,
            "top_p": top_p
        }
        prompt=json.dumps(prompt)
        input_token = claude.count_tokens(prompt)
        response = bedrock.invoke_model(
            body=prompt,
            modelId= params['endpoint-llm'], # "anthropic.claude-v2",
            accept="application/json",
            contentType="application/json"
        )
        answer=response['body'].read().decode()
        answer=json.loads(answer)['completion']
        text = answer
        output_token = claude.count_tokens(answer) ###### count the number of tokens used for output
        total_token_consumed = input_token + output_token ###### count the number of total tokens used
        words=len(text.split()) ###### count the number of words used
        reason = ""
    elif 'ai21-j2-mid' in params['model_name'].lower():
        prompt={
          "prompt":  qa_prompt,
          "maxTokens": max_tokens,
          "temperature": temperature,
          "topP":  top_p,
          "stopSequences": ["Human:"],
          "countPenalty": {"scale": 0 },
          "presencePenalty": {"scale": 0.5 },
          "frequencyPenalty": {"scale": 0.8 }
        }
        prompt=json.dumps(prompt)
        response = bedrock.invoke_model(body=prompt,
                                modelId=params['endpoint-llm'], 
                                accept="application/json", 
                                contentType="application/json")
        answer=response['body'].read().decode() 
        input_token = len(json.loads(answer)['prompt']['tokens'])
        output_token = len(json.loads(answer)['completions'][0]['data']['tokens']) ###### count the number of tokens used for output
        answer=json.loads(answer)['completions'][0]['data']['text']
        text = answer
        total_token_consumed = input_token + output_token ###### count the number of total tokens used
        words=len(text.split()) ###### count the number of words used
        reason = ""
    elif 'command' in params['model_name'].lower():
        # p is a float with a minimum of 0, a maximum of 1, and a default of 0.75
        # k is a float with a minimum of 0, a maximum of 500, and a default of 0
        # max_tokens is an int with a minimum of 1, a maximum of 4096, and a default of 20
        # num_generations has a minimum of 1, maximum of 5, and default of 1
        # return_likelihoods defaults to NONE but can be set to GENERATION or ALL to
        #   return the probability of each token
        prompt = {
            "prompt":  qa_prompt ,
            "max_tokens": params['max_len'],
            "temperature": params['temp'],
            "k": 50,
            "p": params['top_p']
        }
        prompt=json.dumps(prompt)
        response = bedrock.invoke_model(
            body=prompt,
            modelId= params['endpoint-llm'], # "anthropic.claude-v2",
            accept="application/json",
            contentType="application/json"
        )
        
        answer=response['body'].read().decode()
        answer=json.loads(answer)['generations'][0]['text']
        text = answer
        output_token = 200 ## This is just dummy number
        words=len(text.split()) ###### count the number of words used
        reason = ""
    elif 'llama2' in params['model_name'].lower():
        prompt = {
            "prompt":  "[INST] "+qa_prompt + "[/INST]" ,
            "max_gen_len": params['max_len'],
            "temperature": params['temp'],
            "top_p": params['top_p']
        }

        prompt=json.dumps(prompt)
        response = bedrock.invoke_model(
            body=prompt,
            modelId= params['endpoint-llm'],
            accept="application/json",
            contentType="application/json"
        )

        body = response.get('body').read().decode('utf-8')
        response_body = json.loads(body)
        text = response_body['generation'].strip()
        output_token = response_body['generation_token_count'] ## This is just dummy number
        words = len(text.split()) ###### count the number of words used
        reason = ""
    elif 'mistral' in params['model_name'].lower() or 'mixtral' in params['model_name'].lower():
        prompt = {
            "prompt":  "[INST] "+qa_prompt + "[/INST]" ,
            "max_tokens": params['max_len'],
            "temperature": params['temp'],
            "top_p": params['top_p'],
            "top_k": 50
        }

        prompt=json.dumps(prompt)
        response = bedrock.invoke_model(
            body=prompt,
            modelId= params['endpoint-llm'],
            accept="application/json",
            contentType="application/json"
        )

        body = response.get('body').read().decode('utf-8')
        response_body = json.loads(body)
        text = response_body['outputs'][0]['text']
        #output_token = response_body['generation_token_count'] ## This is just dummy number
        #output_token = 200
        output_token = num_tokens_from_string(text,encoding_name="cl100k_base")
        words = len(text.split()) ###### count the number of words used
        reason = ""        

    return text, output_token, words, reason ###### return the generated text, number of tokens, number of words and reason for stopping the text generation



### Modified one for Amazon Bedrok
def q_response(query,doc,params): ###### q_response function    
    prompt=f"""
            Context: {doc}
            
            Answer the question as truthfully as possible using the above provided context and if the answer is not contained within the context provided, say "I don't know"
            Strict Instruction: provide an answer for question. Answer "don't know",if the answer is NOT available in the context.
                        
            According to the context provided, {query}
            
            """     
    text, t1, t2, t3=bedrock_llm_call(params,prompt) ###### call the bedrock_llm_call function
    text_final=text ###### create the final answer with the result in the context
    return text_final ###### return the final answer
'''_________________________________________________________________________________________________________________'''




def chat_bedrock_call(message_dict=[{"role":"user","content":"Hello!"}], model="anthropic.claude-v2", max_tokens=120,temperature=0.1):
    bedrock = boto3.client(service_name='bedrock-runtime')
    qa_prompt = message_dict["content"]
    prompt = {
        "prompt": "\n\nHuman:" + qa_prompt + "\n\nAssistant:",
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 0.99
    }
    prompt=json.dumps(prompt)
    input_token = claude.count_tokens(prompt)
    response = bedrock.invoke_model(
        body=prompt,
        modelId=model,
        accept="application/json",
        contentType="application/json"
    )
    answer=response['body'].read().decode()
    response_dict=json.loads(answer)
    answer=json.loads(answer)['completion']    

    response_text=answer ###### get the response text
    words=len(response_text.split()) ###### count the number of words used
    response_tokens = claude.count_tokens(answer) ###### count the number of tokens used for output
    total_tokens = input_token + response_tokens ###### count the number of total tokens used
    
    return response_text, response_dict, words, total_tokens, response_tokens  ###### return the generated text, response dictionary, number of words, total number of tokens and number of tokens in response
'''_________________________________________________________________________________________________________________'''

#### create_dict_from_session function to create a message dictionary from the session state to include chat behavior####
#### This function takes no inputs and works on two session state variables####
#### pastinp: the list of past user inputs ####
#### pastresp: the list of past assistant responses
#### This function returns the following outputs: ####
#### mdict: the message dictionary ####
def create_dict_from_session(): ###### create_dict_from_session function
    mdict=[] ###### initialize the message dictionary
    if (len(st.session_state['pastinp']))==0: ###### check if the session state is empty
        mdict=[] ###### if the session state is empty, return an empty message dictionary
        return mdict ###### return the empty message dictionary
    elif (len(st.session_state['pastinp']))==1: ###### check if the session state has only one message
        mdict=  [ ###### if the session state has only one message, create a message dictionary with the message
                    {"role":"user","content":st.session_state['pastinp'][0]}, 
                    {"role":"assistant","content":st.session_state['pastresp'][1]} 
                ]
        return mdict   ###### return the message dictionary
    elif (len(st.session_state['pastinp']))==2: ###### check if the session state has only two messages
        mdict=  [ ###### if the session state has only two messages, create a message dictionary with the messages
                    {"role":"user","content":st.session_state['pastinp'][0]},
                    {"role":"assistant","content":st.session_state['pastresp'][1]},
                    {"role":"user","content":st.session_state['pastinp'][1]},
                    {"role":"assistant","content":st.session_state['pastresp'][2]}
                ]
        return mdict ###### return the message dictionary
    else: ###### if the session state has more than two messages
        for i in range(len(st.session_state['pastinp'])-3,len(st.session_state['pastinp'])): ###### loop through the session state to create a message dictionary with the last three messages
            mdict.append({"role":"user","content":st.session_state['pastinp'][i]}) ###### add the user message to the message dictionary
            mdict.append({"role":"assistant","content":st.session_state['pastresp'][i+1]}) ###### add the assistant message to the message dictionary
        return mdict    ###### return the message dictionary
'''_________________________________________________________________________________________________________________'''


#### Modified for Bedrock
def q_response_chat(query,doc,mdict): ###### q_response_chat function
    prompt=f"Answer the question below only and only from the context provided. Answer in detail and in a friendly, enthusiastic tone. If not in the context, respond in no other words except '100', only and only with the number '100'. Do not add any words to '100'.\n context:{doc}.\nquestion:{query}.\nanswer:" ###### create the prompt asking Bedrock to generate an answer with the question and document as context and '100' as the answer if the answer is not in the context
    mdict.append({"role":"user","content":prompt}) ###### add the prompt to the message dictionary
    response_text, response_dict, words, total_tokens, response_tokens=chat_bedrock_call(message_dict=mdict) ###### call the chat_gpt_call function
    text_final=response_text ###### create the final answer with the result in the context
    return text_final ###### return the final answer
'''_________________________________________________________________________________________________________________'''

#### search_context function to search the database for the most relevant section to the user question ####
#### This function takes the following inputs: ####
#### db: the database with embeddings to be used for answering the question ####
#### query: the question to be answered ####
#### This function returns the following outputs: ####
#### defin[0].page_content: the most relevant section to the user question ####
def search_context(db,query): ###### search_context function
     defin=db.similarity_search(query) ###### call the FAISS similarity_search function that searches the database for the most relevant section to the user question and orders the results in descending order of relevance
     return defin[0].page_content ###### return the most relevant section to the user question
'''_________________________________________________________________________________________________________________'''


def summarizer(prompt_data,params,initial_token_count):    
    """
    This function creates the summary of each individual chunk as well as the final summary.
    :param prompt_data: This is the prompt along with the respective chunk of text, at the end it contains all summary chunks combined.
    :return: A summary of the respective chunk of data passed in or the final summary that is a summary of all summary chunks.
    """
    bedrock = boto3.client(service_name='bedrock-runtime',region_name=params['Region_Name'])
    if initial_token_count > 2500: ## if the token count of the document is more than 2500, prefer using Claude v2 for Summarization
        prompt = {
            "prompt": "\n\nHuman:" + prompt_data + "\n\nAssistant:",
            "max_tokens_to_sample": params['max_len'],
            "temperature": params['temp'],
            "top_k": 50,
            "top_p": params['top_p']
        }
        prompt=json.dumps(prompt)
        input_token = claude.count_tokens(prompt)
        response = bedrock.invoke_model(
            body=prompt,
            modelId= "anthropic.claude-v2",  ###params['endpoint-llm'],
            accept="application/json",
            contentType="application/json"
        )
        
        answer=response['body'].read().decode()
        answer=json.loads(answer)['completion']
    else:

        if 'claude2' in params['model_name'].lower() or 'claude instant' in params['model_name'].lower() or 'claude' in params['model_name'].lower():
            
            prompt = {
                "prompt": "\n\nHuman:" + prompt_data + "\n\nAssistant:",
                "max_tokens_to_sample": params['max_len'],
                "temperature": params['temp'],
                "top_k": 50,
                "top_p": params['top_p']
            }
            prompt=json.dumps(prompt)
            input_token = claude.count_tokens(prompt)
            response = bedrock.invoke_model(
                body=prompt,
                modelId= params['endpoint-llm'],
                accept="application/json",
                contentType="application/json"
            )
            
            answer=response['body'].read().decode()
            answer=json.loads(answer)['completion']
            
        elif 'ai21-j2-mid' in params['model_name'].lower():
            prompt={
            "prompt":  prompt_data,
            "maxTokens": params['max_len'],
            "temperature": params['temp'],
            "topP":  params['top_p'],
            "stopSequences": ["Human:"],
            "countPenalty": {"scale": 0 },
            "presencePenalty": {"scale": 0.5 },
            "frequencyPenalty": {"scale": 0.8 }
            }
            prompt=json.dumps(prompt)
            response = bedrock.invoke_model(body=prompt,
                                    modelId=params['endpoint-llm'], 
                                    accept="application/json", 
                                    contentType="application/json")
            answer=response['body'].read().decode() 
            answer=json.loads(answer)['completions'][0]['data']['text']
        elif 'command' in params['model_name'].lower():
            # p is a float with a minimum of 0, a maximum of 1, and a default of 0.75
            # k is a float with a minimum of 0, a maximum of 500, and a default of 0
            # max_tokens is an int with a minimum of 1, a maximum of 4096, and a default of 20
            # num_generations has a minimum of 1, maximum of 5, and default of 1
            # return_likelihoods defaults to NONE but can be set to GENERATION or ALL to
            #   return the probability of each token
            prompt = {
                "prompt":  prompt_data ,
                "max_tokens": params['max_len'],
                "temperature": params['temp'],
                "k": 50,
                "p": params['top_p']
            }
            prompt=json.dumps(prompt)
            response = bedrock.invoke_model(
                body=prompt,
                modelId= params['endpoint-llm'], # "anthropic.claude-v2",
                accept="application/json",
                contentType="application/json"
            )
            
            answer=response['body'].read().decode()
            answer=json.loads(answer)['generations'][0]['text']
        
        elif 'llama2' in params['model_name'].lower():
            prompt = {
                "prompt":  "[INST] "+prompt_data + "[/INST]" ,
                "max_gen_len": params['max_len'],
                "temperature": params['temp'],
                "top_p": params['top_p']
            }

            prompt=json.dumps(prompt)
            response = bedrock.invoke_model(
                body=prompt,
                modelId= params['endpoint-llm'],
                accept="application/json",
                contentType="application/json"
            )

            body = response.get('body').read().decode('utf-8')
            response_body = json.loads(body)
            answer = response_body['generation'].strip()
        elif 'mistral' in params['model_name'].lower() or 'mixtral' in params['model_name'].lower():
            prompt = {
                "prompt":  "[INST] "+prompt_data + "[/INST]" ,
                "max_tokens": params['max_len'],
                "temperature": params['temp'],
                "top_p": params['top_p'],
                "top_k": 50
            }

            prompt=json.dumps(prompt)
            response = bedrock.invoke_model(
                body=prompt,
                modelId= params['endpoint-llm'],
                accept="application/json",
                contentType="application/json"
            )

            body = response.get('body').read().decode('utf-8')
            response_body = json.loads(body)
            answer = response_body['outputs'][0]['text']
  
  
    return answer


def generate_summarized_content(info,params,token): ###### summary function
    # We need to split the text using Character Text Split such that it should not increase token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 10000,
        chunk_overlap  = 2000,
        length_function = len,
    )
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=10000, 
        chunk_overlap=2000,
        length_function=len,
        add_start_index=True
    )
    texts = splitter.create_documents([info])
    # Creating an empty summary string, as this is where we will append the summary of each chunk
    summary = ""
    # looping through each chunk of text we created, passing that into our prompt and generating a summary of that chunk
    for index, chunk in enumerate(texts):
        # gathering the text content of that specific chunk
        chunk_content = chunk.page_content
        # creating the prompt that will be passed into Bedrock with the text content of the chunk
        prompt = f"""\n\nHuman: Provide a detailed summary for the chunk of text provided to you:
        Text: {chunk_content}
        \n\nAssistant:"""
        # passing the prompt into the summarizer function to generate the summary of that chunk, and appending it to
        # the summary string
        summary += summarizer(prompt,params,token)

    return summary

'''_________________________________________________________________________________________________________________'''

#### summarize function to generate a summary of a document ####
#### This function takes the following inputs: ####
#### info: the document to be summarized ####
#### models: the model to be used for generating text ####
#### This function returns the following outputs: ####
#### text: the generated summary ####
def summary(info,params,token): ###### summary function
    initialize_summary_session_state()
    if st.session_state.summary_flag:
        print("Already Summarized")
        summary = st.session_state.summary_content
    else:
        summary = generate_summarized_content(info,params,token)
        st.session_state.summary_flag = True
        st.session_state.summary_content = summary
    final_summary_prompt = f"""\n\nHuman: You will be given a set of summaries from a document. Create a cohesive 
    summary from the provided individual summaries. The summary should very crisp and at max 700 wrods. 
    Summaries: {summary}
            \n\nAssistant:"""
    
    #prompt="Review the summaries from multiple pieces of a single document below:\n"+info+".\n Merge the summaries into a single coherent and cohesive narrative highlighting all key points and produce the summary in 500 words." ###### create the prompt asking LLM to generate a summary of the document
    with st.spinner('Summarizing your uploaded document'): ###### wait while Bedrock response is awaited
        text = summarizer(final_summary_prompt,params,token)
    return text ###### return the generated summary

'''_________________________________________________________________________________________________________________'''


#### talking function to generate key points of a document ####
#### This function takes the following inputs: ####
#### info: the document to be summarized ####
#### models: the model to be used for generating text ####
#### This function returns the following outputs: ####
#### text: the generated key points ####
def talking(info,params,token): ###### talking function
    initialize_summary_session_state()
    if st.session_state.summary_flag:
        print("Already Summarized")
        summary_for_talking_points = st.session_state.summary_content
    else:
        summary_for_talking_points = generate_summarized_content(info,params)
        st.session_state.summary_flag = True
        st.session_state.summary_content = summary_for_talking_points

    prompt="In short bullet points, extract all the main talking points of the text below:\n"+summary_for_talking_points+".\nDo not add any pretext or context. Write each bullet in a new line." ###### create the prompt asking Bedrock to generate key points of the document
    with st.spinner('Extracting the key points'): ###### wait while Bedrock response is awaited
       text = summarizer(prompt,params,token) ###### call the summarizer function
    return text ###### return the generated key points
'''_________________________________________________________________________________________________________________'''


#### questions function to generate questions from a document ####
#### This function takes the following inputs: ####
#### info: the document to be summarized ####
#### models: the model to be used for generating text ####
#### This function returns the following outputs: ####
#### text: the generated questions ####
def questions(info,params,token): ###### questions function
    initialize_summary_session_state()
    if st.session_state.summary_flag:
        summary_for_questions_gen = st.session_state.summary_content
    else:
        summary_for_questions_gen = generate_summarized_content(info,params)
        st.session_state.summary_flag = True
        st.session_state.summary_content = summary_for_questions_gen
    prompt="Extract ten questions that can be asked of the text below:\n"+summary_for_questions_gen+".\nDo not add any pretext or context." ###### create the prompt asking openai to generate questions from the document

    with st.spinner('Generating a few sample questions'): ###### wait while Bedrock response is awaited
        text = summarizer(prompt,params,token) ###### call the summarizer function
    return text ###### return the generated questions
'''_________________________________________________________________________________________________________________'''

