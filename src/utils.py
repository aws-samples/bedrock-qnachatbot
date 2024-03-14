'''
utils.py consists of all template, input and other utility functions of loki
'''

import streamlit as st ###### Import Streamlit library
from streamlit_chat import message ###### Import message function from streamlit_chat library to render chat
from configparser import ConfigParser ###### Import ConfigParser library to read config file for greeting
from chat import initialize_chat ###### Import initialize_chat function from chat.py to initialise chat upon clearing session state
from PIL import Image ###### Import Image from PIL library to save image uploaded by user
import json


bucket='<S3_Bucket_Name>'
prefix='gen-ai-qa'
region_name='<AWS_Region>'


application_metadata = {
    'models-llm':[
        {'name':'Claude2', 'endpoint':"anthropic.claude-v2"},
        {'name':'AI21-J2-mid', 'endpoint':'ai21.j2-mid'},
        {'name':'Claude', 'endpoint':"anthropic.claude-v1"},
        {'name':'Claude Instant', 'endpoint':"anthropic.claude-instant-v1"},
        {'name':'Command', 'endpoint':"cohere.command-text-v14"},
        {'name':'Titan', 'endpoint':"amazon.titan-text-express-v1"},
        {'name':'Llama2-70b', 'endpoint':"meta.llama2-70b-chat-v1"},
        {'name':'Llama2-13b', 'endpoint':"meta.llama2-13b-chat-v1"},
        {'name':'mistral-7b', 'endpoint':"mistral.mixtral-8x7b-instruct-v0:1"},
        {'name':'mixtral-8x7b', 'endpoint':"mistral.mixtral-8x7b-instruct-v0:1"}
       ],
    'models-emb':[
        {'name':'Titan', 'endpoint':'amazon.titan-embed-text-v1'},
        ],
    'summary_model':'cohere-gpt-medium',
    'region':region_name,
    'kendra_index':'<update-Your-Kendra-IndexID>',
    'datastore':
        {'bucket':bucket, 'prefix':prefix},
    'opensearch':
        {'es_username':'username', 'es_password':'password', 'domain_endpoint':'<OpenSeach Domain Endpoint>'},    
}
json.dump(application_metadata, open('application_metadata_complete.json', 'w'))

APP_MD    = json.load(open('application_metadata_complete.json', 'r'))
MODELS_LLM = {d['name']: d['endpoint'] for d in APP_MD['models-llm']}
MODELS_EMB = {d['name']: d['endpoint'] for d in APP_MD['models-emb']}
MODEL_SUM = APP_MD['summary_model']
REGION    = APP_MD['region']
BUCKET    = APP_MD['datastore']['bucket']
PREFIX    = APP_MD['datastore']['prefix']


####
config_object = ConfigParser() ###### Read config file for greeting
config_object.read("./loki-config.ini") #
greeting=config_object["MSG"]["greeting"] #
###


#### function to display document input options and return the input choice and uploaded file
#### this function is called from the main.py file
def input_selector():

        input_choice=st.sidebar.radio("#### :blue[Choose the Input Method]",('Document','Weblink','YouTube','Audio','Image','PPT'))
        if input_choice=="Document":
            with st.sidebar.expander("📁 __Documents__",expanded=True):
                uploaded=st.file_uploader(label="Select File",type=['pdf','txt'],on_change=clear)
        elif input_choice=="Weblink":
            with st.sidebar.expander("🌐 __Webpage__",expanded=True):
                uploaded=st.text_input('Enter a weblink',on_change=clear)
        elif input_choice=="YouTube":
            with st.sidebar.expander("🎥 __YouTube__",expanded=True):
                uploaded=st.text_input('Enter a YT link',on_change=clear)
        elif input_choice=="Audio":
            with st.sidebar.expander("🎙 __Audio__",expanded=True):
                uploaded=st.file_uploader('Select File',type=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav'],on_change=clear)
        elif input_choice=="Image":
            with st.sidebar.expander("🎙 __Text from Image__",expanded=True):
                uploaded=st.file_uploader('Select File',type=['jpg','jpeg','png'],on_change=clear, disabled=False)
                if uploaded:
                    image=Image.open(uploaded)
                    loc='./Assets/'+str(uploaded.name)
                    image.save(loc)
        elif input_choice=="PPT":
             with st.sidebar.expander("🖼️ __Powerpoint__",expanded=True):
                 uploaded = st.file_uploader("Select your PPT", type=['ppt', 'pptx'], accept_multiple_files=False) 
             

                 
        
        return input_choice, uploaded

#### function to display document input options and return the input choice and uploaded file
#### this function is called from the main.py file
def input_selector_rfp(Page_name,params):
    if Page_name == "RFP":
        input_choice = "Document"
        with st.sidebar.expander("📁 __Documents__",expanded=True):
            uploaded=st.file_uploader(label="Select Company profile",type=['pdf','txt'],on_change=clear,accept_multiple_files=True)
    return input_choice, uploaded


def input_selector_csv(Page_name,params):
    if Page_name == "csvBot" or Page_name == "CUR":
        input_choice = "CSV"
        with st.sidebar.expander("📁 __CSV__",expanded=True):
            uploaded = st.sidebar.file_uploader(label="Upload your CSV file",type="csv",on_change=clear)
    return input_choice, uploaded

def select_models(page):
     with st.sidebar:
        llm_model_name = st.selectbox("#### :blue[Select LLM Model]", options=MODELS_LLM.keys())
        emb_model_name = st.selectbox("#### :blue[Select Embedding Model]", options=MODELS_EMB.keys())
        if page == "rag":
            retriever = st.selectbox("#### :blue[Select Retriever]", options=["Opensearch","Kendra"])
            if "OpenSearch" in retriever:
                K=st.slider('Top K', min_value=0., max_value=10., value=1., step=1.)
                engine=st.selectbox('KNN algorithm', ("nmslib", "lucene"), help="Underlying KNN algorithm implementation to use for powering the KNN search")
                m=st.slider('Neighbouring Points', min_value=16.0, max_value=124.0, value=72.0, step=1., help="Explored neighbors count")
                ef_search=st.slider('efSearch', min_value=10.0, max_value=2000.0, value=1000.0, step=10., help="Exploration Factor")
                ef_construction=st.slider('efConstruction', min_value=100.0, max_value=2000.0, value=1000.0, step=10., help="Explain Factor Construction")            
                chunk=st.slider('Token Chunk size', min_value=100.0, max_value=5000.0, value=1000.0, step=100.,help="Token size to chunk documents into Vector DB")

        st.header(":blue[Inference Parameters]")  
        max_len = st.slider('Max Length', min_value=50, max_value=2000, value=300, step=10)
        top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=.01)
        temp = st.slider('Temperature', min_value=0., max_value=1., value=0.01, step=.01)
        
        if page == "rag" and "OpenSearch" in retriever:
            params = {'action_name':'Document Quer', 'endpoint-llm':MODELS_LLM[llm_model_name],'max_len':max_len, 'top_p':top_p, 'temp':temp, 
                      'model_name':llm_model_name, "emb_model":MODELS_EMB[emb_model_name], "rag":retriever,"K":K, "engine":engine, "m":m,
                     "ef_search":ef_search, "ef_construction":ef_construction, "chunk":chunk, "domain":st.session_state['domain'],'Bucket': BUCKET,'Prefix': PREFIX,'Region_Name': REGION}
        elif page == "rag" and "Kendra" in retriever:
            params = {'model_name': llm_model_name, 'endpoint-llm':MODELS_LLM[llm_model_name], "emb_model":emb_model_name, 'endpoint-emb': MODELS_EMB[emb_model_name],'max_len':max_len, 'top_p':top_p, 'temp':temp,'action_name': "Document Query",'Bucket': BUCKET,'Prefix': PREFIX,"rag":retriever,'Region_Name': REGION}
        else:
            params = {'model_name': llm_model_name, 'endpoint-llm':MODELS_LLM[llm_model_name], "emb_model":emb_model_name, 'endpoint-emb': MODELS_EMB[emb_model_name],'max_len':max_len, 'top_p':top_p, 'temp':temp,'action_name': "Document Query",'Bucket': BUCKET,'Prefix': PREFIX,"rag":"",'Region_Name': REGION}
            
            
     return params

#### display function for the first column of the app homepage and info page
#### this function is called from the main.py file
def first_column():
            st.markdown("<p style='text-align:center; color:blue;'><u><b>About Me</b></u></p>",unsafe_allow_html=True)
            st.markdown("<p style='color:#5A5A5A;'>🤝 I am a QnA agent, based on RAG (Retrieval Augmented Generation Architecture) that answers questions by reading assets (like documents, Power point slides, videos & audios) provided by you.</p>",unsafe_allow_html=True)
            st.write(" ")
            st.write(" ")
            st.write(" ")            
            st.markdown("<span style='color:#5A5A5A;'>🤝 I am built on [Streamlit](https://streamlit.io/) using [Amazon Bedrock](https://aws.amazon.com/bedrock/) powered large language models and a diverse set of document loaders developed by [LangChain](https://python.langchain.com/en/latest/index.html). Also [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/) is used as in-memory Vector store.</span>", unsafe_allow_html=True)
            st.write(" ")
            st.write(" ")
            st.markdown("<p style='color:#5A5A5A;'>🤝 Wide ranges of input choices available viz. Documents(.pdf and .txt), web ulrs(single page), YouTube links, Audio files and text from Images and Power Points are enabled. Websites with embedded links and Heavy Spreadsheets are in the upcoming versions.</p>", unsafe_allow_html=True)
            st.write(" ")        
            st.write(" ")


#### display function for the second column of the app homepage and info page
#### this function is called from the main.py file
def second_column():
            st.write(" ")
            st.write(" ")
            st.write("")
            st.markdown("<span style='color:#5A5A5A;'>👉🏽 You can then choose the asset you want to chat on. From the radio buttons on the sidebar. Presently you can select 📜 documents or 🔗 links to webpages, YouTube videos, images basis your choice.",unsafe_allow_html=True)
            st.write(" ")
            st.markdown("<span style='color:#5A5A5A;'>👉🏽 LOKI is ready ✌. You can ask your question. Also, explore summary tab to generate document summary, extract talking points and look at sample questions.",unsafe_allow_html=True)
            

#### display function for the third column of the app homepage and info page
#### this function is called from the main.py file
def third_column():
            st.markdown("<p style='text-align:center;color:blue;'><u><b>Roadmap & Suggestions</b></u></p>",unsafe_allow_html=True)
            st.markdown("<p style='color:#5A5A5A;'>🟢 Spreadsheets and Codes as inputs. Ability to handle multiple inputs, complete websites, content repositories etc.</p>",unsafe_allow_html=True)
            st.write(" ")
            st.write(" ")
            st.markdown("<p style='color:#5A5A5A;'>🟢 Analysis of spreadsheets with charts📊 and insights✍. Analysis of other forms of dataframes/datasets.",unsafe_allow_html=True)
            st.markdown("<p style='color:#5A5A5A;'>🟢 Use models and embeddings that are 3rd party and hosted in Amazon Sagemaker.",unsafe_allow_html=True)
            st.markdown("<p style='color:#5A5A5A;'>🟢 Configurable options for other Vector store like Pinecode, Weaviate, ChromaDB, etc.",unsafe_allow_html=True)
            st.markdown("<p style='color:#5A5A5A;'>🟢 Configurable options for Prompts for the user to select and drive the LLM Response accordingly.",unsafe_allow_html=True)
            st.write(" ")
            st.write(" ")
            #st.markdown("<span style='color:#5A5A5A;'>🎯 Please leave your suggestions, issues, features requests, etc. by filling out [this form](https://forms.gle/uxfHYVhUNtGus8J97). <b>You may be surprised with a ☕🍔🍺reward💸!! 😀😀😀</b><span>",unsafe_allow_html=True)
            #st.markdown("<span style='color:#5A5A5A;'>🎯 I am under regular development. You can also view my source code and contribute [here](https://github.com/abhinav-kimothi/VIDIA.I).</span>", unsafe_allow_html=True)

#### display function for the header display
def heads():
    
    st.markdown("<h3 style='text-align:center;'>👋🏽 Hey There! I am <span style='color:#4B91F1'>LOKI</span>!⚡</h3>",unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center;'>I answer questions after reading documents, webpages, images with text, YouTube videos, audio files and spreadsheets.</span></p>
    """,unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:center;'> ✨ You can ask me anything ✨</h6>",unsafe_allow_html=True)
    

#### display function for the contact info
def contact():

    st.markdown("LinkedIn :[Anand Mandilwar](https://www.linkedin.com/in/anand-mandilwar/)")
    st.markdown("Made by :[Anand Mandilwar](https://anandmandilwar.com)")
    st.markdown("[Reference Architercture](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=RAG_Architecture_Pattern.drawio#R7R1pl6K49tfMOfM%2BdB325SOoICouCKh8eYcdlE12%2BfUvsbTKraZreqq6582UXa0QkhDufpN7w294L27F3MwCOXXc6DcMcdrf8P5vGIYSOAF%2BYMnhVMJQ6HOJn4fOqey1YBl27qkQOZVWoeMWVxXLNI3KMLsutNMkce3yqszM87S5rual0fVdM9N37wqWthndl65CpwyeSxkSeS0fuqEfnO%2BMIqcrsXmufCooAtNJm4sifPAb3svTtHw%2BitueG0HoneHy3E544%2BrLwHI3Kd%2FTIA92Xr5btwymZFXXtrw%2FcL9Rz73UZlSdHvg02PJwhoCfp1V2f7PT%2FWs3L932ESpM69zD69MCOnHT2C3zA6h3akWf4HOiEOJEMc0rtGmafC4LLiCN4ad25gnD%2FkvPr0AAByc4%2FAmYYPT3gQJgkjgu7AX9DeebICzdZWba8GoDOAGUBWUcnS47ZhG81L2H5B8i5ha8b4LxG0U9EYAwafT0jV%2BB9RvKPBHI5Qe7gzNDoNddPIA6Abr5JLh%2FNNg%2FDdIE8XQJJYy9hjT5xGLsPXTfwM8ldFGaeELRc9sPBzD6fQADIZXBwzA%2BysVLcEIAhUAwclHoJ6CsTLOL0olpudE8LcIyTOFVKy3LNAYVIniBN%2B2df0ReL43S%2FHgv3Dt%2BQJXjzbgie5bfCCgxzyde2EJ086fx9IOyhIKfg4DABNtJ0KcQyBsvBGSRP9ngjpjgmKUJfmB5AX4tIGGLElDHtzqNvgGoCyhAueCFkVseMvdb5njfUIx5yhL%2FwynmAv2A%2FzASZynm%2FE0%2FwP8bdT5exL2LEp7VFXIpwZBrmgBwTXfuGadJmkCaAZCNLtBMECx5rBzdkIgNoOzm76Ghx6Rnns5e%2BkmrMgoTMJyzKQBHe6bouPWhifJUu64ZP5mVE6b%2FhTTwmYICZZ9QmsZYkkUZnKJxgrqSFDj%2BRKMsipMMghEUOLiTGSj9RLP4c1MWZ2jmgUSmnliaxBGGIlB4gH4SyeDfJ5nv4vG70uSZns6GFvpuJF%2FR57vIwGwK3Hk6i7n30DF9%2FDwSTp9HP0DZogiGEAiDUjhGkdc6naKfCJzBaJRCCBYe3NMP0EVAqVAI%2FvyNPVA6GP2EMwhLYufvzxI6xDuEzs4t7eAtJZClISSCQQ3gXJwl0A%2BLqU8XR7ckFxe26T6lZQA0VZY2bn58nntC6w%2FIPkZ8Kl0BmUHhyCtpXZEVihJPFEEiFEpjDM6SCHlPV%2FgTxSIETaGn70fGzBt1PpyuyD9JVw%2Flg5cm5QUOwD8BDoEHqHNC9%2FXaY9kw6NPQ8XuD0q7I8qOo64K84dhPnjOKnc9PD%2F9nGOlWQBLAoEpKE8Aq%2F2%2Fu%2BiF4uMN%2FzyLz84gT6LxL2rwhTuCjUBiC0%2Bc%2F5u9NnNQjqUdF5QlNV2RK7av0fOFbcUQoMHQRlMnaI8DP18GRD3%2F7qV3F7rETKfHd4khJGDLPU9stivNtoK17vNNzozveAHgp%2F6RVdyq6o8lb0o1DxzlaWI88tVdfDvlEcsKBCXbpdtFX1ASAe0c%2BOIY%2BIa%2FOHUPdU8%2BnTT2gzB21aAUALYb0j04NwHNWlf8yFH4j8CecougXJGLXEoFB7nD4Dl%2F783CI%2FHWOR7%2FP8ScuD6HX%2Bty7lZ%2BroU%2BgwiQFmuvRVQxeXWZRWJYPr%2BPwei%2Bokt3F5e9JkWtS%2BEUzM8DZ%2BiNuJxgg9O%2FVBSAv9pJYHkx7UR9ALASx3A6bFPWFXsJypaYUbP4u2%2BVmSuZHZ1wc1zOrIy7fsDAezrlcTcucZ2GOMyw493yKCceZk16o8zOlQcain3LgM11qwUDzwVF%2FAc%2B7HreBJxtvWargYJxhUX%2BB8qMFIvvacFQbcVQYoOpSivXIdxYaCumBpanlIADF%2FNgc4As15TNJXXBhIPHoYLTc8IuF3wwEtTpU0ihfaIsByuuOoAvNZNMrhEW5Evga4%2FWdhg5X7UaeHfrttkcnZFlvpx2cDcKnruv9hvF0F%2BF0hpDccGGuD%2FlUETSuCJdkkQbcGjVsjxtPUM5OB9Kgx0%2F62x4%2B3iXzfqPJfGHNKbq2iulhzjMNZmzgIofw%2FIc6qrDJDjzHSzPZGjVBywuFOSzmwkAarSOu6u1lC9RrcaHL5BE4GqwETBxttm23EmZDdwXGttEIl7SZrawPlDXHMXqwMtrKm2i%2BkFZZ6EVLR2rXSqp1PWzXbIVBehgCVSKsx4ciXW3yZr1pikJJ87boE4Ap%2BCVD7deggqoKRsrUcdYLqNKwVcmrJNtcGs9jr1lyOIdTaMV2Em3warPZzFYjvxo3ohHRclcZ4nI4FsLlvIcWW3nd23m%2BombyUNmAe7SS5FqybPRqwAa8tInxuJxyrRoMhgy%2FoCrrkMqKBa8N3FmxbOhFLqzDtJuIqxkotIat52om0mRMTIQJwrCzcFPYhNo2nbVpOzkc9dt4RvRyVh7uh9L6sBixPO5NEV5xUWsGHo9PdaZVchu09IMlyjBeJyA7giBT0t7kdtCTXtAkbszBiplOnXVh2IiMNsqESWV9HnRMKOkmPmt7XQx6nEZtC0QjQ8y5lg8JwvenA78%2F3yyUlsvjwKUTTsHFLvNGO2K6xNatLbgRsRV2TVCM6IFN8webkyyPqgLd5eWW8TdK5863ddU4gEzBHQIaGzkuhADZH3NZKHH5ut7OM7nkpLbny8bK1w%2BrKbie9%2Fs4Iy04TdxLMkZJcruSuGmUMRNX4uYQrnITDAZaK8vFYMCZhjSQyWLUW3EGoO1RvllMtSayNv28DIbtgRtt%2FXEGmnH2GvCSzdXVovWQJpE3%2FtBugh4n9czBVh0sLFnj5cGqHnCrwXad%2BpBo02wgzQVpkI3HS54ji4A3ttnCzzQ%2FKnwsU9akuA17yY4rJR6X1pRmUTvHk0btcohEPKpzDLo4uFtrtOwHe16hBg0iGFif3M8EStwj3B7lWWo8pXoJMkjSCa2NvZSQiaaXBNOhKM74whYZTwjcwdLu7QhxRbR5EKy3%2B5pH3alpixgh0n6Fc7WtdI3SMcuO6RtEH28W8yCtGxk8BD9hzLqJPCCyMV62mJXXADgLfXjKDRm9GTvpoGpGua%2B60oJYhFkw2YuSNqA3SzWQZ4OKUFw%2FqnvyMjSygU1JOYDdRgUwnIS9SoqJeBrw88G2kua2wg%2FSeKdaUuwak95y2C4n%2B1gqpRiJ8GXPGeHaEspWTg8lawdAxrmLQZEq9qJHJPRh4I1QJpkftLmmL9IsUPBlEAjLUTRfaXNjsYqD9XiNaKmZ7iOl24c2GmbZbK2ZZTo11A6VPG3KpOPlzhhbu0zV7eX4gOr7qG%2FmuOLu9TqTEGxq7NZbJBhldpntsZA2l07mYEFBJStSZKlBggzn2R4AJ0bZ6RxRwHgX88OujVbYyi0ingzXY93ezw%2BOu1r3ubwfyULQiZOdzKQG30Zrcclq414Wj6NGr4ssjJZ7XTrsnXE626uCTqNqX59MdX2flNqE2mJ7Y6h1U5VBN8o%2BWOUWqlE5q%2FKtKspxdIj8SexVmxEZsftURW1cd908o8I1HCKrWujCRtD%2B0p6KEkLGkWYqnUpNJwPTyDJmu%2BimjpMOyx6Te1Hml8Sg81UlGGkqFTp2AsnDNanxaksa0h4N84VbrVBgRkQ5XG1wUgubVUhMb2zK5ZE8cXeUja8watTprioonjkZKvJwQKS74aikVWW9Kp3eFBEgx1cojRUzKkzy1k1H7NrKho5JE1DyZzk193LIlbFHtlCkE%2F0sFde%2BE%2FrdrDdo1NmKT%2Bv%2BIVb0QqYa3m%2FR%2FkaWyY1l9afFNCA6eWuOp720Mbc8YWoEbrRhIhq2Wba%2BHPOhoaqKaaKjxJap1abY4arad80yO0q7TsbDTrc2jrFDY7cybKLrOreWcdZfWF5TZsFWW01bhBylB8uSi2lbuDYyLtdjbJnP9kWe7bf6nl5JVTlFS1ztjWQ7K3axvuj2zkw0ill7sGZZAZ8PQ63FtFxijWWN2JkT7exisnOMuTUCwOdRm7Ym5bQMHJZ1O5Xe1EUPKhmPQn1y1W%2FGwzFhbra4OuTsTbSFPcr5CCuSnSOvsq6eqTM368LKXpHQZqsmIXUQuv3ayxxVP6jrcenM2GRPD%2FvYuDbIKTG2crZYltS0GrKogw%2FnTp%2FegpbEarjduWLbUtM6rHcaYiwcDU%2BZrdqRSVkwFd1u1rvIneMZ6iolxdBoXkP7mRbnheStLWo%2B0zd2KZFut6s9F2UMixp6DtvhVjypeh5rDyneK6AuoD0S0K5QDsFx5FFzLm16mj%2Ffc%2F6mlyzkNT8iZMlfCvwYmVCVmjGLJbMqSa4Q6yatuE6ylovDqD%2FeyW6g78UDMjIzaqr6e7dK5lkQz3bIIV%2Fu5CYJAooBlbJq2DWQ1g01a916i%2ByraC5VwKTYmYiiSlsD8apw3axJpUJHMC6ALwcrZD5J0SAQVVEAUrOjVxN6g5RhWdnlIawGdTurhyGoOtsMtN1Ibgog3RWIVLmzZGqAImrYjus%2Brnozk4kVJ6bsbpwM9WIay5gHDRB%2BBIw9azp2MyinGXlITL2uZ0TowZn3%2Bn5vuxknfskG4rrnZWKyF1RkKmtdpi6jCat1zhrde2SGr5LE7HXCet2DZkekUFq9XbOeZmzaPSVbZNzGC9Qgx4ddDmT4VItFcnDAvNpkaI7KSkuRsO166NmUOFvMDoNpMhv2lMTYkV2%2BzVeCgK1WDHhQYYd7ievqOrnGNmI3x4k%2BkAIGEEzgro7V6Qyw%2BwUMni3m5By6DtyE50pTaNNqPTEbtxUSxLKEw2yya010sh3Z2qSKHDEPELab7Iu0jfQmL3q7dIKH1HztkMu94BLlpJ9UTL%2Bm0r0WYoqQLAtGjzt2WEMJCP7PayzaEPmUwplV2BR1FIQ6OnKZJNY307UA6QwKmf18GwAu4fca6qL5emoUK620x2hEuySOVyUW7fdrMlNxIF7JeiLOdMFwWVzl6YbQD4AqhHEpUjsLUAk93kl5vHUNS9jT1HoarapVOS5h50GcO1S21unOnkyhxewR%2BSzy9rWLZuMywQkNmudBAuVgD2uDfRfXk7KdoSQG4VeE00RhgzmdsUCWbMuJk9DGyEn0Ui1DloxMlM3NuZXRaLkMHG8MV31zbOTC35Jh6%2FVaZNWaQvmCEnbQXDqsaegQYDY7mZcNtJ%2BGos8e%2BuTGY7aINtcDshqGyl5pvWrKaShQm%2FiUsRlXIXqzcqIQZYCfALharzV7ttVtG9nuEVLdoq4776zQa1T%2F4EYj8LS8vdziOtLtadajs2oq4x0CbiogOIGRs460m2Gi5k6LAUXCdrBTVprQwVJeY4prTfLINvOu8OZJI7h2HrAl0MxVXUXAs6kMLDf17bLwWoql1yqLNxU%2BrEYV9FEzvVatZG1vc5LNulkLJWU1Ywa0h9sHOodSJ8KViqqhNucJbdguyNWQ3LDIkF0D%2ByjKF6QzQWmhgkRFT7YH2YOPzShsvwNtJq6LdzRwLaBmmx8yxpyP8dIGZz6NBbBT6G3hxlZ02Y5azwP0iGeTpqAvqU9C0h1C36NPAkdcGHH4oF%2BP2Ti3KHZdO1BfHkdcImy9arw%2BD6wOHiIP9zF2AlpQ0BxdDXFojVjDcgcxkpDErPNQGTJ917beaHJg3UXePDc11jg9AnwHyWxWchxhmYN03MxKh5srBDfboNxsLyM6SbWjPuwWiq4B0Ydemzlh2E1IcFhX8Gmw4QKB4DFgh5XWeDMPZUUaHeAtVIRz1oq2nYhQAlDKcEgbC7HhMlucQVqu%2BdqaKM%2BuCFJDQJBtsJpq0B%2FcQnnc07GCYw8jV8eG0J3WAcMuV8YU%2BMJ7eN4DHnKA7TgennAD%2BGVyz5%2F%2BLt1wfAOOeAl%2BDVJOuqyHpJx8ed49HwrqAolG8FDDpgdjpVc2FgAEKEmdwNKhZurKxhGCqdrPthaGVJuVUm9irXJFtDndm3H7GWGt%2BdJYK4E0DEpLJLtZ4rNSLOwsbBTNYnAeo0C0DWop5IlZEpT2UMnny9FG1Tl%2FgbEHY8lhky1XydBf43sh50v99mCIG0rbKeJrr6OdsUVCc6ggdj%2BtJ7iDOwcSlw9kbcd2Las7crZkGjlkDnKIwvaljUeVIwrEZEV20uGl997rHRRR7zY4sICHSmZhhG%2FHSjyLhdACrDtfSuFL%2FfP%2F82hinTBXG%2FCcAeIMOQoQXGUfyMAQ2dBYkhBetbEOMiNkWvBstbQlxpf9zMVje0oTI8JYtYYiRqUBRmmu2Gq%2B1dp5j41cMYpnEZ%2B5og7kMxjfEUdRbf1xX5ubvg53fQ2%2B35cOJLm9m9bWCo2sZAEgd0Ul%2Feb56baZbmP6AWCxhD06%2FdfewF0fQfgFw019hHU0GihgDKD2M22IR1qBtVITOskOAynUXGoKr4u%2BQOtZbQmKMJizpXuUM0KIlsjCHK%2FGu4W8WEyWfglMxsVusa%2FihT8SFV5WB%2BpysxHHUrz328LLhcnBGPf7gqLOeU8fT0K%2BXfRCYRwJSmki6mblW2ZQ9jEkWk2y0q0ZOhFwT%2B9qlU7wvGLlEY6xLl13RuXhRmJgBkbpwk7SNpV91LGcFRaSllIVWy9wXI%2F0kGJrSxOA6Etl2k4cSqx4jNMygYxEheMFztrUYh3Pub1C%2Bd6hd9gJPjvPmLm6wNC4P8f1eEnlxFYY5kMvWQhy3ADtzgmbWWQHs25mO7rCea7LZywNBfNYckU%2BnXlYyKywMFgRe57WmL0qF52Jyb1pr4OWnceuN3Ix2xqWugAuzFQujIiumeGsH6GLfS40okGk6Y6x08VwVQyTGl2k3U5oBD6nu8RvNWjimYrADRbKAMWzvGSMergRK3R%2BNNAXDhC1jojDGSmx3gnc1vT8pJtlnD0rDHs6IiYzb8slIc8s6yHeeIs4ih0o4x1Cx8SkpjIw0sqpeZw8FK5qz6p1iq%2B6eLAdD%2FmMVn1r6%2BCk52DTGfTfOFYZ2f2hNeuzYruOoxmykpg6Ho17Qq%2Fgyc6lGcoKN2k0cg30MIvGwMb3ZLEbNmQznvMzLhGpqLcoSFyb8M4yUBQrYtd4QWsT3XcGkaxaFJDSUG%2FNhtBw101sxdlZy2FlpVFBjapbbbCn3TAaC52%2BOoTqobCXFE%2F3HbvIE0He9WWpnEezRIIKc5YlzIErOEWZIyYYFyjix3BGwjG8lW2kk0EahaIfQv92RY%2BkYKkf2m5RT%2FKRgSx6bVHPxWaxX8tTfT%2FTp%2FZ4PhnpoBdhWhMdBdUNshtGo6JEw1gj5j2nzUgxFXr7GbSy7bTj9fke4aJ5iy7ltnfo%2BD6eaAOs5wixugQWzEEPh8luZXhDohH6VAYVYy9eFO5sdQBOebLbMhq5kIFL6U5SkxKqRdbXqnrbekI68EbOGNymk2fTKOataG%2FGOcNBC8HUfS5LowZjdYZjAwCpoTts8v3Y7gtaHNOiMhzVO8Yay%2BmaPjQHOFtIMGZrd03GWC4BQT8pAH%2BZ9FzEtMXi4ASIiNqDEbXhe9J%2BlUlVm%2FOFH%2Fk9cZCrC2OcKJ43sycc0OK6wWhlxLvqbhfFGI5zrhYp23G%2FaXpQExOSJzSTec%2Bnw0hfEj7FzcJ%2BRAJC42h0WWiYpuSHRBmsjIRkyb7XT5PFEebcvKZ9l3LjVaKNE8Ym2so1JxNzzppcMTW3zAgtQtJYWBs%2BTdepn1RW29sPfWmjikGyjMLd2mlqtCXEeTbE15OGW2O7sJpF5iLKczvbzNXMtaYyvwTwWIcNdOClkSbFlUIi1hYlFD9tdM%2Fi5hrmRwjbDFCTXEBzpi8vGMnl%2BmUSu2OdqaB5UwWePukbXDGnZzuzWRv8cZ7voMNZqXwrbpshYKdyIk7TfimytBL3EdSYiCrumILlFJy8yKEfoMw8lNIR0CX0ALmoRXcrtNcsWKCLBgQHHPJyu%2FTmgbQWGSivhzJtlNqSchsT3xHrwSQH5jG8cbJKjuaakDfOzEJTm9s6h9VqnsS7UMtNFGrvBCIo04Gc30OimBgHPTeHtVipUt7f1HFHExKgZLXW%2B8CxY%2Bkgn3pUPbZ8rx1JDenqMrYeQiMSTgynw1IcC5wvtraxO5hEf69NzLWt2hW%2FpyeFOIV3jM1DNj%2BYtLPfJBnQavi%2B0Uq2IrZTdhlAC9hQEX%2Ff1MuB0d8zrbThZ0o9lLqJ1WeC2cqy0JyXid0i63w7WxKSrveSaClx0ULs8dAFGy%2FGyKEdoQxBbW1mzkFT295Ze6Psx3y2g46U49kQdAPWj5QNBDox5Hbr%2FbQX%2B7uG4jdbCfrQiThOJ6MkT%2FqughbQ0eFGB8GZIUN0mUrPNl8kqLtltYh7vZdVugcB%2Fu9buHtzlY5inqjLRTryepGOfGJQ9uJzt2BHo0%2F4g%2BBZ%2BrPW6Njvr9G5icPBZBO4kh6ZRRHa1wue16ujbhuW69MVeLyB5U%2Fk6azfXlTrH84nCXiU9eXJRSt4%2BtrseHZu9zxU17nLcnnf2muRVrnt%2FkG9EyJKM%2Ffd8nsLnX8YIH1OqbnE6bksdyOzDOvrZ3iE6NMd5sfIutcgEPw6RP%2Bl33MXz495avVKLvcdMTcdITcdPcPhrqMj3b089o%2BTIvoohP9PxhYgb8QWcLHZHeOHeNfJU3t3HxjwuxqWJqwxiC3XgcEF%2F3lveMBVAN4x%2FOwIMJIHfwCEvef%2FJKjagyVPcILnrvBRGX1fiN5XAz%2FoozvcFj4qo%2B8L0ftq8Ow86uvCR2U0eT%2Fi29bog9boTWvw9yfjGW8CFxGUY5j%2BXcDQZXbG3y568WGkYu4%2B87Bkw%2FHw4PT56LqWdSLtz1NuOHujzrAnFLtUZ%2FfBtI%2Bi%2Bs9lH67M0Hekfvx9tdnxbO7mIQAGpKOPUHHfVV3o95N7PlN3HcMRr2iKQZ6om3jVdyswYHx9v7dP1mLYI4vqg7SY6CYcsMsRLst%2BPAD2%2FVGRQM6E3SnT9V61QcFqVmVanMTbnwjM%2FCz5RCNPCHJvXr%2BE%2FsPo6nsJ9YC8PyJ%2B8iF14I%2FyQD%2BIOnq5a8JpQWDBuNbRgvkLYdIfHzv7CeoIu1FHD0MgqZ%2BK30%2Fk%2FqVZu5f26T8cuwTLPGHoRU43cY3sG1eaemB7fBLmH0dGfwBnvxUZrQMbMYWB8kvwA2ngd4GTlsv3Oygno9I%2BRMdcYvz7UczWM44n1kvBS8zt7NkkP8v8k%2FgnH4Y%2Bv7WRwtvzJ2cJfVbg1B1aUexBEDPzaXi952j03qiMojAr3uKXS8a7s%2FevkkLPLoJgxmEEIaCaQRqbsB7oEvA8f3JF%2BsRrmXKCAvbI03GO%2Fz4GNxh6rVAfTF9hD1gO%2ByzUnDu%2BQM0De%2F9fgRriJhvgnJn1y1BzP5eD%2F0tRQ7LXEo26z2T%2Buah5x7Y4%2F6hU5dtZIQQ9zQrdTrV0Ve4%2BVceUtL%2BM9XOi1Hk3BPpejxHoE%2F4g%2F%2B6covPxiL%2FPv3vTUvjKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FnKx%2FmAfJy%2FGJ5wu%2BZ2v3fiJ6bXPF5z%2B4oGfWc06F9E%2FSteX1618gD7PzMgjLgPCCP%2BpSEQx023L0P50GtGpagn7H6r458aFUHcL46T%2F1Js4eT1mxwwlv7l6CGxO%2FRQ%2F1L0EOzNizbOWS%2B%2FDDf3b8%2B4T7dD7jPuPkwR%2Fj6BKR2gxsRM%2FOoYJoIc39ZWfKXwXdzhK4Xv%2FTFjf5MUvr8mKMjr9AjynGD7E9LzHguKe2OY%2FrcK8duNAX61EKc%2BwFF561UOCieCy78rYKChC24AfZbKh1u9A7xhCPBj3NyEjArENfINXs1tiHu7rI7ZDmYCq%2FXSOAMOA5TM%2F3JvB71Jcvx2H1FMsg%2BIh%2Fgs4qHf8fI9mMJ6FptpXgapnyZmNHgtvUkreq0zSaGMPgJ565bl4SSMIfyvsfvXtpF4gJKbtNt7hPzF9Ffy5j2W6Nmo%2FbOZrxSF%2FnFHn711wx3%2B2f9fwf6dVLb3uLrXsh2%2Fn4n4LNn%2BGDuPXtpxi51PT4t%2FyFLvAu3PyUXHkJt04fMEwJ9lRuCh%2FXFHn82M73gX3Y%2BhG3li6CuMU9%2FZCQGc3G5o8Il7%2FZzz8r63rQ%2F2YHOEP8iY%2BFUESZz33zk7%2Brf78byXIAnyuqO7HYI%2BmyAfJZ58x7a00txx82%2F2s5Tmjq3z3799uyz%2Fz5EskB%2FIj3%2FNoAatc%2FB9ehHVvnIhRoQsB8Zm%2BYad%2BZ87bvobJFh%2FZ0oPv3Y38LN%2B%2FlV58g%2FyAz8uT941gf8AarykTd%2BnzMNZqL9Edbc3PXcM7I3kcdcvOVKX3ee%2B9TsG%2FXQARuTy4PFdfn95NiBbgCt1hFSYABqOzdML8o70%2FEK3z8O5HuL%2F6bNfsuh%2FvvOIn%2BkZ%2FhSGJchrFw8%2FT4BfMCz7YO7m0xj2QRr8vVFx4eK5kZU2b3t3x8vgOEjzsIPv4Yyu0fGOF6sXQGuV90bMsVgIo9ecU3B%2Bchmpo1XxwPIBhRdNwNlFgwc%2B50Pf9P71zpczh8QfmD70Z%2B0Mhd7bOT%2B6CdQj7UB9kOHN3CwHsj9q51DXWg7F4H5AF6uON0N7w%2BwB5GEeLqqdpn7ettPu4g7I82L0K8c9d%2FqxO029x637YskfYck%2F4EcSe3VV7nybT3SL3%2BPXMO%2F0a543i%2FiF%2FH4XrIEDqN68mv2HuZ5kgF96wfXYp3A9ydw4VSep9bksj%2F3dWP5X8u5DbqN%2FbHNG7Cfq4D9YRf%2BlPEm9tXMegTJP5A%2FOhr3EYb2sQlJPyM%2BdEHu48cUHOZvKqx%2FWA02gXwGRUVZ5Ahe3%2Fq89Li9P40s3%2BrT72D%2FRuTwG%2B1JmDJ25xCqya1%2FyH%2FSgJy96cZromt9OdP0jvWmcvlmRIe9XZFDkZ7rT%2BN%2FOdv8bKPKzsn6fIn%2BwtvCBtvSvVce3M%2FbkbYD6u9eibqxU4paiP8gavp1gJtGfYA3jnziJfApKgcsFQRWbybco3LlHxBZZmhTuP0Y2YjR1642RyL14%2FLm76N67Ocy9ePxXxBNg7B16iAd5LT81ngC%2FD%2Fi9x86fW2B%2B8cqeaJa59MzQJ4R9ie35Ra4Z9sZGxT8pLuFmepE4p5H8WV1Af6efT%2FbCzhric7ww6HC9LasRaDtDVKe%2Fndd6%2FyHim8aukXoOxr9cJ3qwXfLnie5Hu4t%2BEJa%2F57v8n2Py%2BzuGoujHoBLmtaYQtK%2F8DWPQjzkToPB%2F)")


#### function to clear the cache and initialize the chat
def clear(greeting=greeting):
    with st.spinner("Clearing all history..."):
        st.cache_data.clear()
        if 'history' in st.session_state:
            del st.session_state['history']
        if 'pastinp' in st.session_state:
            del st.session_state['pastinp']
        if 'pastresp' in st.session_state:
            del st.session_state['pastresp']
        if 'summary_flag' in st.session_state:
            del st.session_state.summary_flag
        if 'summary_content' in st.session_state:
             del st.session_state.summary_content

        initialize_chat(greeting)


#### function to clear the cache and initialize the chat
def clear_new():
    with st.spinner("Clearing all history..."):
        st.cache_data.clear()
        if 'generated' in st.session_state:
            del st.session_state['generated']
        if 'past' in st.session_state:
            del st.session_state['past']
        if 'messages' in st.session_state:
            del st.session_state['messages']
        if 'domain' not in st.session_state:
            st.session_state['domain'] = 1

        initialize_chat(greeting)        

### This function for writing chat history into a string variable called hst
### This function is called from the main.py file
### This function is called when the user clicks on the download button
def write_history_to_a_file():
    hst=""
    st.session_state['history']=[]
    st.session_state['history'].append("LOKI says -")
    st.session_state['history'].append(st.session_state['pastresp'][0])
    for i in range(1,len(st.session_state['pastresp'])):
        st.session_state['history'].append("Your Query - ")
        st.session_state['history'].append(st.session_state['pastinp'][i-1])
        st.session_state['history'].append("LOKI's response - ")
        st.session_state['history'].append(st.session_state['pastresp'][i])

    for item in st.session_state['history']:
        hst+="\n"+str(item)
    
    return hst
