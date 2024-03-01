'''
utils.py consists of all template, input and other utility functions of vidia
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
        {'name':'Llama2-13b', 'endpoint':"meta.llama2-13b-chat-v1"}
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
    st.markdown("[Reference Architercture](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=RAG_Architecture_Pattern.drawio#R5V3rl6I40%2F9r5px9Pkwf7pePIIKoKCKg8uU53EG5yV3%2B%2BjextadbnZne3e6ZZ9%2Ft3VESQhKqKr%2BqSirxCz5Ke6m0i0jJPT%2F5giFe%2FwUXvmAYSuAE%2BII5p0sOzbLPOWEZe5e8bxnrePAvmcglt4k9v3pTsM7zpI6Lt5lunmW%2BW7%2FJs8sy794WC%2FLkbauFHfp3GWvXTu5zN7FXR8%2B5DIl8y5%2F4cRhdW0aRy53Uvha%2BZFSR7eXdqyx8%2FAUflXleP1%2Bl%2FchPIPWudHl%2BTvzO3ZeOlX5Wv%2BeBMjoE5WHbM5hWNEPf8%2BHY%2F0o919LaSXN54Utn69OVAmGZN8WlmF%2FWfv%2BI7rZzLY7c9wt9eVsgJ36e%2BnV5AkUuFbGXJy4ScqVs943a9JWm0StKY%2Fgl075wOHyp%2BRsRwMWFDn%2BCJhj9c6IAmmSeD2tBv%2BB8F8W1vy5sF97twEgAeVGdJpfbnl1FL2W%2FS8nXFPs%2Br75Lxq%2FUWzp%2BRZknAnn9h90TlnpEWAI8%2BEmk%2FWjKfhoxsRtiMk8kyt4T8BH9UJp4Ql8KfzgJ0Z%2BTECBNAS%2Fj9AxurwkGSRUDdOOSOMxAXp0Xr3LntuMnal7FdZzDu05e13kKCiTwBm%2B7h%2FDMnlGe5OW5LTw4%2F4Ei58a4qngGYQTk2NdEEPeQofylP0JU1xC9OUgITHS9DH2KAY4EMWB8%2BeSCFjHRs2sbfMH8Cnw7ACarGvD%2Fa5snX3GQgwIOikGc%2BPWp8L8WXvAVxZinIgs%2FXCZe8RuMKYzEWYq5ftIP%2BP%2BdMh%2BPU%2B%2BShGedg7yGIeStTAC65gf%2FytMsz6DMAMomr9hMECx5LpzciIgLSOqX75Ghx6JnX1Iv9eRNncQZ6M5Vn8PeXiU67UNoZzy1vm%2BnT3bjxfl%2FoQx8IhRgTxTNkDRKYM%2Bf7BtgwPEnGmVRnGQQjKDAxR1EoNQT8u1p8Ik%2FgFzqiaVJHGEoAoUX6CdJDP5zifkpG38KJs%2FidDWW0Hfz%2BI14vksK7K7Cvacryr1HjOnz3yNs%2BizxQZ%2FwN9zHmDfyQ9FPBM5gNEohBAsv7uWHfGKBSqEQ%2FPkTeyQ%2F9BPOICyJXT8%2FC3KId0DOwa%2Fd6HsqoMhjKAPjFtC0uuLPXwapTwejW4lLK9f2n%2FI6AnqqyDu%2FPL%2FPvZwJY1LAiN%2BGSihKPFEEiVAokDecJRHyXqzwJ4pFCJpCL58PxOp7ZT5crMg%2FKVYP0SHIs%2FoVC8B%2FIuwCDzjnxf63e4%2BRYSzQ0HX7jqC9kcqPEq5X0g37fvF9Ueyavrz8nxlHt%2FBIAGsqq21Aq%2FK%2FpR%2FG4OVO%2F70C5mfJJv7EvhZNhiDeCieBPlEYgtPX%2F5n%2FbeGkHoEeldQXNr0RU%2BrY5NcbX6szQ4GVi6BM0Z8Jfr0PrkL4LeRuk%2FrnSuQs9KuzJGGIWuauX1XXZqChe27p%2BaG7sQH4V%2F9Jk%2B6SdSeTt6Kbxp53Nq8eOWLfXDXkM6EOfeuLYcj9DMHLJMJrCfm0CQL253Dle6F%2FHcB5WUd5mGd2Mv6We0O8b2XmOUSLMxf3fl2fLrBgN3X%2ByPeFDX2X2pesKm9K1%2F%2FBC9GPGVD6iV3H7dv6H1Hz8qh6VobfGIe8ZRxKkG%2BrqO0y9OvLUzc8eenG3%2FCU32Hu%2FrP4hCKfwigCvWUU8msZhdwx6hUwXsAwhp79Mx465RUK0SdQYJ4DBf%2FoLgbvroskruuH93F4fxQ12SG%2BThs8nIhCPmgi6sfzoHfDBb3HORR%2FgHPUZ%2BEc%2Bo7Z4V8wgAD9ytP28vw5sYOJJ%2FKaFPrXN4XT65TqlzGgBlRur7TUx4zGi2HwPDp%2BVPBz8JW64ulvGrbYPb6ubfA2GDJOHd%2BDY7J6KC9nw%2Fktj99viJQ%2BsKsuyw3I1RI%2BvzDJfyGFH43F96xV%2FGAgfHfkIk8oQWNvefGc%2Bpsc%2FnoLzG8ryIOg8j8Hke9t3meeAt5g4qj07Rry2QTUzMt7YP3jpcQ3SfjPbwfYOzX3%2BwH2fj3kSlJkDb4gBf8QOXm9%2Fs89ke%2FoefX%2F3FNynknHf05V55kFc%2Bcl42WOf%2FnsY18diotzSn4MKyj6ZsLigU3PEPecYD6NE8wdJ4wKCjsinBcigHtWNPUdyf9BntePGULf%2BFgPnPJHa12f5mKh9z4Wekd9UE9cVN%2Bj0Wvtcjt98naO%2BjrjItppnEAK6HaUpzYsB6oE4MVfZnYE4luedqECdsf0Lxjunf%2F7GNagJHUDW8Qdb7BH67iftj5%2Bb64%2FsAv%2FFbzBqZ%2BrlF%2FLm%2FvlYfxfyhvqHf7Ur%2BXNO9yp%2F1erJzdT7AiCMozwaKJ6aEr%2FqTnr278Pl2%2B5jjP3XCcecJ3Ankjysxh%2Fb1x813i7Ts7%2F1ZAMzw%2Fs5mwpfodJD4My3sRtXMM0ziEYOPecxMRzaMUoNvml1iEzKcw58LdYG9HYCMGVsILpYcTtYGIXrGsdXMwKLBFWKD9dIUpoTKatlSaVBYqu5dRMQm9loJBaLE2txxHI5mf2GF%2FpOV%2FI%2BoqLI5lHx9P1jl%2Btwm4s6s2pkaflyliNUd70RFPs5rtRJa7qjci3GG8eDHSy6XfK8iT0%2BxGdkXW7XwwwXARf%2BH7wBePpIcHpAiG5ycrensqFJhpcFa%2FJKo%2B4LWq5ATebo5ybj%2BXxiJ8L%2BxE%2BO2Sq0BkKXzkqRbdOtTipPNNh1o5CnyNYYDSKp4u74sRzvLxUnGkX9bxY2ZNKFcfydJtwzeioOKBcj4tDoUzB1XgjYtJ0t%2B%2BHjbic%2BBvQt51B%2BKTL7BVzrG05jjGjjdU3wdwIxbwp4iBZe3K%2F1XJjGGGHbi%2BO89MEiJa4nZ2qfLMru%2B2uqyotL%2FtKIIBrwa8Z6rgFBXRdtHKmTYtRRNWWq8tBI7v22nrue8uSExXG2FT7ebLDm91ut9xMw2bWSVZCK0NjSevJTIzX6git9sp2dAhCTS%2BUibYDbfSy7DuKYo1aMA54eZfiab3gej0aTxh%2BRTXOKVc0B94b%2B8tq3dGrUtzG%2BTCXNkuQ6Uz6wDdspCuYlIgzhGGX8a5yCb3vBmfXD0o8Ffp0SYxKVpkcJ%2FL2tJqyPB4sEF7zUWcJXo%2FPTabXShc8GUZrgDHBICIHgiBz0t2VbjSSX9gk7ezxhlksvG1luYiCdtqcyRVTjQYmlk0bX%2FajIQU1LpK%2Bj5qMIVSu52OCCMPFOBTU3UrruTKNfDrjNFwaimB6IBZrbNu7op8Qe%2FHQRdWUHrs0f3I52QmoJjJ9XumZcKcNvrpvm84DYgpaiGhs6vmQAqQw44pY5sptu1cLpebkfhQq1iY0T5sFuF8KAs7IK86QjrKCUbLSb2RukRTM3Jc5FdJV6aLx2OgVpRqPOduSxwpZTUcbzgKyPS13q4XRJc5OKOto0p%2B46T6cFeAxzt2CseRybbPqA6TLlF04cbtoxMkje7zXxytHMXhlvGnH3Ga83%2BYhFNq8GMuqKI%2BL2WzNc2QV8da%2BWIWFESZViBXalpT28Sg7cLXM4%2FKWMhzq4AXytF9PkIRHTY5BVyd%2F70zXQnTkNWrcIaKFCeRxKVLSEeGOKM9SswU1ypBxls9pYxbkhEJ0oyxaTCRpyVeuxARi5I%2FX7uhASBuiL6Nouz%2B2POovbFfCCIkOG5xrXW3otIFZD4xgEQLerdQobzsFvAQ%2FZ%2By2SwKA2RivOMwm6ACdRQEmuQljdjMvHzfdtAx1X14Rq7iI5kdJNsb0bq1HynLcEJofJu1IWcdWMXYpuQS02%2BmAhvN41MgpkS4iXh3vG1l1NX6cpwfdkVPfmo%2FWk349P6ZyLadIgq9H3hQ31hBbOTOWnQMgGeevxlWuuasRkdGncTBFmUw9GaphrvIi0vB1FInraaJuDNVabdJoO9siRm7nx0QbjrGLxkWx3Bp2nS8sfUDlwFgw%2BWx9sGbOodBNdz07oeYxEewS1%2Fyj2RYygi2sw3aPRNPCrYsjFtP22is8LKqobENKLDXOkIlaHAFxUpRdqIgG%2BrtST4c%2B2WAbv0p4Mt7OTPeonjx%2FsxW4UkgUMRqk%2BUFhcovvk620Zo3ZqEhnSWe2VREn66Mpn47eLF8eddGkUV0w5wvTPGa1Maf22NGaGMNCZ9Cddow2pYMaVMnqfK9LSpqcknCeBs1uSibsMddRFzd9vyyoeAu7yOoOunIRVFi7C0lGyDQxbG3QqcV8bFtFwexXw8Lz8kk9YsogKcKaGA%2BhrkVTQ6diz82gePg2NdvsSUs%2BonG58psNCkyHpIThiF7uYMsGSemdS%2Fk8Umb%2BgXLxDUZNB9PXRS2w5xNNmYyJ%2FDCZ1rSubTe1N1ogIhzxDUpj1ZKKs7L38ym7dYqJZ9MERP6ipNSghKMyDcgeQjohFLm0Db04HJajcacvN3zeCqdUMyuF6viwR4WdopA7xxEW1SIiBmVvzxajvLP3PGEbBG71cSZZrl33oZLysaXrmm2j08xVqM2uOuC6Lvh2XZzRblDweDCdnWcd0NRvLJcYhsFvFZwNV07Q1UW0NzaLHiGn%2BclxlGrRV76LzOrtDFuXy2NVFse9eaQ3clMv0BrXR1PFLapDaq6Go7eUrGrZn5xlUcH3w1BntajXWOc4U3bpJQe3mh88S3WmgPg86tLOvF7Ukcey%2FqDTu7YaQSUTUGhIboRuNpkR9m6P6xPO3SV7WKNSTrEqO3jKphjapb70iyFu3A0JjbZmHlMncThug8LTzZO%2BndXeks2O9ETAZq1FLoiZU7LVuqYWzYRFPXyiegK9B08Sm8n%2B4Et9Ty3auD0YiLXyDDxn9vpAAuObaeh%2Btz0kvooXqK%2FVFEOjZQtX6mlJreRg61Dq0ty5tUz6w6ENfJSxHGoSeOyAO%2Bm8GQWsO6H4oIK6gA5IILtiPQHXSUCpXN6NjFA9cuFulK2ULT8lFDlci%2FwMmVONXjCrNbOpSa6S2i5vuEF21qvTVJgdFD8yj9IJmdoFtdDDo99kahGlywNyKtcHpcuiiGJAoaKZDB2UdUsver%2FdI8cmUeUGmBQHG9F0eW8hQRNvuy2pNegURv%2Fz9XiDqPMcjSJJl0SAmgO9mdM7pI7rxq1PcTNu%2B2U7iUHR5W5sHKZKVwF01yBTlcFRqDGK6HE%2FawVcD5Y2k2peSrnDLJuY1SJVsAAaIPwUGHvOYuYXEKcZZUIsgmFkJejJU0dCONrvZllYs5G0HQWFlB1FHVkoxlDo62TOGoO3RY8BWeCbLLNHg7jdjqDZkWiU0e63bGBYu%2F5IKQ6Z9ukKtcjZ6VACDF8YqUSOT1jQ2gzNUUXtaDK2304Cl5KWq%2BVpvMiWk5GWWQdyKPflRhSxzYYBLyoe8CDzfdMkt9hOGlScEAAKWACYQKueM5gMcChEDKZWKqlC34Gb81xti33ebOd25%2FdihjiOeFrOD72NzvdT15g3iSeVEcIO82OV94nZldXokM%2FxmFK3Hrk%2Bij5Rz4WsYYSWyo9GjGlitq4YMx3YSQsREPxTWyzZEeWCwplN3FVtEsUmOvWZLDV3i60I5QyCzFHdR2CU8EcD9dFyu7CqjVG7MzShfRLHmxpLjsctWeg4gFeynUtLU7R8Ftd5uiPME5AKcVZL1MEBUkLPDnKZ7n3LEY80tV0km2ZTz2pYeZSWHlVsTXpw5wtoMQdEuUyCY%2BujxazOcMKA5nmUQRwcYX10HNJ2XvdLlMQg%2Fap4kWlspNIFC7BkX8%2B9jLamXmbWeh2zZGKjbGmrTkGj9TryghkMCy%2BxqQ%2B%2Fa4Ztt1uJ1VsK5StKPEBz6bSFIbgi5rJzte6g%2FTSRQvYkkLuA2SOGakZkM4m1o9YHzYIzUKA28QXjMr5GjJb1XCPqCL8QcLPdGu5yb7ousj8ipL5HfV8dnDjo9PDkJ1Pwtry73uMmMhxpNqCLZqHgAwIaFRGcwMjlQLrdJNNLr8eAImEHWCkrz%2BlorWwxzXfmZeLa5VAFataJvltGbA00c9M2CfBsGgsrbXO%2FroKeYumtzuJdg0%2BaaQOd1MJsdSfbuvuSZIth2UOkbJbMmA5w90SXEHUSXGuoFmpznjAm%2FYrcTMgdi0zYLbCPknJFenOUFhsoVPR8f1IC%2BNqMxgoDeGbu%2B%2FhAA9cCajb1VDC2OsNrF6RCGotgpdDbwq295LMDtVUj9Mxnm6agL2nOY9KfQN9DIAnwNeXwsdDO2LR0KHbbelBfnntcI2y76QKBB1YHD5mHhxg7B09Q0BzdTHBojTiT%2BgA5kpHEcghQBQ564HsH0%2FmJ9Vdl9%2FyotcXpKRh3UMyWNccRjj3OZ92y9jhVI7jlDuWWRwUxSaqfCrBaCF1jQoBemz1n2F1McNhQ8Xm04yKR4DFgh9XObKfGiiZPT7AJHeG8rWbs5xJEAEqbTGhrJXVc4UpLKMst3zpz7dkVQeBKqUj20WZhQH9wD%2FF4ZGIVx56mvolNoDttggG73lgL4AsfYXoEPOQIO3A8THBj%2BGFzz3%2FCId9xfAeueBl%2BjHNOfl0OyTnldXp4vhT1FZJM4aWBLU7WxmxcLAIM0LI2g7kTwza1nSdGC10o9g6GNLuN1u5So%2FEltLu0zfhCQThbvra2WiRPotqRyGGZhaycigcHmybLFKRTFEDbuJVjnlhmUe1OtFJdT3e6yYUrjD1Zaw6b77lGgf4aP4q5UBb6kyXtKOOgSd9qnR6sPRLbEw1xhbydwxnDE4krJ7J1U7dV9AO5XDOdEjMnJUbh87WLJ40nicR8Qw7y6aX20bcWNMkcdjiwgCda4WBE6KZaukzF2AFDV13L8Uv5679rb1KTsDc78J4R4k04Cghc457IyJLY2FqTkF6ttY0KK2Z68G6tvCdmr%2BtRpfPzlCElhLXpLU1Kagv00t6wjbo3enXEJr6UpMuEL3zJBPgM%2BnfmUdI6P65rd1PX6a6u8c%2FrMgGSu4dF62zQxMlWgHJvpETont9uX5guZp4AF2tYoyd8qw20%2BojCLxzu2jOtk%2BlYA30ApZ9lQzrLCiyV29BJ9hgoofba0HhTCkXaLFpH1MSxysJlZoAzYozWyMqebWaHlbJazddhDUzG1WF1bNJVOJU0XtHH%2Bnq3k2Zyegz7KijF%2BcmaCYKo6SofmLN5zPerUSzOElGrbUTfbULHjmoBQ5LNvKj9lqEzEQ%2FModXpDC8bVpniGOvT7WA1AW5lFmZhlCkeZGPXuGcdyzlxJRs51bDtCsfNxIwptnUMEUBfrtBu5lFSw2OcUYhkImkcL3LOrpXaVOWOGhUGp9HpIIasWjCqvsLQVFBxM11TJbEXJ%2BUkyFaiknZAu3Pibpm40XJYup6pcYHv8wVLQ2Ceyb7E58sAi5kNFkcb4sjTBnPUlWqwMWW0GA3QsgvY7U6plnvL0VfAhVkolZXQLTNZCgm6OpZiJ1lEnh8YN19NNtUka9FVPhzETuRLesjC3oAmnq2J3HiljVG8KGvGaic7qUHVs4G%2B8gDUehIOZ6Sk9iByezsIs2FZcO6ystzFlJgvgz2XxTyzbid4F6zSJPUgxnuEiUlZSxWgp43X8jh5qnzdXTbbHN8M6Xg%2Fm%2FAFrYfO3sPJwMMWS%2Bi%2Fcaw2dYWJsxRYqd%2BmyRLZyEybTmcjcVTx5ODTDOXEuzyZ%2BhZ6WiYzYOMHijRMOrKbqfySyyQqGa0qEjfmvLeONM1J2C1e0cbcDL1xougOBVAa6q3lBBrupo1tOLfoOaxuDCpqUX1vjI%2B0HyczcTA3p1g%2FVe6a4mnBc6syE5WDoMi1miwzGSrMZZExJ67iNE1FbNAvkMXP4IyEZwUb18rn4zyJpTCG%2Fu2GnsrR2jz1w6qdl1MLWY36qlWlbnXcKgvzuDQX7kydT01Qi7hoiYGC6gY5TJJpVaNxahDqyOsLUsrF0XEJrWw3H3hTPSJcovboWulHp4EX8MwYYyNPTPU1sGBOZjzJDhsrmBCdKFAFVIyjdFX5y80JOOXZYc8Y5EoBLqU%2Fz21KbFaFYDTtvg%2FEfBxMvRloZlCWiyTlneRopyXDQQvBNkOuyJMOY02GYyNAqYk%2F6crjzBVEI01pSZtM2wPjzJR8S5%2B6E5wtJBi7d4euYByfgKSfV2B82bQqYcZqdfIiRELd8ZTa8SP5uCnkpi%2F5KkzCkTQu9ZU1y7QgWLpzDmhx02KMOuF9%2FXBIUgzHOd9ItP1M6LoR1MSEHIjdXB2FdJyYayKkuGUsJCQQNI5G15WBGVp5yrTxxspIlhQCIc9WZ5pzakuHPuWnm8yYZYxL9I1vz%2Be2ytpctbD3zBStYtJaOTs%2Bz7d5mDVOPzpOQnmnS1G2TuLD1utatCcktZjg23nHbbFD3CwTe5WUpVvsVL3wnYXCrwE9tnEHHXh5ashpo5GIs0cJLcw7M3A41cDCBGG7MWqTK2jOCMqKkX1OqLPUn5lMA82bJgrMuWBxlUovD3a3tfjzPN%2FJhLNS5V7adxMwnOq5tMiFWmJpLRUQ1JpLOu7ZouNVnLKCEUEzbRmglImAKqEHyCU9etigo27FAl00JjjgkNf7daBG8lZiIF5PFNqqjTXldzZ%2BILbjeQnMY9hwtsnO5ppYdt7SQXOX23unzUbN0kNslDYKtXcGGVSYAOePUCjm1sks7UkrNbpcCrs2HWhCBpKst6YAHDuWjspFQLUzJwz6qdyRvqlg2wk0IuHEcD6ppZnIhVLvWoeTTQhHY25vXd1t%2BCM9r6QFbDG1T4V6smnvuMsKoNXwY2fUbEPsF%2Bw6ghawpSPhsWvXY0s4Mr2845daO5GHuSMw0XLjOGjJK8RhVQyhW6wJ2TRHWbKWuWQljXjogs1WM%2BTUT1GGoPYuo3LQ1HYPztGqhZQvDtCR8gIXkm7Mhom2g0QnJtxhe1yM0vDQUfxuL0MfOpNm%2BXyalZnga2gFHR1uehK9JTJB17n8bPMlon5YN6t0NPqgyFWMfbouaV2W3QjiftmNhnvh7lfe6E9bb30Up%2F8nN24g39m4IfkZBwQD4Yrir%2B%2FS%2BMigRxg2ew3S%2BhNhSB%2FEfQp9w33yQTgX%2FSvDh%2FA%2FuUXDT5y8ex20fM4AN6K8jAe4hSo5BxR7HDxSBLIssasqdt%2Fy023K9mX1%2FW2IFnj0Et%2FMnLljl%2FWrtN%2FHNQxwRp%2FOAXYw%2BRziTAMZfk5%2Fi3GGidOrxG2E818Nl%2F7yzsjoH26QJx9w%2BZr3N4Nhb8KgCIR8QlCGJdjL39v6nqO372Kf7zepnLHr1SkZ%2BNtWqJt6vxNTDeTCPr0qdhmg330Z9CZw6HpQzjehf67xQ6N6ifvArT8LiOx38PASLLlqfMhCUS3ztKjfi4z%2FyJjwZ4j5%2FgkiYMDRJPv3JP%2FzA72J%2B3ixPxXo%2FWEK9UHE%2BBd4XBOUqotwHS%2FCVfxIuH5%2FmDl%2Bc3YM%2FsAU%2BrVh5gT2QdrwOgB%2FhS68KKyrKvyxGnxRed%2B03O7Lq91Df1PlvWMz0DXG66ebgYjfqkTxu9O2boTuvXrzxcC%2FKjDypqIPUpS3o4m4PeXsZ%2BXJN%2BU%2FSbG%2B4wCD1wOs8kO4ufHNEPu08fQXx8bD3XMvhuobK5X84eD8wEF21VZ%2FecfdL7JUb2NQ2b84yCj6JxV90CCjb1UW%2FSsGzX049Nq3Szd6ZW%2B8sQbujI%2BX4oBffmuf3yjOgM2Q2peDFO7th%2FdspPpH2qPPGPQje5RArucCXXf5%2FD1p%2F9WbEon73XLEPe%2F%2BFXsbMPbtmSo4S72Er%2F%2Bu7Q3EfZQ7%2BS9lD8nc4Cl%2B7wL8Ut6Q9x4A9S%2FlDY28BUH8OnP023hzv42eS%2B3hrL943ytz9%2FAFNn6vtj7I9f5jDq0IUGJuZ2Fz3uyBnE9yhh643dpxctZGkEJ1FMNu3fbvP%2B%2Bd6XmzvemtVuTBK46e%2FwENiY1gzhNEkLvMR3n0fSZ6Xwx8oY9auM18lEffZ6L3xWDq2uu3mY%2FyaPK%2Bx7dPow%2BeRm%2BePpsUf%2BpQtNutWSh32Zp1O%2FReznf9nzsC7eFxZ8DyOtvWsgv7Aw2x56u3pZzLsPqYYwXIt8dwkCh6Byg0cw8o17yPB5R3nNL4u7zRL%2B90Bn%2BX6%2FYyjXFl5u20xrvXFagbqbjVH598OMt16vmVEND%2FUo2P3mwEJh4cMf9LNf51tfR%2FcYBeZ3iAv4gQbxcjKYT%2B8Szsg8XI9473d8%2Bw%2Fi5guJk3JW5PInkvLuA39dwugX8yLFD3xqbm100JjbmoSe3saxIf%2FDO5qiLPADb8v5gcoX6MEMgTit7MjVzP2f%2B78%2B03p5J%2B3uwI9Z7D2r6hxiX%2B4xUHb1HiZdqYJejXE8fAGGVfMj7j6LXfNcTp67nrf1f30zgCT4V%2F%2BXt75DfKIE8MCwDkJuLgVyHA%2FTSN9m36FHgP5zglSEwIC0AeXjmeD5xMJy89v%2FzqPqtu7lxx%2BcfXr6%2Fz%2F%2FPQ7QxK%2BEMPyO2BV5%2FU2jmokLJTaNhkTlW8cqA%2FvK3vBEP8vwDSn%2FxuCvIE3KG3c5PX36n4p0wzU%2B%2F4kZqfmV%2BPzuf6Xwa%2BD3J6SPwt60nifdFTH8a6D4j4%2FN5R3fqpOIc%2BYIjGSeDzD2A3lTHATZjFNWcLHaIlIvmZX54XoP4DUl%2Fh3dKF7pULABWaVnbmnZE2BQbW%2BSCff3kAKX1jed3%2FBgb1SPhvTzt9h3sGw8pffm%2FtWei%2B%2FWwdPv4%2F)")


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
