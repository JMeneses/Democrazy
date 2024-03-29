# ESCRUTIN.AI - Political program analist
#----------------------------------------
# Exploring Tools: streamlit or chainlit or gradio, langchain, chromaDB, PyPDF

import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import streamlit as st



# Master function
def setupPoliticalRAG(user_initial_query):
    #---------------------------------------------------------
    # LLM (RAG Strategy / Retrieval Chain / Vector Embeddings)
    #---------------------------------------------------------
    OpenAikey = st.secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(openai_api_key=OpenAikey, model_name="gpt-3.5-turbo-1106", temperature=0)
    openai_embeddings = OpenAIEmbeddings(openai_api_key=OpenAikey)
    openai_embeddingsf = embedding_functions.OpenAIEmbeddingFunction(api_key=OpenAikey,model_name="text-embedding-ada-002")

    # 1. Indexing: Load, Split and Chroma VectorStore

    persistent_client = chromadb.PersistentClient() 
    collection_list = persistent_client.list_collections()
    print(collection_list)
    collection = None
    
    if (len(collection_list)==0):
        # 1.1 Load
        loaderVolt = PyPDFLoader("data/volt_programa_legislativas_2024.pdf")
        loaderPCP = PyPDFLoader("data/pcp_programa_legislativas_2024.pdf")
        loaderBE = PyPDFLoader("data/bloco_esquerda_programa_legislativas_2024.pdf")
        loaderLivre = PyPDFLoader("data/livre_programa_legislativas_2024.pdf")
        loaderIL = PyPDFLoader("data/il_programa_legislativas_2024.pdf") 
        loaderChega = PyPDFLoader("data/chega_programa_legislativas_2024.pdf")
        loaderAD = PyPDFLoader("data/ad_programa_legislativas_2024.pdf")
        loaderPS = PyPDFLoader("data/ps_programa_legislativas_2024.pdf")
        #Falta o PAN (erro 404 no website)
        #Falta a ND (erro no PDF, salvo como imagem o loader não retira o texto)
   
        pagesVolt = loaderVolt.load()
        pagesPCP = loaderPCP.load()
        pagesBE = loaderBE.load()
        pagesLivre = loaderLivre.load()
        pagesIL = loaderIL.load()
        pagesChega = loaderChega.load()
        pagesAD = loaderAD.load()
        pagesPS = loaderPS.load()
        
        # 1.2 Split
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        #------------------------
        # VOLT
        splitVolt = text_splitter.split_documents(pagesVolt)
        strVolt = []
        idsVolt = []
        metadataVolt = []
        n = 0
        for doc in splitVolt:
            strVolt.append(doc.page_content)
            idsVolt.append("Volt"+str(n))
            metadataVolt.append({"source": "Programa eleitoral do partido Volt"})
            n += 1
        #------------------------
        # PCP
        splitPCP = text_splitter.split_documents(pagesPCP)
        strPCP = []
        idsPCP = []
        metadataPCP = []
        n = 0
        for doc in splitPCP:
            strPCP.append(doc.page_content)
            idsPCP.append("PCP"+str(n))
            metadataPCP.append({"source": "Programa eleitoral do partido comunista português (PCP)"})
            n += 1
        #------------------------
        # BLOCO DE ESQUERDA
        splitBE = text_splitter.split_documents(pagesBE)
        strBE = []
        idsBE = []
        metadataBE = []
        n = 0
        for doc in splitBE:
            strBE.append(doc.page_content)
            idsBE.append("BE"+str(n))
            metadataBE.append({"source": "Programa eleitoral do partido bloco de esquerda (BE)"})
            n += 1
        #------------------------
        # LIVRE
        splitLV = text_splitter.split_documents(pagesLivre)
        strLV = []
        idsLV = []
        metadataLV = []
        n = 0
        for doc in splitLV:
            strLV.append(doc.page_content)
            idsLV.append("LV"+str(n))
            metadataLV.append({"source": "Programa eleitoral do partido Livre"})
            n += 1
        #------------------------
        # INICIATIVA LIBERAL
        splitIL = text_splitter.split_documents(pagesIL)
        strIL = []
        idsIL = []
        metadataIL = []
        n = 0
        for doc in splitIL:
            strIL.append(doc.page_content)
            idsIL.append("IL"+str(n))
            metadataIL.append({"source": "Programa eleitoral do partido Iniciativa Liberal (IL)"})
            n += 1
        #------------------------
        # CHEGA
        splitChega = text_splitter.split_documents(pagesChega)
        strChega = []
        idsChega = []
        metadataChega = []
        n = 0
        for doc in splitChega:
            strChega.append(doc.page_content)
            idsChega.append("CHEGA"+str(n))
            metadataChega.append({"source": "Programa eleitoral do partido CHEGA)"})
            n += 1
        #------------------------
        # AD
        splitAD = text_splitter.split_documents(pagesAD)
        strAD = []
        idsAD = []
        metadataAD = []
        n = 0
        for doc in splitAD:
            strAD.append(doc.page_content)
            idsAD.append("AD"+str(n))
            metadataAD.append({"source": "Programa eleitoral do partido Aliança Democrárica (AD), composto pelo PSD e CDS)"})
            n += 1
        #------------------------
        # PS
        splitPS = text_splitter.split_documents(pagesPS)
        strPS = []
        idsPS = []
        metadataPS = []
        n = 0
        for doc in splitPS:
            strPS.append(doc.page_content)
            idsPS.append("PS"+str(n))
            metadataPS.append({"source": "Programa eleitoral do partido socialista (PS))"})
            n += 1
        
        
        # 1.3 Create Chroma DB (one Collection for each political party)
        
        collection_volt = persistent_client.create_collection(name="legislativas_2024_volt",embedding_function=openai_embeddingsf)
        collection_pcp = persistent_client.create_collection(name="legislativas_2024_pcp",embedding_function=openai_embeddingsf)
        collection_be = persistent_client.create_collection(name="legislativas_2024_be",embedding_function=openai_embeddingsf)
        collection_livre = persistent_client.create_collection(name="legislativas_2024_livre",embedding_function=openai_embeddingsf)
        collection_il = persistent_client.create_collection(name="legislativas_2024_il",embedding_function=openai_embeddingsf)
        collection_chega = persistent_client.create_collection(name="legislativas_2024_chega",embedding_function=openai_embeddingsf)
        collection_ad = persistent_client.create_collection(name="legislativas_2024_ad",embedding_function=openai_embeddingsf)
        collection_ps = persistent_client.create_collection(name="legislativas_2024_ps",embedding_function=openai_embeddingsf)

        collection_volt.add(ids=idsVolt, documents=strVolt, metadatas=metadataVolt)
        collection_pcp.add(ids=idsPCP, documents=strPCP, metadatas=metadataPCP)
        collection_be.add(ids=idsBE, documents=strBE, metadatas=metadataBE)
        collection_livre.add(ids=idsLV, documents=strLV, metadatas=metadataLV)
        collection_il.add(ids=idsIL, documents=strIL, metadatas=metadataIL)
        collection_chega.add(ids=idsChega, documents=strChega, metadatas=metadataChega)
        collection_ad.add(ids=idsAD, documents=strAD, metadatas=metadataAD)
        collection_ps.add(ids=idsPS, documents=strPS, metadatas=metadataPS)
    else:
        
        # 1.4 Load Chroma DB (one Collection for each political party)
        
        collection_volt = persistent_client.get_collection(name="legislativas_2024_volt",embedding_function=openai_embeddingsf)
        collection_pcp = persistent_client.get_collection(name="legislativas_2024_pcp",embedding_function=openai_embeddingsf)
        collection_be = persistent_client.get_collection(name="legislativas_2024_be",embedding_function=openai_embeddingsf)
        collection_livre = persistent_client.get_collection(name="legislativas_2024_livre",embedding_function=openai_embeddingsf)
        collection_il = persistent_client.get_collection(name="legislativas_2024_il",embedding_function=openai_embeddingsf)
        collection_chega = persistent_client.get_collection(name="legislativas_2024_chega",embedding_function=openai_embeddingsf)
        collection_ad = persistent_client.get_collection(name="legislativas_2024_ad",embedding_function=openai_embeddingsf)
        collection_ps = persistent_client.get_collection(name="legislativas_2024_ps",embedding_function=openai_embeddingsf)
    
    
    # 2. Retrieval and LLM Generation
    

    # 2.1 Regular prompt to extract metadata from the query (wich political parties should be adressed at the answer)

    prompt_goal = """
    Consider the following list of Portuguese political parties: Partido Bloco de Esquerda (also known by its acronym BE);\
    Partido Volt (also known by its acronym Volt); Partido Comunista Português (also known by its acronym PCP);\
    Partido Nova Democracia (also known by its acronym ND); Partido Livre (also known by its acronym Livre);\
    Partido Iniciativa Liberal (also known by its acronym IL); Partido Chega (also known by its acronym CHEGA);\
    Partido Aliança Democrática (also known by its acronym AD); and Partido Socialista (also known by its acronym PS).\
    From the user question that is delimited by triple backticks:´´´{user_query}´´´, extract which Portuguese political\
    parties from the list are being questioned and required to be adressed to completely answwer the user question.\
    Return the required Portuguese political parties acronyms in a JSON format, inside an array with the key 'partidos'.\
    If the user question is intended to all political parties or its not specific, return all of the Portuguese political parties\
    declared in the list. Also format the user question keeping its meaning, removing any political party reference,\
    including any name or acronym of the political parties from the list).\
    The resultant rephrased question should be returned inside the previous JSON format, inside the key 'questao'.
    """
    
      
    prompt_template = PromptTemplate.from_template(prompt_goal)
    regular_chain = prompt_template | llm
    json_string = regular_chain.invoke({"user_query": user_initial_query})
    json_object = json.loads(json_string.content)
    user_initial_query = json_object["questao"]
    print(json_object["partidos"])
    print(json_object["questao"])


    # 2.2 From the appropriate vector collection, select the data for the RAG


    vectorstore_volt = None
    vectorstore_pcp = None
    vectorstore_be = None
    vectorstore_livre = None
    vectorstore_il = None
    vectorstore_chega = None
    vectorstore_ad = None
    vectorstore_ps = None
    
    output_political_review = "Recolha da resposta em documentação oficial do(s) partido(s) político(s):\n\n"
    
    if 'Volt' in json_object["partidos"]:
        if vectorstore_volt is None:
            vectorstore_volt = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_volt",
                embedding_function=openai_embeddings,
            )
        qa_chain_volt = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_volt.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_volt = qa_chain_volt.invoke({'query': user_initial_query})
        output_political_review += " \n VOLT \n"+"\n"+result_volt['result']+"\n"
        
    if 'PCP' in json_object["partidos"]:
        if vectorstore_pcp is None:
            vectorstore_pcp = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_pcp",
                embedding_function=openai_embeddings,
            )
        qa_chain_pcp = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_pcp.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_pcp = qa_chain_pcp.invoke({'query': user_initial_query})
        output_political_review += " \n PCP \n"+"\n"+result_pcp['result']+"\n"
        
    if 'BE' in json_object["partidos"]:
        if vectorstore_be is None:
            vectorstore_be = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_be",
                embedding_function=openai_embeddings,
            )
        qa_chain_be = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_be.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_be = qa_chain_be.invoke({'query': user_initial_query})
        output_political_review += " \n BE \n"+"\n"+result_be['result']+"\n"

    if 'Livre' in json_object["partidos"]:
        if vectorstore_livre is None:
            vectorstore_livre = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_livre",
                embedding_function=openai_embeddings,
            )
        qa_chain_livre = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_livre.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_livre = qa_chain_livre.invoke({'query': user_initial_query})
        output_political_review += " \n Livre \n"+"\n"+result_livre['result']+"\n"

    if 'IL' in json_object["partidos"]:
        if vectorstore_il is None:
            vectorstore_il = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_il",
                embedding_function=openai_embeddings,
            )
        qa_chain_il = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_il.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_il = qa_chain_il.invoke({'query': user_initial_query})
        output_political_review += " \n IL \n "+" \n "+result_il['result']+" \n "

    if 'CHEGA' in json_object["partidos"]:
        if vectorstore_chega is None:
            vectorstore_chega = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_chega",
                embedding_function=openai_embeddings,
            )
        qa_chain_chega = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_chega.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_chega = qa_chain_chega.invoke({'query': user_initial_query})
        output_political_review += " \n CHEGA \n "+" \n "+result_chega['result']+" \n "
    
    if 'AD' in json_object["partidos"]:
        if vectorstore_ad is None:
            vectorstore_ad = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_ad",
                embedding_function=openai_embeddings,
            )
        qa_chain_ad = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_ad.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_ad = qa_chain_ad.invoke({'query': user_initial_query})
        output_political_review += " \n AD \n "+" \n "+result_ad['result']+" \n "
    
    if 'PS' in json_object["partidos"]:
        if vectorstore_ps is None:
            vectorstore_ps = Chroma(
                client=persistent_client,
                collection_name="legislativas_2024_ps",
                embedding_function=openai_embeddings,
            )
        qa_chain_ps = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore_ps.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True
        )
        result_ps = qa_chain_ps.invoke({'query': user_initial_query})
        output_political_review += " \n PS \n "+" \n "+result_ps['result']+" \n "
    
    print(output_political_review)
    with st.chat_message("assistant"):
        st.write(output_political_review)
        
    return output_political_review


# 3. Deployment 
#print(setupPoliticalRAG("Qual a posição na habitação jovem?"))

# Options
st.set_page_config(
    page_title="escrutin.AI",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Esta aplicação foi desenvolvida por João Meneses com recurso ao LLM da OpenAI, numa estratégia de RAG utilizando como fonte de dados os programas políticos oficiais dos principais partidos concorrentes nas legislativas 2024. Esta aplicação não visa qualquer objectivo particular para além de ser demonstrativa da tecnologia dos LLM. A sua utilização e os seus resultados devem ser interpretados de forma crítica e em consideração que a amostragem do modelo pode levar a resposta incompletas ou incorrectas. O criador da aplicação não se responsabiliza por erros, omissões ou quaisquer outros resultantes da utilização da aplicação. O código fonte encontra-se aberto e disponível na plataforma GITHUB."
    }
)

# 3.1 Streamlit
st.title('Eleições Legislativas 2024')

# Present AI
with st.chat_message("assistant"):
    textAI = "Olá! Sou o escrutin.AI e no contexto das eleições legislativas Portuguesas de 2024 estou aqui para te tentar ajudar a realizar buscas simultâneamente nos vários programas eleitorais para que os consigas comparar no teu tempo e nos teus próprios termos.\n \n Faz uma questão ou pedido sobre um tópico ou sobre as propostas/medidas políticas de todos os partidos, de um subconjunto deles, ou apenas a um partido em particular. (ex: Refere as medidas dos partidos com impacto na habitação jovem.)"
    st.write(textAI)

# React to user input
textGuide = "Coloca aqui a tua questão..."
if question := st.chat_input(textGuide):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Ask escrutin.AI
    setupPoliticalRAG(question)



#Exemplos:
#Qual a política salarial de cada partido?
#Refere as medidas dos partidos com impacto na habitação jovem.
#Qual a visão a respeito da globalização?
