import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# הגדרת כותרת לממשק
st.set_page_config(page_title="Dental Research Chat", page_icon="🦷")
st.title("🦷 צ'אט עם מאמרים דנטליים")

# טעינת המפתח מתוך ה-Secrets של Streamlit (במקום קובץ .env)

if not api_key:
    st.error("Missing OpenAI API Key. Please configure it in Streamlit Secrets.")
    st.stop()

# נתיב לבסיס הנתונים (הקובץ שהעלית לגיט)
persist_directory = "." 

# אתחול ה-RAG
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

# בניית ה-Chain
template = "ענה בעברית על בסיס המידע הבא:\n{context}\n\nשאלה: {question}"
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ממשק צ'אט
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("שאל שאלה על המאמרים..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("סורק מאמרים..."):
            response = rag_chain.invoke(user_query)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})