import streamlit as st
import os

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Dental Research AI", page_icon="🦷")
st.title("🦷 צ'אט מחקר דנטלי מתקדם")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing API Key in Secrets")
    st.stop()

persist_directory = "./chroma_db_private"

@st.cache_resource
def init_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # הגדלנו ל-7 קטעים לחיפוש מעמיק יותר
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

    # פרומפט חדש שמכריח אותו לבדוק טוב יותר
    template = """אתה עוזר מחקר אקדמי. עליך לענות על השאלה בצורה מפורטת אך ורק על סמך הטקסט המצורף.
אם מופיע שם של חוקר (כמו Levrini) בטקסט, ציין את הממצאים שלו במדויק.

מידע מהמאמרים:
{context}

שאלה: {question}

תשובה מפורטת בעברית (כולל שם המאמר אם מצאת):"""
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([f"--- מקור: {d.metadata.get('source', 'לא ידוע')} ---\n{d.page_content}" for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

rag_chain = init_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("שאל שאלה ספציפית..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("סורק את כל המאמרים (זה עשוי לקחת כמה שניות)..."):
            response = rag_chain.invoke(prompt_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
