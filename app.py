import streamlit as st
import os

# --- תיקון תאימות ל-SQLite בשרתי Streamlit (חובה!) ---
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

# הגדרות עמוד
st.set_page_config(page_title="צ'אט מאמרים דנטליים", page_icon="🦷")
st.title("🦷 צ'אט עם מאמרים דנטליים")
st.markdown("מערכת RAG המבוססת על המאמרים שהעלית ל-GitHub.")

# --- הגדרת המפתח (Secrets) ---
# וודא שהגדרת OPENAI_API_KEY ב-Settings -> Secrets של Streamlit
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("שגיאה: לא נמצא מפתח API. אנא הגדר אותו ב-Streamlit Secrets.")
    st.stop()

# --- חיבור לבסיס הנתונים ---
# מכיוון שהקובץ chroma.sqlite3 נמצא בתיקייה הראשית בגיט, נשתמש ב-"."
persist_directory = "."

@st.cache_resource
def init_rag():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=api_key
    )
    
    # טעינת מסד הנתונים הקיים
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(
        model_name="gpt-4o", 
        temperature=0, 
        openai_api_key=api_key
    )
    
    template = """אתה עוזר מחקר מקצועי בתחום רפואת השיניים. 
השתמש בקטעי המידע הבאים כדי לענות על השאלה. 
אם התשובה לא נמצאת במידע, אמור שאינך יודע על בסיס המאמרים הקיימים.

הקשר (Context):
{context}

שאלה: {question}

תשובה בעברית:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
         "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return chain

# אתחול המערכת
try:
    rag_chain = init_rag()
except Exception as e:
    st.error(f"שגיאה בטעינת מסד הנתונים: {e}")
    st.stop()

# --- ניהול היסטוריית הצ'אט ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת הודעות קודמות
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# תיבת קלט למשתמש
if prompt_input := st.chat_input("שאל שאלה על המאמרים (למשל: מה המסקנה של לביני?)..."):
    # הוספת הודעת משתמש
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # יצירת תשובה
    with st.chat_message("assistant"):
        with st.spinner("סורק מאמרים ומנסח תשובה..."):
            try:
                full_response = rag_chain.invoke(prompt_input)
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"קרתה שגיאה בזמן יצירת התשובה: {e}")
