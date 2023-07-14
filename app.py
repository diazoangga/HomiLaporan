!pip install streamlit_chat

import os
import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import FAISS
import tempfile
import pandas as pd

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def chat_with_ai(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                    # 'delimiter': ','})
        loader = UnstructuredExcelLoader(tmp_file_path, mode="elements")
        data = loader.load()

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     length_function=len
        # )

        # chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(
                llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
                                retriever=vectorstore.as_retriever())
        
        # st.write(vectorstore)
        def conversational_chat(query):
        
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            
            return result["answer"]
    
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]
            
        #container for the chat history
        response_container = st.container()
        #container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

class FinancialReport():
    def __init__(self, uploaded_file):
        super(FinancialReport, self).__init__()
        self.file = uploaded_file
    
    def preparing_data(self):
        data = pd.read_excel(uploaded_file, 'Sheet1', skiprows = 11)
        self.dataFrame = pd.DataFrame(data)
        self.dataFrame[HEADER_NAME[0]] = pd.to_datetime(self.dataFrame['Waktu'])
        
        self.dataFrame['month'] = self.dataFrame[HEADER_NAME[0]].dt.month
        self.dataFrame['year'] = self.dataFrame[HEADER_NAME[0]].dt.year

    def is_file_right(self, function):
        if self.file is None:
            st.write('The file is not exist, please upload the file')
        else:
            pass

    def hitung_bonus(self, month, year):
        month_last = month - 1 if month > 1 else 12 

        penjualan_last = self.hitung_penjualan(month_last)
        penjualan_last_until_24 = self.hitung_penjualan(month_last, until_last_day=False)
        penjualan_current = self.hitung_penjualan(month, until_last_day=False)  
        rata_current = int(penjualan_current/BONUS_DAY)
        rata_last = int(penjualan_last_until_24/BONUS_DAY)
        # print(rata_current)

        if rata_current < TARGET_PENJUALAN_BONUS[0]:
            bonus_percent = 0
        elif (rata_current > TARGET_PENJUALAN_BONUS[0]) & (rata_current < TARGET_PENJUALAN_BONUS[1]):
            bonus_percent = 2
        elif (rata_current > TARGET_PENJUALAN_BONUS[1]) & (rata_current < TARGET_PENJUALAN_BONUS[2]):
            bonus_percent = 3
        elif (rata_current > TARGET_PENJUALAN_BONUS[2]) & (rata_current < TARGET_PENJUALAN_BONUS[3]):
            bonus_percent = 4
        elif (rata_current > TARGET_PENJUALAN_BONUS[3]) & (rata_current < TARGET_PENJUALAN_BONUS[4]):
            bonus_percent = 5
        elif (rata_current > TARGET_PENJUALAN_BONUS[4]):
            bonus_percent = 5

        bonus = int(penjualan_last*bonus_percent/100)
        
        if bonus > BONUS_MAX:
            bonus = BONUS_MAX

        bonus_bulat = int(round(bonus, -5))
        
        return convert_to_currency(penjualan_current),\
            convert_to_currency(penjualan_last),\
            convert_to_currency(rata_current),\
            convert_to_currency(rata_last),\
            convert_to_currency(bonus),\
            convert_to_currency(bonus_bulat), bonus_percent

    def hitung_penjualan(self, month, until_last_day=True):
        data_mo = self.dataFrame.loc[self.dataFrame['month'] == month]
        data_mo.reset_index(inplace=True, drop=True)
        # st.write(data_mo)
        if until_last_day==True:
            penjualan = data_mo.loc[:, HEADER_NAME[1]].sum()
        else:
            penjualan = data_mo.loc[0:(BONUS_DAY - 1), HEADER_NAME[1]].sum()
        return penjualan

def convert_to_currency(number):
    currency = "Rp. {:,}".format(number)
    return currency

os.environ['OPENAI_API_KEY'] = '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'
# MONTHS = ('Jan', 'Feb', 'Mar',
#           'Apr', 'Mei', 'Jun',
#           'Jul', 'Aug', 'Sept',
#           'Okt', 'Nov', 'Des')

MONTHS = (1,2,3,4,5,6,7,8,9,10,11,12)
YEAR = (2023, 2024, 2025, 2026, 2027)

HEADER_NAME = ['Waktu', 'Penjualan (Rp.)', 'Laba Kotor (Rp.)',
               'Jumlah Transaksi', ' Order / Transaksi (Rp.) ',
               'Refund (Rp.)', 'Komisi (Rp.)', 'Jumlah Produk',
               'produk / Transaksi']

BONUS_DAY = 24
TARGET_PENJUALAN_BONUS = [1650000, 1850000, 2000000, 2250000, 2500000]
BONUS_MAX = 4000000
st.set_page_config(page_title="Laporan HOMI", layout="wide")
with st.sidebar:
    st.title('Laporan HOMI')
    st.markdown('''
    ## About
    This app is a simple :
    - Streamlit
    - Langchain
    - OpenAI
    
    ''')

with st.container():
    uploaded_file = st.file_uploader('Upload your PDF file here', type='xlsx')
    if uploaded_file:
        fin = FinancialReport(uploaded_file)
        fin.preparing_data()
        print(uploaded_file)
        print('data is prepared')

    left_col, right_col = st.columns(2)
    with left_col:
        month_col, year_col = st.columns(2)
        with month_col:
            month_select = st.selectbox(label="Select a month", options=MONTHS)
        with year_col:
            year_select = st.selectbox(label='Select a year', options=YEAR)
    with right_col:
        bonus_button = st.button(label="Perhitungan Bonus")
        ask_button = st.button(label="Ask Anything")
    if bonus_button:
        jual_cur, jual_last, rata_cur, rata_last, bonus, bonus_round, bonus_percent = fin.hitung_bonus(month_select, year_select)
        table_bonus = {'Label': ['Penjualan bulan ini', 
                                 'Penjualan bulan lalu',
                                 'Rata-rata penjualan bulan ini s/d tanggal 24',
                                 'Rata-rata penjualan bulan lalu s/d tanggal 24',
                                 'Bonus (sebelum dibulatkan)',
                                 'Bonus (setelah dibulatkan)'], 
                        'Jumlah': [jual_cur, jual_last, rata_cur, rata_last, bonus, bonus_round]}
        table_bonus = pd.DataFrame(table_bonus).set_index('Label')
        summary_bonus = f'Berdasarkan data penjualan Homi tgl 1/{month_select}/{year_select} s.d {BONUS_DAY}/{month_select}/{year_select} (bruto) tercatat sebesar {jual_cur} dg rata2 penjualan seb {rata_cur}.\
            Rata-rata penjualan harian bulan {month_select}/{year_select} meningkat dari posisi bulan sebelumnya yang tercatat seb {rata_last}. Dengan rata2 penjualan harian bulan ini sebesar {rata_cur}, \
            jumlah total bonus dg rata2 penjualan harian {rata_cur} adalah seb {bonus_percent}% dari penjualan bulan sblmnya seb {jual_last} adalah sebesar {bonus} atau dibulatkan seb {bonus_round}.'
        st.subheader('PERHITUNGAN BONUS')
        st.write(table_bonus)
        st.write('KESIMPULAN: ')
        st.write(summary_bonus)
        
    if ask_button:
        chat_with_ai(uploaded_file)

# local_css("style/style.css")
