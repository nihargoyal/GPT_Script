from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os


loader = DirectoryLoader('/', glob='**/*.txt')
docs = loader.load()

char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_texts = char_text_splitter.split_documents(docs)

openAI_embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vStore = Chroma.from_documents(doc_texts, openAI_embeddings)

model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vStore, return_source_documents=True)

question = "What are the effects of homelessness"
response = model({"query":question})
response['source_documents']
