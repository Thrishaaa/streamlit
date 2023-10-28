from packages import *
from sentence_transformers import SentenceTransformerEmbeddings

#Connecting to Pinecone Server
api_key = "fa8f252e-6725-47d0-9d17-5378a7d8fd55"
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
#Connect to your indexes
index_name = "db-trial"
index = pinecone.Index(index_name=index_name)
index.delete(deleteAll='true', namespace="") #to delete exsisting vectors from index
#print('\n\n----- Pnecone index DESC -----\n\n',index.describe_index_stats(),'\n\n')

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
loader = TextLoader("https://github.com/Thrishaaa/streamlit/blob/main/CCD_AI_Items.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0,separator='***********************************************************************************')
docs = text_splitter.split_documents(documents)
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

#print('\n\n----- Pnecone index DESC (After)-----\n\n',index.describe_index_stats(),'\n\n')
def product_qa(user_input):
    docs = docsearch.similarity_search(user_input)
    content = ''
    for doc in docs:
        content = content + '\n'+ doc.page_content + '\n'
    return content

llm_chat = ChatOpenAI(temperature=1, openai_api_key="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")

def chat_reply(template, user_input):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template), HumanMessagePromptTemplate.from_template("{input}")
    ])
    conversation = ConversationChain(prompt=prompt, llm=llm_chat)
    return conversation.predict(input=user_input)

#print(product_qa('give me some veg items'))

#print(product_qa('List all the beef items'))
'''
with open('/root/Programs/Tree_OpenAi_v2/content/ccd_items_temp.txt', 'r') as file:
        template = file.read()

while True:
    user_input = input("\n Human:   ")
    
    if user_input.lower() in ('exit', 'quit', 'bye'):    
        break
    context = product_qa(user_input)
    template = template + context + "{history}"
    print('\nAI:    ',chat_reply(template,user_input))
'''
