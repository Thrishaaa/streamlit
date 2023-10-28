from collections import deque
from langchain.llms import OpenAI, HuggingFacePipeline  
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
import itertools
import pinecone
from copy import copy,deepcopy
from langchain.memory import ConversationBufferWindowMemory
from transformers import pipeline
import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
#from transformers import Conversation, SystemMessagePrompt, UserMessagePrompt
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json
from collections import deque
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
import transformers 
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI