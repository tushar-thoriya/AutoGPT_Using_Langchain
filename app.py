####### Method 1 : Acces API key of ChatGPT
# from dotenv import load_dotenv
# import os
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

##### Method 2 : Import API key 
from openai_api_key import apikey
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App freamwork
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here ')


# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a youtube video script based on this title TITLE: {title} while leaverageing this wikipedia research: {wikipedia_research}'
)

# Memory
# memory = ConversationBufferMemory(input_key='topic', memory_key='history')
title_memory = ConversationBufferMemory(input_key='topic', memory_key='history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='history')

#LLms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory )
# Simple_sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)
# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
#                                   output_variables=['title', 'script'], verbose=True)

wiki = WikipediaAPIWrapper()

# show stuff to the screen if there's a prompt
if prompt:
    # response = llm.invoke(prompt)
    # st.write(response)

    # response = title_chain.run(topic=prompt)
    # st.write(response)

    # response = Simple_sequential_chain.run(prompt)
    # st.write(response)

    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia History"):
        st.info(wiki_research)