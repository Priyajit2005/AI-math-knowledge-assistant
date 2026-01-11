import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain, LLMMathChain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.agents import AgentType
from langchain_core.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper

st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Uing OpenAI")

api_key=st.sidebar.text_input(label="Groq API Key",type="password")


if not api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

llm=ChatOpenAI(model="gpt-3.5-turbo",api_key=api_key)

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"
)


math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description=(
        "Use this tool ONLY for ONE single mathematical expression at a time. "
        "Do NOT pass multiple expressions, lists, tuples, or comma-separated values."
    )
)


prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain=LLMChain(llm=llm,prompt=prompt_template)


reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


question=st.text_area("Enter your question:")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate responses..."):
            st.session_state.messages.append({"role":"users","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant','content':response})
            st.write("Response:")
            st.success(response)
    else:
        st.warning("Please enter the input")