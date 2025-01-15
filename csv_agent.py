from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

OPENAI_API_KEY = "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"
llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=llm_name)

df = pd.read_csv('salaries_2023.csv').fillna(value=0)

# print(df.head())
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
)

# res = agent.invoke("What is the avarage salary?")
# print(res)

# then let's add some pre and sufix prompt
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

QUESTION = "Which grade has the highest average base salary, and compare the average female pay vs male pay?"

# res = agent.invoke(CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX)

# print(f"Final result: {res['output']}")

#--------------------------------------------------------------------------------------------------
import streamlit as st

st.title("Database AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head())

st.write("### Ask a Question")
question = st.text_input(
    "Enter your question about the dataset:",
    "Which grade has the highest average base salary, and compare the average female pay vs male pay?"
)

if st.button("Run Query"):
    QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.markdown(res["output"])
    