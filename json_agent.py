from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

OPENAI_API_KEY = "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"
llm_name = "gpt-4o-mini"
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=llm_name)


df = pd.read_json('results.json')

from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
)

res = agent.invoke("Resuma o texto do dia 20/12/2024 para um leigo")
print(res)