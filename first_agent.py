from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = "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"
llm_name = "gpt-4o-mini"
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=llm_name)

messages = [
    SystemMessage(
        content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    ),
    HumanMessage(content="Who was the first computer scientist?")
]

# res = model.invoke(messages)
# print(res)

def first_agent(messages):
    res = model.invoke(messages)
    return res

def run_agent():
    print("Simple AI agent: Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print("AI Agent is thinking...")
        messages = [HumanMessage(content=user_input)]
        response = first_agent(messages)
        print("AI Agent: getting response...")
        print(f"AI Agent: {response.content}")

if __name__ == "__main__":
    run_agent()
