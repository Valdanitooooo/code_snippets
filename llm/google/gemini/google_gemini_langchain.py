from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def llm_test():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", transport='rest')
    result = llm.invoke("你是谁")
    print(result.content)


def chain_test():
    system = "You are a helpful assistant who translate English to French"
    human = "Translate this sentence from English to {input}. I love programming."
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", transport='rest', convert_system_message_to_human=True)
    chain = prompt | llm
    result = chain.invoke({"input": "Chinese"})
    print(result.content)


if __name__ == '__main__':
    llm_test()
    chain_test()

