import os

import dashscope
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

llm = ChatTongyi(
    model_name='qwen-max',
    streaming=True,
)
if __name__ == '__main__':
    res = llm.stream([HumanMessage(content="你是谁")], streaming=True)
    for r in res:
        print(r.content, end="")
    messages = [
        SystemMessage(
            content="You are a helpful assistant."
        ),
        HumanMessage(
            content="Translate this sentence from English to Chinese. I love programming."
        ),
    ]
    res = llm.invoke(messages)
    print(res.content)

