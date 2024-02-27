import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'), transport='rest')


def text_test():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("你是谁?", stream=True)
    for chuck in response:
        print(chuck.text)


if __name__ == '__main__':
    text_test()
