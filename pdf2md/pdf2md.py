import asyncio
import os

from pyzerox import zerox

kwargs = {}

custom_system_prompt = None

model = "gpt-4o-mini" # 实际用的Qwen2-VL-7B-Instruct
os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["OPENAI_API_BASE"] = "http://xxxx:9997/v1"
os.environ["OPENAI_BASE_URL"] = "http://xxxx:9997/v1"

async def main():

    directory = "./pdf_all"
    files = [os.path.abspath(os.path.join(directory, file)) for file in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, file))]

    select_pages = None

    output_dir = "./output_test"
    for file in files:
        print(f"=== 开始转换文件：{file}")
        try:
            await zerox(file_path=file, model=model, output_dir=output_dir,
                             custom_system_prompt=custom_system_prompt, select_pages=select_pages, **kwargs)
            print(f"=== 转换文件成功：{file}")
        except Exception as e:
            print(f"=== 转换文件失败：{file}")
            print(f"Exception: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())
