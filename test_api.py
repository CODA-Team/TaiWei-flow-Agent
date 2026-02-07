# save as call_deepseek.py
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ai.gitee.com/v1",  # DeepSeek API base url
)

resp = client.chat.completions.create(
    model="DeepSeek-R1",            # 或 "deepseek-chat"
    messages=[{"role": "user", "content": "reply hello"}],
)
print(resp.choices[0].message.content)