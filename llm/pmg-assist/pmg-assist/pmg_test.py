import pandas as pd
import requests

df = pd.read_parquet("../data/pmg_data.parquet.brotli")

question = df.sample(n=1).iloc[0]['question']

print("question: ", question)

url = "http://localhost:5001/question"


data = {"question": question}

response = requests.post(url,  json=data)
# print(response.content)

print(response.json())