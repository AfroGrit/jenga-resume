#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import minsearch
from openai import OpenAI
import os
from tqdm.auto import tqdm
import json
import random


# In[42]:


client = OpenAI()


# In[43]:


df = pd.read_parquet('_pmg_sample_clean.parquet.brotli')
df['responder'] = df['responder'].str.replace('to ask the ', '', regex=False)
# Convert 'date' column to the desired string format 'YYYY-MM-DD'
df['date'] = df['date'].dt.strftime('%Y-%m-%d')
# Confirm that the 'date' column is now of type 'object'
df['date'] = df['date'].astype('object')


# In[44]:


df.info()


# In[45]:


documents = df.to_dict(orient='records')

index = minsearch.Index(
    text_fields=['date', 'id', 'mp', 'responder', 'question', 'answer'],
    keyword_fields=['id']
)

index.fit(documents)


# #### RAG flow

# In[46]:


def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[47]:


prompt_template = """
You emulate a user of our parliament assistant application.
Formulate 5 questions this user might ask based on the provided parliamentary record.
Make the questions specific to this record.
The record should contain the answer to the questions, and the questions should be
complete and not too short. Use as few words as possible from the record.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
date: {date}
id: {id}
mp: {mp}
responder: {responder}
question: {question}
answer: {answer}
""".strip()


# In[48]:


def build_prompt(query, search_results):
    context = ""
    
    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query, model='gpt-4o-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    #print(prompt)
    answer = llm(prompt, model=model)
    return answer


# In[49]:


question = 'How will the SA Police Service develop mechanisms that will reduce the workload of detectives to ensure a speedy resolution of criminal cases?'
answer = rag(question)
print(answer)


# #### Retrieval evaluation
# 

# In[50]:


df_question = pd.read_parquet('_gt_retrieval.parquet.brotli')
ground_truth = df_question.to_dict(orient='records')


# In[51]:


df_question.head()


# In[52]:


ground_truth[0]


# In[53]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[54]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[55]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# ##### best params

# In[56]:


df_validation = df_question[:100]
df_test = df_question[100:]


# In[57]:


def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)
        
        # Evaluate the objective function
        current_score = objective_function(current_params)
        
        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params
    
    return best_params, best_score


# In[58]:


gt_val = df_validation.to_dict(orient='records')


# In[59]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[60]:


param_ranges = {
    'date': (0.0, 0.0),
    'mp': (0.0, 3.0),
    'responder': (0.0, 3.0),
    'question': (0.0, 3.0),
    'answer': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[61]:


simple_optimize(param_ranges, objective, n_iterations=5)


# In[62]:


def minsearch_improved(query):
    boost = {
        'date': 0,
        'mp': 0.81,
        'responder': 0.51,
        'question': 2.34,
        'answer': 0.67,
    }

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

evaluate(ground_truth, lambda q: minsearch_improved(q['question']))


# In[ ]:





# #### RAG evaluation
# 

# In[63]:


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[64]:


df_sample = df_question.sample(n=200, random_state=1)
sample = df_sample.to_dict(orient='records')


# In[65]:


evaluations = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question) 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[66]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[67]:


df_eval.relevance.value_counts(normalize=True)


# In[68]:


# Write the DataFrame to a Parquet file with Brotli compression
df_eval.to_parquet('rageval_4omini.parquet.brotli', compression='brotli')
print("rag evaluation on gpt 4o mini written to parquet successfully.")


# In[69]:


evaluations_gpt4o = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question, model='gpt-4o') 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)
    
    evaluations_gpt4o.append((record, answer_llm, evaluation))


# In[70]:


df_eval = pd.DataFrame(evaluations_gpt4o, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[71]:


df_eval.relevance.value_counts()


# In[72]:


df_eval.relevance.value_counts(normalize=True)


# In[73]:


# Write the DataFrame to a Parquet file with Brotli compression
df_eval.to_parquet('rageval_4o.parquet.brotli', compression='brotli')
print("rag evaluation on gpt-4o  written to parquet successfully.")


# In[82]:


df_eval


# In[ ]:




