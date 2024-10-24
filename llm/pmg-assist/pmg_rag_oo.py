#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import minsearch
from openai import OpenAI
import os
from tqdm.auto import tqdm
import json
import random
from time import time
import ingest

class ParliamentAssistant:
    """
    Class for performing search and query operations for a Parliament Assistant application.

    Attributes:
        client (OpenAI): The OpenAI client for language model interactions.
        index (Index): Loaded index for performing search queries.
    """
    
    def __init__(self):
        """
        Initialize the assistant by loading the OpenAI client and index.
        """
        self.client = OpenAI()
        self.index = ingest.load_index()

    def search(self, query):
        """
        Search the index based on the provided query with customized boost values.

        Args:
            query (str): The search query.

        Returns:
            list: List of search results matching the query.
        """
        boost = {
            'date': 0,
            'mp': 0.81,
            'responder': 0.51,
            'question': 2.34,
            'answer': 0.67,
        }
        results = self.index.search(
            query=query, 
            filter_dict={}, 
            boost_dict=boost, 
            num_results=5
        )
        return results


class PromptBuilder:
    """
    Class to build prompts for the language model.

    Attributes:
        prompt_template (str): Template for generating questions.
        entry_template (str): Template for formatting search results into context.
    """

    def __init__(self):
        self.prompt_template = """
        You emulate a user of our parliament assistant application.
        Formulate 5 questions this user might ask based on the provided parliamentary record.
        Make the questions specific to this record. The record should contain the answer to the questions,
        and the questions should be complete and not too short. Use as few words as possible from the record.

        QUESTION: {question}

        CONTEXT:
        {context}
        """.strip()

        self.entry_template = """
        date: {date}
        id: {id}
        mp: {mp}
        responder: {responder}
        question: {question}
        answer: {answer}
        """.strip()

    def build_prompt(self, query, search_results):
        """
        Build a prompt by formatting search results into the context.

        Args:
            query (str): The search query.
            search_results (list): List of search results.

        Returns:
            str: Formatted prompt string.
        """
        context = ""
        for doc in search_results:
            context += self.entry_template.format(**doc) + "\n\n"
        
        prompt = self.prompt_template.format(question=query, context=context).strip()
        return prompt


class ModelEvaluator:
    """
    Class to evaluate the relevance of the generated answers using an LLM.

    Attributes:
        client (OpenAI): The OpenAI client for language model interactions.
    """

    def __init__(self, client):
        self.client = client
        self.evaluation_prompt_template = """
        You are an expert evaluator for a RAG system.
        Your task is to analyze the relevance of the generated answer to the given question.
        Based on the relevance of the generated answer, you will classify it
        as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

        Here is the data for evaluation:

        Question: {question}
        Generated Answer: {answer}

        Please analyze the content and context of the generated answer in relation to the question
        and provide your evaluation in parsable JSON without using code blocks:

        {{
          "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
          "Explanation": "[Provide a brief explanation for your evaluation]"
        }}
        """.strip()

    def evaluate_relevance(self, question, answer):
        """
        Evaluate the relevance of the generated answer using the model.

        Args:
            question (str): The question asked.
            answer (str): The generated answer.

        Returns:
            dict: The evaluation result including relevance and explanation.
            int: Number of tokens used in the evaluation process.
        """
        prompt = self.evaluation_prompt_template.format(question=question, answer=answer)
        evaluation = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            json_eval = json.loads(evaluation.choices[0].message.content)
            return json_eval, evaluation.usage.total_tokens
        except json.JSONDecodeError:
            result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
            return result, evaluation.usage.total_tokens


class LLMHandler:
    """
    Class to handle LLM queries and cost calculation.

    Attributes:
        client (OpenAI): The OpenAI client for language model interactions.
    """

    def __init__(self, client):
        self.client = client

    def llm(self, prompt, model='gpt-4o-mini'):
        """
        Query the language model with the provided prompt.

        Args:
            prompt (str): The formatted prompt.
            model (str): The model to be used for the query.

        Returns:
            str: The generated answer from the model.
            dict: Token statistics for the request.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, response.usage

    def calculate_openai_cost(self, model, tokens):
        """
        Calculate the OpenAI cost based on the model and token usage.

        Args:
            model (str): The model name.
            tokens (dict): Token usage statistics.

        Returns:
            float: The calculated cost.
        """
        openai_cost = 0
        if model == "gpt-4o-mini":
            openai_cost = (
                tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
            ) / 1000
        return openai_cost


class RAGSystem:
    """
    Class representing the complete Retrieval-Augmented Generation (RAG) system.

    Attributes:
        assistant (ParliamentAssistant): The ParliamentAssistant instance for search.
        prompt_builder (PromptBuilder): The PromptBuilder instance for prompt generation.
        evaluator (ModelEvaluator): The ModelEvaluator instance for evaluating answers.
        llm_handler (LLMHandler): The LLMHandler instance for managing LLM queries and costs.
    """

    def __init__(self):
        """
        Initialize the RAG system by setting up the necessary components.
        """
        self.assistant = ParliamentAssistant()
        self.prompt_builder = PromptBuilder()
        self.evaluator = ModelEvaluator(self.assistant.client)
        self.llm_handler = LLMHandler(self.assistant.client)

    def rag(self, query, model="gpt-4o-mini"):
        """
        Perform the RAG process for a given query.

        Args:
            query (str): The search query.
            model (str): The language model to be used.

        Returns:
            dict: The final result including answer, relevance, token usage, and cost.
        """
        t0 = time()

        # Search for relevant documents
        search_results = self.assistant.search(query)

        # Build the prompt from search results
        prompt = self.prompt_builder.build_prompt(query, search_results)

        # Generate answer using the language model
        answer, token_stats = self.llm_handler.llm(prompt, model=model)

        # Evaluate relevance of the generated answer
        relevance, rel_token_stats = self.evaluator.evaluate_relevance(query, answer)

        # Calculate costs and timing
        openai_cost_rag = self.llm_handler.calculate_openai_cost(model, token_stats)
        openai_cost_eval = self.llm_handler.calculate_openai_cost(model, rel_token_stats)
        openai_cost = openai_cost_rag + openai_cost_eval
        t1 = time()
        took = t1 - t0

        # Compile final answer data
        answer_data = {
            "answer": answer,
            "model_used": model,
            "response_time": took,
            "relevance": relevance.get("Relevance", "UNKNOWN"),
            "relevance_explanation": relevance.get("Explanation", "Failed to parse evaluation"),
            "prompt_tokens": token_stats["prompt_tokens"],
            "completion_tokens": token_stats["completion_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
            "eval_completion_tokens": rel_token_stats["completion_tokens"],
            "eval_total_tokens": rel_token_stats["total_tokens"],
            "openai_cost": openai_cost,
        }

        return answer_data
