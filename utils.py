import os
import textwrap

from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore


def wrap(text, width=70):
    """
    A word-wrap function that preserves existing line breaks
    """
    # If the text is already wrapped, return it as-is

    # Split the text into lines based on existing newlines
    lines = text.split("\n")

    # Wrap each line separately and preserve existing newlines
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines with newline characters
    return "\n".join(wrapped_lines)


load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"]
)


def elastic_search(query, es_client, index_name):
    """
    Perform a search on the Elasticsearch index using the provided query.
    Return the search results as a list of dictionaries.
    """
    query_dict = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["Question", "Answer"],
                    "type": "best_fields",
                }
            }
        }
    }

    response = es_client.search(index=index_name, query=query_dict, size=5, from_=0)

    result_docs = []

    for hit in response["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs


def build_prompt(query, search_results):
    """
    takes a query and return a formatter prompt for the llm to answer the query
    """
    prompt_template = """
You're a pharmaceutical assistant bot. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""

    for doc in search_results:
        context = context + f"question: {doc['Question']}\nanswer: {doc['Answer']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt):
    """
    invokes an llm using the OpenAI
    """

    response = client.chat.completions.create(
        model="meta.llama3-1-8b-instruct-v1:0",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def rag(query, es_client, index_name):
    """
    complete pipeline for the steps to retrieve and generate using the query/keywords
    """
    search_results = elastic_search(query, es_client, index_name)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


def elastic_vector_search(query_vector, es_client, index_name):
    """
    performs a search on the elasticsearch index using the provided query vector
    """
    query_dict = {
        "field": "QuestionVector",
        "query_vector": query_vector,
        "k": 5,
        "num_candidates": 10000,
    }

    response = es_client.search(index=index_name, knn=query_dict, size=5, from_=0)

    result_docs = []

    # since we are not interested in the vectors, we remove them from the search results
    # and just return the documents
    for hit in response["hits"]["hits"]:
        hit["_source"].pop("QuestionVector")
        hit["_source"].pop("AnswerVector")
        result_docs.append(hit["_source"])

    return result_docs


def rag_vector(query, query_vector, es_client, index_name):
    """
    complete pipeline for the steps to retrieve and generate using the embeddings
    """
    search_results = elastic_vector_search(query_vector, es_client, index_name)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
