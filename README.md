<div align="center">

# Question Answering System for Pharmaceutical Support Channel

</div>

## Problem Overview

The goal is to design a Question Answering system to assist human scientists in a pharmaceutical company by suggesting real-time answers to inquiries regarding medical products

## Approaches

Two approaches have been designed. One uses text-based querying (a.k.a. keyword search) while the other uses vector search (a.k.a. semantic searching) by converting all text (both from the dataset as well as the user query) into embedding vectors using an embedding model.

Why the improved model works better than the baseline?  
This is because the user might not enter the same keywords as the ones present in the database. For example, if the database uses the word hypertension and the user enters ‘high blood pressure’, the keyword based search won’t be relevant enough.

## Stack Used

The following tools have been used to build the models.

- datasets library from HuggingFace (and the ‘Jaymax/FDA_Pharmaceuticals_FAQ’ dataset)
- ElasticSearch as the database as well as the query engine for similarity search (this supports both keyword as well as vector-based search.)
- Llama 3.1 as the LLM for generating text
- SentenceTransformers library for the NeuML/pubmedbert-base-embeddings embedding model. This model has been fine-tuned over medical dataset that works well with our use case

## Evaluation

Evaluating a RAG solution requires two things

- a 'golden' dataset where a set of real-world questions and answers are provided
- a SOTA LLM like GPT4o, Claude, or Gemini for evaluating the models.
  Then, a library such as LlamaIndex or Haystack - amongst many - can be used for evaluating these solutions. These evaluate how well the retrieved context match the answer, the context matches the query, etc.

## Steps

### Install Taskfile

```
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
```

### Run ElasticSearch Container

```
task run
```

### Install Libraries

```
pip install -r requirements.txt
```

## Code

To view the baseline model, see [baseline.ipynb](./baseline.ipynb).

To view the improved model, see [embed.ipynb](./embed.ipynb).
