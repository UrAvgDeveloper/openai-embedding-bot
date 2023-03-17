import os

import openai
from flask import Flask, redirect, render_template, request, url_for

from openai.embeddings_utils import distances_from_embeddings
import openai
import pandas as pd
import numpy as np

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

df=pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/processed/embeddings.csv", index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def answer_question(
    question,
    df=df,
    max_len=200,
    size="ada",
    debug=False,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        # max_len=max_len,
        # size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=200,
            messages=[
                  {"role": "system", "content": f"You are a helpful assistant. You must strictly answer the question based on the context below, and if the question can't be answered based on the context or is not related to the context below, say \"This is not related to Fetch.ai\" and nothing else. Also share me the link that will give me more information.\n\nContext: {context}"},
                  {"role": "user", "content": "Question: What is fetch.ai"},
                  {"role": "assistant", "content": "Our mission is to build the infrastructure required for developing modern, decentralized and peer-to-peer (P2P) applications that are free from centralized rent-seeking."},
                  {"role": "user", "content": f"Question: {question}"}
              ]
        )
        print("resp", response)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""
    
def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.text) for idx, r in df.iterrows()
    }

def create_context(
    question, df, max_len=500, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = get_embedding(question, model=EMBEDDING_MODEL)

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')
    # print(question)
    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len or row['distances'] > 0.2:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return ". ".join(returns)


app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        question = request.form["question"]
        print("question =>",question)
        return redirect(url_for("index", result=answer_question(question, debug=True)))

    result = request.args.get("result")
    return render_template("index.html", result=result)


