import os

import openai
from flask import Flask, redirect, render_template, request, url_for

from openai.embeddings_utils import distances_from_embeddings
import openai
import pandas as pd
import numpy as np

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

df = pd.read_csv('~/crawler/try1/processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


def answer_question(
    question,
    df=df,
    model=COMPLETIONS_MODEL,
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=500,
    stop_sequence='\n\n###\n\n'
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        messages = [
            {"role": "system", "content": f"Answer the question based on the {context} only, and if the question can't be answered based on the context, say \"I don't know\""},
            {"role": "user", "content": "how to To Claim your Rewards Using the Fetch Wallet?"},
            {"role": "assistant",
                "content": "1. Ensure you are logged into your Fetch wallet at /basics/wallet/getting_started.2. From the wallet dashboard select **Claim**.3. The wallet shows you a summary of the transaction. Review it, select a transaction fee, and if you are happy, hit **Approve** to complete the operation.You should now see the rewards added to your **Total Balance**."},
            {"role": "user", "content": "list type of blockchains?"},
            {"role": "assistant",
                "content": "We can distinguish among public, private, consortium and hybrid blockchains."},
            {"role": "user", "content": f"{question}"},

        ]
        # Create a completions using the questin and context
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            max_tokens=max_tokens
        )
       
        return response["choices"][0]["message"]["content"].strip()
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
    question, df, max_len=1800, size="ada"
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
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        question = request.form["question"]
        print("question =>", question)
        return redirect(url_for("index", result=answer_question(question, debug=False)))

    result = request.args.get("result")
    return render_template("index.html", result=result)
