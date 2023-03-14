import os

import openai
from flask import Flask, redirect, render_template, request, url_for

from openai.embeddings_utils import distances_from_embeddings
import openai
import pandas as pd
import numpy as np

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

df=pd.read_csv('~/crawler/try1/processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def answer_question(
    question,
    df=df,
    model="gpt-3.5-turbo",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=500,
    stop_sequence=None
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
        # messages = [
        #     {"role": "system", "content": f"You are a helpful assistant."},
        #     {"role": "user", "content": "How can i see details of any validator?"},
        #     {"role": "assistant",
        #         "content": "To see details of validator visit page https://explore.fetch.ai/validators"},
        #     {"role": "user", "content": "on what basis rewards are paid?"},
        #     {"role": "assistant",
        #         "content": "Rewards are paid on a per-block basis and added to the existing pending rewards"}
        # ]
        # Create a completions using the questin and context
        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=[context, messages],
        #     temperature=0,
        #     stop='\n\n###\n\n'
        # )
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop='\n\n###\n\n',
            model="text-davinci-003",
        )
        # print("response================================================================> ",response)
        return response["choices"][0]["text"].strip()
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




# df.head()
# result = answer_question(
#             df, question="how to To Claim your Rewards Using the Fetch Wallet?", debug=False)
# print("answer =>",result)
        


app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        question = request.form["question"]
        print("question =>",question)
        return redirect(url_for("index", result=answer_question(question, debug=False)))

    result = request.args.get("result")
    return render_template("index.html", result=result)


