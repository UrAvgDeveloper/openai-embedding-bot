import shutil
import json
import re
import os
import openai
import pandas as pd
import tiktoken
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import time

class ChunkFullError(Exception):
    pass

class Done(Exception):
    pass
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
source_file = '~/crawler/try1/processed/output.csv'
destination_dir = '~/crawler/try1/processed_stack'
current_timestamp = int(time.time())
new_filename = '% s_output.csv'%current_timestamp
sub = {}
head = {}
obj = {}

root_dir = '~/crawler/docs'
currentPath = '../docs/docs/basics/staking/how_to_stake.md'

md_files = []
Quit_flag = False
# Files to ignore while traversing
ignoreFile = ['README.md', 'CODE_OF_CONDUCT.md']

tokenizer = tiktoken.get_encoding("cl100k_base")

max_tokens = 500


def findAllMDPaths(root):
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.md') and file not in ignoreFile:
                md_files.append(os.path.relpath(
                    os.path.join(subdir, file), start=os.curdir))


def getTitle(md_text):
    title_pattern = r'^# (.*)$'
    title_match = re.search(title_pattern, md_text, re.MULTILINE)
    if(title_match == None):
        return getCurrentFileName()
    return title_match.group(1)


def getHeadings(md_text):
    heading_pattern = r'^## (.*)$'
    return re.findall(heading_pattern, md_text, re.MULTILINE)


def writeToJson(data, name):
    with open(name, 'w') as f:
        json.dump(data, f, indent=2)


def getCleanTextFromMd(path):
    with open(path, 'r') as f:
        text = f.read()
        text = removeCodeBlocks(convertLinks(removeImages(text)))
        return text


def getCurrentFileName():
    currentFileName = os.path.basename(currentPath)
    return (os.path.splitext(currentFileName))[0]


def setCurrentPath(path):
    global currentPath
    currentPath = path


def getSubHeadingContent(text, subHeading1, subHeading2):
    startIndex = text.rindex(subHeading1) + len(subHeading1)
    endIndex = len(text) if subHeading2 == None else text.index(
        subHeading2) - 4
    subContent = text[startIndex:endIndex]
    return subContent


def removeCodeBlocks(text):
    return re.sub('`.*?`', '', text, flags=re.DOTALL)


def getHeadingContent(text, heading1, heading2):
    startIndex = text.rindex(heading1) + len(heading1)
    endIndex = len(text) if heading2 == None else (text.index(heading2) - 3)
    content = text[startIndex:endIndex]
    # check for subheading
    subHeadings = checkForSubHeading(content)
    # print(subHeadings)

    if len(subHeadings) != 0:
        head[heading1] = ""
        for i in range(0, len(subHeadings)):
            subContent = getSubHeadingContent(
                content, subHeadings[i], subHeadings[i+1] if i+1 < len(subHeadings) else None)
            # sub[subHeadings[i]] = subContent
            head[heading1] = head[heading1] + "," + \
                subHeadings[i] + ":" + subContent
    else:
        head[heading1] = content.strip()


def checkForSubHeading(text):
    sub_heading_pattern = r'^### (.*)$'
    sub_headings = re.findall(sub_heading_pattern, text, re.MULTILINE)
    return sub_headings


def checkForSubSubHeading(text):
    sub_heading_pattern = r'^#### (.*)$'
    sub_headings = re.findall(sub_heading_pattern, text, re.MULTILINE)
    # print(sub_headings)
    return sub_headings


def removeImages(text):
    pattern = r"!\[.*\]\(.*\)"
    return re.sub(pattern, "", text)


def convertLinks(text):
    # Regex pattern to match links in markdown format
    pattern = r"\[([^\]]+)\]\(([^\)]+)\)"

    # Replace each match with the desired format
    converted_text = re.sub(pattern, r"\1 at \2", text)

    return converted_text


def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = re.sub(r'\s{2,}', ' ', serie.strip())
    return serie


def getTitleContent(text, heading1):
    startIndex = 0
    endIndex = len(text) if heading1 == None else (text.index(heading1) - 3)
    content = text[startIndex:endIndex]
    return content


def split_into_many(text, max_tokens=max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence))
                for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


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


def answer_question(
    df,
    question,
    model="gpt-3.5-turbo",
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
            stop=stop_sequence,
            model=COMPLETIONS_MODEL,
        )

        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


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


def processAllHeadings():
    # Find all md file path
    processed_files = set()
    if os.path.exists('processed_files.json'):
        with open('processed_files.json', 'r') as f:
            processed_files = set(json.load(f))
    findAllMDPaths(root_dir)
    flag = 0
    count = 0
    for path in md_files:
        JSONOBJ = {}
        global sub, head
        sub = {}
        head = {}
        if count == 10:
            break
            # raise ChunkFullError("The chunk is full, cannot add another item.")
        if path in processed_files:
            # print("Skipping %s (already processed)" % path)
            continue
        
        print("Processing % s" % path)
        setCurrentPath(path)
        md_text = getCleanTextFromMd(currentPath)
        headings = getHeadings(md_text)
        # print(headings)
        head[getTitle(md_text)] = getTitleContent(
            md_text, None if len(headings) == 0 else headings[0])
        if len(headings) == 0 :
            JSONOBJ[getTitle(md_text)] = head
        for i in range(0, len(headings)):
            JSONOBJ[getTitle(md_text)] = head
            getHeadingContent(
                md_text, headings[i], headings[i+1] if i+1 < len(headings) else None)

        # Convert JSON to dataframe
        df = pd.DataFrame.from_dict(JSONOBJ, orient='index')
        df = df.reset_index()
        df = pd.melt(df, id_vars=['index'])
        df['variable'] = df.variable + ". " + \
            df['value'].apply(remove_newlines)  # remove_newlines(df.value)
        df = df.iloc[:, :-1]

        # Write dataframe to CSV file
        if flag == 0:
            df.to_csv('processed/output.csv', index=False,
                      header=['title', 'text'])
            flag = 1
        else:
            df.to_csv('processed/output.csv', mode='a',
                      index=False, header=False)
        count+=1
        processed_files.add(path)  

    if len(processed_files) == len(md_files):
            print("All files have been processed.")
            global Quit_flag
            Quit_flag = True
            raise Done("\n\ndone processing all files.exiting......")
    

    df = pd.read_csv('processed/output.csv', index_col=0)
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    print("creating embeddings...")
    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(
        input=x, engine=EMBEDDING_MODEL)['data'][0]['embedding'])
    df.to_csv('processed/embeddings.csv', mode='a', header=False)

    print('\nDone.\n')
    flag = 0
    current_timestamp = int(time.time())
    new_filename = '% s_output.csv'%current_timestamp
    shutil.copy2(source_file, f'{destination_dir}/{new_filename}')
    print('output.csv added to processed_stack')
    with open('processed_files.json', 'w') as f:
        json.dump(list(processed_files), f)
    print("Files processed => % s"%len(processed_files) )

    
    raise ChunkFullError("\nPicking other 10 files...")


# processAllHeadings()
while not Quit_flag:
    try:
        if Quit_flag :
            break
        processAllHeadings()
    except KeyboardInterrupt:
        print("Process terminated by user.")
        break
    except Done:
        print("Process terminated by system.")
        break
    except ChunkFullError as e:
        print("% s\n\n"%e)
        continue
    except Exception as e:
        print(f"Error occurred: {e}")
        continue