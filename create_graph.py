from openai import OpenAI
import os
import json
from get_news import get_titles
from pyvis.network import Network
from create_graph_functions import create_graph

# api

client = OpenAI()

prompt = "what is the news about apple?"

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": 
         """Isolate the ticker symbols that are most likely associated with the prompt. 
Return the output in the form of a python list with nothing else."""},
        {
            "role": "user",
            "content": prompt
        }
    ]
)

response = completion.choices[0].message.content
tickers = eval(response[response.index("["):response.rindex("]")+1])
seen = set(tickers)
news = {}
max_tickers = 5
while tickers:
    print(tickers)
    ticker = tickers.pop(0)
    news_temp = get_titles(ticker)
    if not news_temp:
        continue
    news_string = "\n[ARTICLE END]\n".join(news_temp) + "\n[ARTICLE END]"
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": 
            """Create a graph based on the news data. There should be multiple lines of output. 
The first line should be two words separated by a space character representing either the important ticker symbols or 
entities (countries, organizations, important products, etc.). If a subject has multiple words, use hyphens to concatenate.
The second line should be a paragraph description of the edge that connects the two subjects.
Aim for a graph with as many edges as possible (on the order of tens) and follow this format of output.
The goal is for each subject (vertex) to be used in multiple edges to create a well connected graph. 
Nothing else besides this two line format."""},
            {
                "role": "user",
                "content": news_string
            }
        ]
    )
    graph_contents = completion.choices[0].message.content
    graph_contents_list = graph_contents.split("\n")
    graph_contents_list = [i.strip() for i in graph_contents_list if i]
    for i in range(0,len(graph_contents_list),2):
        if i == len(graph_contents_list) - 1:
            break
        subjects = graph_contents_list[i].split()
        description = graph_contents_list[i+1]
        subject1, subject2 = subjects[0], "".join(subjects[1:])
        subject2 = subject2 if subject2 else "UNKNOWN"
        key = tuple(sorted([subject1, subject2]))
        if key in news:
            news[key] = news[key] + "\n" + description
        else:
            news[key] = description
    
    if len(seen) < max_tickers:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": 
                """Isolate the ticker symbols that are most likely associated with the prompt. 
    Return the output in the form of a python list with nothing else. No preamble."""},
                {
                    "role": "user",
                    "content": graph_contents
                }
            ]
        )

        response = completion.choices[0].message.content
        tickers_temp = eval(response[response.index("["):response.rindex("]")+1])
        for j in range(len(tickers_temp)):
            if tickers_temp[j] not in seen:
                tickers.append(tickers_temp[j])
                seen.add(tickers_temp[j])
print(news)

create_graph(news, prompt, output_file="graph.html", report_file="report.html", client=client)