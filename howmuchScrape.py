import json
import re

import requests
from bs4 import BeautifulSoup

url = "https://howmuch.one"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
nav_links = soup.find_all("a", class_="nav-link")
headers = [link.text for link in nav_links]
headers = [
    head
    for head in headers
    if ("Donate" not in head and "Calculator" not in head and "HowMuch" not in head)
]
headers = list(set(headers))
all_data = {}
cur_iter = 0
save_interval = 10


def traverse_electronics(page_header):
    global cur_iter
    response = requests.get(url + "/" + page_header)
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all(
        lambda tag: tag.name == "a" and "nav-link" not in tag.attrs.get("class", [])
    )
    links = [link["href"] for link in links if "twitter" not in link.text]
    for link in links:
        print(f"Cur Iter: {cur_iter}; Cur Link: {link}")
        graph_response = requests.get(url + "/" + link + "/price-history")
        graph_soup = BeautifulSoup(graph_response.text, "html.parser")
        script_tag = graph_soup.find(
            "script", text=lambda x: x and "var datasets = " in x
        )
        script_content = script_tag.string
        data_points = re.search(r"data: \[(.*?)\]", script_content, re.DOTALL).group(1)
        pattern = r"\{[^{}]*\}"
        matches = re.findall(pattern, data_points)
        all_nums = []
        for match in matches:
            numbers = re.findall(r"-?\d+", match)
            numbers = [int(num) for num in numbers]
            all_nums.append(numbers)
        all_data[link] = all_nums
        if cur_iter % save_interval == 0:
            with open("data.json", "w") as file:
                json.dump(all_data, file, indent=4)
        cur_iter += 1


for header in headers:
    traverse_electronics(header)

with open("data.json", "w") as file:
    json.dump(all_data, file, indent=4)
