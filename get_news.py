# import requests

# def get_top_articles(query, api_key, page_size=10):
#     """
#     Retrieves the top news articles related to the given query.
    
#     Parameters:
#         query (str): The search query.
#         api_key (str): Your NewsAPI API key.
#         page_size (int): Number of articles to return.
        
#     Returns:
#         list: A list of dictionaries, each containing the article title and a one-paragraph summary.
#     """
#     url = "https://newsapi.org/v2/everything"
#     params = {
#         "q": query,
#         "pageSize": page_size,
#         "sortBy": "relevancy",
#         "language": "en"
#     }
    
#     headers = {
#         "Authorization": api_key
#     }
    
#     response = requests.get(url, params=params, headers=headers)
    
#     if response.status_code != 200:
#         print(f"Error: Received status code {response.status_code}")
#         print("Response:", response.text)
#         return []
    
#     data = response.json()
#     articles = data.get("articles", [])
#     result = []
#     for article in articles:
#         title = article.get("title")
#         summary = article.get("description")
#         if not summary:
#             summary = "No summary available."
#         result.append({"title": title, "summary": summary})
#     return result

# if __name__ == "__main__":
#     # Replace with your NewsAPI API key
#     api_key = "61223d221c45451db3a2877bd7fe9c90"
#     prompt = "how will nvidia be shaping the future of ai?"
    
#     print(f"Top 10 news articles for: '{prompt}'\n")
#     articles = get_top_articles(prompt, api_key)
    
#     for idx, article in enumerate(articles, start=1):
#         print(f"{idx}. {article['title']}")
#         print(article["summary"])
#         print()

import bs4 as bs
import requests

def get_endpoint(symbol: str) -> str:
  return f"https://seekingalpha.com/api/sa/combined/{symbol}.xml"

def fetch_rss(symbol: str) -> str:
  response = requests.get(get_endpoint(symbol))

  if (response.status_code != 200):
    print("Failed to fetch RSS feed.")
    return None

  return response.text;

def parse_rss(rss: str) -> list[str]:
  soup = bs.BeautifulSoup(rss, "xml")
  items = soup.find_all("item")
  titles = [item.title.text for item in items]
  dates = [item.pubDate.text[item.pubDate.text.find(", ")+2:item.pubDate.text.find(", ")+13] for item in items]
  return [titles[i] + " " + dates[i] for i in range(len(titles))]

def get_titles(symbol: str) -> list[str]:
  rss = fetch_rss(symbol)
  if not rss:
    return None
  titles = parse_rss(rss)
  return titles

# # Example Usage
# titles = get_titles("AAPL")
# print(get_titles("AAPL"))