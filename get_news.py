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

from playwright.sync_api import sync_playwright
import urllib.parse

def get_top_articles(query, from_date, to_date, num_articles=10):
    """
    Uses Playwright to scrape Google News for articles matching the query and published between from_date and to_date.
    
    Parameters:
      query (str): The search query.
      from_date (str): Start date in 'YYYY-MM-DD'.
      to_date (str): End date in 'YYYY-MM-DD' (exclusive; e.g., '2024-07-01' for June articles).
      num_articles (int): Number of articles to retrieve.
      
    Returns:
      list: A list of dictionaries containing the title and summary of each article.
    """
    # Append date constraints to the query (advanced search syntax)
    query_with_dates = f"{query} after:{from_date} before:{to_date}"
    encoded_query = urllib.parse.quote(query_with_dates)
    url = f"https://news.google.com/search?q={encoded_query}"
    
    articles_data = []
    with sync_playwright() as p:
        # Launch in headed mode so you can see what’s happening.
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        )
        page = context.new_page()
        print(f"Navigating to {url}")
        page.goto(url)
        
        # Wait for articles to load; adjust timeout if needed.
        try:
            page.wait_for_selector("article", timeout=15000)
        except Exception as e:
            print("Timeout or error waiting for articles:", e)
            browser.close()
            return []
        
        # Query all article elements.
        articles = page.query_selector_all("article")
        print(f"Found {len(articles)} articles.")
        
        count = 0
        for article in articles:
            if count >= num_articles:
                break
            try:
                # Titles are usually inside <h3> tags.
                title_el = article.query_selector("h3")
                if not title_el:
                    continue
                title = title_el.inner_text().strip()
                
                # Try extracting a summary from a <span> element.
                summary_el = article.query_selector("span")
                summary = summary_el.inner_text().strip() if summary_el else "No summary available."
                
                articles_data.append({"title": title, "summary": summary})
                count += 1
            except Exception as e:
                print("Error processing an article:", e)
                continue
        
        browser.close()
    return articles_data

if __name__ == "__main__":
    query = "how will nvidia be shaping the future of ai?"
    from_date = "2024-06-01"
    to_date = "2024-07-01"  # Exclusive end date for June
    articles = get_top_articles(query, from_date, to_date)
    
    print(f"Top {len(articles)} news articles for: '{query}' between {from_date} and {to_date}")
    for i, article in enumerate(articles, start=1):
        print(f"{i}. {article['title']}")
        print(article["summary"])
        print()
