import requests
from bs4 import BeautifulSoup

def search_sap_help(query):
    # url = 'https://help.sap.com'
    # search_url = f'{url}/search?q={query}'
    
    # try:
    #     response = requests.get(search_url)
    #     response.raise_for_status()
    #     print(f'Successfully accessed {search_url}')
        
    #     # 在这里可以对返回的网页内容进行处理，例如提取相关信息或进行解析操作
    #     # 这里只是简单地打印出网页内容
    #     print(response.text)
        
    # except requests.exceptions.RequestException as e:
    #     print(f'Error accessing {search_url}: {e}')
    query_content = "sap cloud sdk"
    encoded_query = urllib.parse.quote_plus(query_content)
    url = f"https://help.sap.com/http.svc/elasticsearch?area=content&version=&language=en-US&state=PRODUCTION&q={encoded_query}&transtype=standard,html,pdf,others&product=&to=19&advancedSearch=0&excludeNotSearchable=1"
    response = requests.get(url)
    if response.status_code == 200:
        responseJson = response.json()
        if responseJson.get('status') == 'OK':
            #data = response.get('data', {})
            data = responseJson.get('data', {})
            results = data.get('results', [])
            for i, result in enumerate(results, 1):
                snippet_soup = BeautifulSoup(result.get('snippet', 'N/A'), 'html.parser')
                searchResult = f"Result {i}:\nTitle: {result.get('title', 'N/A')}\nDate: {result.get('date', 'N/A')}\nProduct: {result.get('product', 'N/A')}\nURL: https://help.sap.com{result.get('url', 'N/A')}\nSnippet: {snippet_soup.get_text()}\n"
                return searchResult
        else:
            searchResult = "No results found by query from the help.sap.com"
            return searchResult
    else:
        searchResult = "Request failed with status code {response.status_code}"
        return searchResult

if __name__ == "__main__":
    search_sap_help('team member')