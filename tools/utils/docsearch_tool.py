import requests
from bs4 import BeautifulSoup
from googlesearch import search

docs_search_function_description = {
    "name": "docs_search",
    "description": "Search for documentation on web based on the query. Receives a well-written query to be asked on Google Search",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Google search query",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def docs_search(query: str) -> str:
    """ "Search for documentation on web based on the query. Receives a well-written query to be asked on Google Search"""

    print("Searching for docs on google")
    search_result = search(query, num_results=1)
    docs_url = next(search_result)

    documentation = _get_text_from_url(docs_url)

    return documentation


def _get_text_from_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
    else:
        return f"Erro ao acessar a página: {response.status_code}"
