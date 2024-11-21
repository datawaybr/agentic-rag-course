from tools.utils.docsearch_tool import docs_search_function_description
from tools.utils.retrieve_tool import retrieve_data_function_description

toolkit = [
    {
        "type": "function",
        "function": docs_search_function_description,
    },
    {"type": "function", "function": retrieve_data_function_description},
]
