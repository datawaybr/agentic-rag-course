import json

import openai

from tools.toolkit import docs_search, docs_search_function_description

tools = [
    {
        "type": "function",
        "function": docs_search_function_description,
    }
]

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    },
    {"role": "user", "content": "How can I use MLFlow to evaluate LLMs?"},
]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

if response.choices[0].finish_reason == "tool_calls":

    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)

    query = arguments.get("query")

    documentation = docs_search(query=query)

    function_call_result_message = {
        "role": "tool",
        "content": json.dumps({"documentation": documentation}),
        "tool_call_id": response.choices[0].message.tool_calls[0].id,
    }

    messages.append(response.choices[0].message)
    messages.append(function_call_result_message)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    print(response)
