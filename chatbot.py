import json

import openai

from tools.toolkit import toolkit
from tools.utils import docsearch_tool, retrieve_tool
from vdb.vdb_insert import insert


def get_response(messages):

    response = openai.chat.completions.create(
        model="gpt-4o", messages=messages, tools=toolkit
    )

    if response.choices[0].finish_reason == "tool_calls":

        tool_call = response.choices[0].message.tool_calls[0]
        function = tool_call.function
        arguments = json.loads(function.arguments)
        tool_name = function.name

        query = arguments.get("query")

        if tool_name == "retrive_data":
            data = retrieve_tool.retrieve_data_qdrant(query=query)

        elif tool_name == "docs_search":
            data = docsearch_tool.docs_search(query=query)
            insert(data, "ai_docs")

        function_call_result_message = {
            "role": "tool",
            "content": json.dumps({"Tool output": f"{data}"}),
            "tool_call_id": response.choices[0].message.tool_calls[0].id,
        }

        function_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [response.choices[0].message.tool_calls[0]],
        }

        messages.append(function_call_message)
        messages.append(function_call_result_message)

        response = get_response(messages)

    else:
        response = response.choices[0].message.content

    return response
