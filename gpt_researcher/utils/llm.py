# libraries
from __future__ import annotations
import json
from fastapi import WebSocket
from langchain.adapters import openai as lc_openai
from colorama import Fore, Style
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from gpt_researcher.master.prompts import auto_agent_instructions


async def create_chat_completion(
        messages: list,  # type: ignore
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        llm_provider: Optional[str] = None,
        stream: Optional[bool] = False,
        websocket: WebSocket | None = None,
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.
        stream (bool, optional): Whether to stream the response. Defaults to False.
        llm_provider (str, optional): The LLM Provider to use.
        webocket (WebSocket): The websocket used in the currect request
    Returns:
        str: The response from the chat completion
    """

    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 8001:
        raise ValueError(f"Max tokens cannot be more than 8001, but got {max_tokens}")

    # create response
    for attempt in range(10):  # maximum of 10 attempts
        if llm_provider=="google":
            response = await send_google_chat_completion_request(
                messages, model, temperature, max_tokens, stream, llm_provider, websocket
            )
        else:
            response = await send_oepnai_chat_completion_request(
                messages, model, temperature, max_tokens, stream, llm_provider, websocket
            )
        
        return response

    if llm_provider=="google":
        logging.error("Failed to get response from Gemini API")
        raise RuntimeError("Failed to get response from Gemini API")
    else:
        logging.error("Failed to get response from OpenAI API")
        raise RuntimeError("Failed to get response from OpenAI API")


import logging

def convert_messages(messages):
    """
    The function `convert_messages` converts messages based on their role into either SystemMessage or
    HumanMessage objects.
    
    :param messages: It seems like the code snippet you provided is a function called `convert_messages`
    that takes a list of messages as input and converts each message based on its role into either a
    `SystemMessage` or a `HumanMessage`. However, the definition of `SystemMessage` and `HumanMessage`
    classes
    :return: The `convert_messages` function is returning a list of converted messages where each
    message is an instance of either `SystemMessage` or `HumanMessage` based on the role specified in
    the input messages.
    """
    converted_messages = []
    for message in messages:
        if message["role"] == "system":
            converted_messages.append(SystemMessage(content=message["content"]))
        elif message["role"] == "user":
            converted_messages.append(HumanMessage(content=message["content"]))
            
    return converted_messages

async def send_google_chat_completion_request(
        messages, model, temperature, max_tokens, stream, llm_provider, websocket
):
    if not stream:
        llm = ChatGoogleGenerativeAI(model=model, convert_system_message_to_human=True, temperature=temperature, max_output_tokens=max_tokens)
        converted_messages = convert_messages(messages)
        result = llm.invoke(converted_messages)
        
        return result.content
    else:
        return await stream_response(model, messages, temperature, max_tokens, llm_provider, websocket)
    

async def send_oepnai_chat_completion_request(
        messages, model, temperature, max_tokens, stream, llm_provider, websocket
):
    if not stream:
        result = lc_openai.ChatCompletion.create(
            model=model,  # Change model here to use different models
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=llm_provider,  # Change provider here to use a different API
        )
        return result["choices"][0]["message"]["content"]
    else:
        return await stream_response(model, messages, temperature, max_tokens, llm_provider, websocket)


async def stream_response(model, messages, temperature, max_tokens, llm_provider, websocket=None):
    async def send_output(output):
        if websocket is not None:
            await websocket.send_json({"type": "report", "output": output})
        else:
            print(f"{Fore.GREEN}{output}{Style.RESET_ALL}")

    async def stream_chunks(llm, messages):
        async for chunk in llm.stream(convert_messages(messages)):
            yield chunk

    paragraph = ""
    response = ""

    if llm_provider == "google":
        llm = ChatGoogleGenerativeAI(
            model=model, 
            convert_system_message_to_human=True, 
            temperature=temperature, 
            max_output_tokens=max_tokens
        )
    else:
        llm = lc_openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=llm_provider,
            stream=True,
        )

    async for chunk in stream_chunks(llm, messages):
        content = chunk if llm_provider == "google" else chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            response += content
            paragraph += content
            if "\n" in paragraph:
                await send_output(paragraph)
                paragraph = ""

    return response


def choose_agent(smart_llm_model: str, llm_provider: str, task: str) -> dict:
    """Determines what server should be used
    Args:
        task (str): The research question the user asked
        smart_llm_model (str): the llm model to be used
        llm_provider (str): the llm provider used
    Returns:
        server - The server that will be used
        agent_role_prompt (str): The prompt for the server
    """
    try:
        response = create_chat_completion(
            model=smart_llm_model,
            messages=[
                {"role": "system", "content": f"{auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {task}"}],
            temperature=0,
            llm_provider=llm_provider
        )
        agent_dict = json.loads(response)
        print(f"Agent: {agent_dict.get('server')}")
        return agent_dict
    except Exception as e:
        print(f"{Fore.RED}Error in choose_agent: {e}{Style.RESET_ALL}")
        return {"server": "Default Agent",
                "agent_role_prompt": "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."}
