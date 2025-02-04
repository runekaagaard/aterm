import os
import json
import asyncio
import sys
import readline
import argparse
import importlib.util
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

from langgraph.checkpoint.memory import MemorySaver

from langchain_mcp_tools import convert_mcp_to_langchain_tools

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

def p(*args, **kwargs):
    kwargs["flush"] = True
    print(*args, **kwargs)

### App ###

def load_tools_from_file(filepath: str) -> List:
    """Dynamically import tools from a Python file."""
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, '__all__'):
        raise ValueError(f"Tool file {filepath} must define __all__")

    return [getattr(module, name) for name in module.__all__]

async def init(mcp_config_file: str = None, tool_files: List[str] = None):
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=os.environ["ANTHROPIC_API_KEY"])

    tools = []
    if mcp_config_file:
        with open(mcp_config_file) as f:
            config = json.load(f)
            if 'mcpServers' not in config:
                raise ValueError("Config file must contain mcpServers key")
            mcp_tools, cleanup = await convert_mcp_to_langchain_tools(config['mcpServers'])
            tools.extend(mcp_tools)
    else:
        cleanup = None

    if tool_files:
        for tool_file in tool_files:
            tools.extend(load_tools_from_file(tool_file))

    agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
    return agent, cleanup

async def invoke(query, agent):
    config = {"configurable": {"thread_id": "42"}}
    prev_event = None
    async for event in agent.astream_events({"messages": [HumanMessage(content=query)]}, version="v1",
                                            config=config):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if not content:
                continue

            for message in content:
                assert message["type"] in {"tool_use", "partial_json", "text"}
                if message["type"] == "text":
                    text = message["text"]

                    if prev_event and prev_event != "on_chat_model_stream":
                        text = "\n\n" + text.lstrip()

                    p(f"{text}", end="")
        elif event["event"] == "on_tool_start":
            signature = ", ".join("=".join((k, repr(v))) for k, v in event['data'].get('input', {}).items())
            p(f"\n\ntool call: {event['name']}({signature})")
            approval = input("approve? (y/n): ").lower().strip()
            if approval != 'y':
                p("tool call cancelled")
                return
        elif event["event"] == "on_tool_end":
            try:
                tool_result = event['data']['output'].content

            except (KeyError, TypeError, AttributeError):
                tool_result = f"Event data error: {event}"
            p(f"{event['name']} tool result: {tool_result}", end="")
        else:
            pass
        if event["event"] in {"on_chat_model_stream", "on_tool_start", "on_tool_end"}:
            prev_event = event["event"]

    p()

async def get_query():
    session = PromptSession()
    while True:
        with patch_stdout():
            query = await session.prompt_async('> ', prompt_continuation=lambda *a, **kw: "", multiline=True)

            return query.strip()

async def chat(mcp_config: str = None, tool_files: List[str] = None):
    cleanup = None

    try:
        agent, cleanup = await init(mcp_config, tool_files)
        while True:
            query = await get_query()
            p()
            if query:
                await invoke(query, agent)
                p()
    finally:
        if cleanup is not None:
            await cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A terminal LLM chat app with support for langchain tools and mcp servers')
    parser.add_argument('--mcp-config', type=str, help='Path to MCP config JSON file')
    parser.add_argument('--langchain-tools', type=str, action='append', help='Path to langchain tools Python file')

    args = parser.parse_args()
    asyncio.run(chat(args.mcp_config, args.langchain_tools))
