from semantic_kernel.contents import (
    FunctionCallContent,
    FunctionResultContent,
    StreamingTextContent,
)
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from openai import AsyncOpenAI


from WebsiteReaderPlugin import WebsiteChatbotPlugin
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import uvicorn
import json
import html
import os


load_dotenv()

# Emoji icons for accessibility and readability
ICONS = {"user": "👤", "agent": "🤖", "function": "🛠️", "result": "📄"}

# Dummy imports: Replace with your actual imports
# from your_agent import agent, ChatHistoryAgentThread, FunctionCallContent, FunctionResultContent, StreamingTextContent

app = FastAPI()
agent = None  # Define agent as a global variable
# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.on_event("startup")
async def startup():
    global agent  # Declare agent as global to modify it
    endpoint = "https://models.inference.ai.azure.com/"
    # endpoint = "https://models.github.ai/inference"
    model = "gpt-4o"  # "openai/gpt-4.1"
    embeddings_model = "text-embedding-3-large"  # "text-embedding-3-small"
    token = os.environ["OPENAI_API_KEY"]
    client = AsyncOpenAI(base_url=endpoint, api_key=token)

    #   Create an AI Service that will be used by the `ChatCompletionAgent`
    chat_completion_service = OpenAIChatCompletion(
        ai_model_id=model, async_client=client
    )

    AGENT_INSTRUCTIONS = """
    You are a helpful AI Agent that can assist users in finding relevant information and resources within a dataplattform documentation website.
    
    Important: When users asks a question and needs help with information, start by providing information related to the sources from the website. 
    Only suggest general resources when an answer is not relevant to the information from the website. prioritize user preferences. 
    Your goal is to provide accurate and helpful information based on the content of the website and even help generating documentation to missing sections when asked for it.
    """

    wcp = WebsiteChatbotPlugin(client=client, embeddings_model=embeddings_model)
    await wcp.read_index(filename="website_index.csv")

    agent = ChatCompletionAgent(
        service=chat_completion_service,
        plugins=[wcp],
        name="Dimlbot",
        instructions=AGENT_INSTRUCTIONS,
    )

    return agent


class ChatRequest(BaseModel):
    user_inputs: List[str]


async def chat(request: ChatRequest):
    global agent  # Access the global agent variable


@app.post("/chat")
async def chat(request: ChatRequest):
    global active_thread  # Keeps thread memory across function calls (in-memory only)
    user_input = request.user_inputs[-1]

    async def event_stream(active_thread: ChatHistoryAgentThread | None = None):
        async for response in agent.invoke_stream(
            messages=user_input, thread=active_thread
        ):
            active_thread = response.thread  # Update thread each time
            agent_name = response.name
            for item in response.items:
                if isinstance(item, StreamingTextContent) and item.text:
                    # HTML-escape if you need
                    yield item.text

    return StreamingResponse(event_stream(), media_type="text/plain")


# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
