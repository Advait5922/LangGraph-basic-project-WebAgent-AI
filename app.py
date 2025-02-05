from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults  # TavilySearchResults tool for handling search results from Tavily
import os  # os module for environment variable handling
from langgraph.prebuilt import create_react_agent  # Function to create a ReAct agent
from langchain_groq import ChatGroq  # ChatGroq class for interacting with LLMs
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Predefined list of supported model names
MODEL_NAMES = [
    "llama3-70b-8192",  # Model 1: Llama 3 with specific configuration
    "mixtral-8x7b-32768"  # Model 2: Mixtral with specific configuration
]

# Initializing the TavilySearchResults tool with a specified maximum number of results.
tool_tavily = TavilySearchResults(max_results=10)

# Combining the TavilySearchResults and ExecPython tools into a list.
tools = [tool_tavily, ]

# FastAPI application setup with a title
app = FastAPI(title='LangGraph AI Agent')

# Defining the request schema using Pydantic's BaseModel
class RequestState(BaseModel):
    model_name: str  # Name of the model to use for processing the request
    system_prompt: str  # System prompt for initializing the model
    messages: List[str]  # List of messages in the chat

# Defining an endpoint for handling chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically selects the model specified in the request.
    """
    if request.model_name not in MODEL_NAMES:
        # Returning an error response if the model name is invalid
        return {"error": "Invalid model name. Please select a valid model."}

    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=request.model_name)
    agent = create_react_agent(llm, tools=tools, state_modifier=request.system_prompt)

    # Creating the initial state for processing
    state = {"messages": request.messages}

    # Processing the state using the agent
    result = agent.invoke(state)

    # Returning the result as the response
    return result

# Running the application if executed as the main script
if __name__ == '__main__':
    import uvicorn  # Import Uvicorn server for running the FastAPI app
    uvicorn.run(app, host='127.0.0.1', port=8000)  # Start the app on localhost with port 8000