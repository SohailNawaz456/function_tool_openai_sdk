import os
import logging
import random
from dotenv import load_dotenv  # To load environment variables from a .env file
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from rich.console import Console  # For colorful and styled terminal output

# === Suppress noisy logging ===
logging.getLogger("httpx").setLevel(logging.WARNING)  # Only show warnings or errors from httpx

# === Load environment variables ===
load_dotenv()  # Load environment variables from the .env file
set_tracing_disabled(disabled=True)  # Disable tracing for performance or debug simplicity
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Fetch the OpenRouter API key

# === Define a tool function to be used by the agent ===
@function_tool
def karachi_weather(city: str) -> str:
    """
    Get the current weather in Karachi.
    """
    temperature = round(random.uniform(30.0, 38.0), 1)
    return f"The weather in {city} is sunny with a temperature of {temperature}°C."

# === Initialize the OpenRouter Async Client ===
client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# === Create the agent ===
agent = Agent(
    name="My Async Agent",
    instructions=(
        "Always use the karachi_weather tool whenever the user asks about the weather in Karachi. "
        "You are a helpful assistant."
    ),
    model=OpenAIChatCompletionsModel(
        model="mistralai/mistral-small-24b-instruct-2501",
        openai_client=client
    ),
    tools=[karachi_weather]
)

# === Run the agent synchronously with a test prompt ===
result = Runner.run_sync(agent, "what is the weather in karachi?")

# === Print the final result using rich formatting ===
console = Console()
console.print(result.final_output.strip())
