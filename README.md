# 🌤️ Async Weather Agent (Karachi Weather Bot)

This is an AI-powered chatbot built using the [Agent SDK](https://github.com/OpenPipe/agent-sdk) that responds to weather-related queries — specifically for **Karachi**. It demonstrates how to create a custom agent with a tool function using `AsyncOpenAI`, `OpenRouter`, and `function_tool`.

---

## 🚀 Features

- Uses **OpenRouter API** for LLM completions
- Asynchronous agent for efficient interactions
- Custom tool: `karachi_weather` for mocked weather data
- Clean and colorful output using `rich` library
- Loads API keys from a `.env` file

---

## 🧠 Tech Stack

- Python 3.10+
- [OpenPipe Agent SDK](https://github.com/OpenPipe/agent-sdk)
- AsyncOpenAI (via OpenRouter)
- Rich (for terminal formatting)
- dotenv (for environment variables)

---

## 🔧 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/async-karachi-weather-agent.git
cd async-karachi-weather-agent
