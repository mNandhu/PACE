# PACE

Personality Accentuating Conversational Engine (PACE) is a framework designed to enhance conversational AI systems by integrating personality traits, emotional intelligence, and long term memory (using mem0). It combines a Digital Companion with a Personal Assistant to give an integrated experience.

## Installation

1. Clone the repository

2. Install dependencies:
   ```bash
   uv sync
   ```
3. Configure environment:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add required API keys (see `.env.example` and `src/pace/config/constants.py`).
4. Run the application:
   ```bash
   uv run main.py
   ```

## Configuration

Configuration is managed via the `.env` file and `src/pace/config/constants.py`. For detailed setup of memory backends and LLM providers, refer to:

- Mem0 Vector DBs: https://docs.mem0.ai/components/vectordbs/overview
- Mem0 Embedders: https://docs.mem0.ai/components/embedders/overview
- Mem0 Graph Store: https://docs.mem0.ai/components/graph-store/overview

LLM providers (Gemini, Groq, etc.) require API keys in `.env`.

## Usage

Once configured, start PACE:

```bash
uv run main.py
```

The application will start with the configured personality and memory systems, ready for conversational interaction.

## Project Structure

```
PACE/
├── src/pace/           # Main package
│   ├── config/         # Configuration management
│   ├── graph/          # Graph-based conversation logic
│   ├── llm/            # LLM wrapper and rate limiting
│   ├── memory/         # Memory management
│   └── utils/          # Utility functions
├── Assistant/          # Character definitions and chat logs
├── logs/               # Application logs
└── main.py            # Entry point
```

## Troubleshooting

- **Import errors**: Ensure you're using `uv run` to execute the application
- **API key issues**: Verify all required API keys are set in your `.env` file
- **Memory backend errors**: Check that Milvus, Ollama, and Neo4j services are running
- **Debug mode**: Set `PACE_DEBUG=1` in your `.env` file for verbose logging
