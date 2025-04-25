# Content Agent (No Framework)

A standalone content generation application built with Groq and direct tool integrations.

## Features

- Research topics using DuckDuckGo search
- Generate content for various platforms (Blog, LinkedIn, Twitter, Facebook)
- Preview content in platform-specific formats
- Command-line and web interfaces

## Requirements

- Python 3.8+
- Groq API key
- (Optional) Environment variables for finer control

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd examples/agno_ex/content_agent_noframework
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## Usage

### Command-line Interface

Run the CLI version for a simple interface:

```bash
python main.py
```

### Web Interface

Run the web interface for a more user-friendly experience:

```bash
python app.py
```

The web interface will be available at:
- Local URL: http://localhost:7860
- Public URL: A temporary public URL will be provided in the terminal output

## Components

- `models.py`: Data models for the application
- `research.py`: Research tools connecting to DuckDuckGo search
- `writer.py`: Content generation using Groq LLM
- `preview.py`: Preview generation for different platforms
- `main.py`: Core workflow orchestration
- `app.py`: Gradio web interface

## API Keys

### Groq API Key
Sign up at [Groq](https://console.groq.com/) to get your API key.

## Customization

- Add or modify platform templates in `preview.py`
- Adjust prompt templates in `writer.py`
- Modify search parameters in `research.py`

## License

MIT 