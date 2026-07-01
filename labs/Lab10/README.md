# Lab10: LLM Access via File Bridge

In this lab we use cloud-hosted large language models instead of models running locally on the compute cluster.

Because the university compute cluster blocks direct internet/API access from the notebook environment, the notebook does not call the model provider directly. Instead, we use a file-based bridge:

1. The notebook writes a request as a JSON file into `llm_bridge/requests/`.
2. A local server process running outside the restricted notebook environment reads that file.
3. The server sends the actual API request to the model provider.
4. The server writes the response into `llm_bridge/responses/`.
5. The notebook reads the response file and returns the model output.

This keeps the notebook code simple while working around the cluster restrictions.

## Supported Providers

This lab supports two providers:

- `openai`: for OpenAI-hosted models
- `kiconnect`: for the university-provided KI:connect models

The client interface is the same for both providers, so you can switch providers by changing the `provider` and `model` arguments.

## What You Need

You need:

- a running bridge server on a machine/environment that can access the internet
- an API key for at least one provider

You can provide keys in either of these ways:

1. Export them in your shell environment before starting the server
2. Store them in a local `.env` file in `Lab10`

Examples:

```bash
export KICONNECT_KEY=<your_kiconnect_key>
export OPENAI_API_KEY=<your_openai_key>
python -m utils.llm_server_file
```

or

```env
KICONNECT_KEY=your_kiconnect_key_here
OPENAI_API_KEY=your_openai_key_here
```

You only need the key for the provider you want to use.

## Getting a KI:connect Key

Students can get access to the university models through KI:connect:

1. Log in to KI:connect with your HSBI account.
2. Open `Settings`.
3. Create an API key.
4. Either export it in your shell as `KICONNECT_KEY=...` before starting the server, or put it in `Lab10/.env`.

Screenshots for this process can be added here.

## Using OpenAI Instead

Students can also use their own OpenAI API key if they want to test other or more powerful models.

In that case:

1. Create an OpenAI API key in your OpenAI account.
2. Either export it in your shell as `OPENAI_API_KEY=...` before starting the server, or put it in `Lab10/.env`.
3. Use `provider="openai"` in the notebook client.

## Running the Bridge

Start the server from `labs/Lab10`:

```bash
conda run -n mara python -m utils.llm_server_file
```

If you want to avoid storing keys in a file, export them first in the same shell session:

```bash
export KICONNECT_KEY=<your_kiconnect_key>
conda run -n mara python -m utils.llm_server_file
```

Leave that process running while the notebook is using the bridge.

## Notebook Usage

Example usage from the notebook:

```python
from utils.llm_client_file import LLMClient

client = LLMClient(base_dir="./llm_bridge", timeout_s=60)

client.list_model_ids(provider="kiconnect")

answer = client.prompt(
    "Explain prompting in one sentence.",
    provider="kiconnect",
    model="Mistral Small 4",
)
```

To switch to OpenAI, change the provider and model:

```python
answer = client.prompt(
    "Explain prompting in one sentence.",
    provider="openai",
    model="gpt-4o-mini",
)
```

## Summary

- The notebook cannot access the internet directly.
- The bridge solves this by exchanging JSON files between notebook and local server.
- Both `openai` and `kiconnect` are supported.
- Students can provide API keys either through environment variables or through `Lab10/.env`.
