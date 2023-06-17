# Stock Query with OpenAI Functions

This repository provides a Python script that uses OpenAI's GPT-3.5-turbo model and the Langchain toolkit to handle queries about stock prices.

## Description

This script takes a user's input, interprets it using OpenAI's GPT-3.5-turbo model, and employs a custom tool, `StockPriceTool`, to fetch and display the requested stock price information. The tool extracts the necessary information from the user's question, fetches the requested data, and presents it back to the user.

## Prerequisites

Make sure you have Python 3 and pip installed on your machine. This script also requires these Python packages:

- `openai`
- `langchain` version `0.0.200`. Install it with:

  ```bash
  pip install --upgrade --force-reinstall langchain==0.0.200
  ```

## Usage

Run the script using Python:

```bash
python main.py
```

When the script is running, you can input stock-related queries. For example:

```bash
Question: What is the current price of Tesla stock?
```

To quit the script, use `CTRL+C`.

## Note

Please remember that the effectiveness and accuracy of the stock price information provided by the tool depends on the tool's implementation and the data source it uses to fetch stock prices.

## Contributing

If you want to contribute to this project, feel free to open a pull request or an issue to discuss what you'd like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
