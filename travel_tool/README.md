# TravelPOI with OpenAI Functions

This repository contains a Python script that uses the OpenAI's GPT-3.5-turbo model and Langchain's toolkit to provide travel Points of Interest (POI) suggestions.

## Description

The script takes user input requesting travel suggestions and uses Langchain's TravelPOITool to provide responses.

The script uses OpenAI's GPT-3.5-turbo model to understand the user's request and the TravelPOITool to generate a suitable response.

## Prerequisites

Ensure you have Python 3 and pip installed. The script also requires the following Python packages:

- `openai`
- `langchain` version `0.0.200`. You can install it with this command:

  ```bash
  pip install --upgrade --force-reinstall langchain==0.0.200
  ```

## Usage

Run the script using Python from the command line:

```bash
python main.py
```

Once the script is running, you can input your travel questions. For example, to find points of interest in Paris, you would input:

```bash
Question: What are some points of interest in Paris?
```

To stop the script, use `CTRL+C`.

## Note

Please note that the quality of the suggestions can vary and depend on the underlying model's understanding of the query.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
