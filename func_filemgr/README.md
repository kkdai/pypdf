# File Management with OpenAI Functions

This repository contains a Python script that uses the OpenAI's GPT-3.5-turbo model and Langchain's toolkit to process file management commands.

## Description

The script takes user input for various file management tasks and uses Langchain's file management tools to execute these tasks. It supports a variety of operations, such as moving, copying, reading, writing, deleting files, and listing directories.

The script uses OpenAI's GPT-3.5-turbo model to understand the user's command and the appropriate file management tool to execute the command.

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

Once the script is running, you can input your file management commands. For example, to move a file named 'foo' to a location named 'bar', you would input:

```bash
Question: move file foo to bar
```

To stop the script, use `CTRL+C`.

## Note

The provided code also contains commented-out examples of how to use specific tools like `MoveFileTool` and `DeleteFileTool`. These examples demonstrate how to create an OpenAI function from a tool and use it with a model to predict messages.

For example, to delete all `.md` files from the Document folder:

```bash
Question: delete *.md from Document folder
```

Please note that all the file management tasks will take place in the file system of the machine where the script is running. Use with caution.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
