# PyPDF: Python-based PDF Analysis with LangChain

`PyPDF` is a project that utilizes `LangChain` for learning and performing analysis on PDF documents. It uses a combination of tools such as `PyPDF`, `ChromaDB`, `OpenAI`, and `TikToken` to analyze, parse, and learn from the contents of PDF documents.

## Installation

Before you can use `PyPDF`, ensure you have the required dependencies installed. You can install them by running the following commands in your terminal:

```bash
pip install langchain
pip install pypdf
pip install chromadb
pip install openai
pip install tiktoken
```

## Setting Up Your Environment

In order to use `OpenAI` with this project, you'll need to export your `OpenAI` API key to your environment variables. You can do this by executing the following command in your terminal:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Please replace `"your_openai_api_key"` with your actual `OpenAI` API key.

## Running Your First Analysis

To get started with your first analysis, you'll need a PDF file. For this purpose, we will use the Bitcoin white paper, which you can download by running the following command in your terminal:

```bash
curl -o paper.pdf https://bitcoin.org/bitcoin.pdf
```

Once you've downloaded the PDF, you're ready to start using `PyPDF`!

---

For a detailed walkthrough of the project, please refer to our full documentation, linked in the repository. Enjoy exploring `PyPDF` and feel free to raise any issues or contribute to this project.
