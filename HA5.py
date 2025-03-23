from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI


def summarize_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary


url = "https://itcareerhub.de/ru"
summary = summarize_url(url)
print(summary)