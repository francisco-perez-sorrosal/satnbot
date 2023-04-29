import logging

import click
import coloredlogs
import pinecone
from langchain.document_loaders import GitbookLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone

# Logging
logger = logging.getLogger("document_loader")
logger.setLevel(logging.DEBUG)
# Create a coloredlogs handler
coloredlogs_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
coloredlogs.install(level="DEBUG", logger=logger, fmt=coloredlogs_format)


@click.group()
def main():
    pinecone.init(environment="us-west1-gcp-free")
    logger.info(f"Pinecone indexes: {pinecone.list_indexes()}")


@main.group()
def index():
    """Index commands."""


@index.command()
@click.option("--name", default=None, help="Your username", type=str)
@click.option("--url", default=None, help="url to scan to create the index from")
@click.option("--vector_size", default=1536, type=int, help="vector size of the index")
@click.option("--chunk_size", default=1000, type=int, help="chunk size to create the index")
def create(name, url, vector_size, chunk_size):
    logger.info("Load langchain loader")
    loader = GitbookLoader(url, load_all_paths=True, base_url=url)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logger.debug(f"Docs:\n{docs}")
    embeddings = OpenAIEmbeddings()

    if len(pinecone.list_indexes()) == 0:
        metadata_config = {"indexed": ["color"]}

        logger.info(f"Creating index {name} in Pinecone with vector size {vector_size}")
        pinecone.create_index(name, dimension=vector_size, metric="cosine", pods=1, pod_type="p1.x1")
    logger.info(f"Storing documents in {name} index")
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=name)
    logger.info(f"Index:\n{docsearch}")


@index.command()
@click.option("--name", default="satnbot-idx", help="Your username", type=str)
@click.option("--query", default="How do you create a pinecone index with langchain", help="Your username", type=str)
def use(name, query):
    embeddings = OpenAIEmbeddings()
    logger.info(f"Retrieving index {name} from Pinecone\nQuery:{query}")
    try:
        docsearch = Pinecone.from_existing_index(name, embeddings)
        docs = docsearch.similarity_search(query)
        logger.info(f"Docs retrieved {len(docs)}")
        for d in docs:
            logger.warn(f"{d.metadata['title']}")
            logger.info(f"{d.page_content}")
    except Exception as e:
        logger.error(f"{e}\n Error for pinecone index {name}!!!")


if __name__ == "__main__":
    main()
