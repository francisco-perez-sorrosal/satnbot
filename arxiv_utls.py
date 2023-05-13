import os
from urllib.request import urlretrieve

import arxiv
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from prompts import ARXIV_MD_SUMMARIZATION_PROMPT, ARXIV_SUMMARIZATION_PROMPT
from utils import get_gai_completion


def download_pdf_from_arxiv(arxiv_id: str, filename="downloaded-paper.pdf"):
    paper = next(arxiv.Search(id_list=[arxiv_id]).results())
    dest_file = paper.download_pdf(filename=filename)
    return paper, dest_file


def extract_text_from_arxiv_pdf(arxiv_id_or_url, chars_per_chunk, overlap_chars):
    if arxiv_id_or_url.startswith("http"):
        print(f"Downloading paper from {arxiv_id_or_url}... as downloaded-paper.pdf")
        path = os.path.join("./", "downloaded-paper.pdf")
        dest_file, _ = urlretrieve(arxiv_id_or_url, path)
        title = "N/A"
    else:
        paper, dest_file = download_pdf_from_arxiv(arxiv_id_or_url)
        title = paper.title
    loader = PyPDFLoader(dest_file)
    test_splitter = RecursiveCharacterTextSplitter(  # Set a really small chunk size, just to show.
        chunk_size=chars_per_chunk, chunk_overlap=overlap_chars, length_function=len
    )
    return title, loader.load_and_split(text_splitter=test_splitter)


def summarize_arxiv_paper(text, style, style_items, language):
    if style == "paragraph":
        summary_style = f"{style_items} paragraph"
    elif style == "bulletpoints":
        summary_style = f"{style_items} bullet points"
    elif style == "sonnet":
        summary_style = "a sonnet style"
    else:
        summary_style = "a single sentence"

    prompt = f""""
        Summarize the technical text, delimited by triple
        backticks, in {language} in {summary_style}:
        ```{text}```
        """

    return get_gai_completion(prompt)


def summarize_arxiv_paper_lc(docs, in_depth, style, style_items, language):
    args = {"input_documents": docs, "language": language}

    PROMPT = ARXIV_MD_SUMMARIZATION_PROMPT

    if not in_depth:
        PROMPT = ARXIV_SUMMARIZATION_PROMPT

        if style == "paragraph":
            summary_style = f"{style_items} paragraph"
        elif style == "bulletpoints":
            summary_style = f"{style_items} bullet points"
        elif style == "sonnet":
            summary_style = "a sonnet style"
        else:
            summary_style = "a single sentence"

        args["summary_stype"] = summary_style

    chain = load_summarize_chain(
        OpenAI(temperature=0, max_tokens=1049),
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=PROMPT,
        combine_prompt=PROMPT,
    )
    output = chain(args, return_only_outputs=True)
    print(output)
    if not in_depth:
        return output["output_text"]
    else:
        return f'{output["output_text"]}\n\n{output["intermediate_steps"]}'
