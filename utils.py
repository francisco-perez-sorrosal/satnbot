import re
from io import StringIO
from typing import BinaryIO, List, cast

import openai
from cleantext import clean
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import FileOrName, open_filename


# TODO Remove if dependency on langchain stays
def pdf2text(pdf_file: FileOrName):
    # pdfminer Boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    with open_filename(pdf_file, "rb") as fp:
        fp = cast(BinaryIO, fp)
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)

    # Get text from StringIO and remove nonprintable characters and trailing stuff
    text = sio.getvalue()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text).rstrip()

    device.close()
    sio.close()

    return text


def chunk_text(text: str, chunk_len: int = 256, do_overlap: bool = False, overlap_size=15) -> List[str]:
    # Split text into smaller chunks
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_len])
        if do_overlap:
            i = i + chunk_len - overlap_size
        else:
            i = i + chunk_len
    return chunks


def clean_text(text: str, lang: str = "en") -> str:
    cleaned_text = clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        normalize_whitespace=True,
        no_line_breaks=False,
        strip_lines=True,
        keep_two_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        no_emoji=True,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        replace_with_punct="",
        lang=lang,
    )
    return cleaned_text


def get_gai_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    print("HERE")
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # model's ramdomness degree for output
    )
    print("HERE too")
    return response.choices[0].message["content"]
