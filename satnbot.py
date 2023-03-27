import csv
import json
import os
import pickle

import discord
import openai
import requests
from bs4 import BeautifulSoup
from cleantext import clean
from discord.ext import commands

from utils import chunk_text, download_from, pdf2text

# Load environment variables
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

model_id = "gpt-3.5-turbo"
# model_id="gpt-4"

intents = discord.Intents.all()
intents.messages = True
intents.message_content = True

history_file = "history.pkl"


class DiscordChatGPT4(commands.Bot):
    def __init__(self, intents):
        super().__init__(intents)
        try:
            print(f"Loading command history from {history_file}")
            with open(history_file, "rb") as file:
                self.history = pickle.load(file)
                print(self.history)
        except FileNotFoundError:
            print("No history!")
            self.history = {}
        # self.history = []

    def save_history(self):
        print(self.history)
        with open(history_file, "wb") as file:
            pickle.dump(self.history, file)

    def add_to_history(self, c):
        self.history[len(self.history) + 1] = c

    def get_history(self):
        return self.history

    async def on_ready(self):
        print("Logged in as")
        print(self.user.name)
        print(self.user.id)
        print("------")

    async def on_message(self, message):
        if message.author == self.user:
            return

        # Check if the message contains a mention of the bot
        bot_mention = f"<@{self.user.id}>"
        if bot_mention not in message.content:
            return

        # Remove the bot mention from the message content
        text_content = message.content.replace(bot_mention, "").strip()

        print(f"User message: {text_content} {type(text_content)}")

        image_content = None
        if message.attachments:
            image_content = message.attachments[0].url

        input_content = [{"role": "user", "content": text_content}]
        if image_content:
            input_content[0]["content"]["image"] = image_content

        completion = openai.ChatCompletion.create(model=model_id, messages=input_content)
        self.add_to_history(input_content)
        self.save_history()  # TODO Improve this

        response = completion.choices[0].message.content
        print(f"Text ({len(response)}): {response}")
        await message.channel.send(response[:2000])


# Create and run DisordChatGPT4 instance
bot = DiscordChatGPT4(intents=intents)


# Commands
@bot.slash_command(name="history")
async def history(ctx):
    print(f"Requesting history")
    history_items = bot.get_history().items()
    items_list = []
    for k, v in history_items:
        content = v[0]["content"]
        if len(content) > 15:
            content = content[:15] + "..."
        items_list.append(f"{k}: {content}")
    chat_gpt_cmd_history = "\n".join(items_list)
    await ctx.respond(chat_gpt_cmd_history)


@bot.slash_command(name="h")
async def history_command(ctx, idx: int):
    await ctx.defer()
    print(f"Requesting history index: {idx} on:\n{bot.history}")
    input_content = bot.get_history().get(idx, None)
    print(f"\n\n\nInput content: {input_content}\n\n\nType: {type(input_content)}")
    if not input_content:
        await ctx.respond("History is empty!")
    else:
        completion = openai.ChatCompletion.create(model=model_id, messages=input_content)
        response = completion.choices[0].message.content
        await ctx.respond(response[:2000])


@bot.slash_command(pass_context=True, name="syf")
async def scrape_y_finance(ctx, format: str = "human-readable", ticker_count: int = 50):
    await ctx.defer()
    page = requests.get(f"https://finance.yahoo.com/most-active?offset=0&count={ticker_count}")
    soup = BeautifulSoup(page.content, "html.parser")
    doc = " ".join([x.get_text() for x in soup.find_all("tr", class_="simpTblRow")])
    sep = "|" if format == "human-readable" else "\t"
    prompt = f"""
    From the content following the separator --- below, find and extract the 20 first Stock Symbol, Company Name, Price, Change and % Change
    values and present them in a {format} table properly aligned, separated and formatted following structure:
    Number[SEP]Stock Symbol[SEP]Company Name[SEP]Price[SEP]Change[SEP]% Change
    1[SEP][SS][SEP][CN][SEP][P][SEP][C][SEP][C%]
    2[SEP][SS][SEP][CN][SEP][P][SEP][C][SEP][C%]
    2[SEP][SS][SEP][CN][SEP][P][SEP][C][SEP][C%]
    ...

    where [SEP] represents the separator {sep} and [SS], [CN], [P], [C], [C%] are the different placeholders for
    the Stock Symbol, Company Name, Price, Change and % Change values extracted.
    ---
    {doc}
    """
    print(prompt)
    input_content = [{"role": "user", "content": prompt}]
    completion = openai.ChatCompletion.create(model=model_id, messages=input_content)

    response = completion.choices[0].message.content
    print("===========".center(10))
    f_ext = "tsv" if format == "tsv" else "txt"
    with open(f"output.{f_ext}", "w", newline="") as f_output:
        csv_output = csv.writer(f_output, delimiter=sep)
        for line in response.splitlines():
            line_elems = line.split(sep)
            csv_output.writerow(line_elems)
    print(response)
    if format == "human-readable":
        await ctx.respond(response)
    else:
        # area=ctx.message.channel
        await ctx.send("Download file!", file=discord.File(f"output.{f_ext}"))


class TagFilter(commands.Converter):
    async def convert(self, ctx, tags):
        filter = tags.split("&")
        print(f"filter: {filter}")
        return filter


def check_substrings(main_string, substrings):
    return all(substring in main_string for substring in substrings)


@bot.slash_command(pass_context=True, name="axs")
async def arxiv_sanity_summary(ctx, filter_tags: TagFilter, filter_count: int = 3):
    await ctx.defer()

    url = "http://www.arxiv-sanity.com/top?timefilter=year&vfilter=all"
    # url = 'https://arxiv-sanity-lite.com/?q=&rank=time&tags=&pid=&time_filter=3&svm_c=0.01&skip_have=no'
    res = requests.get(url)
    text = res.text

    soup = BeautifulSoup(text, "html.parser")
    script = soup.find(lambda tag: tag.name == "script" and "var papers =" in tag.text)
    start = script.text.index("[")
    end = script.text.rfind("]")
    json_data = script.text[start:end]
    var_tags_idx = json_data.rfind("var tags")

    json_data = json_data[: var_tags_idx - 2]
    # print(json_data)
    papers = json.loads(json_data)

    filtered_no = 0
    output_string = f"\n**Last {filter_count} papers on {filter_tags} from arxiv sanity**\n\n"
    for paper in papers:
        if check_substrings(paper["tags"], filter_tags):
            title = paper["title"]
            print(f"Adding paper: {title}")
            authors = paper["authors"]
            paper_id = paper["id"]
            tags = paper["tags"]
            paper_template = f"""
                **[{title}](https://arxiv.org/abs/{paper_id})**
                Authors: _{authors}_
                Arxiv ID: {paper_id}
                Tags: {tags}
            """
            output_string += paper_template
            filtered_no += 1
        if filtered_no == filter_count:
            break
    print(f"Discord Text Length: {len(output_string)}. Will be cut to 2000")
    await ctx.respond(output_string[:2000])


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


@bot.slash_command(pass_context=True, name="ax")
async def arxiv_summary(
    ctx,
    arxiv_id: str = "1706.03762",
    language: str = "english",
    style: str = "paragraph",
    style_items: int = 1,
    chunks: int = 10,
    chars_per_chunk: int = 1024,
):  # Transformers paper arxiv
    await ctx.defer()
    paper = download_from(arxiv_id)
    title = paper.title
    print(f"Paper title: {title}")
    whole_text = pdf2text("downloaded-paper.pdf")
    chunk_list = chunk_text(whole_text, chunk_len=chars_per_chunk)
    total_chars = chunks * chars_per_chunk
    print(
        f"Chunks: {len(chunk_list)}\nSending {chunks} of {chars_per_chunk} chars each to Chat GPT (Total ~{total_chars})"
    )
    chat_gpt_text = " ".join(chunk_list[:chunks])
    chat_gpt_text = clean_text(chat_gpt_text)
    if style == "paragraph":
        summary_style = f"{style_items} paragraph"
    elif style == "bulletpoints":
        summary_style = f"{style_items} bullet points"
    elif style == "sonnet":
        summary_style = "a sonnet style"
    else:
        summary_style = "a single sentence"
    prompt = f""""
        Please, summarize this paper in {language} in {summary_style}:
        {chat_gpt_text}
        """
    print(f"Len prompt {len(chat_gpt_text)}, words {len(prompt.split())}")
    input_content = [{"role": "user", "content": prompt}]
    completion = openai.ChatCompletion.create(model=model_id, messages=input_content)
    cgpt_summary = completion.choices[0].message.content
    print(f"Text ({len(cgpt_summary)}): {cgpt_summary}")
    summary = f"""
\n\n
**{title}**\n
_SUMMARY in {language} ({summary_style})_\n
{cgpt_summary}
    """
    print(f"Discord Text Length: {len(summary)}. Will be cut to 2000")
    await ctx.respond(summary[:2000])


bot.run(DISCORD_TOKEN)
