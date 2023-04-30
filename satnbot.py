import asyncio
import csv
import json
import os
from typing import List

import discord
import openai
import requests
from bs4 import BeautifulSoup
from cleantext import clean
from discord.ext import commands, tasks
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# from history import ChatHistory
from memory import Memory
from utils import chunk_text, download_from, pdf2text

DISCORD_CHUNK_LEN = 2000
# Load environment variables
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

model_id = "gpt-3.5-turbo"
# model_id="gpt-4"

intents = discord.Intents.all()
intents.members = True
intents.messages = True
intents.message_content = True
intents.guild_messages = True

memory_file = "memory.pkl"


class DiscordChatGPT4(commands.Bot):
    def __init__(self, intents):
        super().__init__(intents)
        llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_id, temperature=0)
        self.memory = Memory(memory_file)
        self.conversation = ConversationChain(
            llm=llm, verbose=True, memory=self.memory.episodic  # We pass only the episodic part of the global memory
        )

    @tasks.loop()
    async def status_task(self) -> None:
        # self._get_websocket()
        # bot.fetch_webhook()
        channel = self.get_channel(1086445347997503512)
        print(f"Sending recurrent message to channel {channel} after 30 secs")
        await channel.send(
            f"""
        <@{self.user.id}> @myself return one of these 4 things at random:
        1) a useful Python trick
        2) a best practice in Python
        3) a popular code snippet in Python explained
        4) a common mistake in Python and how to avoid it"""
        )
        await asyncio.sleep(30)

    async def on_ready(self):
        print(f"------\nLogged in as {self.user.name} ({self.user.id})\n------")
        # self.status_task.start()

    async def on_message(self, message: discord.message.Message):
        if message.author == self.user and "@myself" not in message.content:
            return

        # TODO: Remove as we don't want to specify the name of the bot to chat in the chatbot channel (can't figure it out in Discord yet)
        # Check if the message contains a mention of the bot
        bot_mention = f"<@{self.user.id}>"
        if bot_mention not in message.content:
            return

        print(message.channel.id)
        print(message.content)
        print(message)

        # for m in list(message.channel.history(limit=10)):
        #     print(m)
        if message.channel.name != "chatbot":
            print("Chat with AI only in the 'chatbot' channel!")
            return

        # Remove the bot mention from the message content
        bot_mention = f"<@{self.user.id}>"
        text_content = message.content
        if bot_mention in message.content:
            text_content = message.content.replace(bot_mention, "").strip()
        print(text_content)
        text_content = text_content.replace("@myself", "")

        print(f"User message: {text_content} {type(text_content)}")

        image_content = None  # TODO add multimodal content
        if message.attachments:
            message.attachments[0].url

        output = self.conversation.predict(input=text_content)

        print(f"Message received {output}")
        self.memory.save_memory()  # TODO Improve this
        print(f"History:\n{self.memory}")

        response = output
        print(f"Text ({len(response)}): {response}")
        # await message.channel.send(response[:2000])
        chunk_list = chunk_text(response, chunk_len=DISCORD_CHUNK_LEN)
        print("*" * 100)
        print(f"msg: {message}")
        print("=" * 100)
        if message.author.id == self.user.id:
            ctx = await self.get_context(message)
        else:
            ctx = message.channel
        await discord_multi_response(ctx, chunk_list)


# Create and run DisordChatGPT4 instance
bot = DiscordChatGPT4(intents=intents)


async def discord_multi_response(ctx, chunks: List[str], is_send: bool = True):
    print(f"Sendng {len(chunks)} chunks")
    for chunk in chunks:
        try:
            assert len(chunk) <= DISCORD_CHUNK_LEN
            print(f"Sendng chunks of {len(chunk)} chars")
            if is_send:
                await ctx.send(chunk)
            else:
                await ctx.respond(chunk)
        except AssertionError:
            print(f"Cutting chunk:\n{chunk}\n\nto:\n{chunk[:DISCORD_CHUNK_LEN]}")
            if is_send:
                await ctx.send(chunk[:DISCORD_CHUNK_LEN])
            else:
                await ctx.respond(chunk[:DISCORD_CHUNK_LEN])


# Commands
@bot.slash_command(name="history")
async def history(ctx):
    print(f"Requesting history")
    history_items = bot.memory.episodic.chat_memory.messages
    print(history_items)
    items_list = []
    for i, e in enumerate(history_items):
        content = f"{e.content}"
        if len(content) > 50:
            content = content[:50] + "..."
        items_list.append(f"{i}. [{e.type}]: {content}")
    chat_gpt_cmd_history = "\n".join(items_list)
    chunk_list = chunk_text(chat_gpt_cmd_history, chunk_len=DISCORD_CHUNK_LEN)
    await discord_multi_response(ctx, chunk_list, is_send=False)


# async def on_member_join(member):
#     await ctx.send(me)


@bot.slash_command(name="h")
async def history_command(ctx, idx: int):
    await ctx.defer()
    print(f"Requesting history index: {idx} on:\n{bot.memory}")
    input_content = bot.memory.episodic.chat_memory.messages[idx]
    print(f"\n\n\nInput content: {input_content.content}\n\n\nType: {type(input_content)}")
    if not input_content:
        await ctx.respond("History is empty!")
    else:
        response = bot.conversation.predict(input=input_content.content)
        chunk_list = chunk_text(response, chunk_len=DISCORD_CHUNK_LEN)
        await discord_multi_response(ctx, chunk_list, is_send=False)


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
        chunk_list = chunk_text(response, chunk_len=DISCORD_CHUNK_LEN)
        await discord_multi_response(ctx, chunk_list, is_send=False)
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
    chunk_list = chunk_text(output_string, chunk_len=DISCORD_CHUNK_LEN)
    await discord_multi_response(ctx, chunk_list, is_send=False)


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
    chunk_list = chunk_text(response, chunk_len=DISCORD_CHUNK_LEN)
    await discord_multi_response(ctx, chunk_list, is_send=False)


bot.run(DISCORD_TOKEN)
