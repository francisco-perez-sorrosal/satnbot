import csv
import os

import discord
import openai
import requests
from bs4 import BeautifulSoup
from discord.ext import commands

# Load environment variables
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

model_id = "gpt-3.5-turbo"
# model_id="gpt-4"

intents = discord.Intents.all()
intents.messages = True
intents.message_content = True

class DiscordChatGPT4(commands.Bot):
    def __init__(self, intents):
        super().__init__(intents)
        self.history = []

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

        print(f"User message: {text_content}")

        image_content = None
        if message.attachments:
            image_content = message.attachments[0].url

        input_content = [{"role": "user", "content": text_content}]
        if image_content:
            input_content[0]["content"]["image"] = image_content

        completion = openai.ChatCompletion.create(model=model_id, messages=input_content)
        self.history.append(input_content)

        response = completion.choices[0].message.content
        await message.channel.send(response)


# Create and run DisordChatGPT4 instance
bot = DiscordChatGPT4(intents=intents)


# Commands
@bot.slash_command(name="history")
async def history(ctx):
    print(f"Requesting history")
    chat_gpt_cmd_history = {i: bot.history[i][0]["content"] for i in range(len(bot.history))}
    await ctx.respond(chat_gpt_cmd_history)
    # await ctx.respond("chat_gpt_cmd_history")


@bot.slash_command(name="h")
async def history_command(ctx, idx: int):
    print(f"Requesting history index: {idx}")
    chat_gpt_cmd_history = {i: bot.history[i] for i in range(len(bot.history))}
    print(f"cmd history: {chat_gpt_cmd_history}")
    input_content = chat_gpt_cmd_history.get(idx, None)
    print(f"IC\n{type(input_content)}")
    print(f"IC\n{input_content}")
    if not input_content:
        await ctx.respond("History is empty!")
    else:
        completion = openai.ChatCompletion.create(model=model_id, messages=input_content)
        response = completion.choices[0].message.content
        await ctx.respond(response)


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


@bot.slash_command(pass_context=True, name="ax")
async def arxiv_summary(ctx, arxiv_id: str = "1605.08386v1"):
    from utils import download_from, pdf2text, chunk_text
    await ctx.defer()
    paper = download_from(arxiv_id)
    title = paper.title
    print(f"Paper title: {title}")
    whole_text = pdf2text("downloaded-paper.pdf")
    chunks = chunk_text(whole_text, chunk_len=1024)
    print(f"Chunks: {len(chunks)} Sending 10 to Chat GPT")
    chat_gpt_text = " ".join(chunks[:10])
    prompt = f""""
Please, summarize this paper:\n
{chat_gpt_text}
"""
    input_content = [{"role": "user", "content": prompt}]
    completion = openai.ChatCompletion.create(model=model_id, messages=input_content)
    cgpt_summary = completion.choices[0].message.content
    print(f"Text: {cgpt_summary}")
    summary = f""""
Paper downloaded!: {title}
SUMMARY\n
-------\n
{cgpt_summary}
    """
    print("KK")
    await ctx.respond(summary)
    

bot.run(DISCORD_TOKEN)
