import logging

from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory

import logging_config
from episodic_store import EpisodeId
from prompts import EPISODE_MERGING_PROMPT

log_file = "satnbot.log"
logging_config.setup_logger("satnbot", log_file, level=logging.DEBUG)


llm = OpenAI(temperature=0)
memory = ConversationSummaryBufferMemory(llm=llm)
chain = LLMChain(llm=llm, prompt=EPISODE_MERGING_PROMPT, verbose=True)
output = chain.predict(
    episodes_summary="""
Human: Hi, what's up?
AI:  Hi there! I'm doing great. I'm spending some time learning about the latest developments in AI technology. How about you?
Human: Just working on writing some documentation!
"""
)

outputs = output.split("#", 2)
print(outputs)

episode_keywords = [w.strip() for w in outputs[0].split(",")]
episode_keywords = sorted(episode_keywords)
episode_hrid = ",".join(episode_keywords)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
episode_embeddings = embeddings.embed_query(episode_hrid)
episode_id = EpisodeId(episode_hrid=episode_hrid, embedding=episode_embeddings)
print(episode_id)
