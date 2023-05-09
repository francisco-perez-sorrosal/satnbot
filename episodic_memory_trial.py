import logging

from langchain import ConversationChain, PromptTemplate
from langchain.llms import OpenAI

import logging_config
from memory import ConversationEpisodicMemory

log_file = "satnbot.log"
logging_config.setup_logger("satnbot", log_file, level=logging.DEBUG)
logger = logging.getLogger("satnbot")

_DEFAULT_EPISODIC_MEMORY_CONVERSATION_TEMPLATE = """You are an assistant to a human, powered by a large language model trained by OpenAI.

You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations
and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive,
allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large
amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access
to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text
based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of
topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here
to assist.

Episodic memory:
{episode}

Current conversation:
{history}
Last line:
Human: {input}
You:"""

EPISODIC_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
    input_variables=["episode", "history", "input"],
    template=_DEFAULT_EPISODIC_MEMORY_CONVERSATION_TEMPLATE,
)

llm = OpenAI(temperature=0)
memory = ConversationEpisodicMemory(llm=llm)
conversation = ConversationChain(llm=llm, verbose=True, prompt=EPISODIC_MEMORY_CONVERSATION_TEMPLATE, memory=memory)

output = conversation.predict(input="Hi there!")

logger.info(f"First interaction ouput:\n\t{output}")
logger.info(f"Memory at end of interaction 1:\n\t{memory.episode_store}")
logger.info("@" * 100)
logger.info("\n" * 3)

output = conversation.predict(input="Hi! I'm Francisco. Nice to meet you!")

logger.info(f"Second interaction ouput:\n\t{output}")
logger.info(f"Memory at end of interaction 2:\n\t{memory.episode_store}")
logger.info("@" * 100)
logger.info("\n" * 3)

output = conversation.predict(input="Would like to know more about you.")

logger.info(f"Third interaction ouput:\n\t{output}")
logger.info(f"Memory:\n\t{memory.episode_store}")
logger.info("@" * 100)
logger.info("\n" * 3)

output = conversation.predict(input="Oh wow. You are such an amazing tool! What can you do?")

logger.info(f"Fourth interaction ouput:\n\t{output}")
logger.info(f"Memory:\n\t{memory.episode_store}")
logger.info("@" * 100)
logger.info("\n" * 3)

output = conversation.predict(input="Do you remember my name?")

logger.info(f"Fifth interaction ouput:\n\t{output}")
logger.info(f"Memory:\n\t{memory.episode_store}")
logger.info("@" * 100)
logger.info("\n" * 3)


output = conversation.predict(
    input="I'd like to introduce you also to my friend, John. Please, explain also to him your capabilities."
)
logger.info(f"Sixth interaction ouput:\n\t{output}")
logger.info(f"Memory:\n\t{memory.episode_store}")
logger.info("@" * 100)
logger.info("\n" * 3)


# _input = {"input": "Hi, there!"}
# m1 = memory.load_memory_variables(_input)
# logger.info(m1)
# memory.save_context(
#     _input,
#     {"ouput": "Hello! How can I assist you today?"}
# )

# m1 = memory.load_memory_variables({"input": 'Hello!'})
# logger.info(m1)


# _input = {"input": "Hello! I'm francisco"}
# m1 = memory.load_memory_variables(_input)
# logger.info(m1)

# memory.save_context(
#     _input,
#     {"ouput": "Hi Francisco! How are you today? Hoper you are good!"}
# )
