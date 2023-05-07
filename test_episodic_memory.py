import logging

from langchain import ConversationChain, PromptTemplate
from langchain.llms import OpenAI

import logging_config
from memory import ConversationEpisodicMemory

log_file = "satnbot.log"
logging_config.setup_logger("satnbot", log_file, level=logging.DEBUG)

llm = OpenAI(temperature=0)
memory = ConversationEpisodicMemory(llm=llm)


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

Context:
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


conversation = ConversationChain(llm=llm, verbose=True, prompt=EPISODIC_MEMORY_CONVERSATION_TEMPLATE, memory=memory)

output = conversation.predict(input="Hi there!")

print(f"First interaction ouput:\n\t{output}")

output = conversation.predict(input="Would like to know more about you.")

print(f"Second interaction ouput:\n\t{output}")


output = conversation.predict(input="Oh wow. You are such an amazing tool!")

print(f"Third interaction ouput:\n\t{output}")


# _input = {"input": "Hi, there!"}
# m1 = memory.load_memory_variables(_input)
# print(m1)
# memory.save_context(
#     _input,
#     {"ouput": "Hello! How can I assist you today?"}
# )

# m1 = memory.load_memory_variables({"input": 'Hello!'})
# print(m1)


# _input = {"input": "Hello! I'm francisco"}
# m1 = memory.load_memory_variables(_input)
# print(m1)

# memory.save_context(
#     _input,
#     {"ouput": "Hi Francisco! How are you today? Hoper you are good!"}
# )
