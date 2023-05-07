from typing import Any, Dict, List

from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain

# from langchain.schema import BaseMemory, messages_from_dict, messages_to_dict
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel, BaseMessage, get_buffer_string
from pydantic import Field
from sentence_transformers import util

from episodic_store import (
    BaseEpisodicMemoryStore,
    EpisodeId,
    InMemoryEpisodicMemoryStore,
)
from logging_config import logger

# logger = logging.getLogger()


_DEFAULT_EPISODE_IDENTIFICATION_TEMPLATE = """You are an AI assistant helping yourself to keep track of facts about relevant episodes
that the humans that interact with you have.
Based on the last line of the human dialogue with the AI, categorize the input part of dialogue that the human and the AI are having using five keywords.
Return the output following these rules:
1. The list has to be comma separated.
2. The list has to be sorted alphabetically.
3. To sort alphabetically, use first letter of the word and the order of the letters of the english alphabet.
4. Filter that list to avoid stop words. Use an exhaustive list of stop words in english to filter.
5. If after filtering the list has less than five words, add more keywords that are not on the list of stop words until it has five words.
6. Just respond with a list of the five keywords that best represent the category of the conversation.

EXAMPLE
Last line of conversation:
Human: Hello, I'm a human named Francisco.
Output: ai, greetings, human, interaction, introduction
END OF EXAMPLE

Last line of conversation:
Human: {input}
Output:"""

EPISODE_IDENTIFICATION_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=_DEFAULT_EPISODE_IDENTIFICATION_TEMPLATE,
)

_DEFAULT_EPISODE_SUMMARIZATION_TEMPLATE = """You are an AI assistant helping yourself to keep track of facts about relevant episodes that the
humans that interact with you have.
Update the summary of the provided episode in the "Episode" section based on the last line of the human dialogue with the AI.
If you are writing the summary for the first time, return a single sentence.
The update should only include facts that are relayed in the last line of conversation about the provided dialogue, and should only contain facts
about the provided episode.

If there is no new information about the provided episode or the information is not worth noting (not an important or relevant fact to remember
long-term), return the existing summary unchanged.

Full conversation history (for context):
{history}

Episode to summarize:
{episode}

Existing summary of {episode}:
{summary}

Last line of conversation:
Human: {input}
AI: {output}
Updated summary:"""

EPISODE_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["episode", "summary", "history", "input", "output"],
    template=_DEFAULT_EPISODE_SUMMARIZATION_TEMPLATE,
)


class ConversationEpisodicMemory(BaseChatMemory):
    """Entity extractor & summarizer to memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    embeddings: HuggingFaceEmbeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    llm: BaseLanguageModel
    episode_identification_prompt: BasePromptTemplate = EPISODE_IDENTIFICATION_PROMPT
    episode_summarization_prompt: BasePromptTemplate = EPISODE_SUMMARIZATION_PROMPT
    episode_cache: List[EpisodeId] = []
    k: int = 3
    chat_history_key: str = "history"
    episode_store: BaseEpisodicMemoryStore = Field(default_factory=InMemoryEpisodicMemoryStore)

    # def __init__(self, memory_file):
    #     super().__init__()
    #     try:
    #         logger.debug(f"Loading historic memory from {memory_file}")
    #         with open(memory_file, "rb") as file:
    #             memory_content = pickle.load(file)
    #             # self.episodic.load_memory_variables(memory_content)
    #             # episode = messages_from_dict(memory_content)
    #             # logger.debug(type(episode))
    #             # self.episodic.chat_memory.messages = episode
    #             # logger.debug(f"loaded episode:\n{self.episodic.chat_memory.messages}")

    #     except FileNotFoundError:
    #         logger.debug("No history!")
    #         # self.episodic.load_memory_variables({})

    @property
    def buffer(self) -> List[BaseMessage]:
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return ["episode", self.chat_history_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""

        logger.debug(f"=" * 100)
        logger.debug(
            f"LOAD VARIABLES\n\tInputs: {inputs}\n\tInput key: {self.input_key}\n\tMemVars: {self.memory_variables}"
        )
        logger.debug(f"=" * 100)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        logger.debug(f"PIK: {prompt_input_key}")
        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        logger.debug(
            f"CALLING episode identification CHAIN with:\n\tHistory: {buffer_string}\n\tInput: {inputs[prompt_input_key]}"
        )
        chain = LLMChain(llm=self.llm, prompt=self.episode_identification_prompt)
        output = chain.predict(
            history=buffer_string,
            input=inputs[prompt_input_key],
        )
        logger.debug(f"CHAIN Output:\n\t{output}")
        if output.strip() == "NONE":
            episode_keywords = []
        else:
            episode_keywords = [w.strip() for w in output.split(",")]
        episode_keywords = sorted(episode_keywords)
        episode_hrid = ",".join(episode_keywords)
        episode_embeddings = self.embeddings.embed_query(episode_hrid)
        episode_id = EpisodeId(episode_hrid=episode_hrid, embedding=episode_embeddings)

        closest_conversations = self.episode_store.get_k_closest(episode_id)
        logger.debug(f"CC {closest_conversations}")
        self.episode_cache.append(episode_id)  # list(map(lambda x: x[0], closest_conversations))
        logger.debug(f"len Episodic Cache {len(self.episode_cache)}")

        episode_summaries = {}
        for id, episode_summary in closest_conversations:
            episode_summaries[id.episode_hrid] = self.episode_store.get(id, "")

        if self.return_messages:
            buffer: Any = self.buffer[-self.k * 2 :]
        else:
            buffer = buffer_string

        return {
            self.chat_history_key: buffer,
            "episode": episode_summaries,
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)

        logger.debug(f"*" * 100)
        logger.debug(f"SAVE CTX\n\tInputs: {inputs}\n\tOutputs: {outputs}")
        logger.debug(f"*" * 100)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        input_data = inputs[prompt_input_key]

        chain = LLMChain(llm=self.llm, prompt=self.episode_summarization_prompt)

        logger.info("+" * 100)
        logger.info(f"EP CACHE LEN: {len(self.episode_cache)}")
        logger.info(f"EP CACHE: {[e.episode_hrid for e in self.episode_cache]}")
        logger.info("+" * 100)
        for episode_id in self.episode_cache:
            existing_summary = self.episode_store.get(episode_id, "")
            logger.debug(
                f"CALLING episode summarization CHAIN with:\n\tSummary: {existing_summary}\n\tEp Id: {episode_id}\n\tHistory:\n\t\t{buffer_string}\n\tInput: {input_data}"
            )
            output = chain.predict(
                summary=existing_summary,
                episode=episode_id.episode_hrid,
                history=buffer_string,
                input=input_data,
                output=outputs["response"],
            )
            logger.debug(f"CHAIN setting output for {episode_id}: {output}")
            self.episode_store.set(episode_id, output.strip())

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
        # self.entity_cache.clear()
        # self.episode_store.clear()


# class Memory(BaseMemory):
#     episodic: ConversationBufferMemory = ConversationBufferMemory(return_messages=True)
#     memory_file: str = "memory.pkl"

#     def __init__(self, memory_file):
#         super().__init__()
#         try:
#             logger.debug(f"Loading historic memory from {memory_file}")
#             with open(memory_file, "rb") as file:
#                 memory_content = pickle.load(file)
#                 self.episodic.load_memory_variables(memory_content)
#                 episode = messages_from_dict(memory_content)
#                 logger.debug(type(episode))
#                 self.episodic.chat_memory.messages = episode
#                 logger.debug(f"loaded episode:\n{self.episodic.chat_memory.messages}")

#         except FileNotFoundError:
#             logger.debug("No history!")
#             self.episodic.load_memory_variables({})

#     @property
#     def memory_variables(self) -> List[str]:
#         combined_memory_variables = []
#         combined_memory_variables += self.episodic.memory_variables
#         logger.debug(combined_memory_variables)
#         return combined_memory_variables

#     def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
#         """Return key-value pairs given the text input to the chain.

#         If None, return all memories
#         """
#         return {
#             "episodic": self.episodic.load_memory_variables(inputs),
#         }

#     def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
#         """Save the context of this model run to memory."""
#         self.episodic.save_context(inputs, outputs)

#     def clear(self) -> None:
#         """Clear memory contents."""
#         self.episodic.clear()

#     def add_chitchat(self, input, output):
#         """Add an human-ai pair entry to the historic memory."""
#         self.episodic.chat_memory.add_user_message(input)
#         self.episodic.chat_memory.add_ai_message(output)

#     def get_memory(self, session_name):
#         """Return the history for a given session."""
#         return self.episodic.chat_memory.messages

#     def save_memory(self):
#         logger.debug(f"Saving memory in {self.memory_file}")
#         with open(self.memory_file, "wb") as file:
#             pickle.dump(messages_to_dict(self.episodic.chat_memory.messages), file)
