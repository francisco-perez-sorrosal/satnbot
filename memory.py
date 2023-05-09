from typing import Any, Dict, List, Tuple

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
from prompts import (
    EPISODE_IDENTIFICATION_PROMPT,
    EPISODE_MERGING_PROMPT,
    EPISODE_SUMMARIZATION_PROMPT,
)


class ConversationEpisodicMemory(BaseChatMemory):
    """Entity extractor & summarizer to memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    embeddings: HuggingFaceEmbeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    llm: BaseLanguageModel
    episode_identification_prompt: BasePromptTemplate = EPISODE_IDENTIFICATION_PROMPT
    episode_summarization_prompt: BasePromptTemplate = EPISODE_SUMMARIZATION_PROMPT
    episode_merging_prompt: BasePromptTemplate = EPISODE_MERGING_PROMPT
    relevant_episodes_cache: List[Tuple[float, EpisodeId]] = []
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
        chain = LLMChain(llm=self.llm, prompt=self.episode_identification_prompt, verbose=True)
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

        if not self.episode_store.exists(episode_id):
            logger.info("Setting NEW episode in memory (withouth description bc it's new!!!)")
            self.episode_store.set(episode_id, "")

        closest_conversations = self.episode_store.get_k_closest(episode_id)
        logger.debug(f"K Closests convs (returned)")
        for i, r in enumerate(closest_conversations):
            logger.debug(f"{i}: {r[0]}: {r[1].episode_hrid} - {r[1].embedding[:3]}...: {r[2]}")

        episode_summaries = {}
        self.relevant_episodes_cache = []
        for score, id, episode_summary in closest_conversations:
            self.relevant_episodes_cache.append((score, id))  # list(map(lambda x: x[0], closest_conversations))
            episode_summaries[id.episode_hrid] = episode_summary

        logger.debug(f"Len Relevant Episodic Cache: {len(self.relevant_episodes_cache)}")

        if self.return_messages:
            buffer: Any = self.buffer[-self.k * 2 :]
        else:
            buffer = buffer_string

        return {
            self.chat_history_key: buffer,
            "episode": "\n".join([summary for _, summary in episode_summaries.items()]),
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

        chain = LLMChain(llm=self.llm, prompt=self.episode_summarization_prompt, verbose=True)

        logger.info("+" * 100)
        logger.info(
            f"EP CACHE ({len(self.relevant_episodes_cache)}): {[e.episode_hrid for s, e in self.relevant_episodes_cache]}"
        )
        logger.info("+" * 100)

        merge_threshold = 0.9
        candidate_episodes_to_merge = []
        for score, episode_id in self.relevant_episodes_cache:
            episode_summary = self.episode_store.get(episode_id, "")
            logger.debug(
                f"CALLING episode summarization CHAIN with:\n\tSummary: {episode_summary}\n\tHistory:\n\t\t{buffer_string}\n\tInput: {input_data}"
            )
            output = chain.predict(
                episode_id=episode_id.episode_hrid,
                summary=episode_summary,
                # history=buffer_string,
                input=input_data,
                output=outputs["response"],
            )
            logger.debug(f"CHAIN setting output for {episode_id}: {output}")
            if self.episode_store.exists(episode_id):
                logger.info(
                    f"Episode {episode_id.episode_hrid} EXIST!\n\tCurrent episode summary: {episode_summary if episode_summary else 'N/A' }"
                )
            else:
                logger.info(f"{episode_id.episode_hrid} DOES NOT EXIST!")
            logger.info(f"\tUpdating the episode to:\n\t{output.strip()}")
            self.episode_store.set(episode_id, output.strip())
            if score > merge_threshold:
                logger.debug(f"Adding '{episode_id.episode_hrid}' ({score}) to candidate episodes to merge")
                candidate_episodes_to_merge.append((episode_id, episode_summary))

        if len(candidate_episodes_to_merge) > 1:
            logger.info("^" * 100)
            logger.info(f"MERGING {len(candidate_episodes_to_merge)} episodes")
            logger.info("^" * 100)

            chain = LLMChain(llm=self.llm, prompt=self.episode_merging_prompt, verbose=True)

            summary_of_episodes = ""
            for episode_id, episode_summary in candidate_episodes_to_merge:
                summary_of_episodes += f"{episode_summary}\n\n"
                logger.info(f"Deleting episode {episode_id} from store: {episode_summary}")
                self.episode_store.delete(episode_id)

            logger.info(f"Summary of episodes to merge:\n{summary_of_episodes}")

            output = chain.predict(episodes_summary=summary_of_episodes)
            logger.info(f"MERGING OUTPUT: {output}")
            import json

            outputs = json.loads(output)

            print(outputs)
            episode_keywords = [w.strip() for w in outputs["categories"].split(",")]
            episode_keywords = sorted(episode_keywords)
            episode_hrid = ",".join(episode_keywords)

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            episode_embeddings = embeddings.embed_query(episode_hrid)
            episode_id = EpisodeId(episode_hrid=episode_hrid, embedding=episode_embeddings)
            logger.info(f"\tAdding merging episode to {episode_id.episode_hrid}:\n\t{outputs['summary']}")
            self.episode_store.set(episode_id, outputs["summary"])
            logger.info("^" * 100)

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
