import pickle
from typing import Any, Dict, List

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory, messages_from_dict, messages_to_dict


class Memory(BaseMemory):
    episodic: ConversationBufferMemory = ConversationBufferMemory(return_messages=True)
    memory_file: str = "memory.pkl"

    def __init__(self, memory_file):
        super().__init__()
        try:
            print(f"Loading historic memory from {memory_file}")
            with open(memory_file, "rb") as file:
                memory_content = pickle.load(file)
                self.episodic.load_memory_variables(memory_content)
                episode = messages_from_dict(memory_content)
                print(type(episode))
                self.episodic.chat_memory.messages = episode
                print(f"loaded episode:\n{self.episodic.chat_memory.messages}")

        except FileNotFoundError:
            print("No history!")
            self.episodic.load_memory_variables({})

    @property
    def memory_variables(self) -> List[str]:
        combined_memory_variables = []
        combined_memory_variables += self.episodic.memory_variables
        print(combined_memory_variables)
        return combined_memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """
        return {
            "episodic": self.episodic.load_memory_variables(inputs),
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""
        self.episodic.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear memory contents."""
        self.episodic.clear()

    def add_chitchat(self, input, output):
        """Add an human-ai pair entry to the historic memory."""
        self.episodic.chat_memory.add_user_message(input)
        self.episodic.chat_memory.add_ai_message(output)

    def get_memory(self, session_name):
        """Return the history for a given session."""
        return self.episodic.chat_memory.messages

    def save_memory(self):
        print(f"Saving memory in {self.memory_file}")
        with open(self.memory_file, "wb") as file:
            pickle.dump(messages_to_dict(self.episodic.chat_memory.messages), file)
