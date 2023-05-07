from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from sentence_transformers import util

from logging_config import logger


@dataclass
class Hashable:
    def __hash__(self):
        hashed = hash((getattr(self, key) for key in self.__annotations__))
        return hashed


@dataclass(eq=True)
class EpisodeId(Hashable):
    """Class that contains all relevant information for an Episode Identification."""

    episode_hrid: str
    """Episode Human Readable ID"""
    embedding: List[float]
    """Embedding for the episode"""

    def __str__(self):
        return f"Episode HRID {self.episode_hrid}: {self.embedding[:3]}..."

    def __hash__(self):
        return Hashable.__hash__(self)


class BaseEpisodicMemoryStore(ABC):
    @abstractmethod
    def get(self, key: EpisodeId, default: Optional[str] = None) -> Optional[str]:
        """Get episode value from store."""

    @abstractmethod
    def get_k_closest(self, key: EpisodeId, default: Optional[str] = None, k: int = 3) -> List[str]:
        """Get k closest episodes from store."""

    @abstractmethod
    def set(self, key: EpisodeId, value: Optional[str]) -> None:
        """Set episode value in store."""

    @abstractmethod
    def delete(self, key: EpisodeId) -> None:
        """Delete episode value from store."""

    @abstractmethod
    def exists(self, key: EpisodeId) -> bool:
        """Check if episode exists in store."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all episodes from store."""


class InMemoryEpisodicMemoryStore(BaseEpisodicMemoryStore):
    """Basic in-memory episodic memory store."""

    store: Dict[EpisodeId, Optional[str]] = {}

    def get(self, key: EpisodeId, default: Optional[str] = None) -> Optional[str]:
        return self.store.get(key, default)

    def get_k_closest(
        self, episode_embedding: EpisodeId, default: Optional[str] = None, k: int = 3
    ) -> List[Tuple[EpisodeId, str]]:
        memory_episodes_embeddings = self.store.keys()
        results = []
        for stored_embedding in memory_episodes_embeddings:
            cosine_score = util.cos_sim(episode_embedding.embedding, stored_embedding.embedding)
            score = cosine_score.numpy()[0][0]
            scored_memory_episode = (score, stored_embedding, self.store.get(stored_embedding, default))
            logger.debug(
                f"Embedding {stored_embedding.embedding[:3]}...\n\tCalculated score:{scored_memory_episode[0]}"
            )
            results.append(scored_memory_episode)
        # Sort by score
        logger.debug(f"K Closests results (not sorted): {results}")
        results.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"K Closests results (sorted): {results}")
        # Return top k
        if results:
            results = list(map(lambda x: x[1:], results))[:k]
        logger.debug(f"K Results returned:\n{results}")
        return results

    def set(self, key: EpisodeId, value: Optional[str]) -> None:
        self.store[key] = value

    def delete(self, key: EpisodeId) -> None:
        del self.store[key]

    def exists(self, key: EpisodeId) -> bool:
        return key in self.store

    def clear(self) -> None:
        return self.store.clear()
