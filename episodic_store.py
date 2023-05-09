from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass
from sentence_transformers import util

from logging_config import logger


def float_list_to_str(float_list: List[float]) -> Sequence[float]:
    return tuple(float_list)


@dataclass(eq=True, frozen=True)
class EpisodeId(BaseModel):
    """Class that contains all relevant information for an Episode Identification."""

    episode_hrid: str
    """Episode Human Readable ID"""
    embedding: Sequence[float]
    """Embedding for the episode"""
    _extract_str = validator("embedding", pre=True, allow_reuse=True)(float_list_to_str)

    class Config:
        frozen = True

    # def embedding(self) -> List[float]:
    #     return [float(idx) for idx in self.embedding_str.split(' ')]

    def __str__(self):
        return f"Episode HRID {self.episode_hrid}: {self.embedding[:5]}..."


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
        # logger.debug(f"K Closests results (not sorted): {results}")
        results.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"K Closests results (sorted)")
        for i, r in enumerate(results):
            logger.debug(f"{i}: {r[0]}: {r[1].episode_hrid} - {r[1].embedding[:3]}...: {r[2]}")
        # Return top k
        # if results:
        # results = list(map(lambda x: x[1:], results))[:k]

        return results[:k]

    def set(self, key: EpisodeId, value: Optional[str]) -> None:
        logger.info("&&&&" * 10)
        logger.info(f"Setting key {key.episode_hrid} with value {value}")
        self.store[key] = value

    def delete(self, key: EpisodeId) -> None:
        del self.store[key]

    def exists(self, key: EpisodeId) -> bool:
        return key in self.store

    def clear(self) -> None:
        return self.store.clear()

    def __str__(self):
        result = "Whole Memory:\n"
        for id, summary in self.store.items():
            result += f"{id.episode_hrid} {id.embedding[:3]}...:\n\t{summary}\n"
        return result
