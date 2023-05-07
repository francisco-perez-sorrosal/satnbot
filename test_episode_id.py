from typing import Dict, Optional

import pytest

from episodic_store import EpisodeId


def test_episode_id():
    episode_id = EpisodeId("1", [1.0, 2.0, 3.0])
    print(episode_id)
    assert episode_id.episode_hrid == "1" and episode_id.embedding == (1.0, 2.0, 3.0)
    # assert episode_id.embedding == [1., 2., 3.]

    store: Dict[EpisodeId, Optional[str]] = {}

    store[episode_id] = "test"
    assert len(store) == 1 and store[episode_id] == "test"
