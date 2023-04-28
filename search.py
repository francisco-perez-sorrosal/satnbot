from typing import Optional

import llms
from kagiapi import KagiClient
from llms.llms import Result


class SearchEngine:
    def __init__(self) -> None:
        self.model = llms.init()
        self.kagi = KagiClient("BQBAd2yDQwQ._1jl-SHXWugCVa9o9cbiTjHx8v4CZsc8VvOEEB2deSI")
        self.temperature = 0.1
        self.max_tokens = 200
        print(f"Search engine: {self.kagi}")
        print(f"Search engine: {self.model.list()} (temp={self.temperature}, max_tokens={self.max_tokens})")

    def query(
        self, q: str, q_results: int = 10, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> Result:
        print(type(q), q)
        results = self.kagi.search(q)  # , limit=q_results)
        print(f"Kagi results:\n{results}")
        search_ctx = "\n".join([f"{r['title']} {r['snippet']}" for r in results[:q_results]])

        prompt = f"""
        {search_ctx}
        {q}
        """
        print(f"Prompt to search:\n{prompt}")
        return self.model.complete(
            prompt=prompt,
            temperature=self.temperature if not temperature else temperature,
            max_tokens=self.max_tokens if not max_tokens else max_tokens,
        )
