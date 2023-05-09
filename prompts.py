from langchain import PromptTemplate

_DEFAULT_EPISODE_IDENTIFICATION_TEMPLATE = """You are an AI assistant helping yourself to keep track of facts about relevant episodes
that the humans that interact with you have.
Based on the last line of the human dialogue with the AI, classify the input part of dialogue that the human and the AI are having
using exactly seven keywords related to categories. The list has to be comma separated.

Last line of conversation:
Human: {input}
Output:"""
EPISODE_IDENTIFICATION_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=_DEFAULT_EPISODE_IDENTIFICATION_TEMPLATE,
)

_DEFAULT_EPISODE_SUMMARIZATION_TEMPLATE = """You are an AI assistant helping yourself to keep track of facts about relevant episodes that the
humans that interact with you have.
Update the 'Episode Summary' section below based on the last lines of the human dialogue with the AI.
If you are writing the summary for the first time, return a single sentence.
The update should only include facts that are relayed in the last lines of conversation about the provided dialogue, and should only contain facts
about the provided episode.

If there is no new information about the provided episode or the information is not worth noting (not an important or relevant fact to remember
long-term), return the existing summary unchanged.

Episode summary ({episode_id}):
{summary}

Last lines of conversation:
Human: {input}
AI: {output}
Updated summary:"""

EPISODE_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["episode_id", "summary", "input", "output"],
    template=_DEFAULT_EPISODE_SUMMARIZATION_TEMPLATE,
)

_DEFAULT_EPISODE_MERGING_TEMPLATE = """You are an AI assistant helping yourself keeping track of facts about relevant episodes that the
humans that interact with you have.

You will return a json object after 'New Episode Summary JSON:'. The json object will contain an attribute named 'summary' whose content will
be a string containing the summary of the content that appears after the <Summary of Current Episodes> section below. The update should only
include facts about the episodes content. If there is no new information about the provided content of current episodes or the information is
not worth noting (not an important or relevant fact to remember long-term), return the existing summary unchanged.

The json object will also include an attribute called 'categories' that will have a list of exactly seven keywords related to categories that
best categorize the <Summary of Current Episodes>. The list to be a comma separated string.

<Summary of Current Episodes>

{episodes_summary}

New Episode Summary JSON:"""

EPISODE_MERGING_PROMPT = PromptTemplate(
    input_variables=["episodes_summary"],
    template=_DEFAULT_EPISODE_MERGING_TEMPLATE,
)
