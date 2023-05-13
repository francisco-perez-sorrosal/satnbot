from langchain import PromptTemplate


_DEFAULT_EPISODE_IDENTIFICATION_TEMPLATE = """You are an AI assistant helping yourself building an episodic memory to keep track of facts about the relevant episodes
the humans that interact with you have.
Based on the last line of the human dialogue with the AI, classify the input part of dialogue that the human and the AI are having
using exactly seven keywords that categorize the last line added by the human. Avoid using names. The list has to be comma separated.

[LAST LINE OF DIALOGUE]
Human: {input}
Output:"""
EPISODE_IDENTIFICATION_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=_DEFAULT_EPISODE_IDENTIFICATION_TEMPLATE,
)

_DEFAULT_EPISODE_SUMMARIZATION_TEMPLATE = """You are an AI assistant acting as an episodic memory. You track and extract facts (episodes)
from the interaction with humans.


You will return a json object after [NEW EPISODE JSON] tag. You will update the content of the 'Episode Summary' section below based on
the last lines of the human dialogue with the AI. If you are writing the summary for the first time, return a single sentence.
The update will add new facts that are relayed in the last lines of conversation about the provided dialogue.
You will add the updated summary to an attribute named 'summary' in the JSON to return.

If there is no new information about the provided episode or the information is not worth noting (not an important or relevant fact to remember
long-term), return the existing summary unchanged in the 'summary' attribute of the JSON.

The json object will also include an attribute named 'categories' that will have a comma-separated list of exactly seven keywords related
to categories that best categorize the new summary you are providing.

Episode summary ({episode_id}):
{summary}

Last lines of conversation:
Human: {input}
AI: {output}
[NEW EPISODE JSON]"""

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


ARXIV_SUMMARIZATION_PROMPT_TEMPLATE = """
    Summarize the technical text, delimited by triple
    =, in {language} in {summary_style}:
    ==={text}===
"""

ARXIV_SUMMARIZATION_PROMPT = PromptTemplate(
    template=ARXIV_SUMMARIZATION_PROMPT_TEMPLATE, input_variables=["text", "language", "summary_style"]
)


ARXIV_MD_SUMMARIZATION_PROMPT_TEMPLATE = """
    Write a concise summary of each section and subsection of the text below, delimited below by triple = in {language} language.
    The summary of the sections/subsections should concisely in one paragraph each.
    Do not summarize neither references nor acknowledgements.
    If it exists, summarize also the sections and subsections in the appendices and/or supplementary material that may
    appear at the end of the paper.
    Do not include in the summary or summarize the legend of the figures.

    Return the summary in markdown format. Follow these rules for formatting the markdown:
    1) Heading level 1 should be the title of the paper.
    2) Heading level 2 should be the title of each section.
    3) Heading level 3 should be the summary of each sub-section.
    4) The summary of each section or subsection should be in regular text.
    5) The summary of each section should be followed by a horizontal rule.

    ==={text}===
"""

ARXIV_MD_SUMMARIZATION_PROMPT = PromptTemplate(
    template=ARXIV_MD_SUMMARIZATION_PROMPT_TEMPLATE, input_variables=["text", "language"]
)
