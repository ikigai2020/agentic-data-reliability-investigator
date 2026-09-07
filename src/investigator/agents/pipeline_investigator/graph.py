"""Pipeline Investigator agent (FR-110).

A distinct agent: role ``pipeline_investigator`` (pipeline server only, FR-506), its
own role prompt, and the FR-802 bounded loop. Behaviour is provided by the shared
specialist subgraph; distinctness is in role, permissions, prompt, and tool domain.
"""

from __future__ import annotations

from ..prompts import PROMPT_VERSION, load_prompt
from ..specialist_base import build_specialist_subgraph

ROLE = "pipeline_investigator"
AGENT_NAME = "pipeline_investigator"
PROMPT = load_prompt("pipeline_investigator")
PROMPT_ID = f"pipeline_investigator:{PROMPT_VERSION}"


def build():
    """Return the compiled Pipeline Investigator subgraph."""
    return build_specialist_subgraph(role=ROLE, agent_name=AGENT_NAME)
