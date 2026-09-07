"""Data Investigator agent (FR-120).

A distinct agent: role ``data_investigator`` (observability server only, FR-506), its
own role prompt, and the FR-802 bounded loop. Behaviour is provided by the shared
specialist subgraph; distinctness is in role, permissions, prompt, and tool domain.
"""

from __future__ import annotations

from ..prompts import PROMPT_VERSION, load_prompt
from ..specialist_base import build_specialist_subgraph

ROLE = "data_investigator"
AGENT_NAME = "data_investigator"
PROMPT = load_prompt("data_investigator")
PROMPT_ID = f"data_investigator:{PROMPT_VERSION}"


def build():
    """Return the compiled Data Investigator subgraph."""
    return build_specialist_subgraph(role=ROLE, agent_name=AGENT_NAME)
