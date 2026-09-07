"""Provider-layer exceptions shared by both MCP servers (FR-508).

Providers raise these; server handlers translate them into ``ToolResult.status``
values. Keeping the vocabulary here lets a real provider signal the same conditions
as the fixture provider without changing server or agent contracts (AD-006).
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for provider failures."""


class ProviderNoData(ProviderError):
    """The requested target exists but the provider has no data for it (``no_data``)."""


class ProviderUnavailable(ProviderError):
    """The backend/provider could not be reached (``provider_unavailable``)."""
