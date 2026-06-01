"""
DistributedAgentManager: extends AgentManager with remote agent registration.

Usage
-----
    from maki.distributed.registry import DistributedAgentManager

    manager = DistributedAgentManager(local_backend)

    # local agent (unchanged API)
    manager.add_agent("analyst", role="data analyst", instructions="...")

    # remote agent on another machine
    manager.register_remote("writer", endpoint="http://writer-node:8101", api_key="tok")

    # works exactly like a local agent
    result = manager.assign_task("writer", "write a summary of the findings")
    results = manager.coordinate_agents([
        {"agent": "analyst", "task": "analyse the data"},
        {"agent": "writer",  "task": "write the report"},
    ])
"""
import logging
from typing import Optional

from ..agents.agent_manager import AgentManager
from ..backend import LLMBackend
from .proxy import AgentProxy

logger = logging.getLogger(__name__)


class DistributedAgentManager(AgentManager):
    """
    AgentManager extended with remote-agent support.

    All existing methods (assign_task, coordinate_agents, collaborative_task,
    run_workflow) work unchanged — they only call execute_task() and
    execute_task_with_retry() on whatever is in self.agents, and AgentProxy
    implements both.
    """

    def register_remote(
        self,
        name: str,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AgentProxy:
        """
        Connect to a remote AgentServer and register it under *name*.

        The proxy immediately fetches /info from the remote server to verify
        the connection and populate name/role/agent_id/plugins.

        Args:
            name:     Registry key used in assign_task(), coordinate_agents(), etc.
            endpoint: Base URL of the remote AgentServer, e.g. "http://host:8100".
            api_key:  Bearer token expected by the remote server (None = open).
            timeout:  Per-request HTTP timeout in seconds (default 60).

        Returns:
            The AgentProxy that was registered.

        Raises:
            MakiNetworkError: If the remote server is unreachable.
            MakiTimeoutError: If the connection times out.
            MakiAPIError:     If the server rejects the request (e.g. bad key).
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string")

        proxy = AgentProxy(endpoint=endpoint, api_key=api_key, timeout=timeout)
        self.agents[name] = proxy
        logger.info(
            "Registered remote agent '%s' → %s (server name: '%s')",
            name, endpoint, proxy.name,
        )
        return proxy

    def unregister_remote(self, name: str) -> None:
        """Remove a remote agent from the registry and close its HTTP session."""
        proxy = self.agents.get(name)
        if isinstance(proxy, AgentProxy):
            try:
                proxy._session.close()
            except Exception:
                pass
        self.remove_agent(name)
