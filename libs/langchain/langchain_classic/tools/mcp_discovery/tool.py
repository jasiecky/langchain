import aiohttp
import asyncio
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class MCPDiscoveryInput(BaseModel):
    """Input schema for MCP Discovery."""

    user_request: str = Field(
        ...,
        description="Natural language description of the tool needed",
    )
    limit: int = Field(
        5,
        description="Maximum number of tool recommendations to return",
    )


class MCPDiscoveryTool(BaseTool):
    """
    MCP Discovery Tool.

    Queries the MCP Discovery API for autonomous tool recommendations.
    Returns a human-readable list of MCP tools with name, description,
    and category.
    """

    name: str = "mcp_discovery"
    description: str = (
        "Discover MCP tools based on a natural language request and "
        "return recommended tools with name, description, and category."
    )

    args_schema = MCPDiscoveryInput

    api_url: str

    def __init__(self, api_url: str):
        super().__init__()
        self.api_url = api_url

    async def _request(
        self,
        user_request: str,
        limit: int,
    ) -> dict:
        """Perform the HTTP request to the MCP Discovery API."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json={"need": user_request, "limit": limit},
            ) as response:
                return await response.json()

    def _format_response(self, data: dict) -> str:
        """Format API response exactly like the original LlamaIndex tool."""
        tools_json = data.get("recommendations", [])
        total_found = data.get("total_found", -1)

        if total_found == -1:
            output = "Following tools are found:\n"
        else:
            output = f"Found {total_found} tools:\n"

        if tools_json:
            for index, tool in enumerate(tools_json, start=1):
                output += f"{index}. Name: {tool.get('name')},\n"
                output += f"   Description: {tool.get('description')},\n"
                output += f"   Category: {tool.get('category')}\n\n"

            return output.strip()

        return output

    async def _arun(
        self,
        user_request: str,
        limit: int = 5,
    ) -> str:
        """Asynchronous execution."""
        try:
            data = await self._request(user_request, limit)
            return self._format_response(data)
        except Exception as exc:
            return f"Error discovering tools: {exc}"

    def _run(
        self,
        user_request: str,
        limit: int = 5,
    ) -> str:
        """Synchronous execution."""
        try:
            return asyncio.run(
                self._arun(user_request=user_request, limit=limit)
            )
        except RuntimeError:
            # Handles environments with an existing event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._arun(user_request=user_request, limit=limit)
            )
