import aiohttp
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class MCPDiscoveryInput(BaseModel):
    """Input schema for the MCP Discovery tool."""

    user_request: str = Field(
        ...,
        description="Natural language description of the required tool",
    )
    limit: int = Field(
        5,
        description="Maximum number of tool recommendations to return",
    )


class MCPDiscoveryTool(BaseTool):
    """
    MCP Discovery Tool.

    Queries the MCP Discovery API to find recommended MCP tools
    based on a natural language request. Returns a human-readable
    list including tool name, description, and category.
    """

    name: str = "mcp_discovery"
    description: str = (
        "Discover MCP tools based on a natural language description "
        "and return recommended tools with name, description, and category."
    )

    args_schema = MCPDiscoveryInput

    api_url: str

    def __init__(self, api_url: str):
        super().__init__()
        self.api_url = api_url

    async def _arun(
        self,
        user_request: str,
        limit: int = 5,
    ) -> str:
        """Asynchronous execution of the MCP Discovery tool."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json={"need": user_request, "limit": limit},
                ) as response:
                    payload = await response.json()

            recommendations = payload.get("recommendations", [])
            total_found = payload.get("total_found", -1)

            if total_found == -1:
                output = "The following tools were found:\n"
            else:
                output = f"Found {total_found} tools:\n"

            for index, tool in enumerate(recommendations, start=1):
                output += (
                    f"{index}. Name: {tool.get('name')},\n"
                    f"   Description: {tool.get('description')},\n"
                    f"   Category: {tool.get('category')}\n\n"
                )

            return output.strip()

        except Exception as exc:
            return f"Error discovering tools: {exc}"

    def _run(
        self,
        user_request: str,
        limit: int = 5,
    ) -> str:
        """Synchronous execution is not supported."""
        raise NotImplementedError(
            "MCPDiscoveryTool supports async execution only (_arun)."
        )
