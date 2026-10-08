"""Controlled tool contracts and registry for Stage 4 orchestration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from pydantic import BaseModel, Field, ValidationError


class ToolDefinition(BaseModel):
    """Describes a registered controlled tool and its accepted input schema."""

    name: str
    description: str
    input_schema: Dict[str, Any]


class ToolRequest(BaseModel):
    """A request to invoke one named tool with validated structured arguments."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Structured outcome from a controlled tool invocation."""

    tool_name: str
    success: bool
    output: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolInputError(ValueError):
    """Raised when a request does not satisfy a registered tool's input schema."""


class ControlledTool(ABC):
    """Contract for a controlled tool; implementations define a bounded input model."""

    name: str
    description: str
    input_model: Type[BaseModel]

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )

    @abstractmethod
    async def execute(self, tool_input: BaseModel) -> ToolResult:
        """Run the tool with a schema-validated input object."""


class ToolRegistry:
    """In-memory registry that resolves and invokes explicitly registered tools only."""

    def __init__(self) -> None:
        self._tools: Dict[str, ControlledTool] = {}

    def register(self, tool: ControlledTool) -> None:
        """Register one controlled tool by its non-empty name."""
        normalized = tool.name.strip().lower()
        if not normalized:
            raise ValueError("Tool name cannot be empty.")
        self._tools[normalized] = tool

    def resolve(self, tool_name: str) -> ControlledTool:
        """Resolve a registered tool or raise a clear lookup error."""
        normalized = tool_name.strip().lower()
        try:
            return self._tools[normalized]
        except KeyError as exc:
            raise KeyError(f"No tool is registered with name '{tool_name}'.") from exc

    async def invoke(self, request: ToolRequest) -> ToolResult:
        """Validate a request against the selected tool's schema and execute it."""
        tool = self.resolve(request.name)
        try:
            tool_input = tool.input_model.model_validate(request.arguments)
        except ValidationError as exc:
            raise ToolInputError(
                f"Invalid input for tool '{tool.name}': {exc.errors(include_url=False)}"
            ) from exc

        result = await tool.execute(tool_input)
        if result.tool_name != tool.name:
            raise ValueError(
                f"Tool '{tool.name}' returned a result labelled '{result.tool_name}'."
            )
        return result
