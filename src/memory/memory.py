"""
Sliding Window Conversation Memory
Keeps last N exchanges to avoid context bloat
"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class SlidingWindowMemory:
    def __init__(self, window_size: int = 6):
        """
        window_size: max number of messages to keep (3 exchanges = 6 messages)
        """
        self.window_size = window_size
        self.history: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))
        # Trim to window
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def get_context(self) -> str: #g, readable texts
        if not self.history:
            return ""
        lines = []
        for msg in self.history:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def get_messages(self) -> list[dict]:#g, structured data 
        return [{"role": m.role, "content": m.content} for m in self.history]

    def clear(self) -> None:
        self.history = []

    def __len__(self) -> int:
        return len(self.history)
