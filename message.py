import json
from datetime import datetime


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __str__(self):
        return f"{self.role}: {self.content}"

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class StreamChunk:
    def __init__(
        self,
        content: str,
        model: str,
        done: bool = False,
        role: str = "assistant",
        created_at: datetime = datetime.now(),
    ):
        print("StreamChunk: ", content)
        self.content = content
        self.done = done
        self.model = model
        self.role = role
        self.created_at = created_at

    def to_dict(self):
        return {
            "model": self.model,
            "message": {
                "role": self.role,
                "content": self.content,
            },
            "done": self.done,
            "created_at": self.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        }

    def to_json(self, newline=True):
        return json.dumps(self.to_dict()) + ("\n" if newline else "")
