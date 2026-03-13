from dataclasses import dataclass, field


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationConfig:
    """Sampling / generation hyper-parameters forwarded to Ollama."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048        # maps to num_predict in Ollama
    seed: int = -1                # -1 = random
    stop: list[str] = field(default_factory=list)

    def to_ollama_options(self) -> dict:
        opts: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.max_tokens,
        }
        if self.seed != -1:
            opts["seed"] = self.seed
        if self.stop:
            opts["stop"] = self.stop
        return opts


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float
    done: bool = True

    def __str__(self) -> str:
        return self.content

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.0
