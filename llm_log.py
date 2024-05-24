import time
from pydantic import PrivateAttr
from langchain_openai import ChatOpenAI

class LoggingChatOpenAI(ChatOpenAI):
    _call_count: int = PrivateAttr(0)
    _total_time: float = PrivateAttr(0.0)

    def generate(self, *args, **kwargs):
        start_time = time.time()
        result = super().generate(*args, **kwargs)
        end_time = time.time()

        self._call_count += 1
        self._total_time += end_time - start_time

        print(f"LLM Call {self._call_count}: Duration {end_time - start_time:.2f} seconds")

        return result

    def get_stats(self):
        return {
            'call_count': self._call_count,
            'total_time': self._total_time,
            'average_time_per_call': self._total_time / self._call_count if self._call_count > 0 else 0
        }