import re
from typing import List, Optional
from livekit.agents.tokenize import token_stream, SentenceTokenizer, SentenceStream

class JapaneseSentenceTokenizer(SentenceTokenizer):
    """
    日本語音声対話向け低遅延ストリーミング SentenceTokenizer
    
    【設計思想】
    LLMの生成完了を待たず、句読点（「、」「。」「！」「？」）や改行ごとに
    テキストをチャンク分割して即座に Kokoro TTS に投入します。
    これにより、発話終了から応答音声開始までの体感遅延（Turnaround Time）を
    300〜500ms に抑制します。
    """
    def __init__(self, min_sentence_len: int = 4, comma_split_len: int = 12):
        super().__init__()
        self._min_sentence_len = min_sentence_len
        self._comma_split_len = comma_split_len
        # 分割パターン:
        # 1. 「。」「！」「？」「…」「\n」で終了する句
        # 2. 12文字以上経過した読点「、」「,」
        self._pattern = re.compile(
            rf'([^。！？…\n]+[。！？…\n]+|[^。！？…\n]{{{self._comma_split_len},}}?[、,])'
        )

    def tokenize(self, text: str, *, language: Optional[str] = None) -> List[str]:
        if not text:
            return []
        matches = self._pattern.findall(text)
        remainder = self._pattern.sub("", text)
        result = [m.strip() for m in matches if len(m.strip()) >= 1]
        if remainder.strip():
            result.append(remainder.strip())
        return result if result else [text.strip()]

    def stream(self, *, language: Optional[str] = None) -> SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=self.tokenize,
            min_token_len=self._min_sentence_len,
            min_ctx_len=1,
        )
