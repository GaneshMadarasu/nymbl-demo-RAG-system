import re
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Split on sentence boundaries (.!? + whitespace) OR raw newlines.
    # The newline branch handles unpunctuated content (handwritten OCR
    # output, bullet lists, headers) so the chunker still finds split
    # points instead of collapsing the whole text into one giant chunk.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_token_count: int = 0

    def _encode(s: str) -> list[int]:
        return _enc.encode(s)

    for sentence in sentences:
        sentence_tokens = _encode(sentence)
        if current_token_count + len(sentence_tokens) > max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            # keep trailing sentences that fit within overlap token budget
            overlap_sentences: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                t = len(_encode(s))
                if overlap_count + t <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_count += t
                else:
                    break
            current_sentences = overlap_sentences + [sentence]
            current_token_count = overlap_count + len(sentence_tokens)
        else:
            current_sentences.append(sentence)
            current_token_count += len(sentence_tokens)

    if current_sentences:
        decoded = " ".join(current_sentences)
        if decoded.strip():
            chunks.append(decoded)
    return [c for c in chunks if c.strip()]
