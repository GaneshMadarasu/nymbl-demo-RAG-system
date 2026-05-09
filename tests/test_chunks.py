from backend.chunks import chunk_text


def test_short_text_produces_one_chunk():
    result = chunk_text("Hello world. This is a short sentence.", max_tokens=512)
    assert len(result) == 1
    assert "Hello world" in result[0]


def test_long_text_splits_into_multiple_chunks():
    sentence = "The quick brown fox jumps over the lazy dog. " * 30
    result = chunk_text(sentence, max_tokens=100, overlap=10)
    assert len(result) > 1


def test_overlap_means_second_chunk_shares_tokens_with_first():
    text = " ".join([f"word{i}." for i in range(80)])
    chunks = chunk_text(text, max_tokens=60, overlap=15)
    assert len(chunks) >= 2
    end_of_first = chunks[0].split()[-5:]
    start_of_second = chunks[1].split()[:20]
    shared = set(end_of_first) & set(start_of_second)
    assert len(shared) > 0


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_no_chunk_is_empty_string():
    text = "Sentence one. Sentence two. Sentence three."
    result = chunk_text(text, max_tokens=512)
    assert all(c.strip() for c in result)


def test_unpunctuated_newline_separated_text_splits_on_newlines():
    """Handwritten OCR'd content often has no sentence punctuation but
    uses line breaks. The chunker must still find split points so a long
    document doesn't collapse into a single oversized chunk."""
    lines = [f"Line number {i} with some handwritten content" for i in range(50)]
    text = "\n".join(lines)
    chunks = chunk_text(text, max_tokens=64, overlap=8)
    assert len(chunks) > 1, f"expected multiple chunks, got {len(chunks)}"
