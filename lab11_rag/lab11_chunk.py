def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a string into chunks of size `chunk_size` with an overlap of `overlap`.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The maximum length of each chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    ### TODO: Implement Document Chunking with Overlap

    # 1. Handle edge cases: if text is empty, return empty list.
    if not text:
        return []

    # 2. Check if chunk_size > 0 and overlap < chunk_size. Raise ValueError if invalid.
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    # 3. Calculate the step size (how much to advance the index each time).
    step = chunk_size - overlap
    # step size determines how much we move forward for the next chunk
    # For example, if chunk_size=5 and overlap=2, step=3 means we take 5 characters for the first chunk
    # then move forward by 3 characters to start the next chunk, which will overlap by 2 characters with the previous chunk.

    # 4. Loop through the text and extract chunks of length `chunk_size`.
    chunks = []

    for start_idx in range(0, len(text), step):
        chunk = text[start_idx : start_idx + chunk_size]
        chunks.append(chunk)

        # 5. Stop when you've reached the end of the text.
        if start_idx + chunk_size >= len(text):
            break

    return chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Lab 11: Task 1 - Document Chunking with Overlap")
    print("=" * 60)

    # Test cases
    test_text = "Hello World"
    print(f"Original Text: '{test_text}'")

    print("\nChunk size 5, overlap 2:")
    try:
        result = chunk_text(test_text, chunk_size=5, overlap=2)
        print(f"Chunks: {result}")
        assert result == ["Hello", "lo Wo", "World"], "Test case failed!"

        print("\nEdge Case: Text shorter than chunk size:")
        result_short = chunk_text("Hi", chunk_size=5, overlap=2)
        print(f"Chunks: {result_short}")
        assert result_short == ["Hi"], "Short text test failed!"

        print("\nEdge Case: Empty input:")
        result_empty = chunk_text("", chunk_size=5, overlap=2)
        print(f"Chunks: {result_empty}")
        assert result_empty == [], "Empty input test failed!"

        print("\nAll chunking tests passed!")
    except NotImplementedError:
        print("Task 1 Not Implemented Yet!")
    except Exception as e:
        print(f"Task 1 Error: {e}")
