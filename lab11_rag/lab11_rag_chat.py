import sys

# Import the QA system
from lab11_rag import CourseQASystem, load_course_materials, print_result


def main():
    print("=" * 60)
    print("Interactive Course QA System (Lab 11)")
    print("=" * 60)

    handout_dir = "coe201_handout"

    print("\nInitializing RAG System...")
    rag = CourseQASystem()

    print("\nLoading and Indexing Course Materials...")
    all_pages = load_course_materials(handout_dir)

    try:
        rag.index_pages(all_pages, chunk_size=256, overlap=64)
    except NotImplementedError:
        print(
            "\n[Error]: You must implement the Embedding logic (Task 2) first before you can use the QA system!"
        )
        sys.exit(1)

    print("\nSystem Ready!")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        try:
            query = input("\n[You]: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break

            use_reranker_input = input("Use Reranker? (y/N): ").strip().lower()
            use_reranker = use_reranker_input == "y"

            print(
                f"\nGenerating answer ({'with' if use_reranker else 'without'} reranker)..."
            )
            answer, references, prompt = rag.generate_answer(
                query, k=5, use_reranker=use_reranker
            )

            label = "Reranked" if use_reranker else "Baseline"
            print_result(query, answer, references, prompt, label=label)

        except KeyboardInterrupt:
            break
        except NotImplementedError:
            print(
                "\n[Error]: The requested feature (e.g., retrieve, reranking, generation) is not implemented yet. Finish the TODOs in lab11_rag.py!"
            )
        except Exception as e:
            print(f"\n[Error]: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
