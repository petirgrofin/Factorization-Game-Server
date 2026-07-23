from server.rag.retriever import retrieve_examples

examples = retrieve_examples(
    topic="difference_of_squares",
    difficulty="hard",
    mistakes=[
        "sign_errors",
        "multi_step_factoring"
    ],
    k=5
)

print("\nRETRIEVED EXAMPLES:\n")

for i, ex in enumerate(examples, 1):
    print(f"--- Example {i} ---")
    print(ex)
    print()