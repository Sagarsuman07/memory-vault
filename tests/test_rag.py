from rag.retriever import retrieve

USER_ID = "demo_user"

question = "Who is Manshi Kumari?"

results = retrieve(
    question=question,
    user_id=USER_ID,
)

print("\nNumber of results:", len(results))

for i, (document, distance) in enumerate(results, start=1):
    print("\n--- Result", i, "---")
    print("Distance:", distance)
    print("Metadata:", document.metadata)
    print("Content:", document.page_content[:500])