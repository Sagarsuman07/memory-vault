from config.settings import settings

from memory.memory_service import (
    answer_question,
)


USER_ID = settings.DEMO_USER_ID


while True:

    question = input(
        "\nAsk a question "
        "(type 'exit' to stop): "
    )


    if question.lower() == "exit":

        break


    try:

        result = answer_question(
            question=question,
            user_id=USER_ID,
        )


        print("\n" + "=" * 70)

        print("ANSWER")

        print("=" * 70)

        print(
            result["answer"]
        )


        print("\nGROUNDED:")

        print(
            result["grounded"]
        )


        print("\nSOURCES:")

        for source in result["sources"]:

            print(
                f"- Memory: "
                f"{source['memory_id']}"
            )

            print(
                f"  Chunk: "
                f"{source['chunk_index']}"
            )

            print(
                f"  Distance: "
                f"{source['distance']}"
            )


    except Exception as error:

        print(
            f"\nError: {error}"
        )