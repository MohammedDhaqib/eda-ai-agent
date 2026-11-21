import ollama


def test_ollama():

    print("\nConnecting to Ollama...")

    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "user",
                    "content": "Explain Exploratory Data Analysis in simple words."
                }
            ]
        )

        answer = response["message"]["content"]

        print("\nAI RESPONSE:")
        print("-" * 50)
        print(answer)
        print("-" * 50)

        print("\nOllama connection successful!")

    except Exception as error:
        print("\nOllama connection failed!")
        print(f"Error: {error}")


if __name__ == "__main__":
    test_ollama()