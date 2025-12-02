from chatbot import ResponsibleChatbot


def main():
    bot = ResponsibleChatbot()
    print("Responsible AI Chatbot (type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        result = bot.chat(user_input)
        print("\nAssistant:", result["final_answer"], "\n")
        print("Safety report:")
        print(result["safety_report"])  
        print("-" * 60)


if __name__ == "__main__":
    main()
