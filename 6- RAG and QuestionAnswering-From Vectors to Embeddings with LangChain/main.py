from expert_knowledge_worker import RAGAssistant




def main():
    bot = RAGAssistant(folder_root="knowledge-base",
                       embed_backend="hf",  # no cost
                       store_backend="chroma",
                       debug=True,
                       k=25)

    print(bot.chat(
        "Who received the prestigious IIOTY award in 2023?"))  # if it says "I do not know", it is wrong because we have this info in the files.
    bot.launchGradio()  # opens a chat window

    # visualize vector embedding
    bot.visualise(dims=3)




if __name__ == "__main__":
    main()
