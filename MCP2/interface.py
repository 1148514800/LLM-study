# interface.py

async def run(agent):
    await agent.enter_clients()
    try:
        while True:
            user_input = input("\033[94mUser: \033[0m")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            agent.messages.append({
                "role": "user",
                "content": user_input
            })

            response_message = await agent.chat(agent.messages)
            print("\033[92mAssistant: \033[0m", response_message.content)
    finally:
        await agent.exit_clients()
