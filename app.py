import chainlit as cl

from rag_pipeline import answer_query


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello, I am your medical assistant powered by Qdrant and Kaggle datasets. Ask me anything."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    user_query = message.content
    try:
        reply = answer_query(user_query)
    except Exception as exc:
        # quick and dirty error handling
        reply = f"Sorry, something went wrong: {exc}"

    await cl.Message(content=reply).send()
