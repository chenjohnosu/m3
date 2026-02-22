import click
import sys
from core.prompt_manager import PromptManager
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer


def start_dialogue(analyze_manager):
    """
    Starts an interactive RAG chat loop using the provided AnalyzeManager.

    Args:
        analyze_manager: An initialized instance of AnalyzeManager connected to the
                         active project's vector store and LLM.
    """
    project_name = analyze_manager.project_name
    click.secho(f"\n--- Starting Dialogue for: {project_name} ---", fg="green", bold=True)
    click.echo("  > Type 'exit', 'quit', or 'bye' to leave.")
    click.echo("  > Context is retrieved from the project's vector store.")
    click.echo("------------------------------------------------")

    try:
        # 1. Get the retrieval engine (index) from the manager
        #    We configure it to retrieve the top 5 most relevant chunks.
        retriever = analyze_manager.index.as_retriever(similarity_top_k=5)

        # 2. Get the LLM
        #    We explicitly request the 'synthesis_model' (e.g., Mistral/Llama3)
        #    because dialogue requires better reasoning than simple extraction.
        llm = analyze_manager.get_llm('synthesis_model')

        # 3. Setup Memory
        #    ChatMemoryBuffer keeps a token-limited history of the conversation.
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

        # 4. Initialize the ContextChatEngine
        #    This engine handles the RAG loop:
        #    Query -> Retrieve Context -> Add to Prompt + History -> Call LLM
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=memory,
            system_prompt=PromptManager().get('dialogue_system')
        )

    except Exception as e:
        click.secho(f"🔥 Error initializing chat engine: {e}", fg="red")
        return

    # 5. The Interactive Loop
    while True:
        try:
            # Create a custom prompt string
            prompt_str = click.style(f"[{project_name}] Chat", fg="cyan")
            user_input = click.prompt(prompt_str, prompt_suffix=" > ")

            # Exit conditions
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break

            if not user_input.strip():
                continue

            click.echo("  > Thinking...", nl=False)  # Simple spinner-like effect

            # 6. Stream the response
            #    streaming=True provides a better UX for long answers
            response = chat_engine.stream_chat(user_input)

            # Clear the "Thinking..." line using carriage return
            sys.stdout.write("\r" + " " * 20 + "\r")
            sys.stdout.flush()

            click.secho("Assistant:", fg="magenta", bold=True)

            # Print the stream chunks as they arrive
            response.print_response_stream()

            click.echo("\n")  # Extra spacing between turns

        except KeyboardInterrupt:
            click.echo("\n\nExiting dialogue...")
            break
        except Exception as e:
            click.secho(f"\n🔥 An error occurred during chat: {e}", fg="red")

    click.echo("Dialogue session ended.")