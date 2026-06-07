import gradio as gr
import RAG_functions


def app():
    with gr.Blocks() as demo:

        gr.Markdown("# Multilingual RAG Agent")
        # gr.Markdown("Upload an Arabic or English PDF document, process it, then chat about its contents.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                process_btn = gr.Button("Process and Load Document", variant="primary")
                status_output = gr.Textbox(label="System Status", value="Awaiting PDF upload...")

            with gr.Column(scale=2):
                chatbot_ui = gr.ChatInterface(
                    fn=RAG_functions.ollama_rag_agent,
                    # title="Document Knowledge Base"
                )

        # Connect processing click action
        process_btn.click(fn=RAG_functions.process_pdf, inputs=[file_input], outputs=[status_output])

    demo.launch(inbrowser=True, show_error=True,)

if __name__ == "__main__":
    app()