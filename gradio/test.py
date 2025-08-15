import gradio as gr

def greet(name):
    return f"Hello {name}!"

app = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Greeting App",
    description="Enter your name and get a greeting!"
)

app.launch()