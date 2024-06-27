import gradio as gr
from chat import chat

demo = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="OceanBase 问答机器人-V0.1",
    description="基于混合检索和意图分类的OceanBase 问答机器人"
)
demo.launch()