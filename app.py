"""
NetraChat web app.

Run:      python app.py
Then open http://127.0.0.1:7860 in your browser.

Options:
  --blip            also load the BLIP captioning + question-answering models
                    (better descriptions and open questions; ~2 GB download)
  --weights FILE    YOLO weights to use (default yolov8s.pt; try yolov8m.pt)
  --share           create a temporary public link you can send to others
"""

import argparse

import gradio as gr

from netrachat import NetraChat
from netrachat.nlu.intent import IntentClassifier
from netrachat.vision.annotate import annotate
from netrachat.vision.captioner import VisionLanguageModel
from netrachat.vision.detector import DEFAULT_WEIGHTS, ObjectDetector

EXAMPLE_QUESTIONS = [
    "Describe the image.", "What objects can you see?", "How many people are there?",
    "Is there a dog?", "What colour is it?", "Where is the bus?",
    "What is next to the person?", "Where was this photo taken?", "How sure are you?",
]


def build_app(detector, vlm, intents):
    # Each browser tab gets its own NetraChat (its own memory), but they all
    # share the heavy models, which are loaded only once.
    def new_bot():
        return NetraChat(detector=detector, vlm=vlm, intent_classifier=intents)

    def on_image(image, bot, history):
        bot = bot or new_bot()
        if image is None:
            return None, bot, history
        message = bot.load_image(image)
        boxed = annotate(bot.image, bot.scene.detections)
        history = (history or []) + [
            {"role": "user", "content": "📷 *uploaded an image*"},
            {"role": "assistant", "content": message},
        ]
        return boxed, bot, history

    def on_message(text, bot, history):
        bot = bot or new_bot()
        if not text or not text.strip():
            return "", bot, history
        answer = bot.reply(text)
        history = (history or []) + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
        return "", bot, history

    def on_clear():
        return None, None, None, []

    with gr.Blocks(title="NetraChat") as demo:
        gr.Markdown(
            "# 👁️ NetraChat — Conversational Image Recognition Chatbot\n"
            "Upload an image, then chat with the bot about what is in it."
        )
        bot_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="1. Upload an image")
                image_out = gr.Image(type="pil", label="What NetraChat detected",
                                     interactive=False)
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="2. Ask questions", height=520)
                text = gr.Textbox(placeholder="Type a question and press Enter…",
                                  show_label=False)
                gr.Examples(EXAMPLE_QUESTIONS, inputs=text, label="Try these")
                clear = gr.Button("Start over")
        status = ("BLIP captioning and question answering: **on**"
                  if vlm is not None else
                  "BLIP captioning and question answering: **off** "
                  "(start with `python app.py --blip` to enable)")
        gr.Markdown(status)

        image_in.upload(on_image, [image_in, bot_state, chat], [image_out, bot_state, chat])
        text.submit(on_message, [text, bot_state, chat], [text, bot_state, chat])
        clear.click(on_clear, None, [image_in, image_out, bot_state, chat])
    return demo


def main():
    parser = argparse.ArgumentParser(description="NetraChat web app")
    parser.add_argument("--blip", action="store_true", help="enable BLIP models")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights")
    parser.add_argument("--share", action="store_true", help="public share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading models, please wait…")
    detector = ObjectDetector(args.weights)
    vlm = VisionLanguageModel() if args.blip else None
    intents = IntentClassifier()
    demo = build_app(detector, vlm, intents)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
