"""
Chat with NetraChat in the terminal (no browser needed).

    python chat_cli.py samples/bus.jpg
    python chat_cli.py my_photo.jpg --blip

Commands while chatting:
    /image <path>   load a different image
    /quit           leave
"""

import argparse

from PIL import Image

from netrachat import NetraChat
from netrachat.vision.captioner import VisionLanguageModel
from netrachat.vision.detector import DEFAULT_WEIGHTS, ObjectDetector


def main():
    parser = argparse.ArgumentParser(description="NetraChat in the terminal")
    parser.add_argument("image", nargs="?", help="path to an image file")
    parser.add_argument("--blip", action="store_true", help="enable BLIP models")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights")
    args = parser.parse_args()

    print("Loading models, please wait…")
    bot = NetraChat(detector=ObjectDetector(args.weights),
                    vlm=VisionLanguageModel() if args.blip else None)
    if args.image:
        print("NetraChat:", bot.load_image(Image.open(args.image)))
    else:
        print("NetraChat: Hello! Load an image with: /image path/to/photo.jpg")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in ("/quit", "/exit"):
            break
        if text.startswith("/image "):
            path = text[len("/image "):].strip().strip('"')
            try:
                print("NetraChat:", bot.load_image(Image.open(path)))
            except OSError as exc:
                print(f"NetraChat: I could not open that file ({exc}).")
            continue
        print("NetraChat:", bot.reply(text))
        if text.lower().strip("!. ") in ("bye", "goodbye"):
            break


if __name__ == "__main__":
    main()
