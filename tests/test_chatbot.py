"""
Automated tests. Run with:   python -m pytest -q

A fake detector returns a fixed list of objects, so these tests check the
language and dialogue logic quickly, without needing the YOLO model.
"""

import pytest
from PIL import Image

from netrachat import NetraChat
from netrachat.dialogue.manager import _answer_sentence
from netrachat.nlg.grammar import count_phrase, join_list, polish
from netrachat.nlu.entities import extract
from netrachat.nlu.intent import IntentClassifier
from netrachat.vision.detector import Detection, color_name

W, H = 900, 600


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return list(self.detections)


class FakeVLM:
    available = True

    def caption(self, image):
        return "a dog sitting on a couch next to a cat"

    def answer(self, image, question):
        q = question.lower()
        if "holding" in q:
            return "frisbee"
        if "doing" in q:
            return "sleeping"
        if "raining" in q:
            return "no"
        return "living room"


SCENE = [
    Detection("dog", 0.91, (80, 200, 380, 500), color="brown"),     # left, big
    Detection("cat", 0.85, (420, 300, 600, 520), color="black"),    # centre
    Detection("couch", 0.78, (20, 200, 880, 590), color="grey"),
    Detection("person", 0.88, (650, 60, 800, 500), color="blue"),   # right
    Detection("person", 0.80, (820, 80, 890, 480), color="red"),
]


@pytest.fixture(scope="module")
def intents():
    return IntentClassifier()


@pytest.fixture
def bot(intents):
    b = NetraChat(detector=FakeDetector(SCENE), vlm=None, intent_classifier=intents)
    b.load_image(Image.new("RGB", (W, H)))
    return b


# ---------------------------------------------------------------- grammar
def test_count_phrases_are_grammatical():
    assert count_phrase("person", 1) == "a person"
    assert count_phrase("person", 3) == "three people"
    assert count_phrase("umbrella", 1) == "an umbrella"
    assert count_phrase("skis", 2) == "two pairs of skis"
    assert count_phrase("bus", 2) == "two buses"
    assert count_phrase("knife", 4) == "four knives"


def test_join_list_and_polish():
    assert join_list(["a cat", "a dog", "a TV"]) == "a cat, a dog, and a TV"
    assert polish("there is a umbrella  near the the car") == \
        "There is an umbrella near the car."
    assert polish("i see it .it is red") == "I see it. It is red."


def test_color_names():
    assert color_name((200, 30, 30)) == "red"
    assert color_name((30, 70, 200)) == "blue"
    assert color_name((250, 250, 250)) == "white"
    assert color_name((10, 10, 10)) == "black"
    assert color_name((40, 160, 50)) == "green"


# ---------------------------------------------------------------- understanding
def test_entities():
    e = extract("Are there any traffic lights or puppies near the red car?")
    assert e.objects == ["traffic light", "dog", "car"]
    assert e.colors == ["red"]
    assert e.relation == "next to"
    assert extract("are there animals?").categories == ["animal"]
    assert extract("what colour is it").has_pronoun


@pytest.mark.parametrize("text,intent", [
    ("hello there", "greet"),
    ("how many dogs are there?", "count"),
    ("is there a giraffe", "exists"),
    ("what colour is the car", "color"),
    ("where is the cat", "location"),
    ("what is next to the sofa", "relation"),
    ("describe the picture", "describe"),
    ("what objects do you see", "list_objects"),
    ("what is the main object", "main_subject"),
    ("where was this photo taken", "scene_type"),
    ("how confident are you", "confidence"),
    ("what is the man holding", "open_question"),
    ("thank you so much", "thanks"),
    ("bye", "goodbye"),
    ("hw mny cats r there", "count"),
])
def test_intents(intents, text, intent):
    assert intents.predict(text)[0] == intent


# ---------------------------------------------------------------- dialogue
def test_needs_image_first(intents):
    b = NetraChat(detector=FakeDetector(SCENE), intent_classifier=intents)
    assert "upload an image" in b.reply("how many dogs are there?").lower()


def test_opening_message(intents):
    b = NetraChat(detector=FakeDetector(SCENE), intent_classifier=intents)
    msg = b.load_image(Image.new("RGB", (W, H)))
    assert "two people, a dog, a cat, and a couch" in msg


def test_count(bot):
    assert bot.reply("How many people are there?") == "There are two people in the image."
    assert bot.reply("how many dogs?") == "There is one dog in the image."
    assert bot.reply("How many horses are there?") == "I cannot see any horses in the image."
    assert "two animals: a dog and a cat" in bot.reply("how many animals are there")


def test_exists(bot):
    assert bot.reply("Is there a dog?") == "Yes, there is a dog on the left of the image."
    assert bot.reply("is there a giraffe?") == "No, I cannot see any giraffes in this image."
    assert "a dog and a cat" in bot.reply("are there any animals?")


def test_unknown_object_is_handled_honestly(bot):
    assert "not trained to recognise" in bot.reply("is there a helicopter?")


def test_colour_and_pronoun_memory(bot):
    assert bot.reply("What colour is the dog?") == "The dog looks brown."
    assert bot.reply("where is it?") == "The dog is on the left of the image."
    assert bot.reply("Is the cat white?") == "No, the cat looks black rather than white."
    assert "dressed mostly in" in bot.reply("what colour are the people")


def test_follow_up_question(bot):
    bot.reply("how many dogs are there?")
    assert bot.reply("what about the people?") == "There are two people in the image."


def test_location_of_group(bot):
    assert bot.reply("where are the people?") == \
        "There are two people, both on the right of the image."


def test_multiple_objects_in_one_question(bot):
    assert bot.reply("how many people and dogs are there?") == \
        "There are two people and one dog in the image."
    assert bot.reply("is there a laptop or a tv?") == \
        "No, I cannot see any laptops or TVs in this image."


def test_describe_one_object(bot):
    answer = bot.reply("tell me about the couch")
    assert answer.startswith("The couch is large, grey, and sits in the centre")


def test_small_talk_and_nonsense(bot):
    assert bot.reply("ok").startswith("Okay!")
    assert bot.reply("tell me a joke").startswith("Sorry, I can only answer questions")
    assert bot.reply("asdfgh qwerty").startswith("Sorry, I did not understand")


def test_relation(bot):
    answer = bot.reply("what is next to the dog?")
    assert answer.startswith("Around the dog, I can see")
    assert bot.reply("where is the dog relative to the cat?") == \
        "The dog is to the left of the cat."


def test_main_subject_and_scene(bot):
    assert "a couch" in bot.reply("what is the main object?")
    assert bot.reply("where was this photo taken?") == ("Based on the objects I can see "
                                                        "(a couch), it looks like a living room.")
    assert bot.reply("is this a kitchen?").startswith("I do not think this is a kitchen.")
    assert bot.reply("is this indoors or outdoors?").startswith("It is most likely indoors")


def test_open_question_without_blip_is_honest(bot):
    assert "object detection alone" in bot.reply("what is the man holding?")


def test_every_reply_is_well_formed(bot):
    questions = ["hi", "describe the image", "list all objects", "how many cats",
                 "is there a cat", "what color is it", "where is the couch",
                 "what is above the dog", "how sure are you", "help", "who are you",
                 "what is the main subject", "thanks", "bye"]
    for q in questions:
        a = bot.reply(q)
        assert a[0].isupper(), a
        assert a[-1] in ".!?", a
        assert "  " not in a, a


# ---------------------------------------------------------------- with BLIP
def test_with_vision_language_model(intents):
    b = NetraChat(detector=FakeDetector(SCENE), vlm=FakeVLM(), intent_classifier=intents)
    msg = b.load_image(Image.new("RGB", (W, H)))
    assert "It looks like a dog sitting on a couch next to a cat." in msg
    assert b.reply("what is the man holding?") == "The man is holding a frisbee."
    assert b.reply("is it raining?") == "No, it does not look that way."


def test_answer_sentence():
    assert _answer_sentence("what is she doing?", "playing tennis") == \
        "She is playing tennis."
    assert _answer_sentence("what is the boy eating", "apple") == "The boy is eating an apple."
    assert _answer_sentence("is the dog happy?", "yes") == "Yes, it looks that way."
