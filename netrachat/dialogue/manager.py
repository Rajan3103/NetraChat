"""
The dialogue manager: the "brain" of NetraChat.

For every message it runs the same four steps:

  1. UNDERSTAND  - intent classifier: what kind of question is this?
                   entity extractor : which objects / colours does it mention?
  2. REMEMBER    - resolve "it" / "they" to the object discussed last
                   (conversation memory), and handle follow-ups like
                   "what about the cars?".
  3. LOOK        - query the Scene built from the object detector (and the
                   optional BLIP caption / question-answering model).
  4. RESPOND     - build a reply from grammar-aware templates and run it
                   through the grammar polisher, so it is always correct.
"""

from __future__ import annotations

import re
from collections import Counter

from PIL import Image

from ..nlg.grammar import (be_verb, count_phrase, join_list, number_word,
                           polish_paragraph, with_article)
from ..nlu.entities import Entities, extract, normalize
from ..nlu.intent import IntentClassifier
from ..scene import SCENE_HINTS, Scene
from ..vocabulary import (CATEGORIES, CATEGORY_FORMS, COCO_FORMS, PHRASE_TO_LABEL,
                          plural, singular)

IMAGE_INTENTS = {"describe", "list_objects", "count", "exists", "color", "location",
                 "relation", "main_subject", "scene_type", "confidence", "open_question"}

INDOOR_SCENES = {"a kitchen", "a living room", "a bedroom", "a dining scene",
                 "an office or workspace", "a bathroom"}

# Words that follow "the/a/any" but are not objects the user is asking about.
_NOT_OBJECTS = {
    "image", "picture", "photo", "pic", "scene", "main", "object", "objects", "thing",
    "things", "item", "items", "colour", "color", "left", "right", "top", "bottom",
    "middle", "centre", "center", "background", "foreground", "number", "biggest",
    "largest", "same", "other", "most", "first", "second", "whole", "total", "lot",
    "kind", "type", "sort", "place", "photo's", "image's", "one", "ones", "count",
    "list", "description", "caption", "help", "few", "bit", "uploaded", "big",
    "small", "little", "new", "old", "red", "blue", "green", "white", "black",
    "yellow", "orange", "purple", "pink", "brown", "grey", "gray", "good", "nice",
    "detection", "detections", "answer", "focus", "subject", "setting",
}

_FOLLOW_UP = re.compile(r"^(and|what about|how about|and what about|ok and|also)\b")
_OBJECT_INTENTS = {"count", "exists", "color", "location", "relation", "confidence"}
_SOCIAL = {"greet", "goodbye", "thanks", "help", "identity", "acknowledge",
           "out_of_scope", "fallback"}


class NetraChat:
    """A conversational chatbot that answers questions about an uploaded image.

    Parameters
    ----------
    detector : object with a `.detect(PIL.Image) -> list[Detection]` method.
               Defaults to the YOLOv8 ObjectDetector.
    vlm      : optional VisionLanguageModel (BLIP) for captions and open
               questions. Pass `None` to run with the detector only.
    """

    def __init__(self, detector=None, vlm=None, intent_classifier=None):
        if detector is None:
            from ..vision.detector import ObjectDetector
            detector = ObjectDetector()
        self.detector = detector
        self.vlm = vlm
        self.intents = intent_classifier or IntentClassifier()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.image: Image.Image | None = None
        self.scene: Scene | None = None
        self.focus: list[str] = []        # labels discussed most recently
        self.last_intent: str | None = None
        self.history: list[tuple[str, str]] = []

    @property
    def vlm_ready(self) -> bool:
        return self.vlm is not None and getattr(self.vlm, "available", False)

    # ------------------------------------------------------------ image input
    def load_image(self, image: Image.Image) -> str:
        """Analyse a new image and return a short opening message."""
        self.image = image.convert("RGB")
        detections = self.detector.detect(self.image)
        caption = self.vlm.caption(self.image) if self.vlm is not None else None
        self.scene = Scene(detections, self.image.width, self.image.height, caption)
        self.focus = []
        self.last_intent = None

        if not detections and not caption:
            msg = ("I have received your image, but I could not recognise any objects "
                   "in it. You can still ask me questions, or try a clearer photo.")
        else:
            parts = ["Thanks for the image!"]
            if caption:
                parts.append(f"It looks like {self._clean_caption(caption)}.")
            if detections:
                parts.append(f"I can see {self._inventory()}.")
            parts.append('Ask me anything about it, for example: '
                         f'"{self._example_question()}"')
            msg = " ".join(parts)
        self.history.append(("<image>", msg))
        return msg

    # --------------------------------------------------------------- dialogue
    def reply(self, message: str) -> str:
        message = (message or "").strip()
        if not message:
            return "Please type a question about the image."

        ents = extract(message)
        intent, prob = self.intents.predict(message)
        intent = self._adjust_intent(message, intent, prob, ents)

        if intent in IMAGE_INTENTS and self.scene is None:
            answer = ("Please upload an image first, then I can answer questions "
                      "about it.")
        else:
            handler = getattr(self, f"_on_{intent}")
            answer = handler(message, ents)

        if intent not in _SOCIAL:
            self.last_intent = intent
        answer = polish_paragraph(_split_sentences(answer))
        self.history.append((message, answer))
        return answer

    def _adjust_intent(self, message, intent, prob, ents: Entities) -> str:
        """Use conversation context to correct or refine the predicted intent."""
        text = normalize(message)
        # "what about the cars?" -> repeat the previous kind of question.
        if _FOLLOW_UP.match(text) and (ents.objects or ents.categories) \
                and len(text.split()) <= 6:
            return self.last_intent if self.last_intent in _OBJECT_INTENTS else "describe"
        # "where is the dog relative to the cat?" asks how two objects relate.
        if len(ents.objects) >= 2 and re.search(
                r"\b(relative to|compared to|in relation to|with respect to)\b", text):
            return "relation"
        # A colour word + object, e.g. "is the bus red?", is a colour check.
        if ents.colors and ents.objects and intent == "exists" and \
                re.match(r"^(is|are) (the|it|they)\b", text):
            return "color"
        # The classifier is unsure and the message names nothing we know:
        # ask the user to rephrase instead of guessing.
        if prob < 0.35 and not (ents.objects or ents.categories or ents.has_pronoun):
            return "fallback"
        if self.intents.is_gibberish(message):
            return "fallback"
        return intent

    # ======================================================= social intents
    def _on_greet(self, msg, ents):
        if self.scene is None:
            return ("Hello! I am NetraChat. Upload an image and ask me anything "
                    "about it: what is in it, how many objects there are, their "
                    "colours and where they are.")
        return "Hello! Ask me anything about the image you uploaded."

    def _on_goodbye(self, msg, ents):
        return "Goodbye! It was nice talking with you. Come back with another image any time."

    def _on_thanks(self, msg, ents):
        return "You are welcome! Is there anything else you would like to know about the image?"

    def _on_acknowledge(self, msg, ents):
        if self.scene is None:
            return "Okay! Upload an image whenever you are ready."
        return "Okay! What else would you like to know about the image?"

    def _on_out_of_scope(self, msg, ents):
        return ("Sorry, I can only answer questions about the image you upload. "
                "Try asking me what is in it, how many objects there are, or what "
                "colour something is.")

    def _on_fallback(self, msg, ents):
        return ("Sorry, I did not understand that. Could you rephrase it? For "
                "example, you can ask \"What is in the image?\" or \"How many "
                "people are there?\"")

    def _on_identity(self, msg, ents):
        deep = (" and a BLIP vision-language model for captions and open questions"
                if self.vlm_ready else "")
        return ("I am NetraChat, an image recognition chatbot. I use a YOLOv8 deep "
                f"learning model to detect objects{deep}, plus a trained intent "
                "classifier to understand your questions.")

    def _on_help(self, msg, ents):
        return ("Upload an image, then ask questions such as: \"Describe the image.\", "
                "\"How many people are there?\", \"Is there a dog?\", "
                "\"What colour is the car?\", \"Where is the bus?\", "
                "\"What is next to the person?\", \"What is the main object?\" "
                "or \"Where was this photo taken?\". You can also say \"it\" or "
                "\"they\" to refer to the object we just talked about.")

    # ======================================================= image intents
    def _on_describe(self, msg, ents):
        s = self.scene
        present = [o for o in ents.objects if s.count(o)]
        if present:
            return self._describe_object(present[0])
        if ents.objects:
            return self._missing_object_answer(ents)
        out = []
        if s.caption:
            out.append(f"It looks like {self._clean_caption(s.caption)}.")
        if s.detections:
            out.append(f"I can see {self._inventory()}.")
            main = s.main_subject()
            out.append(f"The most prominent object is {with_article(singular(main.label))}"
                       f" {s.position(main)} of the picture.")
            kind = s.scene_type()
            if kind:
                out.append(f"Overall, this looks like {kind}.")
            self.focus = [main.label]
        elif not s.caption:
            out.append("I could not recognise any familiar objects in this image.")
        return " ".join(out)

    def _describe_object(self, label) -> str:
        """'Tell me about the dog' -> size, colour, position and neighbours."""
        s = self.scene
        dets = s.of(label)
        self.focus = [label]
        if len(dets) > 1:
            return (f"There {be_verb(len(dets))} {count_phrase(label, len(dets), article=False)}"
                    f"{self._spread(dets)}. The largest one is "
                    f"{s.size_word(max(dets, key=lambda d: d.area))}.")
        d = dets[0]
        if label == "person":
            look = f"The person is dressed mostly in {d.color}" if d.color else "There is a person"
            out = [f"{look} and is {s.position(d)} of the image."]
        else:
            colour = f", {d.color}," if d.color else ""
            out = [f"The {singular(label)} is {s.size_word(d)}{colour} and sits "
                   f"{s.position(d)} of the image."]
        near = s.neighbours(d, k=2)
        if near:
            out.append(f"Close to it, I can see {join_list(_group_names(near))}.")
        out.append(f"I am {d.confidence:.0%} sure it is {with_article(singular(label))}.")
        return " ".join(out)

    def _on_list_objects(self, msg, ents):
        s = self.scene
        if not s.detections:
            return "I did not detect any objects that I know in this image."
        lines = []
        for label in s.labels():
            best = max(d.confidence for d in s.of(label))
            lines.append(f"{count_phrase(label, s.count(label), article=False)} "
                         f"({best:.0%} confidence)")
        self.focus = s.labels()
        n_total = len(s.detections)
        kinds = len(s.counts)
        return (f"I detected {number_word(n_total)} "
                f"{'object' if n_total == 1 else 'objects'} of "
                f"{number_word(kinds)} {'kind' if kinds == 1 else 'kinds'}: "
                f"{join_list(lines)}.")

    def _on_count(self, msg, ents):
        s = self.scene
        targets = self._targets(ents, allow_focus=True)
        if ents.categories and not ents.objects:
            return self._count_category(ents.categories[0])
        unknown = self._unknown_object(msg, ents)
        if unknown and not ents.objects:
            return self._unknown_answer(msg, unknown)
        if not targets:
            n = len(s.detections)
            return (f"There {be_verb(n)} {number_word(n)} "
                    f"{'object' if n == 1 else 'objects'} that I can recognise in total: "
                    f"{self._inventory()}.")
        found = [l for l in targets if s.count(l)]
        missing = [l for l in targets if not s.count(l)]
        parts = []
        if found:
            first = s.count(found[0])
            counted = join_list([count_phrase(l, s.count(l), article=False) for l in found])
            parts.append(f"There {be_verb(first)} {counted} in the image")
        if missing:
            parts.append(f"I cannot see any {join_list([plural(l) for l in missing], 'or')}"
                         f"{' in the image' if not found else ''}")
        self.focus = found or self.focus
        return ". ".join(parts) + "."

    def _count_category(self, cat):
        s = self.scene
        found = [l for l in s.labels() if l in CATEGORIES[cat]]
        n = sum(s.count(l) for l in found)
        sing, plur = CATEGORY_FORMS[cat]
        if n == 0:
            return f"I cannot see any {plur} in the image."
        self.focus = found
        detail = join_list([count_phrase(l, s.count(l)) for l in found])
        noun = sing if n == 1 else plur
        return f"There {be_verb(n)} {number_word(n)} {noun}: {detail}."

    def _on_exists(self, msg, ents):
        s = self.scene
        if ents.categories and not ents.objects:
            cat = ents.categories[0]
            found = [l for l in s.labels() if l in CATEGORIES[cat]]
            sing, plur = CATEGORY_FORMS[cat]
            if not found:
                return f"No, I cannot see any {plur} in this image."
            self.focus = found
            return f"Yes, I can see {join_list([count_phrase(l, s.count(l)) for l in found])}."
        unknown = self._unknown_object(msg, ents)
        if unknown and not ents.objects:
            return self._unknown_answer(msg, unknown)
        targets = self._targets(ents, allow_focus=True)
        if not targets:
            return ("Which object should I look for? For example, ask "
                    "\"Is there a dog?\"")
        found = [l for l in targets if s.count(l)]
        missing = [l for l in targets if not s.count(l)]
        if not found:
            return (f"No, I cannot see any {join_list([plural(l) for l in missing], 'or')} "
                    "in this image.")
        self.focus = found
        if len(found) == 1 and s.count(found[0]) == 1:
            d = s.of(found[0])[0]
            answer = (f"Yes, there is {with_article(singular(found[0]))} "
                      f"{s.position(d)} of the image.")
        else:
            first = s.count(found[0])
            answer = (f"Yes, there {be_verb(first)} "
                      f"{join_list([count_phrase(l, s.count(l)) for l in found])} in the image.")
        if missing:
            answer += f" However, I cannot see any {join_list([plural(l) for l in missing], 'or')}."
        return answer

    def _on_color(self, msg, ents):
        s = self.scene
        targets = self._present_targets(ents)
        if targets is None:
            return self._missing_object_answer(ents)
        if not targets:
            return self._image_colors()
        sentences = []
        for label in targets:
            dets = s.of(label)
            colors = [self._object_color(label, d, len(dets)) for d in dets]
            if ents.colors and len(dets) == 1:
                asked = ents.colors[0]
                if colors[0] == asked:
                    sentences.append(f"Yes, the {singular(label)} looks {asked}")
                else:
                    sentences.append(f"No, the {singular(label)} looks {colors[0]} "
                                     f"rather than {asked}")
            elif label == "person":
                sentences.append(self._clothing_colors(colors))
            elif len(dets) == 1:
                sentences.append(f"The {singular(label)} looks {colors[0]}")
            else:
                sentences.append(self._group_colors(label, colors))
        self.focus = targets
        return ". ".join(sentences) + "."

    def _on_location(self, msg, ents):
        s = self.scene
        targets = self._present_targets(ents)
        if targets is None:
            return self._missing_object_answer(ents)
        if not targets:
            main = s.main_subject()
            if main is None:
                return "I could not find any objects to locate in this image."
            targets = [main.label]
        sentences = []
        for label in targets:
            dets = s.of(label)
            if len(dets) == 1:
                sentences.append(f"The {singular(label)} is {s.position(dets[0])} of the image")
            else:
                sentences.append(f"There {be_verb(len(dets))} "
                                 f"{count_phrase(label, len(dets), article=False)}"
                                 f"{self._spread(dets)}")
        self.focus = targets
        return ". ".join(sentences) + "."

    def _on_relation(self, msg, ents):
        s = self.scene
        targets = self._present_targets(ents)
        if targets is None:
            return self._missing_object_answer(ents)
        if not targets:
            main = s.main_subject()
            if main is None:
                return "There are no objects in this image that I can relate to each other."
            targets = [main.label]
        anchor = max(s.of(targets[0]), key=lambda d: d.area)
        name = singular(anchor.label)
        # Two objects named: "where is the dog relative to the cat?"
        if len(targets) >= 2:
            other = max(s.of(targets[1]), key=lambda d: d.area)
            self.focus = targets[:2]
            return (f"The {name} is {s.relation(anchor, other)} the "
                    f"{singular(other.label)}.")
        others = [d for d in s.detections if d is not anchor]
        if ents.relation and ents.relation != "next to":
            others = [d for d in others if s.relation(d, anchor) == ents.relation]
            if not others:
                return f"I cannot see anything {ents.relation} the {name}."
            names = _group_names(others)
            self.focus = [targets[0]]
            return f"{_cap(join_list(names))} {_is_are(others)} {ents.relation} the {name}."
        near = s.neighbours(anchor, k=3)
        if not near:
            return f"The {name} is the only object I can recognise in this image."
        pron = "them" if anchor.label == "person" else "it"
        grouped = Counter((d.label, s.relation(d, anchor)) for d in near)
        bits = []
        for (label, rel), n in grouped.items():
            if label == anchor.label:   # "another person", "two other cars"
                what = (f"another {singular(label)}" if n == 1
                        else f"{number_word(n)} other {plural(label)}")
            else:
                what = count_phrase(label, n)
            bits.append(f"{what} {rel} {pron}")
        self.focus = [targets[0]]
        return f"Around the {name}, I can see {join_list(bits)}."

    def _on_main_subject(self, msg, ents):
        s = self.scene
        main = s.main_subject()
        if main is None:
            if s.caption:
                return f"The picture seems to show {self._clean_caption(s.caption)}."
            return "I could not find a clear main subject in this image."
        self.focus = [main.label]
        return (f"The main subject appears to be {with_article(singular(main.label))}. "
                f"It is {s.size_word(main)} and sits {s.position(main)} of the image.")

    def _on_scene_type(self, msg, ents):
        s = self.scene
        kind = s.scene_type()
        text = normalize(msg)
        out = []
        if self.vlm_ready:
            vqa = self.vlm.answer(self.image, "where was this picture taken?")
            if vqa:
                out.append(f"This picture seems to be taken in {with_article(vqa)}"
                           if not vqa.startswith(("a ", "an ", "the ")) else
                           f"This picture seems to be taken in {vqa}")
        if kind:
            clues = [l for l in s.labels() if l in SCENE_HINTS.get(kind, ())] or s.labels()
            evidence = join_list([count_phrase(l, s.count(l)) for l in clues[:3]])
            out.append(f"Based on the objects I can see ({evidence}), it looks like {kind}")
            if re.search(r"\b(indoor|indoors|outdoor|outdoors|inside|outside)\b", text):
                place = "indoors" if kind in INDOOR_SCENES else "outdoors"
                if kind == "a photo of people":
                    place = None
                if place:
                    out.insert(0, f"It is most likely {place}")
        asked = re.search(r"\bis (?:this|it) (?:a|an) ([a-z ]+?)(?: or |$)", text)
        if asked and kind:
            place = asked.group(1).strip()
            if place in kind:
                return (f"Yes, it looks like {with_article(place)}, based on the objects "
                        f"I can see ({evidence}).")
            elif place not in ("indoor", "outdoor"):
                out.insert(0, f"I do not think this is {with_article(place)}")
        if not out:
            return ("I am not sure what kind of place this is, because the objects I "
                    "recognise do not point to a particular setting.")
        return ". ".join(out) + "."

    def _on_confidence(self, msg, ents):
        s = self.scene
        targets = self._present_targets(ents)
        if not targets:
            targets = s.labels()[:3]
        if not targets:
            return "I did not detect any objects, so I have no confidence scores to share."
        bits = []
        for label in targets:
            best = max(d.confidence for d in s.of(label))
            noun = singular(label) if s.count(label) == 1 else plural(label)
            bits.append(f"{best:.0%} sure about the {noun}")
        self.focus = targets
        return (f"I am {join_list(bits)}. These scores come from the YOLOv8 detector; "
                "anything above about 50% is usually reliable.")

    def _on_open_question(self, msg, ents):
        s = self.scene
        if self.vlm_ready:
            question = self._resolve_pronouns(msg)
            answer = self.vlm.answer(self.image, question)
            if answer:
                return _answer_sentence(question, answer)
        # Detector-only fallback: be honest and give what we do know.
        targets = [t for t in self._targets(ents, allow_focus=True) if s.count(t)]
        if targets and re.search(r"\b(big|small|large|huge|tiny|size)\b", normalize(msg)):
            d = max(s.of(targets[0]), key=lambda x: x.area)
            share = d.area / float(s.width * s.height)
            return (f"The {singular(targets[0])} looks {s.size_word(d)} in this picture: "
                    f"it covers about {share:.0%} of the image.")
        info = ""
        if targets:
            label = targets[0]
            d = max(s.of(label), key=lambda x: x.area)
            n = s.count(label)
            which = "it is" if n == 1 else "the largest one is"
            info = (f" What I can tell you is that there {be_verb(n)} "
                    f"{count_phrase(label, n)}, and {which} {s.position(d)} of the image.")
        elif s.detections:
            info = f" What I can tell you is that I can see {self._inventory()}."
        return ("I cannot answer that kind of question with object detection alone."
                + info + " Enable the BLIP question-answering model "
                "(see the README) for open-ended questions.")

    # ===================================================== helper methods
    def _targets(self, ents: Entities, allow_focus: bool) -> list[str]:
        if ents.objects:
            return ents.objects
        if allow_focus and ents.has_pronoun and self.focus:
            return list(self.focus)
        return []

    def _present_targets(self, ents: Entities):
        """Objects asked about that are in the image.

        Returns None if the user named objects but none are in the image,
        [] if the user named nothing and there is no conversation focus.
        """
        if ents.categories and not ents.objects:
            found = [l for l in self.scene.labels() if l in CATEGORIES[ents.categories[0]]]
            return found or None
        targets = self._targets(ents, allow_focus=True)
        if not targets:
            return self.focus[:] if self.focus else []
        present = [t for t in targets if self.scene.count(t)]
        if not present:
            return None
        return present

    def _spread(self, dets) -> str:
        """Where a group of objects is: ', both on the left' or ': one ..., two ...'."""
        spots = Counter(self.scene.position(d) for d in dets)
        if len(spots) == 1:
            everyone = "both" if len(dets) == 2 else f"all {number_word(len(dets))}"
            return f", {everyone} {next(iter(spots))} of the image"
        return ": " + join_list([f"{number_word(c) if c > 1 else 'one'} {p}"
                                 for p, c in spots.most_common()])

    def _missing_object_answer(self, ents):
        names = ([plural(o) for o in ents.objects] or
                 [CATEGORY_FORMS[c][1] for c in ents.categories])
        return (f"I cannot see any {join_list(names, 'or')} in this image, "
                f"so I cannot answer that. I can see {self._inventory()}."
                if self.scene.detections else
                f"I cannot see any {join_list(names, 'or')} in this image.")

    def _unknown_object(self, msg, ents) -> str | None:
        """Return a noun the user asked about that the detector does not know."""
        text = normalize(msg)
        for m in re.finditer(r"\b(?:the|a|an|any|some|many)\s+([a-z]+)", text):
            word = m.group(1)
            if word in _NOT_OBJECTS or word in PHRASE_TO_LABEL or word in COCO_FORMS:
                continue
            if word in {"there", "it", "they", "them", "you", "this", "that", "of", "in"}:
                continue
            return word
        return None

    def _unknown_answer(self, msg, word):
        if self.vlm_ready:
            answer = self.vlm.answer(self.image, msg)
            if answer:
                return _answer_sentence(msg, answer)
        return (f"I was not trained to recognise \"{word}\", so I cannot say for sure. "
                "I can recognise 80 kinds of everyday objects, such as people, "
                "vehicles, animals, furniture, food and electronics. "
                f"In this image I can see {self._inventory()}."
                if self.scene.detections else
                f"I was not trained to recognise \"{word}\", so I cannot say for sure.")

    def _object_color(self, label, det, n_same) -> str:
        """Colour of one object, asking BLIP when it is available."""
        if self.vlm_ready and n_same == 1:
            ans = self.vlm.answer(self.image, f"what color is the {singular(label)}?")
            if ans and ans.split()[0] in _COLOR_SET:
                return ans
        return det.color or "of an unclear colour"

    @staticmethod
    def _clothing_colors(colors) -> str:
        """For people we describe clothing, never skin."""
        if len(colors) == 1:
            return f"The person is dressed mostly in {colors[0]}"
        counts = Counter(colors)
        if len(counts) == 1:
            everyone = "Both" if len(colors) == 2 else f"All {number_word(len(colors))}"
            return f"{everyone} people are dressed mostly in {colors[0]}"
        parts = [f"{number_word(c) if c > 1 else 'one'} {'is' if c == 1 else 'are'} "
                 f"dressed mostly in {col}" for col, c in counts.most_common()]
        return f"Of the {number_word(len(colors))} people, {join_list(parts)}"

    def _group_colors(self, label, colors) -> str:
        counts = Counter(colors)
        n = len(colors)
        if len(counts) == 1:
            everyone = "Both" if n == 2 else f"All {number_word(n)}"
            return f"{everyone} {plural(label)} look {colors[0]}"
        if n == 2:
            return f"One {singular(label)} looks {colors[0]} and the other looks {colors[1]}"
        parts = [f"{number_word(c)} {'looks' if c == 1 else 'look'} {col}"
                 for col, c in counts.most_common()]
        return f"Of the {number_word(n)} {plural(label)}, {join_list(parts)}"

    def _image_colors(self) -> str:
        s = self.scene
        if not s.detections:
            return "I cannot pick out any objects to describe the colours of."
        cols = [d.color for d in s.detections if d.color][:5]
        main = s.main_subject()
        self.focus = [main.label]
        return (f"The main object, the {singular(main.label)}, looks {main.color}. "
                f"The colours I notice on the objects are {join_list(_unique(cols))}.")

    def _inventory(self) -> str:
        s = self.scene
        return join_list([count_phrase(l, s.count(l)) for l in s.labels()])

    def _example_question(self) -> str:
        s = self.scene
        if not s.detections:
            return "Describe the image."
        label = s.labels()[0]
        if s.count(label) > 1:
            return f"How many {plural(label)} are there?"
        return f"Where is the {singular(label)}?"

    def _resolve_pronouns(self, msg: str) -> str:
        """Replace 'it'/'they' by the object in focus (helps BLIP answer)."""
        if not self.focus:
            return msg
        label = self.focus[0]
        noun = f"the {singular(label)}" if self.scene.count(label) == 1 else f"the {plural(label)}"
        msg = re.sub(r"\bit\b", noun, msg, count=1, flags=re.IGNORECASE)
        return re.sub(r"\bthey\b", noun, msg, count=1, flags=re.IGNORECASE)

    @staticmethod
    def _clean_caption(caption: str) -> str:
        caption = caption.strip().rstrip(".")
        caption = re.sub(r"^(there is|this is|an image of|a picture of|a photo of)\s+",
                         "", caption, flags=re.IGNORECASE)
        return caption


# ---------------------------------------------------------------- utilities
_COLOR_SET = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
              "black", "white", "grey", "gray", "silver", "gold", "tan", "beige"}


def _unique(items):
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _group_names(dets) -> list[str]:
    c = Counter(d.label for d in dets)
    return [count_phrase(l, n) for l, n in c.most_common()]


def _is_are(dets) -> str:
    return "is" if len(dets) == 1 else "are"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text) if s]


_YES_NO = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|will|would|should)\b")
_ACTION = re.compile(
    r"^what (is|are) (.+?) (holding|wearing|eating|carrying|looking at|playing with|"
    r"riding|drinking|reading|watching|playing|sitting on|standing on|using)\b")
_DOING = re.compile(r"^what (is|are) (.+?) doing\b")


def _answer_sentence(question: str, answer: str) -> str:
    """Wrap a short VQA answer ("frisbee", "yes") into a full sentence."""
    q = normalize(question)
    a = answer.strip().rstrip(".").lower()
    if _YES_NO.match(q):
        if a == "yes":
            return "Yes, it looks that way."
        if a == "no":
            return "No, it does not look that way."
    m = _ACTION.match(q)
    if m:
        be, subj, verb = m.groups()
        return f"{_subject(subj)} {be} {verb} {_noun_phrase(a)}."
    m = _DOING.match(q)
    if m:
        be, subj = m.groups()
        return f"{_subject(subj)} {be} {a}."
    if a.isdigit():
        return f"I think the answer is {number_word(int(a))}."
    return f"I think the answer is {_noun_phrase(a)}."


_PRONOUN_SUBJ = {"he": "He", "she": "She", "they": "They", "it": "It", "you": "You"}


def _subject(subj: str) -> str:
    return _PRONOUN_SUBJ.get(subj, _cap(subj))


def _noun_phrase(a: str) -> str:
    """Add an article to a bare singular noun answer ("frisbee" -> "a frisbee")."""
    words = a.split()
    if len(words) <= 2 and words[0] not in {"a", "an", "the", "his", "her", "their",
                                            "nothing", "yes", "no"} \
            and not a.endswith("s") and not words[0].endswith("ing") \
            and words[0] not in _COLOR_SET:
        return with_article(a)
    return a
