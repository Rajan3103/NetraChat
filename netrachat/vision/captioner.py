"""
Optional "deep" vision-language models (BLIP from Salesforce).

  * Captioning: writes one sentence describing the whole image.
  * Visual Question Answering (VQA): answers a free-form question about the
    image, e.g. "what is the man holding?" or "is it raining?".

These models are large (about 1 GB each) and are downloaded from Hugging Face
the first time they are used. If they cannot be loaded (no internet, not
installed), NetraChat still works: it answers from the object detector alone.
"""

from __future__ import annotations

import logging

from PIL import Image

log = logging.getLogger(__name__)

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
VQA_MODEL = "Salesforce/blip-vqa-base"


class VisionLanguageModel:
    def __init__(self, caption_model: str = CAPTION_MODEL, vqa_model: str = VQA_MODEL):
        self.caption_model_name = caption_model
        self.vqa_model_name = vqa_model
        self._caption = None
        self._vqa = None
        self.available = True
        self.error = None

    def _load(self):
        if self._caption is not None or not self.available:
            return
        try:
            import torch
            from transformers import (BlipForConditionalGeneration,
                                      BlipForQuestionAnswering, BlipProcessor)

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._caption = (
                BlipProcessor.from_pretrained(self.caption_model_name),
                BlipForConditionalGeneration.from_pretrained(
                    self.caption_model_name).to(self._device).eval(),
            )
            self._vqa = (
                BlipProcessor.from_pretrained(self.vqa_model_name),
                BlipForQuestionAnswering.from_pretrained(
                    self.vqa_model_name).to(self._device).eval(),
            )
        except Exception as exc:  # no internet, missing package, out of memory...
            self.available = False
            self.error = str(exc)
            self._caption = self._vqa = None
            log.warning("BLIP models unavailable, using detector-only mode: %s", exc)

    def _generate(self, pair, image, text=None) -> str | None:
        import torch

        processor, model = pair
        inputs = processor(image.convert("RGB"), text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    def caption(self, image: Image.Image) -> str | None:
        self._load()
        if not self.available:
            return None
        return self._generate(self._caption, image)

    def answer(self, image: Image.Image, question: str) -> str | None:
        self._load()
        if not self.available:
            return None
        return self._generate(self._vqa, image, question)
