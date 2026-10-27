# 👁️ NetraChat: Conversational Image Recognition Chatbot

**Problem Statement 1604, Bharat Electronics Limited (BEL)**

NetraChat ("Netra" means *eye* in Sanskrit) is a chatbot that **looks at an image you upload** and **holds a conversation about it**. It recognises the objects in the picture, understands your questions in everyday English, remembers what you were talking about, and always replies in grammatically correct sentences.

```
NetraChat: Thanks for the image! I can see four people and a bus. Ask me anything
           about it, for example: "How many people are there?"
You:       what colour is the bus?
NetraChat: The bus looks blue.
You:       where is it?
NetraChat: The bus is in the centre of the image.
You:       what is next to it?
NetraChat: Around the bus, I can see three people in front of it.
You:       are there any animals?
NetraChat: No, I cannot see any animals in this image.
You:       is there a helicopter?
NetraChat: I was not trained to recognise "helicopter", so I cannot say for sure. I can
           recognise 80 kinds of everyday objects, such as people, vehicles, animals,
           furniture, food and electronics. In this image I can see four people and a bus.
You:       where was this photo taken?
NetraChat: Based on the objects I can see (a bus), it looks like a street scene.
```

---

## 1. How it meets the expected solution

| Requirement from the problem statement | How NetraChat does it |
|---|---|
| Detect the objects in the image | **YOLOv8** deep-learning detector (80 object classes), plus colour, position, size and spatial relations for every object |
| Have a related dialogue about the image | A **dialogue manager** with 18 trained intents: describe, list, count, exists, colour, location, relations, main subject, scene type, confidence, open questions, and small talk |
| Understand the sender's messages | A **trained intent classifier** (TF-IDF features and Logistic Regression, with typo-tolerant character n-grams) plus an **entity extractor** with 150+ synonyms ("kids" → person, "sofa" → couch, "animals" → every animal class) |
| Predict which response is relevant | Conversation memory: **"it", "they"** refer to the last object discussed, and **"what about the cars?"** repeats the previous kind of question |
| Replies correct lexically and grammatically | A **grammar layer**: a/an articles, singular/plural agreement ("1 bus" / "2 buses", "a person" / "three people"), number words, lists with commas, and a final polish of capitals and punctuation. Tests check every reply |
| "After training" | The intent classifier is trained by `scripts/train_intent.py`; the detector can be fine-tuned on your own objects with `scripts/train_detector.py` |
| Open-ended questions ("what is the man holding?") | Optional **BLIP** vision-language models for captions and visual question answering (`--blip`) |

---

## 2. Step-by-step setup (assumes no prior Python experience)

### Step 1: Install Python
1. Go to <https://www.python.org/downloads/> and download **Python 3.10 or newer** (3.11 or 3.12 recommended).
2. **Windows:** in the installer, tick **"Add python.exe to PATH"**, then click *Install Now*.
3. Check that it worked. Open a terminal (Windows: press <kbd>Win</kbd>, type `cmd`, press Enter; macOS: open *Terminal*; Linux: open your terminal) and type:
   ```bash
   python --version
   ```
   You should see something like `Python 3.12.4`. On macOS/Linux, if `python` is not found, use `python3` everywhere below.

### Step 2: Get the project
Either download the ZIP from GitHub (green **Code** button → *Download ZIP*) and unzip it, or use git:
```bash
git clone https://github.com/Rajan3103/NetraChat.git
cd NetraChat
```
All the commands below are run **inside this folder**.

### Step 3: Create a virtual environment
A virtual environment is a private box of Python packages just for this project.
```bash
python -m venv .venv
```
Activate it (you need to do this every time you open a new terminal):
- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

You will now see `(.venv)` at the start of your terminal line.

### Step 4: Install the packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This downloads PyTorch, YOLOv8, Gradio and the other packages (about 1–3 GB, so it can take a few minutes).

> Have an NVIDIA GPU? Install the GPU version of PyTorch first, following <https://pytorch.org/get-started/locally/>. Everything also works on a normal CPU.

### Step 5: Train the language model
```bash
python scripts/train_intent.py
```
This trains the intent classifier, prints its accuracy, and saves it to `models/intent_model.joblib`. It takes a few seconds. (A trained copy is already included, so this step is optional.)

### Step 6: Get sample images (optional)
```bash
python scripts/download_samples.py
```

### Step 7: Run the chatbot
**Web app (recommended):**
```bash
python app.py
```
Open **<http://127.0.0.1:7860>** in your browser. Upload an image on the left and chat on the right. The lower image shows the boxes the detector found. The first run downloads the YOLO model (about 22 MB) automatically.

**Terminal version:**
```bash
python chat_cli.py samples/bus.jpg
```
Type `/image path/to/another.jpg` to switch images, and `/quit` to leave.

### Step 8 (optional): Turn on BLIP for open-ended questions
```bash
python app.py --blip
```
The first run downloads two BLIP models (about 2 GB) from Hugging Face. With BLIP on, NetraChat also writes a caption for each image and answers free questions such as *"What is the man holding?"* → *"The man is holding a frisbee."* Without BLIP, NetraChat tells you honestly that it cannot answer those questions.

### Step 9: Run the tests
```bash
python -m pytest -q
```
You should see `36 passed`.

---

## 3. What you can ask

| Type | Examples |
|---|---|
| Describe | "Describe the image", "What's in this pic?", "Tell me about the dog" |
| List | "What objects can you see?", "List everything" |
| Count | "How many people are there?", "How many people and dogs?", "How many animals?" |
| Exists | "Is there a cat?", "Any vehicles?", "Is there a laptop or a TV?" |
| Colour | "What colour is the car?", "Is the bus red?", "What colour are they?" |
| Location | "Where is the dog?", "Where are the people?" |
| Relations | "What is next to the bus?", "What is left of the cat?", "Where is the dog relative to the cat?" |
| Main subject / size | "What is the main object?", "Is the couch big?" |
| Scene | "Where was this taken?", "Is this indoors or outdoors?", "Is this a kitchen?" |
| Confidence | "How sure are you?", "How confident are you about the tie?" |
| Follow-ups | "Where is **it**?", "What colour are **they**?", "**What about** the cars?" |
| Open questions (BLIP) | "What is the man holding?", "What are they doing?", "Is it raining?" |
| Small talk | "Hi", "Who are you?", "Help", "Thanks", "Bye" |

---

## 4. How it works

```
                ┌──────────────────────── IMAGE ────────────────────────┐
 upload ──►     │ YOLOv8 detector ─► boxes + labels + confidence        │
                │      └─► colour (HSV pixel voting) · position · size  │ ─► Scene
                │ BLIP captioner (optional) ─► one-sentence caption     │
                └───────────────────────────────────────────────────────┘
                ┌──────────────────────── MESSAGE ──────────────────────┐
 question ──►   │ Entity extractor ─► objects, categories, colours,     │
                │                     relation words, pronouns          │
                │ Intent classifier (trained) ─► e.g. "count" (0.97)    │
                └───────────────────────────────────────────────────────┘
                                         │
                       Dialogue manager: memory ("it" = last object),
                       follow-ups, picks the handler for the intent,
                       queries the Scene (or BLIP for open questions)
                                         │
                       Grammar layer: articles, plurals, agreement,
                       lists, capitals and punctuation
                                         ▼
                                  grammatical reply
```

### Project layout
```
netrachat/
├── app.py                    ← web app (Gradio)
├── chat_cli.py               ← terminal chat
├── requirements.txt
├── data/intents.json         ← training sentences for the intent classifier
├── models/                   ← trained intent model (+ downloaded YOLO weights)
├── netrachat/
│   ├── vocabulary.py         ← 80 object classes, plurals, synonyms, categories
│   ├── scene.py              ← positions, relations, main subject, scene type
│   ├── vision/
│   │   ├── detector.py       ← YOLOv8 wrapper + colour estimation
│   │   ├── captioner.py      ← optional BLIP caption + question answering
│   │   └── annotate.py       ← draws boxes on the image
│   ├── nlu/
│   │   ├── entities.py       ← finds objects / colours / relations in text
│   │   └── intent.py         ← trainable intent classifier
│   ├── nlg/grammar.py        ← grammar-correct sentence building
│   └── dialogue/manager.py   ← the conversation brain
├── scripts/
│   ├── train_intent.py       ← train + evaluate the language model
│   ├── train_detector.py     ← fine-tune YOLO on your own objects
│   └── download_samples.py
└── tests/test_chatbot.py     ← 36 automated tests
```

---

## 5. Results

| Component | Result |
|---|---|
| Intent classifier, 5-fold cross-validation on 608 sentences it did not train on (18 intents) | **80.9%** overall, **89.1%** on image questions, 94–99% on count / exists / colour |
| Object detector (YOLOv8s, official COCO benchmark) | 44.9 mAP50-95 on 80 classes. Use `--weights yolov8m.pt` for 50.2 |
| Grammar | Every reply checked by tests: capital letter, end punctuation, no double spaces, correct plural/article forms |
| Speed (CPU) | about 0.2–0.5 s per image for detection; replies are instant |

The hardest intents to tell apart are the small-talk ones ("ok" vs "thanks") and off-topic requests. Misclassifying these does little harm. To improve accuracy, add more example sentences to `data/intents.json` and run `python scripts/train_intent.py` again.

---

## 6. Customising

- **Better detection:** `python app.py --weights yolov8m.pt` (slower, more accurate).
- **New object types** (e.g. drones, helmets, PCB parts): label your images and follow the instructions at the top of `scripts/train_detector.py`, then run `python app.py --weights runs/detect/netrachat/weights/best.pt`. Add the new names to `COCO_FORMS` in `netrachat/vocabulary.py`.
- **Teach it new ways of asking:** add sentences to `data/intents.json` and retrain.
- **Share with others:** `python app.py --share` gives a temporary public link.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Reinstall Python and tick "Add to PATH" (Windows), or use `python3` |
| `ModuleNotFoundError` | Activate the virtual environment (Step 3), then `pip install -r requirements.txt` |
| YOLO weights fail to download | Check your internet/proxy, or download `yolov8s.pt` from the Ultralytics GitHub releases and put it in `models/` |
| `--blip` says the models are unavailable | Hugging Face must be reachable for the first download; NetraChat keeps working without it |
| Port 7860 is busy | `python app.py --port 7861` |

## 8. Limitations
- The default detector knows 80 COCO object classes. Other objects need fine-tuning or `--blip`.
- Colours are estimated from pixels, so lighting and shadows can mislead them. With `--blip`, the colour comes from the vision-language model instead.
- For people, NetraChat describes clothing colour only, never skin.
