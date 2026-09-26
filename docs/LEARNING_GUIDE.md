# The Complete Learning Guide to This Repo

> **Who this is for:** someone who is new to Python, new to pandas, new to
> machine learning, and new to Vue. You know some words. That's enough.
>
> **How to read it:** top to bottom the first time. It is long on purpose.
> Every file in the repo is explained, most of them line by line. When a
> symbol (like `[`, `{`, `**`, `?.`, `=>`) appears for the first time, it is
> explained right there or in the "brackets and symbols" chapter.
>
> Keep the code open next to this guide. When the guide says
> `dataset.py` line 43, open that file and look at line 43.

---

## Table of contents

- [Part 0 — The whole story in one page](#part-0--the-whole-story-in-one-page)
- [Part 1 — The biggest question: does the model learn by itself?](#part-1--the-biggest-question-does-the-model-learn-by-itself)
- [Part 2 — Map of every file](#part-2--map-of-every-file)
- [Part 3 — Words you will meet (glossary)](#part-3--words-you-will-meet-glossary)
- [Part 4 — Brackets and symbols: what every squiggle means](#part-4--brackets-and-symbols-what-every-squiggle-means)
- [Part 5 — Python crash course (only what this repo uses)](#part-5--python-crash-course-only-what-this-repo-uses)
- [Part 6 — The tools around the code (uv, npm, .env, git)](#part-6--the-tools-around-the-code-uv-npm-env-git)
- [Part 7 — The data: Sentiment140 and `download_data.py`](#part-7--the-data-sentiment140-and-download_datapy)
- [Part 8 — pandas crash course](#part-8--pandas-crash-course)
- [Part 9 — `dataset.py` line by line](#part-9--datasetpy-line-by-line)
- [Part 10 — Machine learning explained from zero](#part-10--machine-learning-explained-from-zero)
- [Part 11 — `train.py` line by line](#part-11--trainpy-line-by-line)
- [Part 12 — `predict.py` line by line](#part-12--predictpy-line-by-line)
- [Part 13 — How the web works (just enough)](#part-13--how-the-web-works-just-enough)
- [Part 14 — `db.py` and SQL line by line](#part-14--dbpy-and-sql-line-by-line)
- [Part 15 — `facebook.py` line by line](#part-15--facebookpy-line-by-line)
- [Part 16 — `get_token.py` line by line](#part-16--get_tokenpy-line-by-line)
- [Part 17 — `app.py` (Flask) line by line](#part-17--apppy-flask-line-by-line)
- [Part 18 — JavaScript crash course](#part-18--javascript-crash-course)
- [Part 19 — Frontend tooling: Node, npm, Vite, Tailwind](#part-19--frontend-tooling-node-npm-vite-tailwind)
- [Part 20 — Vue crash course](#part-20--vue-crash-course)
- [Part 21 — Frontend files line by line](#part-21--frontend-files-line-by-line)
- [Part 22 — The components line by line](#part-22--the-components-line-by-line)
- [Part 23 — Full journeys: follow one action through every file](#part-23--full-journeys-follow-one-action-through-every-file)
- [Part 24 — Things I noticed while writing this guide](#part-24--things-i-noticed-while-writing-this-guide)
- [Part 25 — Experiments to try (the best way to learn)](#part-25--experiments-to-try-the-best-way-to-learn)
- [Part 26 — FAQ](#part-26--faq)
- [Part 27 — Future extension: Bangla, Banglish, sarcasm, and higher accuracy](#part-27--future-extension-bangla-banglish-sarcasm-and-higher-accuracy)

---

## Part 0 — The whole story in one page

Imagine you own a shop that has a Facebook Page. People write comments under
your posts. Some are happy ("love this!"), some are angry ("broke after two
days"), and some are just questions ("what is the price?"). You want a screen
that tells you, at a glance, how people feel.

Reading 1,000 comments by hand is slow. So we teach a computer program to
guess the feeling of a comment. That guess is called **sentiment**, and
teaching the computer is called **machine learning (ML)**.

The project has **two separate lives**. This is the most important idea in the
whole guide:

```
 LIFE 1: TRAINING  (you run it by hand, once in a while)
 ──────────────────────────────────────────────────────
  Stanford website
        │  download_data.py
        ▼
  1.6 million old tweets, each already marked happy/sad  (a CSV file)
        │  dataset.py  (clean the text, split into 3 piles)
        ▼
  train.py  (the computer studies the tweets and learns word weights)
        │
        ▼
  models/sentiment_pipeline.joblib   ← the "brain", saved as a file
  models/model_meta.json             ← the brain's report card


 LIFE 2: USING  (the web app, runs every time you open the dashboard)
 ──────────────────────────────────────────────────────────────────
  You click "View latest expression" in the browser (Vue)
        │  api.js sends a request
        ▼
  app.py (Flask web server)
        │  facebook.py asks Facebook for comments
        │  db.py saves them in a small database file (app.db)
        │  predict.py loads the brain file and guesses each comment's mood
        │  db.py saves the guesses and counts everything up
        ▼
  JSON (plain text data) goes back to the browser
        │
        ▼
  Vue components draw tiles, a donut chart, a bar chart, and lists
```

Life 1 makes the brain. Life 2 only **uses** the brain. Life 2 never changes
the brain.

---

## Part 1 — The biggest question: does the model learn by itself?

**Short answer: No.** In this project the model is trained **once** (each time
you run `train.py`), saved to a file, and then only **used** for predictions.
It does **not** learn from your Facebook comments. It does **not** get smarter
by itself over time.

Here is an analogy. Think of a student who studies for an exam from a textbook.

- **Training** = the student studying the textbook (1.6 million example tweets
  with the correct answer written next to each one).
- **The saved model file** = what is inside the student's head after studying.
- **Prediction** = the student answering new questions (your Facebook
  comments). While answering, the student does **not** open the textbook
  again and does **not** learn anything new. They just use what they already
  know.

### Why can't it just learn from my Facebook comments?

To learn, a model needs **examples with the correct answer attached**. These
are called **labels**. A tweet in Sentiment140 comes with a label: 0 (sad) or
4 (happy). Your Facebook comments come with **no label**. Nobody has written
"this one is negative" next to them. If the model learned from its **own**
guesses, it would just repeat its own mistakes louder. That is like a student
marking their own exam and then studying their own wrong answers.

### What would it take to make it learn from new data?

You would need to:

1. Collect new comments.
2. Have a **human** mark each one as positive / negative / neutral (label it).
3. Add those labeled comments to the training data.
4. Run `train.py` again. This creates a **new** brain file that replaces the old one.
5. Restart the web app (`app.py`) so it loads the new file. (It keeps the old
   one in memory until restart. Part 12 explains why.)

That is called **retraining**, and it is always something **you** start by
hand in this project. There are fancier setups where a model updates itself
continuously ("online learning"), but this project deliberately doesn't do
that. It is built for learning, and a model that changes by itself is much
harder to understand and to trust.

### How often should I retrain?

Only when something changes:
- you changed the cleaning or the settings in `train.py`, or
- you got new labeled data, or
- you ran a quick small experiment (`--sample 20000`) that **overwrote** the
  good model with a worse one. `train.py` always overwrites the saved file, so
  run it again at the size you actually want.

---

## Part 2 — Map of every file

```
sentiment-analysis/
├── CLAUDE.md                 notes for the AI assistant about this project
├── README.md                 the "how to run it" page
├── .gitignore                list of files git must NOT save (secrets, big data)
├── docs/
│   ├── dashboard-light.png   screenshot
│   ├── dashboard-dark.png    screenshot
│   └── LEARNING_GUIDE.md     ← you are here
│
├── backend/                  everything that runs in Python on your computer
│   ├── .python-version       says "use Python 3.12"
│   ├── pyproject.toml        list of Python libraries this project needs
│   ├── uv.lock               exact versions of those libraries (auto-made)
│   ├── .venv/                the installed libraries live here (auto-made)
│   ├── .env.example          template for your secret settings
│   ├── .env                  YOUR secret settings (Facebook token). Never share.
│   │
│   ├── download_data.py      step 1: download the tweet dataset
│   ├── dataset.py            step 2: read, clean, split the tweets
│   ├── train.py              step 3: teach the model, test it, save it
│   ├── predict.py            load the saved model and guess moods
│   │
│   ├── db.py                 read/write the SQLite database file
│   ├── facebook.py           talk to Facebook's servers
│   ├── get_token.py          one-time helper to get a Facebook access token
│   ├── app.py                the web server (Flask) that joins it all
│   │
│   ├── data/                 (auto-made, not in git)
│   │   ├── training.1600000.processed.noemoticon.csv   the tweets, 239 MB
│   │   ├── sentiment140.zip                             the download (safe to delete after unzipping)
│   │   └── app.db                                       the database
│   ├── models/               (auto-made by train.py, not in git)
│   │   ├── sentiment_pipeline.joblib   the trained brain (~1 MB)
│   │   └── model_meta.json             the brain's report card
│   └── static/               (auto-made by `npm run build`, not in git)
│       ├── index.html
│       └── assets/index-XXXX.js, index-XXXX.css
│
└── frontend/                 everything that runs in the browser
    ├── package.json          list of JavaScript libraries + commands
    ├── package-lock.json     exact versions (auto-made)
    ├── node_modules/         the installed libraries (auto-made)
    ├── index.html            the one HTML page
    ├── vite.config.js        settings for the Vite build tool
    ├── tailwind.config.js    settings for Tailwind (CSS helper)
    ├── postcss.config.js     settings for PostCSS (runs Tailwind)
    └── src/
        ├── main.js           starts the Vue app
        ├── App.vue           the main page, holds all the pieces
        ├── api.js            the only file that talks to the backend
        ├── format.js         small helpers for dates and percents
        ├── theme.js          light/dark mode
        ├── style.css         colors and base styles
        └── components/
            ├── KpiTiles.vue        the 4 number boxes at the top
            ├── SentimentDonut.vue  the donut (ring) chart
            ├── TimelineChart.vue   the bar chart by day
            ├── NeedsAttention.vue  list of the most negative comments
            ├── PostBreakdown.vue   table of posts with color bars
            ├── TryModel.vue        "type a sentence and test the model"
            └── ModelInsights.vue   "what the model learned"
```

### Who calls whom

An arrow `A → B` means "A uses B". A file only ever uses files **below** it in
this list. That rule keeps things simple: if you change `db.py`, you know that
`dataset.py` can't be affected.

```
app.py ─────→ predict.py ─→ dataset.py (only for clean_text)
   │    └───→ db.py
   └────────→ facebook.py

train.py ───→ dataset.py

download_data.py   (stands alone)
get_token.py       (stands alone, writes .env)
```

`app.py` never uses `train.py`. The web app must never train. `predict.py` is
the "doorway" between the ML world and the web world.

---

## Part 3 — Words you will meet (glossary)

Skim this now; come back when a word confuses you.

| Word | Simple meaning |
|---|---|
| **Sentiment** | The feeling in a piece of text: positive, negative, or neutral. |
| **Machine learning (ML)** | Letting a computer find patterns in examples instead of a human writing rules. |
| **Model** | The thing that makes the guesses. Here: a list of numbers (weights), one per word. |
| **Training / fitting** | The step where the model looks at examples and picks its numbers. `fit` is the code word. |
| **Prediction / inference** | Using a trained model on new text. `predict` is the code word. |
| **Dataset** | A big table of examples. |
| **Label** | The correct answer attached to an example (0 = negative, 1 = positive). |
| **Feature** | One thing the model can look at. Here: "does the word *love* appear, and how much?" |
| **Vectorizer** | The tool that turns text into numbers (features). |
| **TF-IDF** | The specific way we turn text into numbers (Part 10). |
| **n-gram** | A group of n words in a row. unigram = 1 word ("bad"), bigram = 2 words ("not bad"). |
| **Stop words** | Very common words like "the", "is", "and", often thrown away. |
| **Logistic Regression** | The kind of model we use. It adds up word weights and squashes the total into a probability. |
| **Probability** | A number from 0 to 1. 0.9 = "very likely". |
| **Weight / coefficient** | How much one word pushes toward positive (+) or negative (−). |
| **Pipeline** | Vectorizer + model glued together into one object. |
| **Train / validation / test split** | Three separate piles of data: one to study, one to practice, one for the final exam. |
| **Accuracy** | Out of all guesses, the fraction that were right. |
| **ROC-AUC** | A score from 0.5 (coin flip) to 1.0 (perfect) for how well the model *ranks* happy above sad. |
| **Cross-validation (CV)** | Training and testing several times on different slices to see how steady the score is. |
| **Overfitting** | Memorizing the examples instead of learning the pattern. Good on practice, bad on new data. |
| **joblib** | A Python library that saves Python objects (like our pipeline) to a file and loads them back. |
| **pandas** | A Python library for working with tables (like Excel inside Python). |
| **NumPy** | A Python library for fast math on lists of numbers. pandas and scikit-learn are built on it. |
| **scikit-learn / sklearn** | The Python library that provides TF-IDF, Logistic Regression, splitting, and scoring. |
| **CSV** | "Comma-separated values": a text file where each line is a row and commas separate columns. |
| **JSON** | A text format for data: `{"name": "Rahim", "age": 12}`. Both Python and JavaScript read it easily. |
| **API** | A set of URLs that a program (not a person) calls to get or send data. |
| **Endpoint** | One URL of an API, like `/api/analytics`. |
| **Backend** | The code running on the server (here: Python/Flask). |
| **Frontend** | The code running in the browser (here: Vue). |
| **Flask** | A small Python library for making web servers. |
| **Vue** | A JavaScript library for building web pages from reusable pieces (components). |
| **Component** | A reusable piece of a web page, like a Lego brick. One `.vue` file = one component. |
| **SPA** | "Single-page application": one HTML page, and JavaScript redraws parts of it. |
| **Database / SQLite** | An organized store of tables. SQLite keeps it all in one file (`app.db`). |
| **SQL** | The language you use to ask a database questions. |
| **Token** | A secret password-like string that proves to Facebook you are allowed to read the Page. |
| **Graph API** | Facebook's API. |
| **Environment variable** | A named setting that lives outside the code (e.g. `FB_PAGE_ID`). |
| **Library / package / dependency** | Code someone else wrote that we download and use. |
| **Module** | One Python file (or JS file) that can be imported. |
| **Function** | A named block of code you can run by calling its name. |
| **Argument / parameter** | A value you hand to a function. |
| **Return value** | The value a function hands back. |
| **Import** | Bringing code from another file or library into this one. |
| **Seed** | A number that makes "random" choices repeatable. Same seed → same "random" result. |

---

## Part 4 — Brackets and symbols: what every squiggle means

The same bracket can mean different things depending on where it sits. Here
is every one you will see in this repo.

### Round brackets `( )`

| Where | Meaning | Example from the repo |
|---|---|---|
| After a name | **Call** (run) a function | `clean_text(t)` runs `clean_text` on `t` |
| After `def name` | List of **parameters** the function accepts | `def clean_text(text: str)` |
| Around math | **Grouping**, like in school math | `(page.value - 1) * PER_PAGE` |
| Bare, with commas | A **tuple** (a fixed list that can't change) | `ngram_range=(1, 2)` |
| Around a long expression | Just lets it continue over several lines | `return (\n a \n)` |
| After `class Name` | The **parent class** it inherits from | `class GraphError(RuntimeError)` |

### Square brackets `[ ]`

| Where | Meaning | Example |
|---|---|---|
| Alone, with commas | A **list** (an ordered, changeable collection) | `["sentiment", "id", "date"]` |
| After a variable | **Get an item** by position or key | `pages[0]` (first item), `post["id"]` |
| After a variable, with `:` | **Slice** (a piece of) | `text[:400]` = first 400 characters |
| After a type | **Type detail** | `list[str]` = "a list of strings" |
| After a DataFrame, containing a list | **Pick columns** | `df[["sentiment", "text"]]` |
| After a DataFrame, containing True/False | **Filter rows** | `df[df["clean_text"].str.len() > 0]` |
| After a NumPy array, with a comma | **[rows, columns]** | `proba[:, 1]` = every row, column 1 |
| Inside a regex | A **set of characters** | `[a-z]` = any letter a to z |

**Positions start at 0, not 1.** `pages[0]` is the first page. `pages[1]` is
the second. Negative numbers count from the end: `x[-1]` is the last item.

**Slices** `x[start:stop:step]`:
- `x[2:5]` → items 2, 3, 4 (stop is *not* included).
- `x[:5]` → from the start to 4.
- `x[-15:]` → the last 15 items.
- `x[::-1]` → the whole thing **reversed** (step of −1 walks backwards).

### Curly brackets `{ }`

| Where | Meaning | Example |
|---|---|---|
| Python, with `key: value` | A **dict** (dictionary, a lookup table) | `{0: "negative", 1: "positive"}` |
| Python, inside an f-string | **Insert a value** into text | `f"{dropped:,} rows"` |
| Python, without colons | A **set** (unique items, no order) | `{1, 2, 3}` (not used much here) |
| JavaScript, after `function`/`=>` | A **block** of code | `function apply() { ... }` |
| JavaScript, with `key: value` | An **object** (like a Python dict) | `{ method: 'POST' }` |
| JavaScript, inside `${ }` in backticks | **Insert a value** into text | `` `${path} failed` `` |
| Vue template `{{ }}` | **Show a value** on the page | `{{ tile.label }}` |
| CSS | The block of **style rules** for a selector | `body { background: ...; }` |
| JS import | **Pick named items** from a module | `import { ref, computed } from 'vue'` |
| JS destructuring | **Pull fields out** of an object | `const { theme, toggle } = useTheme()` |

### Angle brackets `< >`

- In HTML and Vue templates they make **tags**: `<div>`, `</div>` (the `/`
  means "closing tag"), `<canvas />` (a tag that opens and closes itself).
- In math they compare: `<` less than, `>` greater than, `<=` less or equal,
  `>=` greater or equal.
- In f-string formats they **align text**: `{name:<14}` = left-align in 14
  spaces, `{x:>9}` = right-align in 9 spaces.

### Other symbols

| Symbol | Language | Meaning |
|---|---|---|
| `=` | both | **Put** a value into a name ("assign"). `x = 5`. |
| `==` | Python | **Is equal?** gives True/False. |
| `===` | JavaScript | **Is equal?** (strict; the one you should use in JS). |
| `!=` / `!==` | Py / JS | Is **not** equal? |
| `!` | JavaScript | **not**. `!loading` = "not loading". |
| `not`, `and`, `or` | Python | Logic words. |
| `&&`, `\|\|` | JavaScript | and, or. |
| `#` | Python | **Comment** — everything after it on that line is ignored by Python. |
| `//` | JavaScript | Comment (in JS). In Python `//` means whole-number division: `7 // 2` = 3. |
| `/* ... */` | JS, CSS | Multi-line comment. |
| `<!-- ... -->` | HTML/Vue template | Comment. |
| `"""..."""` | Python | A multi-line string. When it's the first thing in a file or function, it's a **docstring** (built-in documentation). |
| `.` | both | **"the thing inside"**. `df.sample` = the `sample` function that belongs to `df`. |
| `:` | Python | Ends a line that **starts a block** (`def`, `if`, `for`, `with`, `try`). The block is the indented lines below. |
| `->` | Python | "this function **returns** ...". `def f() -> dict:` returns a dict. Only a hint for humans and tools. |
| `=>` | JavaScript | **Arrow function**. `(x) => x * 2` is a tiny function that doubles. |
| `*` | both | Multiply. In Python, also "unpack a list" in some places. |
| `**` | Python | Power (`2 ** 3` = 8), **or** "unpack a dict into keyword arguments" (see Part 5). |
| `...` | JavaScript | **Spread**: copy all items/fields out. `{ ...d, x: 1 }`. |
| `?.` | JavaScript | **Optional chaining**: "if this exists, go inside; otherwise give `undefined` instead of crashing". |
| `??` | JavaScript | "use the right side **only if** the left is `null` or `undefined`". |
| `? :` | JavaScript | **Ternary** (a one-line if/else): `cond ? a : b`. |
| `x if cond else y` | Python | Python's one-line if/else. |
| `@` | Python | A **decorator** (Part 5). In Vue templates `@click` means "on click". |
| `:` in Vue templates | Vue | `:style="..."` = "this attribute's value is JavaScript, not plain text". |
| `\|` | Python types | "or": `int \| None` = "an int or nothing". In regex: "or". |
| `_` | Python | In numbers, just for readability: `200_000` is 200000. At the start of a name (`_get`) it means "private, internal use". |
| `\` | strings/regex | **Escape**: gives the next character a special meaning (`\n` newline, `\s` whitespace). |
| `r"..."` | Python | **Raw** string: backslashes are kept as-is. Used for regex. |
| `f"..."` | Python | **f-string**: text with `{}` holes that get filled with values. |
| `` `...` `` | JavaScript | **Template literal**: text with `${}` holes. |
| `;` | JS, CSS, SQL | End of a statement. Optional in this JS code, required in CSS and SQL. |
| `--name` | CSS | A **custom property** (a CSS variable), e.g. `--positive`. |
| `--flag` | command line | An **option** to a program, e.g. `train.py --full`. |

---

## Part 5 — Python crash course (only what this repo uses)

### 5.1 Running Python code

A `.py` file is a list of instructions. Python reads it top to bottom and does
each one. `uv run python train.py` means "use the project's Python to run
`train.py`".

### 5.2 Values and names (variables)

```python
SEED = 42                     # a whole number (int)
NEUTRAL_LOW = 0.40            # a decimal number (float)
GRAPH_VERSION = "v21.0"       # text (str, "string")
debug = True                  # True or False (bool)
cv_scores = None              # "nothing here yet" (None)
```

A **name in ALL_CAPS** is a *constant*: a promise, by convention, that nobody
will change it. Python doesn't enforce this, people just respect it.

### 5.3 Strings (text)

```python
text = "Hello"
text.lower()            # "hello"
text.strip()            # removes spaces at both ends
"a,b".split(",")        # ["a", "b"]
", ".join(["a", "b"])   # "a, b"   (glue a list together)
len("hello")            # 5
```

**f-strings** put values inside text:

```python
dropped = 492
f"dropped {dropped:,} rows"        # "dropped 492 rows"
f"{0.80211:.4f}"                   # "0.8021"  (.4f = 4 digits after the dot)
f"{1234567:,}"                     # "1,234,567"  (, = thousands separator)
f"{'val':<14}"                     # "val           " (left-aligned in 14 spaces)
f"{text!r}"                        # 'hello' with quotes (!r = "show it like code")
```

The part after `:` inside `{}` is the **format spec**:
`<` left, `>` right, a number = width, `,` = thousands commas,
`.4f` = 4 decimals, `.1f` = 1 decimal.

### 5.4 Lists, tuples, dicts, sets

```python
COLUMNS = ["sentiment", "id", "date"]    # list: ordered, can change
COLUMNS[0]                               # "sentiment"
COLUMNS.append("text")                   # add to the end

ngram_range = (1, 2)                     # tuple: ordered, can NOT change

LABEL_MAP = {0: 0, 4: 1}                 # dict: key -> value
LABEL_MAP[4]                             # 1
LABEL_MAP.get(7, "missing")              # "missing"  (.get never crashes)
LABEL_MAP.items()                        # pairs: (0, 0), (4, 1)
```

**`.get()` vs `[]`:** `post["message"]` **crashes** (KeyError) if there is no
`message`. `post.get("message", "")` gives `""` instead. Facebook often leaves
fields out, so `facebook.py` uses `.get()` a lot.

### 5.5 `if`, `for`, `while`

```python
if score >= high:
    sentiment = "positive"
elif score <= low:          # elif = "else if"
    sentiment = "negative"
else:
    sentiment = "neutral"
```

**Indentation (the spaces at the start of a line) matters in Python.** It is
how Python knows which lines belong to the `if`. Other languages use `{ }`
for this; Python uses spaces.

```python
for name, config in CONFIGS.items():   # repeat once per item
    print(name)

while True:          # repeat forever...
    ...
    if not cursor:
        break        # ...until break jumps out
```

`continue` means "skip the rest of this round, go to the next item".

**Truthy / falsy:** in an `if`, these count as False: `False`, `None`, `0`,
`""` (empty text), `[]` (empty list), `{}` (empty dict). Everything else
counts as True. So `if not texts:` means "if the list is empty".

### 5.6 Functions

```python
def classify_one(text: str) -> dict:
    """Single-comment convenience wrapper."""
    return classify([text])[0]
```

- `def` = "define a function".
- `classify_one` = its name.
- `(text: str)` = it takes one input called `text`; `: str` is a **type
  hint** saying "this should be a string". Hints don't change how the code
  runs; they help humans and editors.
- `-> dict` = it returns a dict (again, just a hint).
- `"""..."""` = the docstring.
- `return` = hand this value back to whoever called the function.

**Default values:** `def load_raw(sample: int | None = None, seed: int = SEED)`
means that if you don't pass `sample`, it is `None`; if you don't pass `seed`,
it is 42.

**Keyword arguments:** you can name the argument when calling:
`load_raw(sample=200000, seed=42)`. The order then doesn't matter.

**`**kwargs` (two stars):**

```python
def build_pipeline(seed=SEED, **vectorizer_kwargs):
    ...TfidfVectorizer(sublinear_tf=True, **vectorizer_kwargs)
```

- In a `def`, `**vectorizer_kwargs` means "collect **any extra named
  arguments** into a dict called `vectorizer_kwargs`".
- In a call, `**vectorizer_kwargs` means the opposite: "**spread** this dict
  out as named arguments".

So `build_pipeline(seed=42, ngram_range=(1, 2), max_features=10000)` makes
`vectorizer_kwargs = {"ngram_range": (1, 2), "max_features": 10000}`, and then
`TfidfVectorizer(sublinear_tf=True, **vectorizer_kwargs)` becomes
`TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=10000)`.

**lambda** is a tiny unnamed function on one line:
`lambda n: results[n]["val"]["accuracy"]` = "given `n`, return that result's
validation accuracy".

### 5.7 Comprehensions (building a list/dict in one line)

```python
cleaned = [clean_text(t or "") for t in texts]
```

Read it as: "make a list; for every `t` in `texts`, put `clean_text(t or "")`
into it". It's the same as:

```python
cleaned = []
for t in texts:
    cleaned.append(clean_text(t or ""))
```

`t or ""` means "use `t`, but if `t` is empty/None, use `""`". This protects
`clean_text` from receiving `None`.

A **dict comprehension** uses `{key: value for ...}`:

```python
{k: int(len(v)) for k, v in splits.items()}
# {"train": 139655, "val": 29926, "test": 29927}
```

### 5.8 Handy built-in functions

| Function | What it does | Example |
|---|---|---|
| `print(x)` | show x in the terminal | |
| `len(x)` | how many items / characters | `len([1,2,3])` = 3 |
| `zip(a, b)` | walk two lists side by side | `for row, result in zip(pending, results)` |
| `enumerate(x, 1)` | give each item a counter starting at 1 | `for i, page in enumerate(pages, 1)` |
| `max(x, key=f)` | the item with the biggest `f(item)` | picks the best config |
| `sum(x)` | add all up | |
| `round(x, 4)` | round to 4 decimals | |
| `abs(x)` | distance from 0 (no minus sign) | `abs(-0.3)` = 0.3 |
| `float(x)`, `int(x)`, `str(x)`, `bool(x)` | convert type | `bool("")` = False |
| `isinstance(v, tuple)` | "is v a tuple?" | |
| `input("App ID: ")` | ask the user to type something | |

### 5.9 `import`

```python
import re                          # bring in the whole `re` module; use re.sub(...)
from pathlib import Path           # bring in just `Path`
import pandas as pd                # bring in pandas, call it `pd` for short
from dataset import clean_text     # bring in clean_text from OUR file dataset.py
```

When you `import` a file, Python **runs that whole file once** (top to
bottom) and remembers it. This matters in `app.py` (Part 17).

`from __future__ import annotations` (first line of most files) is a small
switch that tells Python "don't evaluate type hints right away". It lets newer
hint styles like `int | None` work everywhere. It changes nothing you'll
notice.

### 5.10 `if __name__ == "__main__":`

At the bottom of many files:

```python
if __name__ == "__main__":
    main()
```

Python sets the special variable `__name__` to `"__main__"` **only** for the
file you ran directly. If another file imports it, `__name__` is the module
name instead (e.g. `"predict"`). So this line means: "run `main()` only if
someone ran *this* file; do nothing if it was just imported". That's why
`predict.py` can be both a module that `app.py` imports **and** a script you
can run to test.

### 5.11 `with` blocks

```python
with open(ZIP_PATH, "wb") as handle:
    handle.write(chunk)
```

`with` opens something and **automatically cleans up** when the block ends,
even if an error happens. For files, cleanup = close the file. For SQLite
connections, cleanup = save (commit) the changes, or undo them if there was
an error.

### 5.12 Errors: `try`, `except`, `raise`

```python
try:
    response = requests.get(...)            # this might fail
except requests.exceptions.RequestException as exc:
    raise GraphError("Could not reach ...") from None
```

- `try:` = "attempt this".
- `except SomeError as exc:` = "if that kind of error happened, run this
  instead; call the error `exc`".
- `raise X` = "throw error X up to whoever called me".
- `from None` = "and hide the original error". Normally Python shows both the
  new error and the one that caused it. Here the original could contain the
  secret token, so we hide it.
- `finally:` (in JS code later) = "run this no matter what".

### 5.13 Classes (just enough)

```python
class GraphError(RuntimeError):
    """Anything that went wrong talking to Facebook."""
```

A **class** is a blueprint for a kind of thing. This line creates a new kind
of error called `GraphError`, which is a special type of the built-in
`RuntimeError`. It has no extra code — it exists only so other code can say
`except facebook.GraphError:` and catch **only** Facebook problems.

`TfidfVectorizer`, `LogisticRegression`, `Pipeline`, `Path`, `Flask` are all
classes from libraries. Writing `TfidfVectorizer(...)` **creates one object**
from the blueprint.

### 5.14 Decorators (`@something` above a function)

```python
@app.get("/api/analytics")
def get_analytics():
    ...
```

A decorator **wraps** the function below it to give it an extra job. Here it
tells Flask: "when someone visits `/api/analytics` with GET, run this
function". Same idea for `@lru_cache(maxsize=1)` in `predict.py`: "remember
this function's answer and don't run it twice".

### 5.15 `pathlib.Path`

```python
DATA_DIR = Path(__file__).resolve().parent / "data"
```

- `__file__` = the path of *this* Python file, e.g.
  `/media/.../backend/dataset.py`.
- `Path(...)` = turn that text into a Path object (which understands folders).
- `.resolve()` = make it a full absolute path.
- `.parent` = the folder that contains it → `/media/.../backend`.
- `/ "data"` = with Path objects, `/` **joins folder names** (it's not division
  here!) → `/media/.../backend/data`.

Why bother? So the code finds its files no matter which folder you were in
when you ran it.

Other Path methods used: `.exists()` (is it there?), `.mkdir(parents=True,
exist_ok=True)` (make the folder, and its parents, don't complain if it
exists), `.stat().st_size` (file size in bytes), `.read_text()`,
`.write_text()`, `.name` (just the filename).

### 5.16 Regular expressions (regex) — `re`

A regex is a tiny pattern language for finding text. `re.sub(pattern,
replacement, text)` finds every match of `pattern` in `text` and replaces it.

| Piece | Means |
|---|---|
| `a`, `http` | exactly those letters |
| `.` | any one character |
| `\S` | any character that is **not** whitespace |
| `\s` | any whitespace (space, tab, newline) |
| `\w` | any "word character": letter, digit, or `_` |
| `+` | "one or more of the thing before" |
| `\|` | or |
| `[a-z]` | any one character from a to z |
| `[^a-z\s]` | any one character that is **NOT** a–z and **NOT** whitespace (`^` at the start of `[ ]` means NOT) |
| `{20,}` | "20 or more of the thing before" |

So `http\S+` = "`http` followed by one or more non-space characters", which
is a whole URL.

---

## Part 6 — The tools around the code (uv, npm, .env, git)

### 6.1 Why we need "package managers"

This project uses code written by other people (pandas, scikit-learn, Flask,
Vue, Chart.js...). A **package manager** downloads those libraries at the
right versions and keeps them in a private folder for this project.

- Python side: **uv** (a fast, modern Python package manager).
- JavaScript side: **npm** (comes with Node.js).

### 6.2 `backend/pyproject.toml`

```toml
[project]
name = "sentiment-analysis"
version = "0.1.0"
description = "Facebook page comment sentiment dashboard - ..."
requires-python = ">=3.12"
dependencies = [
    "scikit-learn>=1.5",
    "pandas>=2.2",
    "numpy>=1.26",
    "joblib>=1.4",
    "flask>=3.0",
    "requests>=2.32",
    "python-dotenv>=1.0",
]
```

- `.toml` is a simple settings-file format. `[project]` starts a section.
- `requires-python = ">=3.12"` → needs Python 3.12 or newer.
- `dependencies` → the libraries. `>=1.5` means "version 1.5 or newer".

| Library | Why we need it |
|---|---|
| `scikit-learn` | TF-IDF, Logistic Regression, splitting data, scoring |
| `pandas` | reading the CSV and working with it as a table |
| `numpy` | fast number arrays (pandas and sklearn use it; `train.py` uses `argsort`) |
| `joblib` | save the trained model to a file, load it back |
| `flask` | the web server |
| `requests` | make HTTP calls (download the dataset, talk to Facebook) |
| `python-dotenv` | read the `.env` file into environment variables |

### 6.3 `uv.lock`, `.venv/`, `.python-version`

- `uv.lock` — written by `uv`. It records the **exact** version of every
  library (and every library *they* need). This makes the install repeatable
  on another computer. Never edit it by hand.
- `.venv/` — the "virtual environment": a private copy of Python plus all the
  installed libraries, just for this project. `uv sync` creates/updates it.
- `.python-version` — one line, `3.12`. Tells uv which Python to use.

`uv run python train.py` = "run `train.py` using the Python in `.venv`", so
the right libraries are always found.

### 6.4 `.env` and `.env.example`

`.env` holds **secrets**:

```
FB_PAGE_ID=1234567890
FB_PAGE_ACCESS_TOKEN=EAAB...very long...
```

- Each line is `NAME=value`. Lines starting with `#` are comments.
- `.env.example` is the empty template that *is* saved in git so people know
  what to fill in. `.env` itself is **never** saved in git (see below).
- `get_token.py` writes `.env` for you.

An **environment variable** is a named setting that the operating system hands
to a running program. `python-dotenv` reads `.env` and pretends those lines
were environment variables. `facebook.py` then reads them with
`os.getenv("FB_PAGE_ID")`.

### 6.5 `.gitignore`

Git is the version-history tool. `.gitignore` lists what git must **not**
track:

```
backend/data/        # 239 MB dataset + your database — too big, and re-downloadable
backend/models/      # the trained model — regenerate with train.py
backend/.venv/       # installed libraries — regenerate with uv sync
*.joblib             # any saved model anywhere
*.zip
backend/static/      # built frontend — regenerate with npm run build
__pycache__/         # Python's speed-up cache files
*.py[cod]            # .pyc/.pyo/.pyd compiled files ([cod] = c or o or d)
.env                 # SECRETS
frontend/node_modules/
frontend/dist/
.vscode/  .idea/  .DS_Store   # editor and Mac clutter
```

A `*` means "anything". So `*.zip` = "every file ending in .zip".

### 6.6 The frontend equivalents

| Python side | JavaScript side |
|---|---|
| `pyproject.toml` | `package.json` |
| `uv.lock` | `package-lock.json` |
| `.venv/` | `node_modules/` |
| `uv sync` | `npm install` |
| `uv run python app.py` | `npm run dev` / `npm run build` |

### 6.7 The PATH problem

`PATH` is a list of folders where your terminal looks for programs. On this
machine, `uv` and `npm` live in folders that aren't on it by default, so you
may need:

```bash
export PATH="$HOME/.local/bin:$PATH"                             # uv
export PATH="/home/minhaz/.nvm/versions/node/v23.5.0/bin:$PATH"  # node + npm
```

`export PATH="NEW:$PATH"` = "put NEW at the front of the list, keep the rest".

---

## Part 7 — The data: Sentiment140 and `download_data.py`

### 7.1 What is Sentiment140?

In 2009, three Stanford students (Go, Bhayani & Huang) collected **1.6 million
tweets**. They needed labels without paying people to read 1.6 million tweets,
so they used a trick: tweets containing `:)` were labeled **positive**, tweets
containing `:(` were labeled **negative**, and then the emoticons were
**removed** from the text (that's what "noemoticon" in the filename means).
Otherwise the model would just learn "`:)` means happy" and nothing else.

Consequences you should know:
- There are only **two** labels. There is **no neutral** in the data. (We
  invent "neutral" later, in `predict.py`.)
- Labels are **noisy**. Someone who wrote "my dog died :(" is sad, but someone
  who wrote "I hate Mondays lol :)" gets labeled positive. The model learns
  from these imperfect labels.
- It's **English only**, and it's 2009 Twitter language ("hehe", "yay", "wanna").

Here are the real first 3 lines of the file:

```
"0","1467810369","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","_TheSpecialOne_","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
"0","1467810672","Mon Apr 06 22:19:49 PDT 2009","NO_QUERY","scotthamilton","is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"
"0","1467810917","Mon Apr 06 22:19:53 PDT 2009","NO_QUERY","mattycus","@Kenichan I dived many times for the ball. Managed to save 50%  The rest go out of bounds"
```

Six columns, separated by commas, each wrapped in `"quotes"`:

| # | Column | Example | Do we use it? |
|---|---|---|---|
| 1 | sentiment | `0` (negative) or `4` (positive) | **Yes — the label** |
| 2 | id | `1467810369` | no |
| 3 | date | `Mon Apr 06 22:19:45 PDT 2009` | no |
| 4 | query | `NO_QUERY` | no |
| 5 | user | `_TheSpecialOne_` | no |
| 6 | text | the tweet | **Yes — the input** |

Notice: **no header row**. The first line is already data. We supply the
column names ourselves.

Also notice: all three start with `0`. The file is **sorted**: the first
800,000 rows are all negative, the last 800,000 all positive. This is a trap
(Part 9 explains).

### 7.2 `download_data.py` line by line

```python
"""One-off: fetch Sentiment140 from its original Stanford host.
...
"""
```
Docstring: explains the file. "One-off" = you run it once.

```python
from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import requests
```
- `sys` = system stuff; we use `sys.exit()` to stop the program with a message.
- `zipfile` = Python's built-in tool for opening `.zip` files.
- `requests` = library for downloading from the web.

The blank line between groups is a style habit: built-in modules first, then
third-party libraries.

```python
URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "sentiment140.zip"
TRAIN_CSV = DATA_DIR / "training.1600000.processed.noemoticon.csv"
```
Four constants: where to download from, and where to put things.
`backend/data/`, `backend/data/sentiment140.zip`, and the CSV inside it.

```python
def download() -> None:
```
A function called `download` that takes nothing and returns nothing
(`None`).

```python
    DATA_DIR.mkdir(parents=True, exist_ok=True)
```
Create `backend/data/` if it doesn't exist.

```python
    if TRAIN_CSV.exists():
        print(f"Already present: {TRAIN_CSV.name} "
              f"({TRAIN_CSV.stat().st_size / 1e6:.0f} MB) - nothing to do.")
        return
```
If the CSV is already there, say so and **stop early** (`return` leaves the
function). `1e6` is scientific notation for 1,000,000, so bytes / 1e6 =
megabytes. `:.0f` = no decimals. Two f-strings next to each other inside
`print(...)` are automatically glued into one string.

```python
    if not ZIP_PATH.exists():
        print(f"Downloading {URL} ...")
        with requests.get(URL, stream=True, timeout=120) as response:
```
If the zip isn't downloaded yet, download it.
- `stream=True`: don't load the whole 81 MB into memory; hand it to us piece by piece.
- `timeout=120`: give up if the server says nothing for 120 seconds.

```python
            response.raise_for_status()
```
If the server answered with an error (like 404 Not Found), raise an error now.

```python
            total = int(response.headers.get("content-length", 0))
            written = 0
```
The server tells us the file size in a **header** called `content-length`.
If it doesn't, use 0. `written` counts how many bytes we've saved so far.

```python
            with open(ZIP_PATH, "wb") as handle:
```
Open the zip file for **w**riting in **b**inary mode (`"wb"`). Binary =
raw bytes, not text.

```python
                for chunk in response.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
                    written += len(chunk)
```
Take the download 65,536 bytes at a time (`1 << 16` means "1 shifted left 16
times" = 2¹⁶ = 65,536), write each piece to disk. `written += x` is short for
`written = written + x`.

```python
                    if total:
                        pct = written / total * 100
                        print(f"\r  {written/1e6:6.1f} / {total/1e6:.1f} MB "
                              f"({pct:5.1f}%)", end="", flush=True)
            print()
```
Show progress. `\r` = "carriage return": jump back to the start of the line,
so the next print **overwrites** the same line (that's how the number counts
up in place). `end=""` = don't add a new line. `flush=True` = show it right
now. The final `print()` just prints a newline when done.

```python
    print(f"Extracting {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for info in archive.infolist():
            print(f"  {info.filename:<55} {info.file_size/1e6:7.1f} MB")
        archive.extractall(DATA_DIR)
```
Open the zip, print each file inside it with its size, then unpack everything
into `data/`. (The zip also contains a small test file we don't use.)

```python
    if not TRAIN_CSV.exists():
        sys.exit(f"Extraction finished but {TRAIN_CSV.name} is missing.")
    print(f"\nReady: {TRAIN_CSV}")
```
Double-check the CSV exists; if not, stop the program with an error message.

```python
if __name__ == "__main__":
    download()
```
Run `download()` only when this file is run directly.

---

## Part 8 — pandas crash course

pandas gives Python **tables**. Think of an Excel sheet you control with code.

### 8.1 DataFrame and Series

- A **DataFrame** is a whole table: rows and named columns.
- A **Series** is **one column** of a DataFrame (a list of values with row
  numbers).
- The row numbers on the left are called the **index**.

```
df (a DataFrame):
        sentiment   text                         clean_text
index
0       0           "@bob this sucks!!"          "this sucks"
1       1           "Love it http://x.co"        "love it"
2       0           "@amy"                       ""

df["text"] (a Series):
0    "@bob this sucks!!"
1    "Love it http://x.co"
2    "@amy"
```

By convention, pandas is imported as `pd`: `import pandas as pd`.

### 8.2 Every pandas operation used in this repo

**Read a CSV into a DataFrame:**
```python
df = pd.read_csv(RAW_CSV, encoding="latin-1", names=COLUMNS)
```
- `encoding="latin-1"`: how to turn the file's bytes into letters. The file
  is not valid UTF-8 (the modern standard), so the default would crash.
- `names=COLUMNS`: the file has no header row, so use these column names.

**Take a random sample of rows:**
```python
df = df.sample(n=200000, random_state=42)
```
`random_state` is the seed: same number → same "random" rows every time.

**Pick columns:**
```python
df["text"]                          # one column  → a Series
df[["sentiment", "text"]]           # two columns → a DataFrame
```
The double `[[ ]]`: the outer `[]` means "select", the inner `[...]` is a
Python list of column names.

**`.copy()`**: make a real independent copy, so later changes don't also
(confusingly) touch the original table. Without it pandas prints a
"SettingWithCopyWarning".

**Change every value in a column using a dict:**
```python
df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})
```
Every 0 stays 0, every 4 becomes 1.

**Run a function on every value:**
```python
df["clean_text"] = df["text"].apply(clean_text)
```
Calls `clean_text` once per row and stores the results in a **new** column.
Assigning to a column name that doesn't exist creates it.

**String tools on a whole column (`.str`):**
```python
df["clean_text"].str.len()        # a Series of lengths: 10, 7, 0
```

**Filter rows with True/False (a "boolean mask"):**
```python
mask = df["clean_text"].str.len() > 0   # True, True, False
df = df[mask]                            # keep only True rows
```
The repo does it in one line: `df[df["clean_text"].str.len() > 0]`.

**Reset the index:**
```python
df = df.reset_index(drop=True)
```
After sampling and filtering, the row numbers are jumbled (e.g. 541, 90213,
7). This renumbers them 0, 1, 2, ... `drop=True` means throw the old numbers
away instead of keeping them as a new column.

**Count each value:**
```python
part["sentiment"].value_counts(normalize=True)
# 0    0.5
# 1    0.5
```
`normalize=True` gives fractions instead of counts. `.mul(100)` multiplies by
100 (→ percent). `.round(1)` → one decimal. `balance.get(0, 0)` → the value
for label 0, or 0 if missing.

**Size:** `len(df)` = number of rows.

---

## Part 9 — `dataset.py` line by line

This file makes **every decision about data**: how to read it, clean it and
split it. `train.py` makes every decision about the **model**. Keeping them
apart means you can try a different model on exactly the same data and
compare fairly.

```python
"""Sentiment140 loading, cleaning and splitting.
...
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split
```
- `re` = regular expressions (Part 5.16).
- `pandas as pd` = tables (Part 8).
- `train_test_split` = a scikit-learn function that randomly cuts a table into
  two parts.

```python
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "training.1600000.processed.noemoticon.csv"
```
Where the CSV lives: `backend/data/training...csv`.

```python
COLUMNS = ["sentiment", "id", "date", "query", "user", "text"]
```
The six column names, since the file has no header row.

```python
LABEL_MAP = {0: 0, 4: 1}
LABEL_NAMES = {0: "negative", 1: "positive"}
```
The file says 0 = negative, 4 = positive. scikit-learn's scoring functions
expect 0 and 1, so we translate 4 → 1. `LABEL_NAMES` turns numbers back into
words for printing.

```python
SEED = 42
```
The seed for all random choices. Any number works; 42 is a programmer joke
(from *The Hitchhiker's Guide to the Galaxy*). Because the seed is fixed, if
you run `train.py` twice you get the **same** sample and the **same** splits,
so results are comparable.

### `clean_text()` — lines 30–45

```python
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs carry no sentiment
    text = re.sub(r"@\w+", " ", text)             # @mentions are just names
    text = re.sub(r"[^a-z\s]", " ", text)         # keep letters only
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text
```

Takes one piece of text, returns a cleaned version. Each line overwrites
`text` with a cleaner version. Let's follow the **real** first tweet of the
dataset through it:

| Step | Code | Result |
|---|---|---|
| start | | `@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D` |
| 1 lowercase | `.lower()` | `@switchfoot http://twitpic.com/2y1zl - awww, that's a bummer.  you shoulda got david carr of third day to do it. ;d` |
| 2 URLs → space | `http\S+\|www\S+` | `@switchfoot   - a  that's a bummer.  you shoulda ...` |
| 3 @mentions → space | `@\w+` | `    - a  that's a bummer.  you shoulda ...` |
| 4 non-letters → space | `[^a-z\s]` | `      a  that s a bummer   you shoulda got david carr of third day to do it   d` |
| 5 squash spaces, trim | `\s+` → `" "`, `.strip()` | `a that s a bummer you shoulda got david carr of third day to do it d` |

Look closely at step 2. **"awww," became "a"**. The pattern `www\S+` was
meant for web addresses like `www.site.com`, but it also matches the "www,"
*inside* "awww,". That's a real little bug. Part 24 talks about it. It's a
good lesson: **always look at the output of your cleaning on real examples.**

Why each step?
1. **lowercase**: so "Love", "LOVE" and "love" count as the same word.
2. **URLs**: a link says nothing about feelings.
3. **@mentions**: `@Rahim` is a name, not a feeling. And if one user always
   tweets happy things, the model might learn "@thatuser = positive", which is
   useless on Facebook.
4. **letters only**: removes punctuation, digits, emoji, and... **every
   non-English letter**. Bangla script (`দারুণ`) disappears completely. So
   does the apostrophe: "that's" → "that s", "can't" → "can t".
5. **tidy spaces**: `\s+` (one or more whitespace) becomes one space;
   `.strip()` removes spaces at the ends.

What it **doesn't** do on purpose: remove stop words ("the", "not"). That is
a decision for the vectorizer (Part 10), and keeping it out of here lets us
experiment with it.

**Very important:** this same function is used during **training** (here) and
during **prediction** (`predict.py` imports it). If the two ever cleaned text
differently, the model would receive text that looks different from what it
studied and quietly make worse guesses.

### `load_raw()` — lines 48–87

```python
def load_raw(sample: int | None = None, seed: int = SEED) -> pd.DataFrame:
```
Takes `sample` (how many rows, or `None` for all) and `seed`. Returns a
DataFrame.

```python
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run:  uv run python download_data.py"
        )
```
If you forgot to download, stop with a helpful message.

```python
    df = pd.read_csv(RAW_CSV, encoding="latin-1", names=COLUMNS)
```
Read all 1.6 million rows into a DataFrame called `df`.

```python
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
```
If a sample size was given and it is smaller than the table, pick that many
rows **at random**.

**The trap:** the file is sorted (all 800,000 negatives first). If you did
`df.head(200000)` ("the first 200,000 rows"), you'd get 200,000 **negative**
tweets and 0 positive. The model would learn "everything is negative" and
still score 100% on that data. Random sampling avoids this.

```python
    df = df[["sentiment", "text"]].copy()
```
Throw away id, date, query and user. Keep only the label and the text.

```python
    df["sentiment"] = df["sentiment"].map(LABEL_MAP)
```
Turn 4s into 1s.

```python
    df["clean_text"] = df["text"].apply(clean_text)
```
Run `clean_text` on every tweet and store the result in a new column.

```python
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped:,} rows that cleaned to an empty string")
```
Some tweets become **empty** after cleaning (a tweet that was only a link or
only `@someone`). An empty text gives the model nothing to learn from, so we
remove those rows and report how many. In the model you have now, 200,000
were sampled and 492 were dropped, leaving **199,508** rows (you can see
`"rows_used": 199508` in `models/model_meta.json`).

```python
    return df
```

### `make_splits()` — lines 90–114

```python
def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
```
Takes a DataFrame, returns a dict whose keys are strings (`"train"` etc.) and
whose values are DataFrames.

Why three piles? A school analogy:

| Pile | Share | School analogy | Rule |
|---|---|---|---|
| **train** | 70% | the textbook | the model learns from it |
| **val** (validation) | 15% | practice tests | use as often as you like to compare settings |
| **test** | 15% | the final exam | use **once**, at the very end |

If you keep changing your settings until the *final exam* score looks good,
the final exam stops being a fair measure. You've effectively seen the
answers. That's why we compare settings on `val` and only touch `test` once.

```python
    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["sentiment"]
    )
```
First cut: 70% → `train`, 30% → `temp`.
- `train, temp = ...` — the function returns two things; this line catches
  both into two names at once ("unpacking").
- `test_size=0.30` → the second part gets 30%.
- `stratify=df["sentiment"]` → keep the **same mix** of positive/negative in
  both parts. If the full data is 50/50, both parts are 50/50.

```python
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["sentiment"]
    )
```
Second cut: split `temp` in half → 15% `val`, 15% `test`.

```python
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
```
Return all three in a dict, each renumbered from 0. With 199,508 rows you get
train 139,655, val 29,926, test 29,927.

### `load_splits()` — lines 117–128

```python
def load_splits(sample: int | None = None, seed: int = SEED) -> dict[str, pd.DataFrame]:
    print(f"Loading Sentiment140 (sample={sample or 'all'})...")
    df = load_raw(sample=sample, seed=seed)
    splits = make_splits(df, seed=seed)
```
A shortcut that does both steps. `sample or 'all'` prints the number, or the
word "all" when `sample` is `None`.

```python
    for name, part in splits.items():
        balance = part["sentiment"].value_counts(normalize=True).mul(100).round(1)
        print(
            f"  {name:5} {len(part):>9,} rows   "
            f"neg {balance.get(0, 0)}% / pos {balance.get(1, 0)}%"
        )
    return splits
```
For each pile, print its size and its positive/negative mix. Output looks
like this (the percentages will be very close to 50, and the same in all three
piles, thanks to `stratify`):

```
  train   139,655 rows   neg 50.0% / pos 50.0%
  val      29,926 rows   neg 50.0% / pos 50.0%
  test     29,927 rows   neg 50.0% / pos 50.0%
```

`{name:5}` pads the name to 5 characters so columns line up.
`{len(part):>9,}` right-aligns the number in 9 characters with commas.

---

## Part 10 — Machine learning explained from zero

This is the heart of the project. Take your time here.

### 10.1 Rules vs. learning

**The old way (rules):** a human writes rules.
"If the comment contains *love*, *great* or *awesome*, say positive. If it
contains *hate*, *bad* or *terrible*, say negative."
Problem: there are thousands of words, and "not bad" contains "bad" but is
positive. You'd never finish writing rules.

**The ML way (learning):** show the computer **many examples with answers**
and let it figure out, by itself, how much each word matters. This is called
**supervised learning** ("supervised" because every example comes with the
right answer, like a teacher supervising).

This problem is called **binary classification**: put each text into one of
**two** classes (positive or negative).

### 10.2 Problem: computers only understand numbers

A model is math. Math needs numbers, not words. So step one is always:
**turn text into numbers**. The tool that does this is called a
**vectorizer** (a "vector" is just a list of numbers).

### 10.3 Bag of words

The simplest idea: make a list of every word the model knows (the
**vocabulary**). For each text, make a row with one slot per vocabulary word,
and put a number in the slot if the word appears.

Vocabulary: `bad, hate, love, not, phone, this`

| text | bad | hate | love | not | phone | this |
|---|---|---|---|---|---|---|
| "love this phone" | 0 | 0 | 1 | 0 | 1 | 1 |
| "hate this phone" | 0 | 1 | 0 | 0 | 1 | 1 |
| "not bad" | 1 | 0 | 0 | 1 | 0 | 0 |

It's called a "bag" because **word order is lost**: "dog bites man" and
"man bites dog" give the same row. Everything is thrown into a bag.

Our real vocabulary has **50,000** slots. A comment uses maybe 10 of them, so
the row is **49,990 zeros and 10 non-zeros**. Storing all those zeros would
waste memory, so scikit-learn uses a **sparse matrix**: it only stores the
non-zero cells ("row 0, column 23804, value 0.236"). You'll see
`X.shape = (1, 50000)` but only 7 stored values in the example below.

### 10.4 TF-IDF: smarter numbers than 0 and 1

Two problems with plain counting:

1. A word that appears 10 times isn't 10× more important. (TF fix below.)
2. Some words appear in almost **every** text ("the", "this", "a"). They tell
   you nothing. A rare word like "furious" tells you a lot. (IDF fix below.)

**TF = Term Frequency** — how many times the word appears in *this* text.
With `sublinear_tf=True` (set in `train.py`), TF is replaced by `1 + log(TF)`.
So 1 time → 1, 2 times → 1.69, 10 times → 3.3. More repeats matter less and
less. "love love love love" doesn't count four times as much as "love".

**IDF = Inverse Document Frequency** — how **rare** the word is across all
training texts. scikit-learn computes it as:

```
IDF(word) = ln( (1 + number_of_texts) / (1 + number_of_texts_containing_word) ) + 1
```

`ln` is the natural logarithm. You don't need to love logarithms; just
remember: **common word → small IDF; rare word → big IDF.** From the real
trained model:

| word | IDF | why |
|---|---|---|
| `the` | 2.32 | appears in a huge share of tweets |
| `not` | 3.73 | common |
| `love` | 4.27 | fairly common |
| `hate` | 5.44 | less common |
| `not bad` | 8.67 | rare phrase |

**TF-IDF = TF × IDF.** Then each row is **normalized** (scaled so that the
sum of squares of the row equals 1). Normalizing means long comments and short
comments are on the same scale.

#### A tiny, real worked example

I ran TF-IDF (with bigrams, explained next) on just three sentences:
`"i love this phone"`, `"i hate this phone"`, `"not bad at all"`.

Vocabulary it built (14 features):
```
all, at, at all, bad, bad at, hate, hate this, love, love this,
not, not bad, phone, this, this phone
```

Notice: **"i" is missing!** The vectorizer's default rule only keeps tokens
of **2 or more letters**. Single letters like "i", "a", "d" are dropped. (So
in the real model, the leftover "a" and "d" from the "Awww ;D" tweet are
dropped too.)

IDF values: words in 1 of 3 sentences get `ln(4/2)+1 = 1.693`; words in 2 of
3 sentences (`phone`, `this`, `this phone`) get `ln(4/3)+1 = 1.288`.

Resulting rows (rounded):

| | all | at | at all | bad | bad at | hate | hate this | love | love this | not | not bad | phone | this | this phone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| i love this phone | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0.52** | **0.52** | 0 | 0 | 0.39 | 0.39 | 0.39 |
| i hate this phone | 0 | 0 | 0 | 0 | 0 | **0.52** | **0.52** | 0 | 0 | 0 | 0 | 0.39 | 0.39 | 0.39 |
| not bad at all | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0 | 0 | 0 | 0 | 0.38 | 0.38 | 0 | 0 | 0 |

See how `love` (unique to one sentence) got a bigger number (0.52) than
`phone` (shared, 0.39)? That's IDF at work.

### 10.5 n-grams: keeping a little bit of word order

An **n-gram** is n words in a row.
- **unigrams** (n=1): `not`, `bad`
- **bigrams** (n=2): `not bad`

`ngram_range=(1, 2)` means "use both unigrams and bigrams". Now "not bad" is
its own feature, and the model can learn that `not bad` is positive even
though `bad` alone is negative. In the real trained model:

| feature | weight |
|---|---|
| `bad` | **−5.87** (pushes negative) |
| `not` | **−5.79** (pushes negative) |
| `not bad` | **+5.18** (pushes positive!) |

### 10.6 Stop words and the "not" trap

**Stop words** are super-common words ("the", "is", "and") that are often
thrown away. scikit-learn has a built-in English list, used when you set
`stop_words="english"`.

The trap: that list contains **"not", "no", "cannot"**. If you remove stop
words, "not bad" becomes just "bad", and "not happy" becomes "happy": the
exact opposite meaning. That's why in `train.py` the bigram settings use
`stop_words=None` (keep everything). The two choices interact: bigrams are
only useful for negation if you **keep** the negation words.

### 10.7 `max_features`: size of the vocabulary

With bigrams, 140,000 tweets contain **hundreds of thousands** of distinct
features. `max_features=50_000` keeps only the **50,000 most frequent**. Rare
features (a typo seen once) are dropped. Smaller vocabulary = smaller, faster
model, and less memorizing of one-off junk.

### 10.8 Logistic Regression: the model itself

After TF-IDF, each comment is a row of 50,000 numbers. The model has **one
weight per feature** (50,000 weights) plus **one extra number** called the
**intercept** (or bias).

To judge a comment, it does two steps:

**Step 1: add up.**

```
z = (weight₁ × value₁) + (weight₂ × value₂) + ... + intercept
```

Only the non-zero features matter (anything × 0 = 0).
A positive weight pushes z **up** (toward positive). A negative weight pushes
z **down** (toward negative).

**Step 2: squash z into a probability between 0 and 1** using the **sigmoid**
function:

```
probability = 1 / (1 + e^(−z))
```

| z | probability of positive |
|---|---|
| −4 | 0.02 |
| −2 | 0.12 |
| 0 | 0.50 |
| +2 | 0.88 |
| +4 | 0.98 |

So a big positive z → close to 1 (positive). A big negative z → close to 0
(negative). z near 0 → about 0.5 (unsure).

That's the entire model. "Logistic" comes from the sigmoid (also called the
logistic function). "Regression" is a historical name; it's used for
classification here.

#### The real model judging a real sentence

I fed `"@Rahim I LOVE this!!! http://x.co 10/10 not bad"` to the actual
trained model.

1. `clean_text` → `"i love this not bad"`
2. TF-IDF → 7 non-zero features (out of 50,000). "i" is dropped (1 letter).
3. Multiply each by its learned weight:

| feature | TF-IDF value | weight | value × weight |
|---|---|---|---|
| `love` | 0.236 | +4.437 | +1.047 |
| `love this` | 0.446 | +2.064 | +0.921 |
| `not bad` | 0.480 | +5.183 | +2.488 |
| `bad` | 0.280 | −5.866 | −1.642 |
| `not` | 0.206 | −5.787 | −1.192 |
| `this` | 0.214 | −0.913 | −0.195 |
| `this not` | 0.590 | −0.434 | −0.256 |
| **sum** | | | **+1.17** |
| + intercept | | | +0.514 |
| **z** | | | **≈ 1.68** |

4. sigmoid(1.68) = **0.843** → 84% chance positive.

Look at `not bad`: `bad` and `not` each pulled strongly negative, but the
bigram `not bad` pulled even harder positive, and it won. That's exactly
why bigrams exist.

### 10.9 So what does "training" actually do?

Training = **finding the 50,001 numbers** (weights + intercept) that make the
model's probabilities match the true labels as closely as possible.

Roughly:
1. Start with all weights at 0 (every comment gets 0.5).
2. Look at all training examples. Measure how wrong the probabilities are.
   This "wrongness" is called the **loss** (for logistic regression, it's
   "log loss": being confidently wrong is punished heavily).
3. Figure out, for each weight, which direction (up or down) would reduce the
   loss, and nudge it that way.
4. Repeat until the loss stops improving.

`max_iter=1000` = "do at most 1000 rounds of nudging". scikit-learn's default
method for this (called `lbfgs`) is a clever version of "nudge downhill".

There's also a built-in brake called **regularization** (on by default,
strength `C=1.0`). It gently punishes huge weights, so the model can't lean
too hard on any one word it happened to see a few times. It's a guard
against memorizing.

What training **learns**: which words are positive/negative and how
strongly. What it **doesn't** learn: grammar, sarcasm, meaning. It's a
very clever word-counting machine.

### 10.10 The Pipeline: why vectorizer and model must travel together

The vectorizer **also learns something during training**: the vocabulary
(which word goes in which column) and the IDF values. The model's weight
#23804 only makes sense if column #23804 is still "love".

The classic beginner bug:
1. Train the vectorizer and the model.
2. Save **only** the model.
3. Later, make a **new** vectorizer and `fit` it on the new comments.
   It builds a **different** vocabulary. Column 23804 might now be "pizza".
4. The model reads "pizza" with love's weight. The predictions are garbage,
   and **no error appears**. The accuracy just silently collapses.

A **Pipeline** glues them into one object:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(...)),     # step 1, named "tfidf"
    ("clf",   LogisticRegression(...)),  # step 2, named "clf" (classifier)
])
```

- `pipeline.fit(texts, labels)` → the vectorizer learns the vocabulary +
  IDF from `texts`, turns them into numbers, and the model learns weights.
- `pipeline.predict(new_texts)` → the vectorizer **transforms** (does *not*
  re-learn) with the saved vocabulary, then the model predicts.
- `joblib.dump(pipeline, ...)` saves **both** together.

Two sklearn words:
- **fit** = learn from data.
- **transform** = apply what was learned.
- **fit_transform** = both at once (only ever on training data!).

### 10.11 Measuring how good the model is

**Accuracy** = correct guesses ÷ all guesses. Your model: **80.2%** on the
test set.

**Confusion matrix**: a 2×2 table of *what really was* vs. *what we
guessed*. Your model's real test numbers (from `model_meta.json`):

|  | guessed negative | guessed positive |
|---|---|---|
| **really negative** | **11,891** ✓ | 3,053 ✗ (false positive) |
| **really positive** | 2,869 ✗ (false negative) | **12,114** ✓ |

Accuracy = (11,891 + 12,114) ÷ 29,927 = 24,005 ÷ 29,927 = **0.802**.

The mistakes are fairly balanced (3,053 vs 2,869), so the model isn't biased
toward one side.

**Precision** (for positive) = of everything we *called* positive, how much
really was? 12,114 ÷ (12,114 + 3,053) = **0.799**.

**Recall** (for positive) = of everything that *really was* positive, how
much did we catch? 12,114 ÷ (12,114 + 2,869) = **0.808**.

**F1** = a single number that balances precision and recall (their "harmonic
mean"). `classification_report` prints all of these for both classes.

**ROC-AUC**: pick one random positive tweet and one random negative tweet.
How often does the model give the positive one a **higher probability**?
0.5 = coin flip, 1.0 = always. Your model: **0.883**. Why do we care? Because
accuracy depends on the cut-off (0.5 by default), and `predict.py` later moves
the cut-offs to make a "neutral" zone. ROC-AUC measures the quality of the
probabilities regardless of where you cut.

**Is 80% good?** For Sentiment140 with a simple model, yes, that's about as
good as it gets. The labels themselves are noisy (the `:)` trick), so even a
perfect model couldn't score 100%. Fancy neural networks reach roughly the
mid-80s on it.

### 10.12 Cross-validation (CV)

One validation score is a single measurement, and it could be lucky. **k-fold
cross-validation** cuts the training data into k slices (folds). It trains k
times; each time, one slice is held out for scoring and the other k−1 are
used to train. You get k scores. Their **mean** is a steadier estimate, and
their **standard deviation** (std) tells you how much the score wobbles.

If config A beats config B by 0.5% but the std is 1%, the difference is just
noise.

`--cv 5` turns this on in `train.py`. Your current model was trained without
it (`"cv_scores": null` in the meta file).

### 10.13 Overfitting and underfitting

- **Overfitting**: the model memorized the training examples (including
  their noise). It scores great on train, worse on new data. Like a student
  who memorized the exact textbook questions.
- **Underfitting**: the model is too simple to capture the pattern. Bad on
  everything.

Separate val/test splits exist to **detect** overfitting. Regularization and
`max_features` help **prevent** it.

### 10.14 Saving and loading: joblib

After training, the pipeline is just a Python object in memory. When the
program ends, it's gone. **joblib** writes the object to a file and reads it
back later:

```python
joblib.dump(pipeline, "models/sentiment_pipeline.joblib", compress=3)  # save
pipeline = joblib.load("models/sentiment_pipeline.joblib")             # load
```

`compress=3` squeezes the file (compression level 3 of 9). Your file is
~0.95 MB. Inside it: the vocabulary (50,000 words → column numbers), the IDF
values, the 50,000 weights and the intercept, plus settings.

This is what lets the web app start instantly without training: it just
loads the file.

**Security note:** loading a joblib file can run code hidden in it. Only load
`.joblib` files you created yourself or fully trust.

### 10.15 What this model cannot do

- **Sarcasm**: "oh great, it broke again" → it sees "great".
- **Long-range meaning**: it only sees words and word pairs, never whole
  sentences.
- **Words it never saw**: new slang, product names, and brand-new words get
  no weight.
- **Bangla**: Bangla script is erased by `clean_text` → the text becomes
  empty → `predict.py` says neutral with confidence 0. Romanized Banglish
  ("khub valo") survives cleaning, but those words mostly aren't in the
  vocabulary, so its guesses on Banglish are unreliable.
- **Neutral**: the data has no neutral class, so neutral is approximated by
  "the model is unsure" (Part 12).
- **Domain shift**: it learned from 2009 tweets. Facebook shop comments in
  2026 use different words. Example from the real model:
  `"when will you restock this item"` → score 0.385 → **negative**. It's a
  neutral question, but words like "when", "will", "restock" carry small
  weights learned from tweets, and together they nudge it just under the 0.40
  line.

---

## Part 11 — `train.py` line by line

Run with `uv run python train.py`. It does four phases:

1. Train 3 different configurations, compare them on **val**, pick a winner.
2. (Optional) cross-validate the winner.
3. Score the winner **once** on **test**.
4. Save the winner to `models/`.

### Imports (lines 15–35)

```python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
```
- `argparse` — reads options you type after the command (`--full`, `--sample 50000`).
- `datetime`, `timezone` — for the "trained at" timestamp.
- `json` — to write the report card file.

```python
import joblib
import numpy as np
```
- `joblib` — to save the pipeline.
- `numpy as np` — for `np.argsort` (sorting weights).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
```
The scikit-learn tools. The round brackets `( ... )` around the four metric
names just let the import span several lines.

```python
from dataset import LABEL_NAMES, SEED, load_splits
```
From **our** `dataset.py`: the label names, the seed, and the loader.

```python
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
META_PATH = MODELS_DIR / "model_meta.json"
```
Where to save. (`predict.py` has the same three lines, so it looks in the same
place.)

### `CONFIGS` — the experiments (lines 52–56)

```python
CONFIGS: dict[str, dict] = {
    "unigram-5k": dict(ngram_range=(1, 1), max_features=5_000, stop_words="english"),
    "bigram-10k": dict(ngram_range=(1, 2), max_features=10_000, stop_words=None),
    "bigram-50k": dict(ngram_range=(1, 2), max_features=50_000, stop_words=None),
}
```
- `CONFIGS: dict[str, dict] = ...` — a variable with a type hint: a dict whose
  keys are strings and whose values are dicts.
- `dict(a=1, b=2)` is another way to write `{"a": 1, "b": 2}`.
- Three experiments:

| name | words | vocab size | stop words |
|---|---|---|---|
| unigram-5k | single words only | 5,000 | removed |
| bigram-10k | words + word pairs | 10,000 | kept |
| bigram-50k | words + word pairs | 50,000 | kept |

Real validation results from your last run (from `model_meta.json`):

| config | val accuracy | val ROC-AUC |
|---|---|---|
| unigram-5k | 0.7584 | 0.8391 |
| bigram-10k | 0.7937 | 0.8753 |
| bigram-50k | **0.7984** | **0.8809** ← winner |

Adding bigrams (and keeping "not") gave the biggest jump: +3.5 points. Going
from 10k to 50k features added only +0.5 more.

### `build_pipeline()` (lines 59–76)

```python
def build_pipeline(seed: int = SEED, **vectorizer_kwargs) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, **vectorizer_kwargs)),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
```
- `**vectorizer_kwargs` collects the config options (Part 5.6) and passes them
  on to `TfidfVectorizer`.
- `Pipeline([...])` takes a **list** of **tuples**. Each tuple is
  `("name", step)`. The names let you get a step back later with
  `pipeline.named_steps["tfidf"]`.
- `sublinear_tf=True` → use `1 + log(count)` (Part 10.4).
- `max_iter=1000` → allow up to 1000 rounds of weight-nudging.
- `random_state=seed` → repeatable.

This function **creates** an empty, untrained pipeline. Nothing has learned
anything yet.

### `evaluate()` (lines 79–91)

```python
def evaluate(pipeline: Pipeline, X, y, label: str) -> dict:
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]
```
- `X` = the texts, `y` = the true labels. (Capital X / small y is a maths
  habit: X is a table of inputs, y is a column of answers.)
- `predict(X)` → a 0 or 1 for each text (cut-off 0.5).
- `predict_proba(X)` → a 2-column table: `[P(negative), P(positive)]` per
  text, e.g. `[[0.157, 0.843], ...]`. `[:, 1]` = "all rows (`:`), column 1" →
  just the P(positive) values.

```python
    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
```
Compare guesses with the truth. Note accuracy uses the 0/1 guesses; ROC-AUC
uses the probabilities.

```python
    print(f"\n  {label}:  accuracy {accuracy:.4f}   roc-auc {auc:.4f}")
    return {"accuracy": float(accuracy), "roc_auc": float(auc)}
```
Print, and return both numbers in a dict. `float(...)` converts numpy's number
type into a plain Python float, which `json` knows how to save.

### `show_top_features()` (lines 94–119)

```python
def show_top_features(pipeline: Pipeline, n: int = 15) -> dict[str, list[str]]:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
```
- Pull the two steps out of the pipeline by name.
- `names` = the 50,000 feature names, in column order.
- `classifier.coef_` = the learned weights. Its shape is `(1, 50000)`: one row
  (because it's binary) of 50,000 weights. `[0]` takes that one row.
- A trailing underscore (`coef_`) is sklearn's way of saying "this was
  **learned** during fit".

```python
    top_positive = np.argsort(coefficients)[-n:][::-1]
    top_negative = np.argsort(coefficients)[:n]
```
`np.argsort` doesn't return sorted **values**; it returns the **positions**
that would sort the list, smallest first. Example: weights `[0.5, -2, 3]` →
argsort → `[1, 0, 2]` (position 1 is smallest, then 0, then 2).
- `[-n:]` = the last 15 positions = the 15 **biggest** weights.
- `[::-1]` = reverse, so the biggest comes first.
- `[:n]` = the first 15 = the 15 **most negative** weights.

```python
    print(f"\n  Strongest POSITIVE features:")
    print("   ", ", ".join(f"{names[i]}" for i in top_positive))
    ...
    return {
        "positive": [str(names[i]) for i in top_positive],
        "negative": [str(names[i]) for i in top_negative],
    }
```
Print the words and return them as lists. Your model's actual output:

- **positive:** thanks, cant wait, happy, not bad, good, awesome, nice, yay,
  no problem, love, welcome, great, cool, smile, hehe
- **negative:** sad, miss, poor, missing, hate, sick, wish, sorry, bad, not,
  sucks, died, missed, sadly, hurts

Notice the **bigrams** in the positive list: `cant wait`, `not bad`,
`no problem`. Each contains a negative-looking word ("cant", "not", "no") but
means something happy. (`cant` without an apostrophe is how people typed it on
Twitter. When someone *did* type "can't", `clean_text` turns it into "can t",
the lone 1-letter "t" is dropped by the vectorizer, and it becomes the bigram
"can wait". Small side effects of cleaning like this are everywhere.)

And `wish`, `sorry`, `miss` are in the negative list: in 2009 tweets, "I
wish..." and "I miss..." usually came from sad people.

**Why look at these?** It's the cheapest bug detector. If you saw a username
or `http` in this list, you'd know your cleaning had a leak.

### `main()` — the actual run (lines 122–231)

#### Reading the command-line options

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200_000,
                        help="rows to sample (default 200000)")
    parser.add_argument("--full", action="store_true",
                        help="use all 1.6M rows, overrides --sample")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv", type=int, default=0,
                        help="cross-validation folds on the winner (0 = skip)")
    args = parser.parse_args()
```
- `__doc__` = the docstring at the top of the file, shown by `--help`.
- `--sample 50000` → `args.sample = 50000`. `type=int` converts the typed text
  to a number.
- `--full` has `action="store_true"`: it takes no value; if present,
  `args.full = True`, else `False`.
- `--cv 5` → `args.cv = 5`. Default 0 = skip.

Try `uv run python train.py --help` to see these.

```python
    sample = None if args.full else args.sample
    splits = load_splits(sample=sample, seed=args.seed)
```
If `--full`, sample is `None` (= all rows). Then load, clean and split.

```python
    X_train, y_train = splits["train"]["clean_text"], splits["train"]["sentiment"]
    X_val, y_val = splits["val"]["clean_text"], splits["val"]["sentiment"]
    X_test, y_test = splits["test"]["clean_text"], splits["test"]["sentiment"]
```
Pull out the text column (X) and the label column (y) of each pile.
`a, b = x, y` sets two names in one line.

#### Phase 1: compare configurations

```python
    print("\n" + "=" * 62)
    print("PHASE 1  comparing configurations on validation")
    print("=" * 62)
```
`"=" * 62` = the `=` character repeated 62 times (a divider line). `+` glues
strings.

```python
    results: dict[str, dict] = {}
    for name, config in CONFIGS.items():
        print(f"\n[{name}]  {config}")
        pipeline = build_pipeline(seed=args.seed, **config)
        pipeline.fit(X_train, y_train)
```
For each of the 3 configs: build a fresh pipeline and **train** it on the
training pile. **`pipeline.fit(X_train, y_train)` is the line where the actual
learning happens.** Everything else in this file is preparation and
checking.

```python
        results[name] = {
            "config": {k: list(v) if isinstance(v, tuple) else v
                       for k, v in config.items()},
            "val": evaluate(pipeline, X_val, y_val, "val"),
            "pipeline": pipeline,
        }
```
Store, for this config:
- a copy of its settings, with tuples like `(1, 2)` turned into lists
  `[1, 2]` so they look the same as they will in the JSON file,
- its validation scores,
- the trained pipeline itself (so we don't have to retrain the winner).

```python
    print("\n" + "-" * 62)
    print(f"{'config':<14}{'val accuracy':>16}{'val roc-auc':>16}")
    print("-" * 62)
    for name, result in results.items():
        print(f"{name:<14}{result['val']['accuracy']:>16.4f}"
              f"{result['val']['roc_auc']:>16.4f}")
```
Print a neat comparison table. Inside an f-string that uses `"`, the inner
strings use `'` so the quotes don't clash.

```python
    best_name = max(results, key=lambda n: results[n]["val"]["accuracy"])
    best = results[best_name]
    best_pipeline: Pipeline = best["pipeline"]
    print("-" * 62)
    print(f"winner: {best_name}")
```
`max(results, key=...)` loops over the config **names** and returns the one
whose validation accuracy is highest. Only **val** is used to choose — never
test.

#### Phase 2: cross-validation (only with `--cv`)

```python
    cv_scores = None
    if args.cv:
        ...
        cv_scores = cross_val_score(
            build_pipeline(seed=args.seed, **CONFIGS[best_name]),
            X_train, y_train, cv=args.cv, scoring="accuracy", n_jobs=-1,
        )
        print(f"  folds: {', '.join(f'{s:.4f}' for s in cv_scores)}")
        print(f"  mean {cv_scores.mean():.4f}  std {cv_scores.std():.4f}")
```
Builds a **fresh** pipeline with the winning settings and runs k-fold CV on the
training pile (Part 10.12). `n_jobs=-1` = use all CPU cores in parallel. The
result is a numpy array of k scores; `.mean()` and `.std()` summarize it.
This phase only **reports**; it doesn't change the model that gets saved.

#### Phase 3: the one look at test

```python
    test_metrics = evaluate(best_pipeline, X_test, y_test, "test")
    test_predictions = best_pipeline.predict(X_test)

    print("\n" + classification_report(
        y_test, test_predictions,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], digits=4,
    ))
```
Score the winner on the untouched test pile. `classification_report` prints
precision, recall, F1 and support (how many examples) per class.

```python
    matrix = confusion_matrix(y_test, test_predictions)
    print("  confusion matrix (rows = actual, cols = predicted)")
    print(f"                predicted_neg  predicted_pos")
    print(f"    actual_neg  {matrix[0][0]:>13,}  {matrix[0][1]:>13,}")
    print(f"    actual_pos  {matrix[1][0]:>13,}  {matrix[1][1]:>13,}")
```
`matrix` is a 2×2 array. `matrix[0][1]` = row 0 (actually negative), column 1
(guessed positive) = the false positives.

```python
    top_features = show_top_features(best_pipeline)
```

#### Phase 4: save

```python
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH, compress=3)
```
Make `models/` and save the winning pipeline. **This overwrites** any older
model with no warning.

```python
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config_name": best_name,
        "config": best["config"],
        "seed": args.seed,
        "rows_used": int(len(splits["train"]) + len(splits["val"]) + len(splits["test"])),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "val_metrics": {n: r["val"] for n, r in results.items()},
        "test_metrics": test_metrics,
        "cv_scores": [float(s) for s in cv_scores] if cv_scores is not None else None,
        "confusion_matrix": matrix.tolist(),
        "top_features": top_features,
        "label_names": LABEL_NAMES,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
```
Build the **report card** and save it as JSON:
- `datetime.now(timezone.utc).isoformat()` → current time as text like
  `2026-09-23T06:12:23.373555+00:00`.
- `matrix.tolist()` → numpy array to plain lists (JSON can't store numpy
  arrays).
- `json.dumps(meta, indent=2)` → turn the dict into JSON text, indented 2
  spaces so humans can read it.
- JSON keys are always text, so `LABEL_NAMES = {0: "negative"}` is saved as
  `{"0": "negative"}`.

The web app reads this file to show the accuracy badge and the "What the
model learned" card.

```python
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved pipeline -> {MODEL_PATH.name} ({size_mb:.1f} MB)")
    print(f"Saved metadata -> {META_PATH.name}")
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
```
Print a summary. Run `main()` only when the file is run directly.

---

## Part 12 — `predict.py` line by line

This is the **doorway** between the ML world and the web app. The web app only
ever touches the model through this file. It never imports `train.py`.

```python
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib

from dataset import clean_text
```
- `lru_cache` — a decorator that remembers a function's results (explained
  below).
- `clean_text` — imported from `dataset.py`, **the same function** that
  cleaned the training data. Never copy-paste it; import it.

```python
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
META_PATH = MODELS_DIR / "model_meta.json"
```
Same paths as `train.py` uses to save.

### The neutral band (lines 20–38)

```python
NEUTRAL_LOW = 0.40
NEUTRAL_HIGH = 0.60
```

The model can only say "probability of positive". It has no "neutral" answer,
because it never saw a neutral example. But lots of real comments have no
feeling: "what's the price?", "@Rahim look at this".

The trick: a neutral comment has **no strong words either way**, so its
probability lands **near 0.5**. So we draw a band around the middle:

```
 0.0 ──────────── 0.40 ─────── 0.60 ──────────── 1.0
     NEGATIVE     │   NEUTRAL   │     POSITIVE
```

This band is **not learned**. A human picked it. That's called a
**calibration choice**.
- Wider band (e.g. 0.30–0.70) → more comments called neutral; the
  positive/negative calls that remain are more confident.
- Narrower band (0.45–0.55) → fewer neutrals, more pos/neg calls, more
  mistakes among them.

### `load_pipeline()` (lines 41–53)

```python
@lru_cache(maxsize=1)
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}.\n"
            f"Train one first:  uv run python train.py"
        )
    return joblib.load(MODEL_PATH)
```
Loads the brain file from disk.

`@lru_cache(maxsize=1)` means: **the first time** this function is called, run
it and remember the answer. **Every later call** returns the remembered
answer instantly, without running the function again. ("LRU" = "least
recently used", a rule for which answers to forget when the memory is full;
`maxsize=1` = remember one answer.)

Why? Loading the file takes ~100 ms. Fine once; too slow on every web request.

**Side effect:** if you retrain while `app.py` is running, the web app keeps
using the **old** model it remembered. **Restart `app.py`** after training.

### `load_meta()` (lines 56–60)

```python
@lru_cache(maxsize=1)
def load_meta() -> dict:
    if not META_PATH.exists():
        return {}
    return json.loads(META_PATH.read_text())
```
Read the report card. `json.loads` = turn JSON text into a Python dict. If
the file is missing, return an empty dict `{}` (the UI then knows the model
isn't trained). Also cached.

### `classify()` (lines 63–116)

```python
def classify(texts: list[str],
             low: float = NEUTRAL_LOW,
             high: float = NEUTRAL_HIGH) -> list[dict]:
```
Takes a **list** of comment strings. Returns a **list** of result dicts, one
per comment, in the same order. `low` and `high` default to the neutral band.

Why a list, even for one comment? Because the vectorizer and model are much
faster doing 500 texts in one call than 500 separate calls.

```python
    if not texts:
        return []
```
Empty list in → empty list out.

```python
    pipeline = load_pipeline()
```
Get the (cached) model.

```python
    cleaned = [clean_text(t or "") for t in texts]
```
Clean every comment exactly like training did. `t or ""` protects against a
`None`.

```python
    scores = pipeline.predict_proba(cleaned)[:, 1]
```
**This is the line where the prediction happens.** The pipeline turns the
cleaned texts into TF-IDF numbers (using the saved vocabulary) and runs
logistic regression. `[:, 1]` keeps only P(positive) for each text.

```python
    results = []
    for raw, clean, score in zip(texts, cleaned, scores):
```
Walk through the original text, cleaned text and score of each comment
together. (`raw` isn't used inside the loop; it's there only to keep the
three lists lined up and readable.)

```python
        if not clean:
            results.append({
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0,
                "reason": "no usable text after cleaning",
            })
            continue
```
If cleaning left **nothing** (emoji-only, link-only, Bangla script), the model
has no words to judge. Its output would just be the intercept (a meaningless
~0.63). So we honestly say: neutral, score 0.5, confidence 0, and give a
reason. `continue` skips to the next comment.

```python
        if score >= high:
            sentiment = "positive"
        elif score <= low:
            sentiment = "negative"
        else:
            sentiment = "neutral"
```
Apply the neutral band.

```python
        results.append({
            "sentiment": sentiment,
            "score": round(float(score), 4),
            "confidence": round(float(abs(score - 0.5) * 2), 4),
        })

    return results
```
- **score** = P(positive), rounded to 4 decimals.
- **confidence** = how far from the middle, scaled to 0..1:
  `abs(score − 0.5) × 2`.

| score | confidence |
|---|---|
| 0.50 | 0.0 (no idea) |
| 0.75 | 0.5 |
| 0.25 | 0.5 |
| 1.00 | 1.0 (certain positive) |
| 0.00 | 1.0 (certain negative) |

### `classify_one()` (lines 119–121)

```python
def classify_one(text: str) -> dict:
    return classify([text])[0]
```
Wrap one text into a list, classify, take the first (only) result. Used by the
"Try the model" box.

### The smoke test (lines 124–138)

```python
if __name__ == "__main__":
    samples = [ ... ]
    width = max(len(s) for s in samples) + 2
    for text, result in zip(samples, classify(samples)):
        print(f"{text!r:<{width}} -> {result['sentiment']:<9} "
              f"score={result['score']:.3f} conf={result['confidence']:.3f}")
```
`uv run python predict.py` runs this. `{text!r:<{width}}` = show the text
with quotes, left-aligned, padded to `width` characters (the width is itself
filled in from a variable, hence the nested `{}`). Real output:

```
'absolutely love this product, best purchase ever'       -> positive  score=0.944 conf=0.887
'terrible quality, broke after two days, waste of money' -> negative  score=0.128 conf=0.744
'what is the price?'                                     -> neutral   score=0.453 conf=0.094
'when will you restock this item'                        -> negative  score=0.385 conf=0.230
'not bad at all, pretty happy with it'                   -> positive  score=0.821 conf=0.642
'i cant wait for the next one'                           -> positive  score=0.884 conf=0.768
''                                                       -> neutral   score=0.500 conf=0.000
```

A few more I ran on the real model:

| text | sentiment | score |
|---|---|---|
| `thanks` | positive | 0.999 |
| `not good` | negative | 0.025 |
| `I am not happy with this` | negative | 0.148 |
| `terrible` | negative | 0.091 |
| `the product is ok` | positive | 0.778 |
| `love it but delivery was slow` | positive | 0.689 |
| `Great service!!! 😍` | positive | 0.840 (the emoji is removed; "great service" does the work) |
| `বাজে পণ্য` (Bangla: "bad product") | neutral | 0.5, reason: no usable text |

---

## Part 13 — How the web works (just enough)

### 13.1 Client and server

- The **server** is a program that waits for requests and answers them. Here:
  Flask (`app.py`), running on your computer.
- The **client** is the program that asks. Here: your web browser running the
  Vue app.

### 13.2 URLs, hosts and ports

`http://127.0.0.1:5000/api/analytics`

| part | meaning |
|---|---|
| `http://` | the protocol (the language of the conversation) |
| `127.0.0.1` | the computer's address. `127.0.0.1` always means "this same computer" (also called `localhost`) |
| `:5000` | the **port**: which program on that computer. Flask uses 5000; Vite's dev server uses 5173 |
| `/api/analytics` | the **path**: which thing you want |
| `?limit=10&fields=id` | (optional) the **query string**: extra options |

### 13.3 HTTP requests and responses

A request has a **method**:
- **GET** = "give me something". Should never change anything.
- **POST** = "here's something / do something". Can change things.

A request can also have **headers** (extra info like
`Authorization: Bearer <token>` or `Content-Type: application/json`) and a
**body** (the data being sent, for POST).

A response has a **status code**:

| code | meaning | used where |
|---|---|---|
| 200 | OK | normal success |
| 400 | Bad Request (you sent something wrong) | `/api/classify` with empty text |
| 404 | Not Found | wrong URL |
| 500 | Server crashed | a bug |
| 502 | Bad Gateway (a server *behind* us failed) | Facebook unreachable or token rejected |

### 13.4 JSON

The data format both sides speak:

```json
{
  "sentiment": "positive",
  "score": 0.944,
  "confidence": 0.887
}
```

Python dict ↔ JSON ↔ JavaScript object. Flask's `jsonify()` turns a dict into
JSON; JavaScript's `response.json()` turns JSON into an object.

### 13.5 API

The backend's set of URLs that return JSON (not web pages) is the **API**.
This app's API:

| method | path | what it does |
|---|---|---|
| GET | `/` | sends the web page (`index.html`) |
| GET | `/api/model` | the model's report card + settings |
| GET | `/api/analytics` | all numbers for the charts (reads the database only) |
| POST | `/api/refresh` | fetch from Facebook → save → classify → return new numbers |
| POST | `/api/classify` | classify one sentence you typed |

---

## Part 14 — `db.py` and SQL line by line

### 14.1 What is a database?

A place to keep data **permanently** (it survives restarts), organized into
**tables**. Each table has **columns** (fixed) and **rows** (one per thing).
It's like a spreadsheet you talk to in a language called **SQL**
(Structured Query Language).

**SQLite** stores the whole database in **one file**: `backend/data/app.db`.
No server, no password. Python has it built in (`import sqlite3`).

### 14.2 SQL in 2 minutes

```sql
CREATE TABLE people (id TEXT PRIMARY KEY, name TEXT, age INTEGER);  -- make a table
INSERT INTO people (id, name, age) VALUES ('1', 'Rahim', 12);      -- add a row
SELECT name, age FROM people WHERE age > 10;                       -- read rows
UPDATE people SET age = 13 WHERE id = '1';                         -- change rows
SELECT COUNT(*) FROM people;                                       -- how many rows
SELECT age, COUNT(*) FROM people GROUP BY age;                     -- count per age
SELECT * FROM people ORDER BY age DESC LIMIT 5;                    -- 5 oldest
```

- `--` starts a comment in SQL.
- `*` = all columns.
- `WHERE` filters rows. `GROUP BY` makes one result row per group.
  `ORDER BY ... ASC/DESC` sorts (ascending = small first / descending = big
  first). `LIMIT n` keeps the first n.
- `PRIMARY KEY` = this column is **unique** for each row. No two rows can
  share it.
- SQL keywords are written in CAPITALS by habit; SQL doesn't care.

### 14.3 The schema (lines 15–39)

The **schema** is the design of the tables.

```python
DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"
```

```sql
CREATE TABLE IF NOT EXISTS posts (
    id            TEXT PRIMARY KEY,
    message       TEXT,
    created_time  TEXT,
    permalink     TEXT
);
```
`IF NOT EXISTS` = only create it if it isn't there yet (so running this every
startup is safe). One row per Facebook post:
- `id` — Facebook's post id, like `"1234567890_9876543210"`. Unique.
- `message` — the post's text.
- `created_time` — when it was posted (stored as text).
- `permalink` — the link to the post on Facebook.

```sql
CREATE TABLE IF NOT EXISTS comments (
    id            TEXT PRIMARY KEY,   -- Facebook's comment id
    post_id       TEXT,
    message       TEXT NOT NULL,
    author        TEXT,
    created_time  TEXT,
    sentiment     TEXT,               -- positive | negative | neutral
    score         REAL,               -- P(positive), 0..1
    confidence    REAL,
    classified_at TEXT,
    FOREIGN KEY (post_id) REFERENCES posts(id)
);
```
One row per comment.
- `id` — Facebook's own comment id. **This is the key to not double-counting**
  (below).
- `post_id` — which post this comment belongs to.
- `message ... NOT NULL` — the comment text; it must be present.
- `author` — the commenter's name, often empty (Facebook hides it unless the
  commenter authorized the app).
- `sentiment`, `score`, `confidence`, `classified_at` — filled in **later** by
  the model. `NULL` (empty) means "not classified yet".
- `REAL` = a decimal number.
- `FOREIGN KEY (post_id) REFERENCES posts(id)` — documents that `post_id`
  points at a row in `posts`. (SQLite only enforces this if you switch it on,
  which this code doesn't. Here it's mostly documentation.)

```sql
CREATE INDEX IF NOT EXISTS idx_comments_sentiment ON comments(sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_created   ON comments(created_time);
CREATE INDEX IF NOT EXISTS idx_comments_post      ON comments(post_id);
```
An **index** is like the index at the back of a book: it lets the database
find rows by that column quickly instead of reading every row. We index the
columns we filter and group by.

### 14.4 Functions

```python
def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row      # rows behave like dicts
    return conn
```
Open the database file (creating the `data/` folder and the file if needed).
`row_factory = sqlite3.Row` makes each result row usable like
`row["message"]` instead of only `row[1]`.

```python
def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
```
Run all the `CREATE` statements. `executescript` runs several SQL statements
at once. Called once when `app.py` starts.

About `with connect() as conn:` for SQLite: when the block ends **without an
error**, the changes are **saved (committed)**. If an error happens, they are
**undone (rolled back)**. (It doesn't close the connection; Python closes it
later when it's no longer used.)

```python
def save_posts(posts: list[dict]) -> None:
    with connect() as conn:
        conn.executemany(
            """INSERT INTO posts (id, message, created_time, permalink)
               VALUES (:id, :message, :created_time, :permalink)
               ON CONFLICT(id) DO UPDATE SET message = excluded.message""",
            posts,
        )
```
- `executemany(sql, list_of_dicts)` runs the SQL once **per dict**.
- `:id`, `:message` are **named placeholders**: filled from each dict's keys.
  Never paste values straight into SQL text; placeholders make it safe from
  "SQL injection" (someone sneaking SQL commands into their comment text).
- `ON CONFLICT(id) DO UPDATE SET message = excluded.message` — this is called
  an **upsert**: if a post with this id already exists, don't fail; just update
  its message (in case the post was edited). `excluded` means "the new row we
  tried to insert".

```python
def save_comments(comments: list[dict]) -> int:
    if not comments:
        return 0
    with connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        conn.executemany(
            """INSERT INTO comments
                   (id, post_id, message, author, created_time)
               VALUES (:id, :post_id, :message, :author, :created_time)
               ON CONFLICT(id) DO NOTHING""",
            comments,
        )
        after = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    return after - before
```
- Count rows before, insert, count after. The difference = **how many were
  new**.
- `.fetchone()` = get the first result row; `[0]` = its first column (the
  count).
- **`ON CONFLICT(id) DO NOTHING`** — the important bit. Facebook sends back
  the same old comments every time you refresh. Because `id` is Facebook's own
  comment id and it's the PRIMARY KEY, trying to insert a comment we already
  have just does nothing. You can press Refresh 100 times and each comment
  is stored exactly once. This property ("doing it twice = doing it once") is
  called **idempotent**.
- Notice it doesn't insert sentiment. New comments start with
  `sentiment = NULL`.

```python
def unclassified(limit: int = 1000) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT id, message FROM comments WHERE sentiment IS NULL LIMIT ?",
            (limit,),
        ).fetchall()
```
Get up to 1000 comments that have **no sentiment yet**.
- `IS NULL` (not `= NULL`; in SQL you must write `IS NULL`).
- `?` is a **positional** placeholder, filled from the tuple `(limit,)`.
  The trailing comma matters: `(limit)` is just `limit` in brackets, but
  `(limit,)` is a **one-item tuple**.
- `.fetchall()` = all result rows as a list.

This is why refresh stays fast: only **new** comments get classified, not all
of them every time.

```python
def save_classifications(rows: list[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["classified_at"] = now
    with connect() as conn:
        conn.executemany(
            """UPDATE comments
                  SET sentiment = :sentiment, score = :score,
                      confidence = :confidence, classified_at = :classified_at
                WHERE id = :id""",
            rows,
        )
```
Stamp each result with the current time, then write the model's answers into
the matching comment rows.

### 14.5 `analytics()` — everything the dashboard shows (lines 114–173)

```python
        totals = {r["sentiment"]: r["n"] for r in conn.execute(
            """SELECT sentiment, COUNT(*) AS n FROM comments
                WHERE sentiment IS NOT NULL GROUP BY sentiment"""
        )}
```
SQL: count comments per sentiment. Result rows like
`("negative", 18), ("neutral", 18), ("positive", 35)`. `AS n` names the count
column `n`. The dict comprehension turns them into
`{"negative": 18, "neutral": 18, "positive": 35}`.

```python
        total = sum(totals.values())
```
18 + 18 + 35 = 71.

```python
        timeline = [dict(r) for r in conn.execute(
            """SELECT SUBSTR(created_time, 1, 10) AS day, sentiment, COUNT(*) AS n
                 FROM comments
                WHERE sentiment IS NOT NULL AND created_time IS NOT NULL
             GROUP BY day, sentiment
             ORDER BY day"""
        )]
```
Count comments **per day per sentiment**. `SUBSTR(text, 1, 10)` = "the
characters from position 1, 10 of them" (SQL counts from 1, not 0). It cuts
`2026-09-23T05:58:14+0000` down to `2026-09-23`. Result rows like
`{"day": "2026-09-23", "sentiment": "positive", "n": 35}`.
`dict(r)` turns each `sqlite3.Row` into a real dict so Flask can make JSON of
it.

**Why not SQLite's `DATE()` function?** That's what this line used to say,
and it was a bug: `DATE()` only understands a timezone written as `+00:00`
(with a colon) or `Z`. Facebook writes `+0000`, so `DATE()` gave up and returned
`NULL` for every row, and the chart showed one undated bar. Part 24.1 has the
story.

```python
        per_post = [dict(r) for r in conn.execute(
            """SELECT p.id, p.message, p.permalink,
                      COUNT(c.id) AS total,
                      SUM(c.sentiment = 'positive') AS positive,
                      SUM(c.sentiment = 'negative') AS negative,
                      SUM(c.sentiment = 'neutral')  AS neutral
                 FROM posts p JOIN comments c ON c.post_id = p.id
                WHERE c.sentiment IS NOT NULL
             GROUP BY p.id
             ORDER BY total DESC"""
        )]
```
One row per post with counts.
- `posts p` and `comments c` give the tables short nicknames (`p`, `c`).
- `JOIN comments c ON c.post_id = p.id` = pair each post with its comments.
  A post with zero classified comments disappears from the result (that's how
  a plain `JOIN` works).
- `SUM(c.sentiment = 'positive')`: in SQLite, a comparison gives **1** if true
  and **0** if false. Adding them up counts the positives. Neat trick.
- `ORDER BY total DESC` = most-commented post first.

```python
        attention = [dict(r) for r in conn.execute(
            """SELECT id, message, author, created_time, sentiment, score
                 FROM comments
                WHERE sentiment = 'negative'
             ORDER BY score ASC LIMIT 50"""
        )]
```
The **"Needs attention"** list: negative comments, the **lowest** score
(= most negative) first, at most 50.

```python
    return {
        "totals": {
            "positive": totals.get("positive", 0),
            "negative": totals.get("negative", 0),
            "neutral": totals.get("neutral", 0),
            "total": total,
        },
        "percentages": {
            k: round(totals.get(k, 0) / total * 100, 1) if total else 0.0
            for k in ("positive", "negative", "neutral")
        },
        "timeline": timeline,
        "per_post": per_post,
        "needs_attention": attention,
    }
```
Package it all into one dict. `.get(k, 0)` makes sure missing sentiments show
as 0. In the percentages, `if total else 0.0` avoids **dividing by zero**
when the database is empty. With your data: positive 49.3, negative 25.4,
neutral 25.4.

This dict is exactly the JSON the frontend receives.

---

## Part 15 — `facebook.py` line by line

Facebook's API is called the **Graph API**. You ask it URLs like
`https://graph.facebook.com/v21.0/<page-id>/posts` and it answers in JSON.
To prove you're allowed, you send an **access token**.

```python
from __future__ import annotations

import os
import re

import requests
```
- `os` — to read environment variables (`os.getenv`).
- `re` — regex, used to hide tokens in error messages.
- `requests` — to make the HTTP calls.

```python
GRAPH_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"
```
Facebook versions its API; we pin v21.0 so their changes don't surprise us.

```python
PAGE_ID = os.getenv("FB_PAGE_ID", "").strip()
PAGE_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "").strip()
```
Read the two settings from the environment (which `app.py` filled from
`.env`). If missing, use `""`. `.strip()` removes stray spaces or newlines.

**Important:** these lines run **once, when the file is first imported**. If
`.env` hadn't been loaded yet at that moment, both would be empty forever.
That's why `app.py` calls `load_dotenv()` **before** `import facebook`.

```python
class GraphError(RuntimeError):
    """Anything that went wrong talking to Facebook, with no token in the text."""
```
Our own error type (Part 5.13), so `app.py` can catch exactly Facebook
problems.

```python
def is_configured() -> bool:
    return bool(PAGE_ID and PAGE_TOKEN)
```
True only if **both** are non-empty. `bool(...)` makes sure the answer is
exactly True/False.

```python
def _redact(text: str) -> str:
    return re.sub(r"EAA[A-Za-z0-9_\-]{20,}", "<redacted-token>", text)
```
Facebook tokens start with `EAA`. This finds "EAA followed by 20 or more
letters/digits/underscores/dashes" and replaces it with `<redacted-token>`.
(`\-` is an escaped dash, so it means a literal `-` rather than a range.)
The `_` at the start of the name = "internal helper, not for other files".

```python
def _get(path: str, **params) -> dict:
    try:
        response = requests.get(
            f"{GRAPH_BASE}/{path}",
            params=params,
            headers={"Authorization": f"Bearer {PAGE_TOKEN}"},
            timeout=30,
        )
```
The one function that actually calls Facebook.
- `path` like `"12345/posts"`.
- `**params` collects options like `fields=...`, `limit=10` into a dict;
  `requests` turns them into the query string `?fields=...&limit=10`.
- **The token goes in a header** (`Authorization: Bearer EAA...`), **not** in
  the URL. URLs get written into logs, error messages and browser history. A
  Page token **never expires**, so one leak is a permanent leak until you
  revoke it by hand.
- `timeout=30` = give up after 30 seconds of silence.

```python
    except requests.exceptions.RequestException as exc:
        raise GraphError(
            "Could not reach graph.facebook.com. Check your internet "
            f"connection and try again. ({type(exc).__name__})"
        ) from None
```
If the network failed (no internet, DNS failure, timeout), turn it into a
friendly `GraphError`. `type(exc).__name__` = the error's class name, like
`ConnectionError`. `from None` hides the original error chain, which might
include request details.

```python
    if not response.ok:
        raise GraphError(
            f"Graph API {response.status_code}: {_redact(response.text[:400])}"
        )
    return response.json()
```
`response.ok` is True for status codes below 400. If Facebook said no (bad
token, missing permission), raise an error with the first 400 characters of
Facebook's explanation, with any token blanked out. Otherwise, turn the JSON
answer into a Python dict and return it.

```python
def fetch_posts(limit: int = 10) -> list[dict]:
    payload = _get(f"{PAGE_ID}/posts",
                   fields="id,message,created_time,permalink_url",
                   limit=limit)
    return [
        {
            "id": post["id"],
            "message": post.get("message", ""),
            "created_time": post.get("created_time"),
            "permalink": post.get("permalink_url"),
        }
        for post in payload.get("data", [])
    ]
```
Ask for the Page's 10 newest posts, only the fields we need. Facebook
answers like:

```json
{
  "data": [
    {"id": "123_456", "message": "New stock!", "created_time": "2026-09-20T10:15:00+0000",
     "permalink_url": "https://facebook.com/..."}
  ],
  "paging": { ... }
}
```

The list comprehension reshapes each post into **our** format (renaming
`permalink_url` → `permalink`). `.get()` is used because a photo-only post has
no `message`.

```python
def fetch_comments(post_id: str, limit: int = 100) -> list[dict]:
    comments: list[dict] = []
    params = dict(fields="id,message,created_time,from", limit=limit,
                  filter="stream")
    path = f"{post_id}/comments"
```
Get **all** comments of one post. `filter="stream"` includes replies to
comments too (not just top-level ones). `from` is the commenter.

```python
    while True:
        payload = _get(path, **params)
        for comment in payload.get("data", []):
            if not comment.get("message"):
                continue                     # sticker/photo-only comment
            comments.append({
                "id": comment["id"],
                "post_id": post_id,
                "message": comment["message"],
                "author": (comment.get("from") or {}).get("name"),
                "created_time": comment.get("created_time"),
            })
```
**Pagination:** Facebook returns at most `limit` (100) comments per answer.
If there are more, you must ask again for the next "page".
- Skip comments with no text (stickers, photos).
- `(comment.get("from") or {}).get("name")`: if `from` is missing,
  `comment.get("from")` is `None`; `None or {}` gives `{}`; `{}.get("name")`
  gives `None`. So a missing author never crashes the code.

```python
        cursor = payload.get("paging", {}).get("cursors", {}).get("after")
        if not cursor or not payload.get("data"):
            break
        params["after"] = cursor
    return comments
```
Facebook's answer includes `"paging": {"cursors": {"after": "QVFIU..."}}`.
That **cursor** is a bookmark: "send this back and I'll give you the next
page". Keep asking with `after=<cursor>` until there's no cursor or no data.
The chained `.get(..., {})` calls safely dig into nested dicts.

```python
def fetch_latest(post_limit: int = 10) -> tuple[list[dict], list[dict]]:
    if not is_configured():
        raise GraphError("Facebook is not configured. ...")

    posts = fetch_posts(limit=post_limit)
    comments: list[dict] = []
    for post in posts:
        comments.extend(fetch_comments(post["id"]))
    return posts, comments
```
The function `app.py` calls. Get the 10 newest posts, then all comments of
each. `.extend(list)` adds all items of a list to the end (unlike `.append`,
which would add the list as one item). Returns **two** things as a tuple.

### Facebook permissions (the gotchas)

- The token needs `pages_show_list`, `pages_read_engagement` **and**
  `pages_read_user_content`. Without the third, posts come back with **zero
  comments** and **no error**. Very confusing if you don't know.
- While your Facebook app is in "Development Mode", it can only read Pages
  **you** administer. That's enough for this project.

---

## Part 16 — `get_token.py` line by line

A one-time helper. Facebook tokens come in kinds:

| Token | Lasts | How you get it |
|---|---|---|
| short-lived **user** token | ~1–2 hours | Graph API Explorer website |
| long-lived **user** token | ~60 days | exchange the short one + your **app secret** |
| **Page** token (from a long-lived user token) | **doesn't expire** | ask `/me/accounts` with the long-lived token |

The script does the exchanges and writes the Page token into `.env`.

```python
from getpass import getpass
from pathlib import Path
import sys

import requests

GRAPH = "https://graph.facebook.com/v21.0"
ENV_PATH = Path(__file__).resolve().parent / ".env"
```
`getpass` = like `input()`, but **doesn't show** what you type (for secrets).

```python
def die(message: str, payload: dict | None = None) -> None:
    print(f"\n  ERROR: {message}")
    if payload:
        error = payload.get("error", {})
        if error:
            print(f"  Facebook says: {error.get('message')}")
            print(f"  (type={error.get('type')} code={error.get('code')})")
    sys.exit(1)
```
Print an error (plus Facebook's explanation if we have one) and stop the
program. `sys.exit(1)`: exit code 1 means "failed" (0 means success).

```python
def exchange_for_long_lived(app_id: str, app_secret: str, short_token: str) -> str:
    print("\n[1/3] Exchanging for a long-lived user token...")
    response = requests.get(
        f"{GRAPH}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": short_token,
        },
        timeout=30,
    )
    payload = response.json()
    if "access_token" not in payload:
        die("could not exchange the token", payload)
    expires = payload.get("expires_in")
    print(f"      got it" + (f" (expires in ~{int(expires) // 86400} days)" if expires else ""))
    return payload["access_token"]
```
Step 1: short → long-lived user token. Needs the **app secret**, which is why
this has to be a script on your computer and not something in a browser.
`expires_in` is in seconds; `// 86400` (seconds in a day, whole-number
division) → days. `"key" not in dict` checks whether a key is missing.

Note: here the tokens **are** sent in the URL (`params`). That's a one-off,
run by hand on your own machine, so it's acceptable. The long-running web
app (`facebook.py`) uses the safer header.

```python
def list_pages(long_token: str) -> list[dict]:
    response = requests.get(
        f"{GRAPH}/me/accounts",
        params={"access_token": long_token, "fields": "id,name,access_token"},
        timeout=30,
    )
    payload = response.json()
    if "data" not in payload:
        die("could not list your Pages", payload)
    pages = payload["data"]
    if not pages:
        die("no Pages found. ...")
    return pages
```
Step 2: `/me/accounts` = "the Pages I administer". Each comes with its own
**Page token**.

```python
def verify(page_token: str, page_id: str) -> None:
    response = requests.get(
        f"{GRAPH}/{page_id}/posts",
        params={"access_token": page_token, "limit": 1, "fields": "id"},
        timeout=30,
    )
    payload = response.json()
    if "data" not in payload:
        die("the Page token did not work for reading posts", payload)
    print(f"      works ({len(payload['data'])} post(s) visible)")
```
Step 3: try reading 1 post with the token, to prove it works before saving it.

```python
def write_env(page_id: str, page_token: str) -> None:
    if ENV_PATH.exists():
        answer = input(f"\n  {ENV_PATH.name} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("\n  Not written. Add these lines yourself:\n")
            print(f"    FB_PAGE_ID={page_id}")
            print(f"    FB_PAGE_ACCESS_TOKEN={page_token}")
            return
    ENV_PATH.write_text(
        f"FB_PAGE_ID={page_id}\nFB_PAGE_ACCESS_TOKEN={page_token}\n"
    )
```
Write `.env`, asking first if it would overwrite one. `[y/N]` is a convention:
the capital letter is the default (pressing Enter = No).

```python
def main() -> None:
    print(__doc__)
    app_id = input("App ID: ").strip()
    app_secret = getpass("App Secret (hidden): ").strip()
    short_token = getpass("Short-lived User token (hidden): ").strip()

    if not (app_id and app_secret and short_token):
        die("all three values are required")

    long_token = exchange_for_long_lived(app_id, app_secret, short_token)
    pages = list_pages(long_token)

    print(f"\n  Found {len(pages)} page(s):")
    for i, page in enumerate(pages, 1):
        print(f"    {i}. {page['name']}  (id {page['id']})")

    if len(pages) == 1:
        chosen = pages[0]
    else:
        raw = input(f"\n  Which page? [1-{len(pages)}]: ").strip()
        try:
            chosen = pages[int(raw) - 1]
        except (ValueError, IndexError):
            die("that was not one of the listed numbers")

    verify(chosen["access_token"], chosen["id"])
    print(f"\n  Page:  {chosen['name']}")
    print(f"  ID:    {chosen['id']}")
    print(f"  Token: {chosen['access_token'][:24]}... ({len(chosen['access_token'])} chars)")
    write_env(chosen["id"], chosen["access_token"])
```
Ask for the three inputs, run steps 1–3, let you pick a Page if you have
several, and save it.
- `enumerate(pages, 1)` numbers them from 1 for humans.
- `pages[int(raw) - 1]` turns the human number (1-based) into a list position
  (0-based).
- `except (ValueError, IndexError)` catches two kinds of error: typing "abc"
  (`int("abc")` → ValueError) or typing "9" when there are 2 pages
  (IndexError).
- `[:24]` shows only the first 24 characters of the token, never the whole
  thing.

---

## Part 17 — `app.py` (Flask) line by line

This file **joins everything**. It is the web server.

```python
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Must run before facebook.py reads the environment at import time.
load_dotenv(Path(__file__).resolve().parent / ".env")
```
**Order matters.** Load `backend/.env` into environment variables **first**...

```python
from flask import Flask, jsonify, request  # noqa: E402

import db  # noqa: E402
import facebook  # noqa: E402
import predict  # noqa: E402
```
...**then** import `facebook`, whose top-level code reads those variables.

`# noqa: E402` tells code-style checkers "I know imports should be at the
top; it's on purpose". (E402 is the rule's code: "module-level import not at
top of file".) Python itself ignores it; it's just a comment.

- `Flask` — the web-server class.
- `jsonify` — turns a dict into a JSON response.
- `request` — the incoming request (to read the body).

```python
app = Flask(__name__, static_url_path="")
db.init_db()
```
- Create the web app. `__name__` tells Flask where this file is, so it can
  find the `static/` folder next to it.
- `static_url_path=""` means: serve files from `backend/static/` at the **root**
  of the site. So `backend/static/assets/index-DOx-dWUt.js` is served at
  `http://127.0.0.1:5000/assets/index-DOx-dWUt.js`. Without this, Flask would
  serve them at `/static/assets/...`, but the built HTML asks for `/assets/...`
  and would get 404s.
- `db.init_db()` creates the tables if they don't exist yet. This runs as soon
  as `app.py` is imported/run.

### Route: the page

```python
@app.get("/")
def index():
    return app.send_static_file("index.html")
```
When the browser asks for `/`, send `backend/static/index.html` as-is.

Why not `render_template()` (Flask's usual way)? Flask's template engine,
Jinja, uses `{{ }}`. So does Vue. Jinja would try to fill in every Vue
`{{ ... }}` and crash. The built page doesn't need any server-side filling
anyway.

### Route: analytics

```python
@app.get("/api/analytics")
def get_analytics():
    """Everything the charts need. Reads only - never touches Facebook."""
    return jsonify(db.analytics())
```
`GET /api/analytics` → read the database → return the dict from Part 14.5 as
JSON. Fast, no network.

### Route: refresh (the big button)

```python
@app.post("/api/refresh")
def refresh():
    try:
        posts, comments = facebook.fetch_latest()
    except facebook.GraphError as exc:
        return jsonify({"error": str(exc)}), 502
```
1. Ask Facebook. If that fails, answer with `{"error": "..."}` and status
   **502**. In Flask, returning a **tuple** `(body, status)` sets the status
   code. `str(exc)` = the error message as text. The frontend shows this
   message in red.

```python
    db.save_posts(posts)
    new_count = db.save_comments(comments)
```
2. Save posts (upsert) and comments (skip ones we already have). Remember how
   many were new.

```python
    pending = db.unclassified()
    classified = 0
    if pending:
        results = predict.classify([row["message"] for row in pending])
        db.save_classifications([
            {
                "id": row["id"],
                "sentiment": result["sentiment"],
                "score": result["score"],
                "confidence": result["confidence"],
            }
            for row, result in zip(pending, results)
        ])
        classified = len(results)
```
3. Find every comment without a sentiment. Send **all** their texts to the
   model in **one** call. Pair each database row with its result (`zip`), and
   build the list of updates. Save them.

Note that the `reason` key (for empty-after-cleaning comments) is **not**
saved to the database; only sentiment, score and confidence are.

```python
    return jsonify({
        "posts_seen": len(posts),
        "comments_seen": len(comments),
        "new_comments": new_count,
        "newly_classified": classified,
        "analytics": db.analytics(),
    })
```
4. Reply with a summary **plus** the fresh analytics, so the browser can
   redraw without making a second request.

### Route: classify one sentence

```python
@app.post("/api/classify")
def classify_text():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    return jsonify(predict.classify_one(text))
```
- `request.get_json(silent=True)` reads the JSON body `{"text": "..."}`. With
  `silent=True`, bad or missing JSON gives `None` instead of crashing.
  `or {}` turns `None` into an empty dict, and `.get("text", "")` then gives
  `""`.
- Empty or only spaces → 400 error.
- Otherwise classify and return `{"sentiment": ..., "score": ..., "confidence": ...}`.

### Route: model info

```python
@app.get("/api/model")
def model_info():
    meta = predict.load_meta()
    return jsonify({
        "trained": bool(meta),
        "configured": facebook.is_configured(),
        "neutral_band": [predict.NEUTRAL_LOW, predict.NEUTRAL_HIGH],
        **meta,
    })
```
- `trained` — True if the report card exists (a non-empty dict is truthy).
- `configured` — True if `.env` has both Facebook values.
- `neutral_band` — `[0.4, 0.6]`.
- `**meta` inside a dict literal = **copy all keys of `meta` in here** (config
  name, test metrics, top features...). It's the dict version of "spread".

### Starting the server

```python
if __name__ == "__main__":
    if not predict.MODEL_PATH.exists():
        print("\n  No trained model found.")
        print("  Run this first:  uv run python train.py\n")
        raise SystemExit(1)

    if facebook.is_configured():
        print(f"\n  Facebook page: {facebook.PAGE_ID}")
    else:
        print("\n  Facebook not configured - the dashboard will load but have")
        print("  nothing to show. Run:  uv run python get_token.py")
    print("  Dashboard:     http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
```
- Refuse to start without a trained model (`raise SystemExit(1)` = stop with
  "failed").
- Say whether Facebook is connected.
- `app.run(debug=True, port=5000)` starts the server and **waits forever** for
  requests. `debug=True` shows detailed errors and **restarts automatically
  when you edit a `.py` file**. (Debug mode is only for your own computer;
  never expose it to the internet.) Because of the automatic restarter, you
  may see the startup messages printed twice. That's normal.

Press **Ctrl+C** in the terminal to stop it.

---

## Part 18 — JavaScript crash course

The browser runs **JavaScript (JS)**. Vue is written in JS. Here is only what
this repo uses, compared with Python where it helps.

### 18.1 Variables

```js
const PER_PAGE = 7        // const = the name can't be pointed at something else
let chart = null          // let = the name CAN be changed later
```
JS uses `null` (and also `undefined`) where Python uses `None`.
`undefined` = "never given a value"; `null` = "deliberately empty".

`const` doesn't freeze the **contents**. `const analytics = reactive({...})`
can still have its fields changed; you just can't do `analytics = something else`.

### 18.2 Functions — three ways to write them

```js
function apply() {                       // classic
  document.documentElement.dataset.theme = theme.value
}

async function refresh() { ... }         // async version (see 18.6)

const formatDay = (iso) => iso ? ... : ''    // arrow function
export const getModel = () => request('/api/model')
```
**Arrow functions** `(inputs) => result`:
- `(x) => x * 2` — if the right side is a single expression, it's
  automatically returned.
- `(x) => { ...; return y }` — with `{ }`, you must write `return`.
- `() => ...` — no inputs.

### 18.3 Objects and arrays

```js
const options = { method: 'POST', headers: { 'Content-Type': 'application/json' } }
options.method           // 'POST'
options['method']        // same

const days = ['2026-09-22', '2026-09-23']
days[0]                  // first item
days.length              // 2
```
- Object = Python dict. Keys don't need quotes unless they contain special
  characters (like `'Content-Type'`).
- Array = Python list.

### 18.4 Array tools used here

| Code | Meaning |
|---|---|
| `arr.map((x) => f(x))` | new array with `f` applied to each item (like a list comprehension) |
| `arr.find((x) => cond)` | the first item where `cond` is true, or `undefined` |
| `arr.slice(a, b)` | items from position a up to (not including) b |
| `arr.forEach((x) => ...)` | run something for each item |
| `[...new Set(arr)]` | remove duplicates: a `Set` keeps unique values, `...` spreads them back into an array |
| `arr.sort()` | sort (as text by default) |
| `arr.join(' – ')` | glue into one string |

### 18.5 Handy operators

```js
a === b          // equal (always use ===, not ==)
!x               // not
a && b           // and
a || b           // or. ALSO: "a if a is truthy, otherwise b": props.totals.total || 0
a ?? b           // "a unless a is null/undefined, then b": percentages.positive ?? 0
cond ? a : b     // if cond then a else b
obj?.field       // field if obj exists, else undefined (no crash)
chart?.destroy() // call destroy only if chart exists
```

The difference between `||` and `??` matters when the value is `0`:
`0 || 5` → 5 (because 0 is falsy), but `0 ?? 5` → 0 (because 0 is not
null/undefined). The repo uses `??` for percentages so that a real 0% stays 0.

**Template literals** use backticks:
```js
`${path} failed (${response.status})`     // "/api/model failed (500)"
```

**Spread** `...`:
```js
{ ...d, borderWidth: 2 }     // copy all fields of d, then add/override borderWidth
```

**Destructuring:**
```js
const { theme, toggle } = useTheme()
// same as:  const r = useTheme(); const theme = r.theme; const toggle = r.toggle
```

### 18.6 async / await / Promises

Asking a server takes time (milliseconds to seconds). JS doesn't stop and wait;
instead, `fetch()` immediately returns a **Promise**: an IOU that says "the
answer will arrive later".

```js
async function request(path, options = {}) {
  const response = await fetch(path, options)
  ...
}
```
- `await` = "pause **this function** until the Promise is fulfilled, then give
  me the value". The rest of the page keeps working meanwhile.
- You can only use `await` inside an `async` function.
- An `async` function itself always returns a Promise.

**Errors:** `try { ... } catch (e) { ... } finally { ... }` works like Python's
try/except. `throw new Error('msg')` is like Python's `raise`. `e.message` is
the error's text.

`.catch(() => ({}))` on a Promise = "if this fails, give `{}` instead". The
extra `( )` around `{}` is needed because `() => {}` would be read as an empty
function **body**, not an empty object.

### 18.7 Modules: import / export

```js
// api.js
export const getModel = () => ...           // named export

// App.vue
import * as api from './api'                // everything, under the name api → api.getModel()
import { ref, computed } from 'vue'         // pick named exports
import App from './App.vue'                 // the default export
import './style.css'                        // just load it (for its effect)
```
`'./api'` = the file `api.js` in **this** folder (`./`). `'../theme'` = one
folder **up** (`../`). A name without `./` (like `'vue'`) = a library from
`node_modules`.

### 18.8 The browser's built-in things used here

| Thing | What it is |
|---|---|
| `document` | the whole web page |
| `document.documentElement` | the `<html>` tag |
| `.dataset.theme = 'dark'` | sets the attribute `data-theme="dark"` on that tag |
| `fetch(url, options)` | make an HTTP request |
| `localStorage` | a tiny key-value store in the browser that survives page reloads |
| `matchMedia('(prefers-color-scheme: dark)')` | asks "is the computer set to dark mode?" |
| `getComputedStyle(el)` | the final CSS values applied to an element |
| `new Date(iso)` | turn a date string into a Date object |
| `.toLocaleDateString(undefined, {...})` | format a date for the user's language, e.g. "Sep 23" |
| `n.toLocaleString()` | format a number with separators, e.g. `1,234` |
| `n.toFixed(2)` | number to text with 2 decimals |
| `Math.max`, `Math.min`, `Math.ceil` | biggest, smallest, round up |
| `JSON.stringify(obj)` | object → JSON text |
| `response.json()` | JSON text → object (returns a Promise) |

---

## Part 19 — Frontend tooling: Node, npm, Vite, Tailwind

### 19.1 Why so many tools?

Browsers only understand **HTML, CSS and plain JavaScript**. But we write
`.vue` files and use Tailwind classes. Something must **translate** our source
into what browsers understand. That translator is **Vite** (pronounced
"veet", French for "fast"). Vite runs on **Node.js** (JavaScript outside the
browser). Node comes with **npm**, which installs libraries.

### 19.2 `package.json`

```json
{
  "name": "sentiment-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "chart.js": "^4.4.7",
    "vue": "^3.5.13"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.2.1",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.17",
    "vite": "^5.4.11"
  }
}
```
- `"private": true` — never publish this to the npm registry by accident.
- `"type": "module"` — `.js` files use `import`/`export` (modern style).
- `"scripts"` — shortcuts. `npm run dev` runs `vite`; `npm run build` runs
  `vite build`; `npm run preview` serves the built files to check them.
- `"dependencies"` — needed **in the browser**: `vue` and `chart.js` (draws
  charts).
- `"devDependencies"` — needed only **while building**: Vite, its Vue plugin,
  Tailwind, PostCSS and Autoprefixer.
- `^4.4.7` — "4.4.7 or any newer 4.x.x" (the `^` allows updates that don't
  change the first number).

### 19.3 `index.html`

```html
<!DOCTYPE html>
<html lang="en" data-theme="light">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Page Sentiment</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```
- `<!DOCTYPE html>` — "this is modern HTML".
- `lang="en"` — English (helps screen readers).
- `data-theme="light"` — starting theme; `main.js` replaces it immediately.
- `<meta charset="UTF-8" />` — how to read the letters.
- `<meta name="viewport" ...>` — "fit the width of a phone screen, don't zoom
  out".
- `<title>` — the text in the browser tab.
- `<div id="app"></div>` — an **empty box**. Vue will draw the whole
  dashboard inside it.
- `<script type="module" src="/src/main.js">` — load our JS entry point.

That's the entire page. Everything else is drawn by JavaScript. That's what
"single-page application" means.

### 19.4 `vite.config.js`

```js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
```
- `defineConfig({...})` just returns the object; it exists so editors can
  autocomplete settings.
- `export default` — the one main thing this file gives to whoever imports it
  (Vite).
- `plugins: [vue()]` — teach Vite how to translate `.vue` files.

```js
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:5000', changeOrigin: true },
    },
  },
```
Development server settings.
- Run on port 5173.
- **Proxy**: any request to `/api/...` that reaches Vite gets **forwarded** to
  Flask on port 5000, and Flask's answer is passed back. The browser thinks it
  only talks to 5173. Why? Browsers block a page from one "origin" (host+port)
  reading data from a different origin unless the server allows it (a rule
  called **CORS**). The proxy avoids needing any CORS setup.
  `changeOrigin: true` makes the forwarded request look like it was sent to
  5000 directly.

```js
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
  },
})
```
- `npm run build` writes the finished files into `backend/static/` (one
  folder up, then into backend).
- `emptyOutDir: true` — delete the old contents first.

### 19.5 Dev mode vs build mode

```
DEV  (npm run dev + uv run python app.py — two terminals)
  Browser ──► Vite :5173 ──(translates .vue on the fly, hot reload)
                  └── /api/* ──proxy──► Flask :5000

BUILD (npm run build, then only uv run python app.py)
  src/*.vue, *.js, *.css ──vite build──► backend/static/index.html
                                         backend/static/assets/index-DOx-dWUt.js   (≈290 KB, all JS incl. Vue + Chart.js)
                                         backend/static/assets/index-CI-O5U9j.css  (≈9 KB)
  Browser ──► Flask :5000 (serves the page, the assets AND the API)
```

**Hot reload** (dev mode): save a `.vue` file and the browser updates
instantly without losing its state.

The **built** `index.html` looks like this. Vite rewrote the script tag to
point at the bundle:

```html
<script type="module" crossorigin src="/assets/index-DOx-dWUt.js"></script>
<link rel="stylesheet" crossorigin href="/assets/index-CI-O5U9j.css">
```

The random-looking part (`DOx-dWUt`) is a **hash** of the file's contents. It
changes whenever the content changes, so browsers never use an old cached
copy.

**Rules:**
- Never edit `backend/static/` by hand. It's overwritten on every build.
- After changing anything in `frontend/src/`, run `npm run build` if you're
  using Flask on port 5000 (dev mode on 5173 doesn't need it).

### 19.6 Tailwind CSS

**CSS** = the language for how things look (colors, sizes, spacing).
**Tailwind** = a CSS library of tiny ready-made **classes** you put on HTML
tags. Instead of writing CSS, you write `class="text-sm px-3 py-2 rounded-lg"`.
Tailwind scans your files, finds which classes you used, and generates only
that CSS.

`tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js}'],
  theme: {
    extend: {
      borderColor: { DEFAULT: 'var(--hairline)' },
    },
  },
  plugins: [],
}
```
- `/** @type ... */` — a comment that tells the editor what type this object
  is (for autocomplete). No effect at runtime.
- `content` — which files to scan for class names. `**` = any folders deep,
  `*.{vue,js}` = any file ending in `.vue` or `.js`.
- `extend.borderColor.DEFAULT` — when a tag uses `border` without a color,
  use our theme-aware `--hairline` color instead of Tailwind's light gray
  (which glared in dark mode).

`postcss.config.js`:
```js
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
}
```
**PostCSS** is a CSS processor. Vite passes CSS through it; it runs Tailwind
(generate classes) then **Autoprefixer** (add old-browser variants of CSS
rules automatically). `{}` = default options.

**Tailwind classes used in this repo:**

| class | meaning |
|---|---|
| `min-h-screen` | at least as tall as the window |
| `max-w-6xl` | max width 72rem (≈1152px) |
| `mx-auto` | center horizontally (auto left/right margin) |
| `px-4` / `py-6` | padding left+right 1rem / top+bottom 1.5rem (1 unit = 0.25rem = 4px) |
| `p-4`, `p-3` | padding on all sides |
| `pb-2`, `pl-3`, `pr-3`, `pt-3` | padding bottom / left / right / top |
| `mb-1`, `mb-2`, `mb-3`, `mt-3` | margin bottom / top |
| `mr-auto`, `ml-auto` | push to the left / right (auto margin eats free space) |
| `space-y-4`, `space-y-1` | vertical gap between children |
| `flex` | lay children in a row |
| `flex-wrap` | let them wrap to the next line |
| `flex-1` | grow to fill leftover space |
| `items-center`, `items-start`, `items-baseline` | vertical alignment in a row |
| `gap-3`, `gap-2`, `gap-x-3` | space between children |
| `shrink-0` | never squeeze this item |
| `self-stretch` | stretch to the row's full height |
| `min-w-0`, `min-w-64` | minimum width 0 / 16rem |
| `grid` | lay children in a grid |
| `grid-cols-2` | 2 columns |
| `md:grid-cols-4` | 4 columns **on screens ≥ 768px** (`md:` = "medium and up") |
| `md:col-span-2` | on medium+ screens, take 2 grid columns |
| `text-xs`, `text-sm`, `text-xl`, `text-3xl` | font sizes |
| `font-medium`, `font-semibold` | boldness |
| `uppercase`, `capitalize` | letter case |
| `tracking-wide` | letter spacing |
| `text-left`, `text-right` | alignment |
| `text-white` | white text |
| `truncate` | one line, cut off with "…" |
| `rounded`, `rounded-lg`, `rounded-xl`, `rounded-full`, `rounded-sm` | corner roundness |
| `border`, `border-t` | 1px border all around / only on top |
| `divide-y` | line between each child row |
| `divide-[color:var(--hairline)]` | the `[ ]` is an "arbitrary value": use exactly this CSS color |
| `w-2.5 h-2.5`, `w-1`, `w-8 h-8`, `w-12`, `w-32`, `w-full`, `h-2` | width / height |
| `max-w-xs` | max width 20rem |
| `overflow-x-auto` | scroll sideways if too wide |
| `overflow-hidden` | clip anything outside |
| `relative` | positioning context (Chart.js needs it for sizing) |
| `disabled:opacity-50` | at 50% opacity **when the button is disabled** |
| `disabled:cursor-not-allowed` | show the 🚫 cursor when disabled |

### 19.7 `style.css` — colors, line by line

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```
Three placeholders that Tailwind replaces with: a reset of browser defaults
(`base`), component classes (`components`), and all the utility classes you
used (`utilities`).

```css
:root {
  color-scheme: light;
  --surface:  #fcfcfb;
  --plane:    #f9f9f7;
  --ink:      #0b0b0b;
  --ink-2:    #52514e;
  --muted:    #898781;
  --grid:     #e1e0d9;
  --axis:     #c3c2b7;
  --hairline: rgba(11, 11, 11, 0.10);
  --positive: #2a78d6;
  --neutral:  #898781;
  --negative: #e34948;
}
```
- `:root` = the `<html>` element (the top of the page).
- `--surface: #fcfcfb;` defines a **CSS variable** (custom property). Anywhere
  in CSS you can say `var(--surface)` to use it.
- `#fcfcfb` is a color in **hex**: two digits each for red, green, blue (00 to
  ff). `#ffffff` is white, `#000000` black.
- `rgba(11, 11, 11, 0.10)` = red, green, blue (0–255) and **alpha**
  (transparency, 0–1). 10% opaque near-black: a faint line.
- `color-scheme: light` tells the browser to draw its own widgets (scrollbars,
  inputs) in light style.

What each color is for:

| variable | used for |
|---|---|
| `--surface` | card backgrounds |
| `--plane` | page background, input backgrounds |
| `--ink` | main text |
| `--ink-2` | secondary text |
| `--muted` | small labels, least important text |
| `--grid` | chart grid lines, empty bar track |
| `--axis` | chart axis line |
| `--hairline` | thin borders |
| `--positive` | blue |
| `--neutral` | gray |
| `--negative` | red |

**Why blue/gray/red and not green/yellow/red?** Sentiment is a **scale**
(negative ← neutral → positive), not three unrelated categories. So we use a
**diverging** palette: two opposite hues at the ends and a gray "nothing" in
the middle. Blue vs red also stays distinguishable for most color-blind
people (red/green is the most common color-blindness problem). The colors were
checked with a color-blindness simulator.

```css
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme='light'])) {
    color-scheme: dark;
    --surface:  #1a1a19;
    ...
    --neutral:  #b5b3a8;
    ...
  }
}
```
- `@media (prefers-color-scheme: dark)` = "only if the computer is set to dark
  mode".
- `:root:where(:not([data-theme='light']))` = "the html tag, **unless** it has
  `data-theme="light"`". So: follow the OS's dark mode, unless the user
  explicitly chose light. (`:where()` makes this rule weak ("zero
  specificity") so it doesn't accidentally beat other rules.)
- Dark mode uses a **lighter gray** `#b5b3a8` for neutral, because the light-mode
  gray `#898781` looked almost identical to the dark-mode red for people with
  protanopia (a type of red-blindness).

```css
:root[data-theme='dark'] { ...same dark values... }
```
"If the user explicitly chose dark, use dark values", regardless of the OS.

Why are the dark values written twice? Because they apply in two different
situations: (1) OS is dark and the user hasn't picked light; (2) the user
picked dark. CSS has no simple way to say "A or B" across a media query.

```css
body {
  background: var(--plane);
  color: var(--ink);
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
}
```
Page background and text color from the variables. Font: use the operating
system's own UI font (the list is fallbacks, tried in order).

```css
@layer components {
  .card { @apply rounded-xl; background: var(--surface); border: 1px solid var(--hairline); }
  .num  { font-variant-numeric: tabular-nums; }
}
```
- Two custom classes. `@layer components` puts them in Tailwind's component
  layer, so utility classes can still override them.
- `.card` = the white(ish) rounded box used for every panel. `@apply rounded-xl`
  pulls in a Tailwind class inside our own CSS.
- `.num` = make all digits the same width, so columns of numbers line up.

---

## Part 20 — Vue crash course

### 20.1 The problem Vue solves

Without Vue, when data changes you'd have to find the right HTML tags and
change them by hand: "find the tile that says 71 and change it to 75, find
the donut and redraw it, find the list and add a row...". That gets messy fast.

With Vue you **describe** what the page should look like **for any data**,
and Vue keeps the page matching the data **automatically**. Change the data →
the page updates. This is called **reactivity**.

### 20.2 Components and Single-File Components (`.vue`)

The page is built from **components**, like Lego bricks. `App.vue` is the big
brick that holds the smaller ones (`KpiTiles`, `SentimentDonut`...).

Each `.vue` file has up to three parts:

```vue
<script setup>
  // JavaScript: the data and logic of this component
</script>

<template>
  <!-- HTML-like description of what to show -->
</template>

<style>
  /* optional CSS; this repo uses Tailwind + style.css instead */
</style>
```

`<script setup>` is the modern, short way. Everything you declare at the top
level of it (variables, functions, imported components) can be used directly
in the `<template>`.

### 20.3 Reactive data: `ref`, `reactive`, `computed`

```js
import { ref, reactive, computed } from 'vue'

const loading = ref(false)        // a reactive box holding one value
loading.value = true              // in JS, read/write through .value
```
In the **template** you write just `loading` (Vue adds `.value` for you).

```js
const analytics = reactive({ totals: {}, timeline: [] })   // a reactive object
analytics.totals = {...}          // no .value needed for reactive objects
```

```js
const accuracy = computed(() =>
  model.test_metrics ? (model.test_metrics.accuracy * 100).toFixed(1) : null,
)
```
A **computed** value is a formula. Vue remembers which reactive data the
formula read (`model.test_metrics`) and recalculates **only when that data
changes**. Like a spreadsheet cell `=A1*100`.

How does Vue know what to update? When a template or computed **reads**
reactive data, Vue writes down "this part depends on that data". When the data
is **changed**, Vue re-runs exactly the dependent parts. It only tracks
**reactive** things (refs, reactive objects, props). Reading plain values, such
as a CSS variable from the browser, is **not** tracked. (This caused a real
bug. See Part 24.3.)

### 20.4 Template syntax

| Syntax | Meaning | Example |
|---|---|---|
| `{{ expr }}` | show the value as text | `{{ tile.label }}` |
| `:attr="expr"` | set an attribute from JS (short for `v-bind:attr`) | `:disabled="loading"` |
| `:style="{ color: 'var(--ink-2)' }"` | set inline CSS from a JS object | |
| `:key="x"` | a unique id for each item in a `v-for` loop | `:key="tile.label"` |
| `@event="handler"` | run code when the event happens (short for `v-on:`) | `@click="refresh"` |
| `@keyup.enter="run"` | only when the **Enter** key is released (`.enter` is a modifier) | |
| `v-if="cond"` | only create this element if cond is true | |
| `v-else-if="cond"` | ...otherwise, if this | |
| `v-for="item in list"` | repeat this element for each item | |
| `v-for="n in 5"` | repeat for n = 1, 2, 3, 4, 5 | page-number buttons |
| `v-model="text"` | two-way link between an input and a ref: typing updates `text`, changing `text` updates the input | |
| `ref="canvas"` | give JS a handle to this real HTML element | used for the chart `<canvas>` |
| `<template v-if>` | an invisible wrapper to apply `v-if` to several elements at once | |
| `<Comp :prop="x" />` | use a child component and pass it data | `<KpiTiles :totals="..." />` |

**Quoting:** `:style="{ color: 'var(--ink-2)' }"`. The outer `"` belong to
HTML. Inside is JavaScript: an object `{ color: ... }` whose value is the
**string** `'var(--ink-2)'` (single quotes, so they don't clash with the
double quotes).

**`class` vs `:class`:** plain `class="text-sm"` is fixed text. Both can be
used together.

### 20.5 Props: passing data from parent to child

Parent (`App.vue`):
```vue
<KpiTiles :totals="analytics.totals" :percentages="analytics.percentages" />
```
Child (`KpiTiles.vue`):
```js
const props = defineProps({
  totals: { type: Object, default: () => ({}) },
  percentages: { type: Object, default: () => ({}) },
})
```
- `defineProps` declares what the component accepts.
- `type: Object` — it should be an object (Vue warns in the console
  otherwise).
- `default: () => ({})` — if not given, use an empty object. It must be a
  **function** returning a new object, so every component gets its own copy.
- Props are **read-only** in the child. Data flows **down** (parent → child).
- Props are reactive: when the parent's data changes, the child updates.

### 20.6 Lifecycle and watching

| Function | Runs when |
|---|---|
| `onMounted(fn)` | the component has just been put on the page (its HTML elements now exist) |
| `onBeforeUnmount(fn)` | the component is about to be removed (clean up here) |
| `watch(source, fn)` | whenever `source` changes, run `fn` |
| `watch(source, fn, { deep: true })` | also notice changes **inside** the object, not just replacement |

`watch(() => props.totals, draw, { deep: true })` = "whenever the totals
change, call `draw`".

### 20.7 How the app boots, overall

```
Browser loads index.html
  → <script src="/src/main.js">
    → main.js: set theme, createApp(App).mount('#app')
      → Vue builds App.vue into <div id="app">
        → App.vue's template creates each child component
        → App.vue's onMounted runs: fetch /api/model and /api/analytics
          → data arrives → reactive objects change → Vue redraws what depends on them
```

---

## Part 21 — Frontend files line by line

### 21.1 `src/main.js`

```js
import { createApp } from 'vue'
import App from './App.vue'
import './style.css'
import { useTheme } from './theme'

useTheme().apply()   // set data-theme before first paint, to avoid a flash
createApp(App).mount('#app')
```
1. Import Vue's `createApp`, our root component `App`, and the stylesheet
   (importing CSS in JS is a Vite feature: it gets injected into the page).
2. `useTheme().apply()` — set `data-theme` on `<html>` **before** anything
   is drawn. Otherwise you'd see a flash of the wrong theme.
3. `createApp(App)` — create the Vue app with `App` as the root.
   `.mount('#app')` — draw it inside the element with `id="app"` (`#` means
   "id" in CSS selectors).

### 21.2 `src/api.js`

The **only** file that talks to the backend.

```js
async function request(path, options = {}) {
  const response = await fetch(path, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.error || `${path} failed (${response.status})`)
  return data
}
```
- `options = {}` — default value: no options → GET request.
- `fetch(path, options)` — send the request. The path is **relative**
  (`/api/model`), so it goes to whatever server served the page (Vite in dev,
  Flask in production). That's why the same code works in both.
- `response.json()` — read the body as JSON. `.catch(() => ({}))` — if the
  body isn't JSON (e.g. an HTML error page), use `{}` instead of crashing.
- `response.ok` — true for status 200–299.
- If not ok, **throw** an error with the server's own message (`data.error`,
  e.g. "text is required"), or a generic one. The calling code catches it and
  shows it.
- Otherwise return the data.

```js
export const getModel = () => request('/api/model')
export const getAnalytics = () => request('/api/analytics')
export const refresh = () => request('/api/refresh', { method: 'POST' })

export const classify = (text) =>
  request('/api/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
```
One small function per API endpoint.
- `{ text }` is shorthand for `{ text: text }`.
- `JSON.stringify` → `'{"text":"hello"}'`.
- The `Content-Type: application/json` header tells Flask the body is JSON,
  so `request.get_json()` will read it.

### 21.3 `src/format.js`

```js
export const formatDay = (iso) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : ''
```
Turns `"2026-09-23"` into `"Sep 23"`. If `iso` is empty/null, returns `''`.
`undefined` for the locale = "use the browser's language".

```js
export const percentOf = (n, total) => (total ? `${(n / total) * 100}%` : '0%')
```
Turns `(5, 20)` into `"25%"` (a CSS width). If total is 0, `"0%"` (never
divide by zero).

### 21.4 `src/theme.js`

```js
import { ref, readonly } from 'vue'

const STORAGE_KEY = 'theme'
const systemPrefersDark = () =>
  typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches
```
- `systemPrefersDark()` — is the OS in dark mode?
- `typeof matchMedia === 'function'` — first check that `matchMedia` exists
  (safety for unusual environments). `&&` only evaluates the right side if
  the left is true.

```js
const theme = ref(
  localStorage.getItem(STORAGE_KEY) || (systemPrefersDark() ? 'dark' : 'light'),
)
```
The current theme, as a reactive ref. Start with what the user chose last
time (saved in `localStorage`), or, if nothing saved (`null`), follow the OS.

**Important:** this line is at the **top level of the module**, outside any
function. A module runs **once**, so there is exactly **one** `theme` shared
by every component that uses `useTheme()`. That's a simple way to share state.

```js
const listeners = new Set()
```
A set of functions to call when the theme changes. Charts register here
because Chart.js bakes colors in when it draws. It can't notice a CSS
variable changing, so it must be told to redraw.

```js
function apply() {
  document.documentElement.dataset.theme = theme.value
}
```
Write `data-theme="dark"` (or light) onto `<html>`. The CSS rules in
`style.css` react to that attribute, so all the `var(--...)` colors switch
instantly.

```js
export function useTheme() {
  return {
    theme: readonly(theme),
    apply,
    toggle() {
      theme.value = theme.value === 'dark' ? 'light' : 'dark'
      localStorage.setItem(STORAGE_KEY, theme.value)
      apply()
      listeners.forEach((fn) => fn())
    },
    onThemeChange(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },
  }
}
```
A **composable** (Vue's name for a `useXxx()` function that shares reactive
logic). It returns:
- `theme` — **read-only** view of the ref (components can read it but not
  change it directly).
- `apply` — the function above (`apply,` is shorthand for `apply: apply`).
- `toggle()` — flip dark↔light, **remember** it in localStorage, apply it, and
  call every registered listener (charts redraw).
- `onThemeChange(fn)` — register a listener. It **returns a function** that
  unregisters it (`listeners.delete(fn)`). Components call that when they're
  removed, so dead charts aren't redrawn.

```js
export const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim()
```
Read a CSS variable's current value from JS: `cssVar('--positive')` →
`"#2a78d6"`. This way the hex colors live **only** in `style.css`.

**When to use `cssVar()` and when to use `var()`**. This is the rule the code
follows (its comment says so too):

| You are coloring... | Use | Why |
|---|---|---|
| an HTML element (a `<span>`, `<div>`) | `'var(--positive)'` in `:style` | the **browser** looks it up live, so it follows a theme toggle by itself |
| a Chart.js chart (a `<canvas>`) | `cssVar('--positive')` | a canvas is just pixels; Chart.js needs a real color like `#2a78d6`, and must redraw on toggle |

### 21.5 `src/App.vue` — the whole page

#### Script part

```js
import { ref, reactive, computed, onMounted } from 'vue'
import * as api from './api'
import { useTheme } from './theme'

import KpiTiles from './components/KpiTiles.vue'
import SentimentDonut from './components/SentimentDonut.vue'
import TimelineChart from './components/TimelineChart.vue'
import TryModel from './components/TryModel.vue'
import NeedsAttention from './components/NeedsAttention.vue'
import PostBreakdown from './components/PostBreakdown.vue'
import ModelInsights from './components/ModelInsights.vue'
```
Import Vue tools, all API functions (as `api.xxx`), the theme, and the seven
child components. In `<script setup>`, an imported component can be used in
the template straight away as `<KpiTiles />`.

```js
const { theme, toggle } = useTheme()
```
Get the shared theme and the toggle function.

```js
const analytics = reactive({
  totals: {}, percentages: {}, timeline: [], per_post: [], needs_attention: [],
})
const model = reactive({})
const loading = ref(false)
const error = ref('')
const lastRun = ref('')
```
The page's state:
- `analytics` — starts **empty but with the right shape**, so the template
  doesn't crash before data arrives (e.g. `analytics.totals.total` is
  `undefined` rather than an error).
- `model` — the report card from `/api/model`.
- `loading` — true while the refresh button is working.
- `error` — error text to show in red ('' = none).
- `lastRun` — the summary line after a refresh.

```js
const accuracy = computed(() =>
  model.test_metrics ? (model.test_metrics.accuracy * 100).toFixed(1) : null,
)
```
`0.8021...` → `"80.2"`, or `null` if the model info hasn't loaded.

```js
async function loadAnalytics() {
  Object.assign(analytics, await api.getAnalytics())
}
```
Get the numbers and **copy them into** the existing reactive object.

Why `Object.assign` and not `analytics = ...`? Because `analytics` is a
`const` pointing at one specific reactive object that the template is
watching. Replacing it would disconnect it. `Object.assign(target, source)`
copies every field of `source` into `target`, so the **same** object
changes and Vue notices.

```js
async function refresh() {
  loading.value = true
  error.value = ''
  try {
    const data = await api.refresh()
    Object.assign(analytics, data.analytics)
    lastRun.value =
      `Fetched ${data.comments_seen} comments across ${data.posts_seen} posts · ` +
      `${data.new_comments} new · ${data.newly_classified} newly classified`
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
```
The big button:
1. Show "Fetching…", clear old errors.
2. `POST /api/refresh` and wait.
3. Copy in the fresh analytics → every chart updates.
4. Build the summary line.
5. If anything threw, show the message.
6. `finally` — no matter what, stop the loading state.

```js
onMounted(async () => {
  try {
    Object.assign(model, await api.getModel())
    await loadAnalytics()
  } catch (e) {
    error.value = e.message
  }
})
```
When the page first appears: load the model info, then the analytics. These
two requests happen **one after the other** (the second `await` waits for the
first to finish).

#### Template part

```vue
<div class="min-h-screen">
  <div class="max-w-6xl mx-auto px-4 py-6 space-y-4">
```
Full-height page; a centered column max ~1152px wide with padding and a gap
between sections.

```vue
<header class="flex flex-wrap items-center gap-3">
  <div class="mr-auto">
    <h1 class="text-xl font-semibold">Page Sentiment</h1>
    <p class="text-sm" :style="{ color: 'var(--ink-2)' }">
      Facebook comment analysis &middot; TF-IDF + Logistic Regression
    </p>
  </div>
```
The title block. `mr-auto` pushes everything after it to the right.
`&middot;` is an HTML **entity** for the "·" character.

```vue
  <span
    v-if="accuracy"
    class="text-xs px-2 py-1 rounded-full border num"
    :style="{ borderColor: 'var(--hairline)', color: 'var(--ink-2)' }"
  >model {{ accuracy }}% acc</span>
```
The little pill "model 80.2% acc", only shown once `accuracy` exists.

```vue
  <button
    class="text-sm px-3 py-2 rounded-lg border"
    :style="{ borderColor: 'var(--hairline)' }"
    @click="toggle"
  >
    {{ theme === 'dark' ? 'Light' : 'Dark' }}
  </button>
```
Theme button. It shows the theme you'd **switch to**. Click → `toggle()`.

```vue
  <button
    class="text-sm font-medium px-4 py-2 rounded-lg text-white disabled:opacity-50"
    :style="{ background: 'var(--positive)' }"
    :disabled="loading"
    @click="refresh"
  >
    {{ loading ? 'Fetching…' : 'View latest expression' }}
  </button>
</header>
```
The main button. Blue background. Disabled (and faded) while loading, so you
can't double-click it mid-request.

```vue
<p v-if="error" class="text-sm px-3 py-2 rounded-lg card"
   :style="{ color: 'var(--negative)' }">{{ error }}</p>

<p v-if="lastRun" class="text-sm" :style="{ color: 'var(--ink-2)' }">{{ lastRun }}</p>
```
Error message (red) and the summary line, each only when non-empty.

```vue
<section v-if="model.trained && model.configured === false" class="card p-4">
  <h2 ...>Facebook is not connected</h2>
  ...
  <pre ...>cd backend
uv run python get_token.py</pre>
</section>
```
If the model exists but Facebook isn't configured, show instructions.
`=== false` (not just `!model.configured`) so that it doesn't flash before
`model` has loaded (when `configured` is still `undefined`). `<pre>` keeps
line breaks and spacing exactly as written, which is why its text is not
indented.

```vue
<section v-else-if="model.configured && !analytics.totals.total" class="card p-4">
  <h2 ...>No comments yet</h2>
  ...
</section>
```
Connected, but the database has no classified comments yet → tell the user to
click the button.

```vue
<template v-if="analytics.totals.total">
  <KpiTiles :totals="analytics.totals" :percentages="analytics.percentages" />

  <section class="grid md:grid-cols-5 gap-3">
    <SentimentDonut class="md:col-span-2" :totals="analytics.totals"
                    :percentages="analytics.percentages" />
    <TimelineChart class="md:col-span-3" :timeline="analytics.timeline" />
  </section>

  <NeedsAttention :comments="analytics.needs_attention" />
  <PostBreakdown :posts="analytics.per_post" />
</template>
```
Only when there is at least one classified comment: show the tiles, a row
with the donut (2/5 width) and timeline (3/5 width) on medium+ screens
(stacked on phones), then the attention list and the post table. Each child
receives exactly the slice of `analytics` it needs.

A `class` put on a component (`class="md:col-span-2"`) is automatically added
to that component's outermost element.

```vue
<TryModel />
<ModelInsights :model="model" />
```
Always shown. They work without any Facebook data.

---

## Part 22 — The components line by line

### 22.1 `KpiTiles.vue` — the four number boxes

```js
import { computed } from 'vue'

const props = defineProps({
  totals: { type: Object, default: () => ({}) },
  percentages: { type: Object, default: () => ({}) },
})
```
Receives `totals` (`{positive: 35, negative: 18, neutral: 18, total: 71}`) and
`percentages`. `const props = ...` keeps a handle so the script can read
`props.totals`.

```js
const tiles = computed(() => [
  { label: 'Total',    value: props.totals.total    || 0, pct: null, color: null },
  { label: 'Positive', value: props.totals.positive || 0, pct: props.percentages.positive ?? 0, color: 'var(--positive)' },
  { label: 'Neutral',  value: props.totals.neutral  || 0, pct: props.percentages.neutral  ?? 0, color: 'var(--neutral)'  },
  { label: 'Negative', value: props.totals.negative || 0, pct: props.percentages.negative ?? 0, color: 'var(--negative)' },
])
```
Build a list of 4 tile descriptions. Using a list means the template can draw
all four with one loop. The "Total" tile has no percentage and no color dot
(`null`).

The colors are the **text** `'var(--positive)'`, not a looked-up hex value.
That text goes straight into the square's `style`, and the browser resolves it,
so the square changes color the instant you toggle the theme (Part 24.3).

```vue
<section class="grid grid-cols-2 md:grid-cols-4 gap-3">
  <div v-for="tile in tiles" :key="tile.label" class="card p-4">
```
2 columns on phones, 4 on wider screens. One card per tile. `:key` gives Vue
a stable id per card.

```vue
    <div class="flex items-center gap-2 mb-1">
      <span v-if="tile.color" class="w-2.5 h-2.5 rounded-sm" :style="{ background: tile.color }" />
      <span class="text-xs uppercase tracking-wide" :style="{ color: 'var(--muted)' }">
        {{ tile.label }}
      </span>
    </div>
```
A small colored square (only if the tile has a color) and the label.

```vue
    <div class="text-3xl font-semibold num">{{ tile.value.toLocaleString() }}</div>
    <div v-if="tile.pct !== null" class="text-sm num" :style="{ color: 'var(--ink-2)' }">
      {{ tile.pct }}% of all
    </div>
```
The big number (with thousands separators) and, except for Total, "49.3% of
all".

### 22.2 `SentimentDonut.vue` — the ring chart

```js
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import Chart from 'chart.js/auto'
import { cssVar, useTheme } from '../theme'
```
`chart.js/auto` = Chart.js with every chart type registered automatically.

```js
const props = defineProps({ totals: ..., percentages: ... })

const canvas = ref(null)
let chart = null
const { onThemeChange } = useTheme()
```
- `canvas` — will hold the real `<canvas>` HTML element (via `ref="canvas"`
  in the template). A `<canvas>` is a blank drawing surface; Chart.js paints
  on it.
- `chart` — the Chart.js object. A plain `let`, **not** reactive: Chart.js
  manages its own drawing and doesn't need Vue watching it.

```js
const legend = computed(() =>
  ['positive', 'neutral', 'negative'].map((key) => ({
    key,
    label: key[0].toUpperCase() + key.slice(1),
    value: props.totals[key] || 0,
    pct: props.percentages[key] ?? 0,
    color: `var(--${key})`,
  })),
)
```
The rows under the donut. For each sentiment name:
- `key[0].toUpperCase() + key.slice(1)` → first letter capital + the rest →
  `"Positive"`.
- `props.totals[key]` — square brackets with a variable = "the field whose
  name is in `key`".
- `` `var(--${key})` `` → the text `'var(--positive)'`, which the browser
  resolves live (same reason as in KpiTiles). The **canvas** part of the donut
  below still uses `cssVar()`, because Chart.js needs a real color.

```js
function draw() {
  chart?.destroy()
  if (!canvas.value) return
```
Throw away the old chart (if any), since Chart.js can't just change its
colors. If the canvas doesn't exist yet, stop.

```js
  const surface = cssVar('--surface')
  chart = new Chart(canvas.value, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Neutral', 'Negative'],
      datasets: [{
        data: [props.totals.positive || 0, props.totals.neutral || 0, props.totals.negative || 0],
        backgroundColor: [cssVar('--positive'), cssVar('--neutral'), cssVar('--negative')],
        borderColor: surface,
        borderWidth: 2,
      }],
    },
```
Create a doughnut chart on the canvas. `data` has the three numbers and their
colors in the same order. The border in the **card's background color** makes
a thin 2px gap between slices.

```js
    options: {
      cutout: '62%',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: surface, titleColor: cssVar('--ink'), ... },
      },
    },
  })
}
```
- `cutout: '62%'` — the hole in the middle is 62% of the radius (that's what
  makes it a donut, not a pie).
- `responsive` + `maintainAspectRatio: false` — fill the parent box
  (`height: 220px`) and resize with the window.
- Built-in legend off (we draw our own rows below, with exact numbers).
- Tooltip (the hover box) colored with our theme.

```js
onMounted(draw)
watch(() => props.totals, draw, { deep: true })
const stop = onThemeChange(draw)
onBeforeUnmount(() => { stop(); chart?.destroy() })
```
The chart's **life**:
- Draw when the component first appears (the canvas now exists).
- Redraw whenever the totals change (after a refresh).
- Redraw when the theme flips. Keep the "unsubscribe" function as `stop`.
- When the component is removed: unsubscribe and destroy the chart (free
  memory).

```vue
<div class="card p-4">
  <h2 class="text-sm font-medium mb-1">Overall split</h2>
  <p class="text-xs mb-3" ...>Share of all classified comments</p>

  <div class="relative" style="height: 220px">
    <canvas ref="canvas" />
  </div>

  <ul class="mt-3 space-y-1">
    <li v-for="row in legend" :key="row.key" class="flex items-center gap-2 text-sm">
      <span class="w-2.5 h-2.5 rounded-sm shrink-0" :style="{ background: row.color }" />
      <span :style="{ color: 'var(--ink-2)' }">{{ row.label }}</span>
      <span class="ml-auto num">{{ row.value.toLocaleString() }}</span>
      <span class="num w-12 text-right" :style="{ color: 'var(--muted)' }">{{ row.pct }}%</span>
    </li>
  </ul>
</div>
```
Title, the canvas in a fixed-height box, then one row per sentiment: color
square, name, count (pushed right with `ml-auto`), percent. Note the plain
`style="height: 220px"` (no colon): a fixed CSS string, not JavaScript.

### 22.3 `TimelineChart.vue` — the bars by day

Same life pattern as the donut (`draw`, `onMounted`, `watch`,
`onThemeChange`, `onBeforeUnmount`). What's new is the data reshaping.

```js
const days = [...new Set(props.timeline.map((r) => r.day))].sort()
```
The API sends flat rows:
```
[{day: "2026-09-22", sentiment: "positive", n: 4},
 {day: "2026-09-22", sentiment: "negative", n: 1},
 {day: "2026-09-23", sentiment: "positive", n: 7}, ...]
```
- `.map((r) => r.day)` → `["2026-09-22", "2026-09-22", "2026-09-23", ...]`
- `new Set(...)` → unique days only.
- `[...set]` → back to an array; `.sort()` → in date order (dates in
  `YYYY-MM-DD` form sort correctly as plain text).

```js
const series = (name) =>
  days.map((d) => props.timeline.find((r) => r.day === d && r.sentiment === name)?.n || 0)
```
For one sentiment, a number for **every** day: find the row for that day +
sentiment and take its `n`. If there is no such row, `find` gives `undefined`,
`?.n` gives `undefined` instead of crashing, and `|| 0` turns it into 0.
Chart.js needs every series to have one value per label. This turning of
long rows into columns is called a **pivot**.

```js
  chart = new Chart(canvas.value, {
    type: 'bar',
    data: {
      labels: days.map(formatDay),
```
Bar chart. X-axis labels: `"Sep 22"`, `"Sep 23"`. `days.map(formatDay)` passes
each day to `formatDay`.

```js
      datasets: [
        { label: 'Negative', data: series('negative'), backgroundColor: cssVar('--negative') },
        { label: 'Neutral',  data: series('neutral'),  backgroundColor: cssVar('--neutral')  },
        {
          label: 'Positive',
          data: series('positive'),
          backgroundColor: cssVar('--positive'),
          borderRadius: { topLeft: 4, topRight: 4 },
        },
      ].map((d) => ({
        ...d,
        borderColor: surface,
        borderWidth: { top: 2 },
        borderSkipped: false,
        maxBarThickness: 48,
      })),
    },
```
Three datasets stacked bottom to top: negative, neutral, positive. The top one
(positive) gets rounded top corners.

Then `.map((d) => ({ ...d, ... }))` adds the **same** extra settings to all
three: copy each dataset (`...d`) and add a 2px top border in the card
color (a gap between stacked pieces), and a max bar width of 48px so a
single day doesn't become one giant block.

```js
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
```
Hovering anywhere over a day's column shows all three values for that day
(`mode: 'index'`); you don't have to hit the bar exactly
(`intersect: false`).

```js
      plugins: {
        tooltip: { ...theme colors... },
        legend: {
          position: 'bottom',
          align: 'start',
          labels: { color: cssVar('--ink-2'), boxWidth: 8, boxHeight: 8,
                    usePointStyle: true, pointStyle: 'rectRounded', padding: 14 },
        },
      },
```
Legend at the bottom-left with small rounded squares.

```js
      scales: {
        x: {
          stacked: true,
          grid: { display: false },
          border: { color: cssVar('--axis') },
          ticks: { color: muted, maxRotation: 0, autoSkipPadding: 12 },
        },
        y: {
          stacked: true,
          beginAtZero: true,
          grid: { color: cssVar('--grid'), drawTicks: false },
          border: { display: false },
          ticks: { color: muted, precision: 0, padding: 8 },
        },
      },
```
- `stacked: true` on both axes → bars sit on top of each other instead of side
  by side.
- x: no vertical grid lines; labels never tilted; skip labels if crowded.
- y: start at 0; faint horizontal grid lines; `precision: 0` → only whole
  numbers (you can't have 2.5 comments).

The template is just a card with a title and a 300px-high canvas.

(Before the `DATE()` bug was fixed (Part 24.1), every row's `day` was `null`,
so `days` was `[null]` and the chart showed one bar with an empty label. Now
all 71 of your comments are from one day, so you see one bar labeled
"Sep 23". More bars appear as comments arrive on other days.)

### 22.4 `NeedsAttention.vue` — the most negative comments, with pages

```js
import { ref, computed, watch } from 'vue'
import { formatDay } from '../format'

const props = defineProps({ comments: { type: Array, default: () => [] } })

const PER_PAGE = 7
const page = ref(1)
const pageCount = computed(() => Math.max(1, Math.ceil(props.comments.length / PER_PAGE)))
```
- Up to 50 comments arrive (the server's limit). Show 7 per page.
- `page` — current page number, starting at 1.
- `pageCount` — e.g. 18 comments / 7 = 2.57 → `Math.ceil` rounds **up** → 3
  pages. `Math.max(1, ...)` → at least 1 page even with 0 comments.

```js
const start = computed(() => (page.value - 1) * PER_PAGE)
const visible = computed(() => props.comments.slice(start.value, start.value + PER_PAGE))
```
Page 1 → start 0 → items 0–6. Page 2 → start 7 → items 7–13. `.slice` cuts
out that piece. All of this happens **in the browser**, so paging is instant
(no server request).

```js
watch(() => props.comments, () => { page.value = 1 })
```
When a refresh brings a new list, jump back to page 1. Otherwise you might be
on page 4 of a list that now only has 2 pages, staring at nothing.

```js
const go = (delta) => {
  page.value = Math.min(pageCount.value, Math.max(1, page.value + delta))
}
```
Move by `delta` (−1 = previous, +1 = next), but **clamp** between 1 and
`pageCount`, so you can never go to page 0 or past the end.

Template highlights:

```vue
<section v-if="comments.length" class="card p-4">
```
Hide the whole card when there are no negative comments. (In the template,
props can be used directly without `props.`.)

```vue
<span ...>
  {{ start + 1 }}–{{ Math.min(start + visible.length, comments.length) }}
  of {{ comments.length }}
</span>
```
Shows "1–7 of 18". `start + 1` because humans count from 1.

```vue
<ul class="divide-y divide-[color:var(--hairline)]">
  <li v-for="comment in visible" :key="comment.id" class="py-2 flex gap-3 items-start text-sm">
    <span class="w-1 self-stretch rounded-full shrink-0" :style="{ background: 'var(--negative)' }" />
    <div class="min-w-0 flex-1">
      <p class="truncate">{{ comment.message }}</p>
      <p class="text-xs" ...>
        {{ comment.author || 'Unknown' }} &middot; {{ formatDay(comment.created_time) }}
      </p>
    </div>
    <span class="num text-xs shrink-0" ...>{{ comment.score.toFixed(2) }}</span>
  </li>
</ul>
```
Each comment: a thin red bar on the left, the text on one line (cut with …),
the author (or "Unknown", since Facebook usually hides it) and the date, then
the score on the right (e.g. `0.03`; lower = more negative). `min-w-0` is
needed so `truncate` works inside a flex row.

**`{{ comment.message }}` is safe:** Vue's `{{ }}` always shows text **as
text**. If someone commented `<script>...</script>`, it's displayed, not run.

```vue
<nav v-if="pageCount > 1" ...>
  <button ... :disabled="page === 1" @click="go(-1)">Previous</button>

  <div class="flex gap-1 mx-auto">
    <button
      v-for="n in pageCount"
      :key="n"
      class="w-8 h-8 text-sm rounded-lg num"
      :style="n === page
        ? { background: 'var(--negative)', color: '#fff' }
        : { color: 'var(--ink-2)' }"
      @click="page = n"
    >{{ n }}</button>
  </div>

  <button ... :disabled="page === pageCount" @click="go(1)">Next</button>
</nav>
```
Only shown when there's more than one page. Previous (disabled on page 1),
numbered buttons (the current one is red with white text; the ternary picks
between two style objects), Next (disabled on the last page).
`@click="page = n"` — in the template you can assign directly (Vue adds
`.value` for you).

### 22.5 `PostBreakdown.vue` — the table

```js
import { percentOf } from '../format'
defineProps({ posts: { type: Array, default: () => [] } })
```
No `const props =` here, because the script never reads the props. Only the
template does.

```vue
<section v-if="posts.length" class="card p-4">
  <h2 ...>By post</h2>
  <div class="overflow-x-auto">
    <table class="w-full text-sm">
      <thead>
        <tr ...>
          <th class="pb-2 font-medium">Post</th>
          <th class="pb-2 font-medium text-right">Total</th>
          <th ...>Pos</th> <th ...>Neu</th> <th ...>Neg</th>
          <th class="pb-2 pl-3 font-medium w-32">Mix</th>
        </tr>
      </thead>
```
HTML table: `<table>` holds `<thead>` (header rows) and `<tbody>` (body rows).
`<tr>` = table row, `<th>` = header cell, `<td>` = data cell. On a narrow
phone, `overflow-x-auto` lets the table scroll sideways instead of breaking
the page.

```vue
      <tbody class="divide-y divide-[color:var(--hairline)]">
        <tr v-for="post in posts" :key="post.id">
          <td class="py-2 pr-3 max-w-xs truncate">{{ post.message || post.id }}</td>
          <td class="py-2 text-right num">{{ post.total }}</td>
          <td ...>{{ post.positive }}</td>
          <td ...>{{ post.neutral }}</td>
          <td ...>{{ post.negative }}</td>
```
One row per post. If the post has no text (photo-only), show its id instead.

```vue
          <td class="py-2 pl-3">
            <div class="flex h-2 rounded-full overflow-hidden" :style="{ background: 'var(--grid)' }">
              <div :style="{ width: percentOf(post.negative, post.total), background: 'var(--negative)', borderRight: '2px solid var(--surface)' }" />
              <div :style="{ width: percentOf(post.neutral, post.total), background: 'var(--neutral)', borderRight: '2px solid var(--surface)' }" />
              <div :style="{ width: percentOf(post.positive, post.total), background: 'var(--positive)' }" />
            </div>
          </td>
```
The "Mix" bar: a thin rounded track with three colored pieces side by side.
Each piece's **width** is its share, e.g. `percentOf(3, 12)` → `"25%"`. Same
order as the timeline (negative, neutral, positive), with small gaps between.
It's a tiny stacked bar chart made from plain `<div>`s, no Chart.js needed.

### 22.6 `TryModel.vue` — type a sentence

```js
import { ref } from 'vue'
import { classify } from '../api'

const text = ref('')
const result = ref(null)
const busy = ref(false)
const error = ref('')
```
What's typed, the model's answer, "working?" flag, and an error message.

```js
async function run() {
  if (!text.value.trim()) return
  busy.value = true
  error.value = ''
  try {
    result.value = await classify(text.value)
  } catch (e) {
    error.value = e.message
    result.value = null
  } finally {
    busy.value = false
  }
}
```
Ignore empty input. Otherwise call `POST /api/classify` and store the answer
(or the error).

```vue
<input
  v-model="text"
  placeholder="e.g. the delivery was really fast, thanks"
  class="flex-1 min-w-64 px-3 py-2 rounded-lg border text-sm"
  :style="{ background: 'var(--plane)', borderColor: 'var(--hairline)', color: 'var(--ink)' }"
  @keyup.enter="run"
>
```
- `v-model="text"` — whatever you type is always in `text.value`.
- `placeholder` — the gray hint shown when empty.
- `@keyup.enter="run"` — pressing Enter runs it too.
- `<input>` has no closing tag in HTML; the `>` on its own line just ends the
  opening tag.

```vue
<button ... :disabled="busy" @click="run">
  {{ busy ? 'Working…' : 'Classify' }}
</button>
```

```vue
<p v-if="error" class="mt-3 text-sm" :style="{ color: 'var(--negative)' }">{{ error }}</p>

<div v-if="result" class="mt-3 flex flex-wrap items-center gap-3 text-sm">
  <span class="w-2.5 h-2.5 rounded-sm" :style="{ background: `var(--${result.sentiment})` }" />
  <span class="font-medium capitalize">{{ result.sentiment }}</span>
  <span class="num" ...>P(positive) = {{ result.score.toFixed(3) }}</span>
  <span class="num" ...>confidence {{ (result.confidence * 100).toFixed(0) }}%</span>
  <span v-if="result.reason" class="text-xs" ...>({{ result.reason }})</span>
</div>
```
The square's color: `` `var(--${result.sentiment})` `` builds `'var(--positive)'`,
`'var(--negative)'` or `'var(--neutral)'`, since the sentiment names match the CSS
variable names exactly. (Yes, a template literal with backticks works inside a
Vue attribute.)

Shows e.g. "■ Positive   P(positive) = 0.944   confidence 89%". The `reason`
appears only for texts that cleaned to nothing (e.g. pure Bangla).
`capitalize` makes "positive" display as "Positive".

### 22.7 `ModelInsights.vue` — what the model learned

```js
defineProps({ model: { type: Object, default: () => ({}) } })
```

```vue
<section v-if="model.top_features" class="card p-4">
```
Only shown once the report card has loaded.

```vue
<div class="grid md:grid-cols-2 gap-4 mb-4">
  <div v-for="side in [
    { key: 'positive', label: 'Pushes positive', color: 'var(--positive)' },
    { key: 'negative', label: 'Pushes negative', color: 'var(--negative)' },
  ]" :key="side.key">
```
A `v-for` over a small array written **right inside** the template: two
columns, one per side.

```vue
    <div class="flex flex-wrap gap-1.5">
      <span v-for="feature in model.top_features[side.key]" :key="feature"
            class="text-xs px-2 py-1 rounded border" ...>{{ feature }}</span>
    </div>
```
A loop **inside** a loop: for this side, show each top word as a little chip.
`model.top_features[side.key]` = `model.top_features["positive"]` = the list
`["thanks", "cant wait", ...]` from `model_meta.json`.

```vue
<dl class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm pt-3 border-t" ...>
  <div>
    <dt ...>Config</dt>
    <dd>{{ model.config_name }}</dd>
  </div>
  <div>
    <dt ...>Trained on</dt>
    <dd class="num">{{ model.rows_used?.toLocaleString() }} rows</dd>
  </div>
  <div>
    <dt ...>Test accuracy</dt>
    <dd class="num">{{ (model.test_metrics?.accuracy * 100).toFixed(2) }}%</dd>
  </div>
  <div>
    <dt ...>Neutral band</dt>
    <dd class="num">{{ model.neutral_band?.join(' – ') }}</dd>
  </div>
</dl>
```
`<dl>` = "description list": `<dt>` is a term, `<dd>` its value. Shows
"bigram-50k", "199,508 rows", "80.21%", "0.4 – 0.6". The `?.` protects
against missing fields.

---

## Part 23 — Full journeys: follow one action through every file

These traces tie everything together. Each step names the file and function.

### Journey A — `uv run python train.py`

```
1. Terminal: uv finds .venv, runs Python 3.12 on train.py
2. train.py top-level: imports run (sklearn, joblib, dataset.py ...)
   dataset.py top-level runs: sets DATA_DIR, COLUMNS, LABEL_MAP, SEED
3. train.py bottom: __name__ == "__main__" → main()
4. main(): argparse → args.sample = 200000, args.full = False, args.cv = 0
5. dataset.load_splits(sample=200000, seed=42)
   5a. load_raw():  pd.read_csv (1,600,000 rows)
                    df.sample(200000)            ← random, NOT head()
                    keep sentiment + text
                    map 4 → 1
                    apply(clean_text) on 200,000 tweets
                    drop 492 empty rows → 199,508
   5b. make_splits(): 70/15/15, stratified → 139,655 / 29,926 / 29,927
   5c. print sizes and balance
6. PHASE 1, for each of 3 configs:
   build_pipeline(**config)  → an empty Pipeline(tfidf, clf)
   pipeline.fit(X_train, y_train)
       tfidf: learn vocabulary + IDF from 139,655 texts, turn them into a sparse matrix
       clf:   learn 1 weight per feature + intercept (up to 1000 rounds)
   evaluate(on val) → accuracy, roc-auc
7. max(...) → winner "bigram-50k"
8. PHASE 2 skipped (cv = 0)
9. PHASE 3: evaluate(winner, test) → 0.8021 / 0.8832
            classification_report, confusion_matrix
            show_top_features → the two word lists
10. PHASE 4: joblib.dump → models/sentiment_pipeline.joblib (~0.95 MB)
             json.dumps  → models/model_meta.json
11. Program ends. The trained model now exists only as those two files.
```

### Journey B — opening `http://127.0.0.1:5000` in the browser

```
BACKEND START (earlier): uv run python app.py
   load_dotenv(.env)  → FB_PAGE_ID / FB_PAGE_ACCESS_TOKEN in environment
   import facebook     → PAGE_ID, PAGE_TOKEN read now
   import predict      → constants only; model NOT loaded yet
   Flask(...)          → app created; db.init_db() creates tables if needed
   app.run()           → waiting on port 5000

1. Browser → GET /
   app.py index() → send_static_file("index.html")   (backend/static/index.html)
2. Browser reads the HTML → GET /assets/index-DOx-dWUt.js and .css
   Flask serves them from backend/static/assets/ (thanks to static_url_path="")
3. The JS bundle runs = main.js:
   theme.js top-level: theme = localStorage value or OS preference
   useTheme().apply() → <html data-theme="dark|light">
   createApp(App).mount('#app') → App.vue renders with EMPTY data:
       header, TryModel, (ModelInsights hidden: no top_features yet)
4. App.vue onMounted:
   api.getModel() → fetch GET /api/model
       app.py model_info() → predict.load_meta() → reads model_meta.json (cached)
                           → facebook.is_configured()
       → JSON {trained, configured, neutral_band, config_name, test_metrics, top_features, ...}
   Object.assign(model, ...) → Vue re-renders:
       accuracy pill appears ("model 80.2% acc")
       ModelInsights appears
       "Facebook is not connected" box appears if configured === false
5. loadAnalytics() → fetch GET /api/analytics
       app.py get_analytics() → db.analytics() → 4 SQL queries on app.db
       → JSON {totals, percentages, timeline, per_post, needs_attention}
   Object.assign(analytics, ...) → analytics.totals.total = 71 → the <template v-if> turns on:
       KpiTiles, SentimentDonut, TimelineChart, NeedsAttention, PostBreakdown are created
       each onMounted → Chart.js draws on its <canvas>
```

### Journey C — clicking **View latest expression**

```
1. App.vue <button @click="refresh"> → refresh()
   loading = true → button shows "Fetching…" and is disabled
2. api.refresh() → fetch POST /api/refresh
   (dev mode: Vite on :5173 proxies it to Flask :5000)
3. app.py refresh():
   3a. facebook.fetch_latest()
         is_configured()? no → GraphError → JSON {"error": ...}, status 502 → skip to step 6
         fetch_posts(10)   → _get("PAGE_ID/posts", ...) with Authorization: Bearer header
         for each post: fetch_comments(post_id)
             loop: _get("POST_ID/comments", ...) → keep text comments
                   follow paging.cursors.after until none left
         return (posts, comments)
   3b. db.save_posts(posts)          → INSERT ... ON CONFLICT DO UPDATE (upsert)
   3c. db.save_comments(comments)    → INSERT ... ON CONFLICT DO NOTHING
                                        returns new_count = after - before
   3d. db.unclassified()             → SELECT ... WHERE sentiment IS NULL LIMIT 1000
   3e. predict.classify([messages])
         load_pipeline()  → first time: joblib.load (≈100 ms), then cached forever
         clean_text each message (same function as training)
         pipeline.predict_proba(cleaned)[:, 1]
             tfidf.transform (saved vocabulary, NO re-learning)
             clf: z = Σ weight × value + intercept → sigmoid → P(positive)
         for each: empty? → neutral/0.5/0.0
                   ≥ 0.60 positive, ≤ 0.40 negative, else neutral
                   confidence = |score − 0.5| × 2
   3f. db.save_classifications(...)  → UPDATE comments SET sentiment=..., score=... WHERE id=...
   3g. db.analytics()                → fresh numbers
   3h. jsonify({posts_seen, comments_seen, new_comments, newly_classified, analytics})
4. Back in the browser, api.js request(): response.ok → return data
5. App.vue: Object.assign(analytics, data.analytics)
            lastRun = "Fetched 71 comments across 4 posts · 0 new · 0 newly classified"
   Vue reactivity:
       KpiTiles' computed tiles recalculates → numbers change
       SentimentDonut's watch(totals) → draw() → old chart destroyed, new one drawn
       TimelineChart's watch(timeline) → draw()
       NeedsAttention's watch(comments) → page = 1, list re-rendered
       PostBreakdown re-renders the table
6. finally: loading = false → button re-enabled
   (on error: error = message → red box shown)
```

Clicking it a second time right away: Facebook returns the same comments →
`ON CONFLICT DO NOTHING` skips them all → `new_comments = 0` →
`unclassified()` finds nothing → the model isn't even called. Cheap and safe.

### Journey D — typing "not bad at all" in **Try the model** and pressing Enter

```
1. <input v-model="text">: every keystroke updates text.value
2. @keyup.enter → run() → busy = true
3. api.classify("not bad at all")
   fetch POST /api/classify, header Content-Type: application/json,
   body '{"text":"not bad at all"}'
4. app.py classify_text():
   request.get_json() → {"text": "not bad at all"} → text not empty
   predict.classify_one(text) → classify(["not bad at all"])
       clean_text → "not bad at all" (nothing to remove)
       tfidf → 7 non-zero features (see table below)
       logistic regression → z = −0.24 → sigmoid → P(positive) = 0.440
       0.40 < 0.440 < 0.60 → "neutral"
   jsonify({"sentiment": "neutral", "score": 0.4397, "confidence": 0.1207})
5. TryModel.vue: result.value = {...} → the result row appears:
   gray square, "Neutral", P(positive) = 0.440, confidence 12%
6. finally: busy = false
```

Surprised? I was too. I expected positive. Here is the real breakdown:

| feature | TF-IDF | weight | push |
|---|---|---|---|
| `not bad` | 0.509 | +5.183 | **+2.640** |
| `bad` | 0.297 | −5.866 | **−1.745** |
| `not` | 0.219 | −5.787 | **−1.267** |
| `at all` | 0.404 | −0.974 | −0.394 |
| `bad at` | 0.582 | +0.252 | +0.146 |
| `at` | 0.218 | −0.460 | −0.100 |
| `all` | 0.235 | −0.157 | −0.037 |
| sum + intercept (0.514) | | | **z = −0.24** |

`not bad` fights hard, but `bad` + `not` together pull even harder, and
`at all` (think "not happy at all", "no sleep at all") tips it slightly
negative. The earlier sentence "i love this not bad" came out positive only
because `love` helped. And the smoke-test sentence "not bad at all, pretty
happy with it" is positive thanks to `happy`. **This is what "a word-weight
counting machine" means in practice**, and it's why you should always test
with real sentences.

### Journey E — clicking the **Dark** button

```
1. App.vue <button @click="toggle"> → theme.js toggle()
2. theme.value = 'dark'
   localStorage.setItem('theme', 'dark')   → remembered on next visit
   apply() → <html data-theme="dark">
       style.css :root[data-theme='dark'] now matches → every var(--...) changes
       → all CSS-colored parts (cards, text, borders, list bars) switch instantly
3. listeners.forEach(fn => fn())
       SentimentDonut draw() and TimelineChart draw() → charts re-created with dark colors
4. App.vue's template read `theme` → button text re-renders to "Light"
```

---

## Part 24 — Things I noticed while writing this guide

While tracing every line, I found three real issues. **24.1 and 24.3 are now
fixed** in the code. They're kept here because *how a bug was found and fixed*
teaches more than the fix itself. **24.2 is not fixed yet**, because it needs a
retrain.

### 24.1 The timeline had no dates (bug — ✅ fixed)

**Where:** `db.py`, `analytics()`, the timeline query.

**What happened:** Facebook sends times like `2026-09-23T05:58:14+0000`.
SQLite's `DATE()` only accepts a timezone written as `+00:00` or `Z`, so it
returns `NULL` for every comment. I ran this against your `app.db`:

```
('2026-09-23T05:58:14+0000', None)
('2026-09-23T06:06:48+0000', None)
```

So the timeline groups **all** comments under one `null` day, and the "Over
time" chart shows one bar with no label.

**The fix (now in the code):** take the first 10 characters (`YYYY-MM-DD`)
instead of asking SQLite to parse the whole thing:

```sql
SELECT SUBSTR(created_time, 1, 10) AS day, sentiment, COUNT(*) AS n
```

(That gives the UTC date. A comment written at 1 a.m. in Bangladesh, which is
UTC+6, counts toward the previous day. That's fine for learning.) No
retraining needed. After the fix, the same database gives:

```
[{'day': '2026-09-23', 'sentiment': 'negative', 'n': 18},
 {'day': '2026-09-23', 'sentiment': 'neutral',  'n': 18},
 {'day': '2026-09-23', 'sentiment': 'positive', 'n': 35}]
```

**Lesson:** a function that fails by returning `NULL` instead of raising an
error is dangerous. Nothing crashed; the chart just looked a bit odd. Always
check the output with real data.

### 24.2 `clean_text` eats words containing "www" (quirk — not fixed yet)

**Where:** `dataset.py`, line 41: `re.sub(r"http\S+|www\S+", " ", text)`.

**What happens:** `www\S+` matches "www" **anywhere**, not just at the start
of a web address. So "awww," → "a", and "awwwww so cute" → "a so cute". It's
a small effect on 2009 tweets full of "awww", but a real one.

**A tighter pattern:** `\bwww\.\S+` means "`www.` at the **start of a word**
followed by non-spaces". (`\b` = word boundary, `\.` = a literal dot.)

**Careful:** `clean_text` is shared by training and prediction. If you change
it, you **must retrain** (`uv run python train.py`) and restart `app.py`, so
the model learns from text cleaned the same new way.

### 24.3 Some color dots didn't switch with the theme (UI bug — ✅ fixed)

**Where:** `KpiTiles.vue` (`tiles` computed), `SentimentDonut.vue` (`legend`
computed), `TryModel.vue` (a helper called `colorForSentiment` in the
template).

**What happened:** these read colors with `cssVar(...)`, which asks the browser
directly and gets back a fixed hex value like `#2a78d6`. That is **not** reactive data, so Vue doesn't know those values
depend on the theme. After toggling, the little squares keep the old theme's
blue/gray/red until the numbers change (e.g. after a refresh). The charts
themselves are fine because they subscribe to `onThemeChange`. The difference
is subtle (light `#2a78d6` vs dark `#3987e5`), so it's easy to miss.

**Two possible fixes:**

1. Make each computed also read the reactive `theme`, so Vue recomputes it on a
   toggle. That works, but it's extra code whose only job is to trick Vue.
2. **Don't resolve the color in JavaScript at all.** Put the text
   `'var(--positive)'` in the style and let the **browser** resolve it. CSS
   variables update live on their own, so nothing needs to recompute.

The code now uses fix 2 (simpler: less code, not more). `cssVar()` is kept
only for Chart.js, which draws on a canvas and really needs a hex value, and
which already redraws on toggle through `onThemeChange`. The unused
`colorForSentiment` helper was deleted from `theme.js`.

Checked in a real (headless) browser after the fix: in light mode the squares
are `rgb(42, 120, 214)` = `#2a78d6`; after clicking **Dark** they are
`rgb(57, 135, 229)` = `#3987e5`, with neutral and negative switching the same
way.

**Lesson:** when two systems both try to own the same thing (Vue's reactivity
and CSS's live variables), pick one. Here CSS already knew how to do it.

### 24.4 Behaviors worth knowing (not bugs)

- **Retrain → restart.** `load_pipeline()` is cached with `lru_cache`, so a
  running `app.py` keeps the old model in memory.
- **Edited comments keep their old text.** `save_comments` uses
  `ON CONFLICT DO NOTHING`, so if someone edits a comment on Facebook, the
  stored text (and its sentiment) doesn't change.
- **1000 per refresh.** `unclassified()` has `LIMIT 1000`. With more pending,
  the rest get classified on the next refresh.
- **Only the 10 newest posts** are fetched (`fetch_latest(post_limit=10)`),
  but **all** comments on each of them, every time.
- **`train.py` always overwrites** the saved model, even with a worse one.

---

## Part 25 — Experiments to try (the best way to learn)

Change one thing, run it, look at what changed. Keep a notebook of results.

### 25.1 Play with the neutral band (easy, no retraining)

In `predict.py`, change `NEUTRAL_LOW`/`NEUTRAL_HIGH` to `0.30`/`0.70`. Restart
`app.py`. Try the same sentences in **Try the model**. More of them become
neutral. Note: comments that were **already** classified in the database
keep their old label. Only new ones use the new band. (To reclassify
everything you'd set `sentiment = NULL` for all rows, e.g. with the
`sqlite3` tool, then click refresh.)

### 25.2 Run `predict.py` on your own sentences

Edit the `samples` list at the bottom of `predict.py` and run
`uv run python predict.py`. Try: sarcasm, Banglish, emoji-only, very long
text, a single word. Where is it wrong? Can you explain **why** with Part 10?

### 25.3 Look inside the model

Save this as `backend/explore.py` and run `uv run python explore.py`:

```python
import joblib
from dataset import clean_text

pipeline = joblib.load("models/sentiment_pipeline.joblib")
vectorizer = pipeline.named_steps["tfidf"]
classifier = pipeline.named_steps["clf"]

# The weight of any word or word pair:
for word in ["love", "hate", "not bad", "price", "delivery", "thanks"]:
    column = vectorizer.vocabulary_.get(word)
    if column is None:
        print(f"{word!r}: not in the vocabulary")
    else:
        print(f"{word!r}: column {column}, weight {classifier.coef_[0][column]:+.3f}")

# Explain one sentence, feature by feature:
text = clean_text("I love this but the delivery was slow")
row = vectorizer.transform([text])
names = vectorizer.get_feature_names_out()
for column, value in zip(row.indices, row.data):
    weight = classifier.coef_[0][column]
    print(f"{names[column]:>20}  tfidf={value:.3f}  weight={weight:+.3f}  push={value * weight:+.3f}")
print("P(positive) =", pipeline.predict_proba([text])[0][1])
```

This is the exact calculation from Part 10.8, for any sentence you like.

### 25.4 Cross-validate

```bash
uv run python train.py --cv 5
```
Look at the five fold scores and the std. Is the gap between bigram-10k and
bigram-50k (≈0.005) bigger than the std?

### 25.5 Add a new configuration

In `train.py`, add to `CONFIGS`:

```python
"trigram-100k": dict(ngram_range=(1, 3), max_features=100_000, stop_words=None),
```
Retrain. Does it beat bigram-50k on **val**? Is the model file bigger? Is it
slower? (Remember: it will overwrite the saved model if it wins.)

### 25.6 See what happens with bad data (to understand the traps)

Temporarily change `df.sample(n=sample, ...)` in `dataset.py` to
`df.head(sample)` and run `uv run python train.py --sample 20000`. Look at
the printed balance ("neg 100.0% / pos 0.0%"). It then crashes,
because logistic regression refuses to train on one class:
`ValueError: This solver needs samples of at least 2 classes in the data`. **Then change it
back and retrain at the normal size**, because the experiment overwrote your
model.

### 25.7 Try the stop-word trap yourself

Add `"bigram-10k-stop": dict(ngram_range=(1, 2), max_features=10_000,
stop_words="english")` and compare it with `bigram-10k` on val. Then check
whether `not bad` is still in `top_features`.

### 25.8 Frontend: change something small

- `NeedsAttention.vue`: change `PER_PAGE = 7` to `5`.
- `App.vue`: change the header text.
- Run `npm run dev` and open `http://127.0.0.1:5173` (with `app.py` also
  running). Watch the page update the moment you save.

### 25.9 Full-stack exercise: add a "neutral examples" list

A guided exercise that touches every layer:
1. `db.py`: in `analytics()`, add a query for neutral comments closest to 0.5
   (`WHERE sentiment = 'neutral' ORDER BY ABS(score - 0.5) LIMIT 20`) and put
   it in the returned dict as `"neutral_examples"`.
2. `App.vue`: add `neutral_examples: []` to the initial `analytics`.
3. Create `components/NeutralExamples.vue` modeled on `NeedsAttention.vue`
   (gray bar instead of red).
4. Use it in `App.vue`'s template: `<NeutralExamples :comments="analytics.neutral_examples" />`.
5. `npm run build`, restart `app.py`, look.

If you can do this, you understand how the whole app is connected.

### 25.10 Fix the remaining bug in Part 24

24.1 and 24.3 are done. Read their diffs with `git diff` to see exactly
what changed. 24.2 (the `www` pattern) is yours: change the regex in
`dataset.py`, test it on "awww, so cute www.example.com" in a Python shell,
then retrain and restart `app.py`. Compare the new test accuracy with 0.8021.

---

## Part 26 — FAQ

**Q: Does the model get better the more I use the dashboard?**
No. See Part 1. It only changes when you run `train.py`, and even then it only
learns from Sentiment140, not from your comments.

**Q: Where is "the AI" physically?**
In `backend/models/sentiment_pipeline.joblib`: a ~1 MB file containing a
vocabulary of 50,000 words/pairs, their IDF values, 50,000 weights and one
intercept. That's it. There's no connection to any online AI service.

**Q: Why not use a big modern model (like BERT or an LLM)?**
They'd be more accurate (sarcasm, Bangla with the right model, context), but
they're much bigger, slower, and you can't read them like a list of word
weights. This project's goal is **learning the classical ML workflow**, where
every number can be inspected.

**Q: Why does "what is the price?" come out neutral but "when will you restock
this item" negative?**
The first scores 0.453 (inside the 0.40–0.60 band). The second scores 0.385,
just below 0.40. The model never saw shop questions; it's pattern-matching
on 2009 tweets. Widening the band or training on labeled shop comments would
help.

**Q: Why is Bangla always neutral?**
`clean_text` keeps only `a–z`, so Bangla script becomes an empty string, and
`predict.py` refuses to guess on empty text. Supporting Bangla would need a
Bangla-labeled dataset and a different cleaning step. Part 27 is the full plan.

**Q: What does `score` mean again?**
The model's probability that the comment is positive (0 to 1).
`confidence` is how far that is from 0.5, scaled to 0–1.

**Q: I changed a `.vue` file and nothing changed at `:5000`.**
Run `npm run build` (from `frontend/`). Port 5000 serves the **built** copy in
`backend/static/`. Or develop on `:5173` with `npm run dev`.

**Q: I retrained but the dashboard shows the old accuracy.**
Restart `app.py`. Both the model and the meta file are cached in memory.

**Q: The dashboard is empty and says Facebook isn't connected.**
Run `uv run python get_token.py` in `backend/`, then restart `app.py`.

**Q: Posts appear but with zero comments.**
Your token is missing the `pages_read_user_content` permission.

**Q: I see `TemplateSyntaxError`.**
Someone changed `index()` in `app.py` to use `render_template`. Put back
`send_static_file` (Part 17).

**Q: `uv: command not found` / `npm: command not found`.**
See Part 6.7 (PATH).

**Q: `FileNotFoundError: ... training.1600000...csv`.**
Run `uv run python download_data.py` first.

**Q: `No trained model found`.**
Run `uv run python train.py` first.

**Q: Is it safe to delete `backend/data/app.db`?**
Yes. You lose the stored comments and their labels; the next refresh
re-downloads and reclassifies them from Facebook. (The tables are recreated
automatically on startup.)

**Q: Is it safe to share my `.env`?**
**No.** The Page token in it does not expire. Anyone who has it can read your
Page's data until you revoke it in Facebook's settings. It's in `.gitignore`
so git never saves it. Never paste it into chats or screenshots.

---

## Part 27 — Future extension: Bangla, Banglish, sarcasm, and higher accuracy

This part is a **plan**, not a description of code that exists. Nothing
here is built yet. It answers one question: *"What would I need to do so the
dashboard understands Bangla, Banglish and sarcasm, and gets more answers
right?"*

The honest headline first:

> **The algorithm is not the main problem. The labeled data is.**
> Every fix below needs examples with the correct answer written next to
> them (Part 1). A fancier model trained on the wrong data still gives wrong
> answers. A simple model trained on the right data often does well.

So most of this part is about **data**: where to get it, how to label it,
and how to measure honestly. The model switch comes after.

### 27.1 Why the current model can't just be "fixed"

| What you want | Why it fails today | What actually fixes it |
|---|---|---|
| **Bangla script** (`দারুণ সার্ভিস`) | `clean_text` keeps only `a–z`, so Bangla becomes an empty string. And Sentiment140 has zero Bangla, so there are no Bangla weights to use anyway. | Bangla labeled data **and** a cleaning step that keeps Bangla letters. |
| **Banglish** (`khub valo product`) | The words survive cleaning but were never in the 2009 tweets, so they have no weights. One word has many spellings: `bhalo`, `valo`, `balo`, `bhaloo`. | Banglish labeled data, plus a model that copes with spelling variety (27.6.2). |
| **Sarcasm** (`wow 3 din por reply, darun service 🙂`) | The model adds up word weights (Part 10). It sees `wow` and `darun` and says positive. The clues (the 🙂, the `!!!`, the complaint *next to* the praise) are deleted by `clean_text` or ignored by bag-of-words. | Sarcasm-labeled data and a model that reads **word order and context**, not just word counts. |
| **Real neutral** | Sentiment140 has no neutral class; we fake it with the 0.40–0.60 band (Part 12). | A dataset with a real `neutral` label. |
| **Your Page's language** | It learned from 2009 tweets, not shop comments (Part 10.15). | Labeled comments **from your own Page**. |

Every row ends in "labeled data". That is the pattern to remember.

### 27.2 The big decision: which kind of model

There are four realistic paths. You said you're willing to switch tools, so
all four are on the table.

**A. Upgraded classical model (what you have, improved).** Keep TF-IDF +
Logistic Regression, but train on Bangla/Banglish data, keep Bangla letters,
and use *character* n-grams (27.6.2). Runs on this PC in seconds, every
weight is still readable. But it still can't read word order, so sarcasm stays
out of reach, and accuracy hits a ceiling.

**B. Fine-tuned transformer (the recommended main path).** A **transformer**
is a neural network that reads a whole sentence and pays attention to how
words relate to each other ("great" *after* "broke again" means something
different). Big organisations have already **pretrained** such models on
billions of sentences in 100+ languages, so they already "know" Bangla and a
fair amount of English. You then **fine-tune** it: train it a little more on
your few thousand labeled comments so it learns *your* task. This is the
standard way to get high accuracy on text today.

**C. A large language model (LLM) through an API**, such as Claude. You send
each comment with instructions ("label this sentiment, is it sarcastic?") and
get an answer back. No training at all, and it handles Bangla, Banglish and
sarcasm reasonably from day one. But you pay per comment, it needs the
internet, your commenters' text leaves your computer, and you cannot inspect
why it answered.

**D. Hybrid (recommended long-term).** Use C to **suggest** labels, have a
human check them, then use those checked labels to fine-tune B. B then runs
locally, free, forever. You get most of the LLM's quality at none of its
running cost.

| | A. Classical+ | B. Fine-tuned transformer | C. LLM API | D. Hybrid (C labels → B) |
|---|---|---|---|---|
| Bangla script | OK with data | Good | Good | Good |
| Banglish | Weak–OK | OK–Good | Good | Good |
| Sarcasm | Very weak | OK (hard for everyone) | Best of the four | OK–Good |
| Needs a GPU to train | No | Yes (free cloud GPU is enough) | No | Yes (for B) |
| Cost per comment | Free | Free (runs on your CPU) | Paid | Free after training |
| Works offline | Yes | Yes | No | Yes |
| Can you inspect it | Fully (word weights) | Partly | No | Partly |
| Comments leave your PC | No | No | **Yes** | Only during labeling |

"Good" and "OK" here are rough, relative judgements, not measured numbers.
The only numbers that count are the ones **you** measure on **your** test
set (27.4, 27.9).

**Recommendation:** do Step 0 (27.4) no matter what. Then try A as a
cheap learning exercise, move to B for real accuracy, and use C only as a
labeling assistant (27.8).

### 27.3 New words for this part

| Word | Simple meaning |
|---|---|
| **Transformer** | A kind of neural network that reads a sentence as a whole and learns which words affect which. BERT, RoBERTa and GPT/Claude are all transformers. |
| **Pretrained model** | A transformer someone already trained on huge amounts of text. You download it (hundreds of MB). |
| **Fine-tuning** | Training a pretrained model a bit more on your own labeled examples. Minutes to hours, not months. |
| **Tokenizer** | The transformer's version of the vectorizer: it cuts text into pieces ("subwords") and turns them into numbers. Like the vectorizer, it **must travel with its model** (Part 10.10). |
| **Subword** | A piece of a word. `unhappiness` → `un`, `happi`, `ness`. This is how transformers cope with words they never saw whole, including Banglish spellings. |
| **Hugging Face** | The website + Python libraries (`transformers`, `datasets`) where pretrained models and datasets are shared. The "npm of ML models". |
| **GPU** | A graphics card. Trains neural networks 10–100× faster than a CPU. This PC has none; you'd borrow one for free in Google Colab or Kaggle Notebooks. |
| **Epoch** | One full pass over the training data. Fine-tuning usually needs 2–4. |
| **Logits / softmax** | The model's raw scores per class; softmax turns them into probabilities that add up to 1 (like P(positive) now, but for 3 classes). |
| **Code-mixing** | Switching languages inside one sentence: `product ta valo but delivery late`. Normal on Bangladeshi Facebook. |
| **Transliteration** | Writing one language in another's alphabet. Banglish is Bangla transliterated into Latin letters. |
| **Annotation / annotator** | Labeling data by hand / the person doing it. |
| **Inter-annotator agreement** | How often two people give the same label. Measured with **Cohen's kappa** (0 = chance, 1 = perfect). It's the realistic ceiling for your model: if humans agree 80% of the time, don't expect 95%. |
| **Macro-F1** | F1 (Part 10.11) computed per class, then averaged. Unlike accuracy, it isn't fooled when one class is rare (27.9). |
| **Gold set** | Your hand-checked test set. The "final exam" from Part 9, but for your Page. |

### 27.4 Step 0 — build your own test set first

Do this **before** choosing any model. Public datasets tell you how a model
does on *someone else's* comments. You need to know how it does on *yours*.

1. **Export real comments.** From `backend/`:

   ```bash
   sqlite3 -header -csv data/app.db \
     "SELECT id, post_id, message FROM comments ORDER BY RANDOM() LIMIT 1500" \
     > data/to_label.csv
   ```

   Take more than you need; some will be spam or empty.

2. **Label them by hand** in a spreadsheet (or a labeling tool such as
   Label Studio, Doccano or Argilla). One row per comment:

   | column | values | meaning |
   |---|---|---|
   | `text` | the comment | |
   | `language` | `en` / `bn` / `banglish` / `mixed` | which kind of text it is |
   | `sentiment` | `positive` / `negative` / `neutral` | what the writer **really** feels |
   | `sarcastic` | `0` / `1` | are they saying the opposite of what they mean? |

3. **Write labeling rules down** before you start, with examples. Without
   rules you'll label the same kind of comment differently on Monday and
   Friday. For example:

   | Comment | sentiment | sarcastic | why |
   |---|---|---|---|
   | `khub valo product, thank you` | positive | 0 | plain praise |
   | `দাম কত?` | neutral | 0 | a question, no feeling |
   | `wow 3 din por reply, darun service 🙂` | negative | 1 | praise words, complaint meaning |
   | `অসাধারণ! আবার অর্ডার করব` | positive | 0 | |
   | `product ta valo but delivery onek late` | negative | 0 | mixed; your rule decides (see 27.12) |
   | `tomra eto valo keno, amar pocket khali kore dile 😂` | positive | 1 | sarcastic **teasing** that is actually affectionate |

   The last row matters: **sarcasm is not always negative.** That's why
   `sentiment` records the real feeling and `sarcastic` is a separate column
   (27.6.3).

4. **Get a second opinion.** Ask a friend to label 200 of the same
   comments without seeing yours. Compute Cohen's kappa with
   `sklearn.metrics.cohen_kappa_score(yours, theirs)`. If it's low (below
   ~0.6), your rules are unclear. Fix the rules, not the labels.

5. **Freeze it.** Save ~1,000 labeled comments as
   `backend/data/gold_test.csv`. **Never train on it.** It's the final exam
   for every model you try from now on.

6. **Measure the current model on it.** That number is your baseline.
   You can't claim "higher accuracy" without it.

The gold set is a CSV of **real** comments with **your** labels. It lives
next to the training data, never inside the `comments` table. (The project
rule about not putting fabricated data in the database still applies.)

### 27.5 Where training data comes from

**Your own labeled comments are worth more than any public dataset**, because
they match your Page's topics and writing style. Aim for 2,000–5,000 beyond
the gold set. 27.8 shows how an LLM can speed this up.

Public datasets help you start, and teach the model general Bangla/Banglish.
Sizes below are approximate; **check each dataset's license and read 50 random
rows before trusting it**. Quality varies a lot.

| Need | Dataset | What it is |
|---|---|---|
| Bangla sentiment | **SentNoB** (2021) | ~15k Bangla social-media comments, positive / negative / neutral. Closest to Facebook comments. |
| Bangla sentiment | **BanglaBook** (2023) | ~158k Bangla book reviews, 3 classes. Large but a different topic. |
| Bangla sentiment | **BLP-2023 shared task** data | Bangla sentiment data released for a research competition; a good benchmark to compare against. |
| Banglish / code-mixed | **BnSentMix** (2024) | ~20k code-mixed Bangla–English sentences with sentiment labels (includes a "mixed" class you'll need to map). |
| Bangla sarcasm | **Ben-Sarc** (2023), **BanglaSarc** (2022) | Bangla Facebook/social-media comments labeled sarcastic or not. |
| English with neutral | **TweetEval (sentiment)** | Tweets labeled positive / negative / neutral. Better than Sentiment140 for a 3-class model. |
| English sarcasm | **iSarcasmEval** (SemEval-2022 Task 6) | Tweets where the *authors themselves* marked their sarcasm. |

Most of these are on the Hugging Face Hub or GitHub; search by name.

**Map labels carefully.** Every dataset has its own label names and rules.
Write one small function per dataset in `dataset.py` that converts its labels
to your `positive / negative / neutral` + `sarcastic` scheme, and print the
balance after mapping (like `load_splits()` does now).

### 27.6 Handling each problem

#### 27.6.1 Bangla script

Three things change:

**1. Cleaning must keep Bangla letters.** Bangla lives in the Unicode range
`U+0980`–`U+09FF`. For the classical path (A), `clean_text` would become
roughly:

```python
import unicodedata

BANGLA = "ঀ-৿"

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)                # see point 2
    text = text.lower()                                      # only affects Latin letters
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)       # also fixes Part 24.2
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(rf"[^a-z{BANGLA}\s]", " ", text)           # keep a-z AND Bangla
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

`rf"..."` is a raw f-string: backslashes kept as-is **and** `{BANGLA}` filled
in. The rule from Part 9 still holds: training and prediction must use this
**same** function, and changing it means retraining.

**2. Normalize Unicode.** The same Bangla letter can be stored as different
byte sequences (e.g. `য়` as one character, or as `য` + a dot mark `়`). They
look identical on screen but are different "words" to a computer.
`unicodedata.normalize("NFC", ...)` makes them consistent. BanglaBERT
(27.7) goes further and expects its own normalizer
(`csebuetnlp/normalizer` on GitHub). Use whatever the model's page says.

**3. Watch the stop-word trap again.** Bangla stop-word lists often contain
`না` and `নয়` ("not"). Removing them flips meanings, exactly like the
English `not` problem in Part 10. Safest: don't remove stop words at all.

For the transformer path (B), clean **less**: the model was pretrained on
raw text, punctuation and emoji included, and those carry meaning. Usually
you only remove URLs, replace `@mentions` with a placeholder like `@user`,
and normalize Unicode.

#### 27.6.2 Banglish

Banglish is hard for three reasons: the spelling is not standard (`bhalo`,
`valo`, `balo`), it's often mixed with English inside one sentence, and to a
computer it looks like English (same letters).

Three ways to handle it, cheapest first:

**a) Character n-grams (classical path, easy experiment).** Instead of whole
words, let TF-IDF count **pieces of words**:

```python
"char-2to5": dict(analyzer="char_wb", ngram_range=(2, 5), max_features=200_000),
```

`analyzer="char_wb"` = "character n-grams inside word boundaries".
`bhalo` and `valo` then share the pieces `al`, `alo`, `lo`, so the model
learns something from both. Add it to `CONFIGS` in `train.py` and compare on
val, the same way as Part 25.5. (Your data must include Banglish for this to
help.)

**b) Transliterate to Bangla script first.** Convert `khub valo` →
`খুব ভালো` with a transliteration model (e.g. AI4Bharat's IndicXlit supports
Bengali), then treat it as Bangla. Works well when transliteration is right;
when it's wrong, the error passes straight into the sentiment model.

**c) A transformer that has seen romanized text (recommended).** Google's
MuRIL was pretrained partly on transliterated Indian-language text, and
XLM-RoBERTa saw a lot of messy web text. Fine-tuned with Banglish examples,
they usually handle spelling variety far better than word counts can, thanks
to subwords.

**Detecting the language.** You need this mostly to **report** accuracy per
language, and to route comments if you end up with separate models. Bangla
script is easy to detect; English vs Banglish is not:

```python
BANGLA_CHAR = re.compile(r"[ঀ-৿]")
LATIN_CHAR = re.compile(r"[a-zA-Z]")
BANGLISH_HINTS = {"ami", "tumi", "apni", "valo", "bhalo", "kharap", "khub",
                  "onek", "ache", "nai", "na", "ki", "keno", "koto", "dam", "ta"}

def detect_language(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)   # links/mentions are Latin letters too
    has_bn = bool(BANGLA_CHAR.search(text))
    has_latin = bool(LATIN_CHAR.search(text))
    if has_bn and has_latin:
        return "mixed"
    if has_bn:
        return "bn"
    words = set(text.lower().split())
    return "banglish" if len(words & BANGLISH_HINTS) >= 1 else "en"
```

`words & BANGLISH_HINTS` is **set intersection**: the words that appear in
both sets. This hint list is a crude starting point. Check it against your
gold set's `language` column and grow the list, or later train a tiny
classifier on that column.

#### 27.6.3 Sarcasm, and how to group it

**Separate the two questions.** Don't make sarcasm a fourth sentiment. Ask
two independent questions of every comment:

1. **What does the writer really feel?** → `sentiment`: positive / negative / neutral
2. **Are they saying it indirectly/ironically?** → `sarcastic`: yes / no

Most sarcasm hides a **negative** feeling behind **positive** words. So if you
grouped sarcastic comments by their surface words, you would count
complaints as praise, which is exactly the mistake the current model makes.
But some sarcasm is friendly teasing that's really positive (the last example
in 27.4). That's why you **label the real feeling directly** and never use a
rule like "sarcastic → flip positive to negative".

**Grouping on the dashboard.** With both columns stored, you can choose any
view without retraining:

| View | How it groups | Good for |
|---|---|---|
| **By real feeling** (recommended default) | Sarcastic complaints count as negative, sarcastic teasing as positive. Sarcasm shown as a tag and a filter ("18 of 120 negatives were sarcastic"). | Honest totals |
| **Sarcasm as its own bucket** | positive / negative / neutral / sarcastic | Spotting sarcasm trends |
| **Both** | e.g. "negative (sarcastic)" as a sub-group of negative | Detailed drill-down |

```python
def display_group(sentiment: str, sarcastic: bool) -> str:
    return f"{sentiment} (sarcastic)" if sarcastic else sentiment
```

**Why sarcasm is hard, and what helps:**

- **The clues are what `clean_text` deletes.** 🙂 🙄 😒 😂, `!!!`, `...`,
  quotation marks, `wow`/`waah`. For sarcasm, keep emoji and punctuation, or
  turn emoji into words (the `emoji` library's `demojize()` turns 🙄 into
  `:face_with_rolling_eyes:`).
- **It often needs context.** `"great job"` under a post announcing a price
  cut is sincere; under a post apologising for a late delivery it's probably
  sarcastic. Feed the model both: `post: <post text> </s> comment: <comment>`.
  Your `posts` table already stores `message`, so the context is available.
- **It's rare.** Only a small share of comments are sarcastic. A model that
  always says "not sarcastic" can score very high *accuracy* and be useless.
  Measure the **precision and recall of the sarcastic class** instead, and
  use class weights during training.
- **Use a threshold, like the neutral band.** Only raise the flag when
  P(sarcastic) is high (say ≥ 0.7). A wrong sarcasm flag is more confusing
  to a reader than a missing one. Tune the threshold on val, not test.
- **Expect lower numbers than sentiment.** Even human annotators disagree
  on sarcasm often. Your kappa from 27.4 tells you the realistic ceiling.

**Model setup:** start with **two separate fine-tuned models** (one for
sentiment, one for sarcasm). It's simpler to debug. Later you can merge them
into one model with two outputs ("multi-task"), which is faster to run.

### 27.7 The recommended model: fine-tuning a transformer

**Candidates** (all free on Hugging Face). Try more than one and let val
decide, exactly like `train.py` compares configs today:

| Model id | Good at | Weak at |
|---|---|---|
| `xlm-roberta-base` | One model for English + Bangla + some Banglish; good default | Slightly behind specialists on pure Bangla |
| `csebuetnlp/banglabert` | Bangla script (built by BUET for Bangla) | English and Banglish; needs its normalizer |
| `google/muril-base-cased` | Indian languages incl. romanized text; good Banglish candidate | Needs testing on your data |

**Hardware.** This PC has no NVIDIA GPU, 4 CPU cores and 7 GB RAM. That's
enough to **run** a base-size model (roughly 1 GB of memory once loaded),
but training on a CPU would be painfully slow. Train in **Google Colab** or
**Kaggle Notebooks**, which give free GPU time. For a base model on a few
tens of thousands of short comments, expect minutes to about an hour per
run on a free GPU. Then download the saved model folder into
`backend/models/`.

**New libraries** (training side): `torch`, `transformers`, `datasets`,
`accelerate`. The web app only needs `torch` (the CPU-only build is much
smaller) and `transformers`.

**A minimal fine-tuning script** (`train_transformer.py`, meant for Colab).
Assumes CSVs with `text` and `sentiment` columns:

```python
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL = "xlm-roberta-base"
LABELS = ["negative", "neutral", "positive"]
label2id = {name: i for i, name in enumerate(LABELS)}

ds = load_dataset("csv", data_files={"train": "data/train.csv",
                                     "validation": "data/val.csv"})
ds = ds.map(lambda row: {"label": label2id[row["sentiment"]]})

tokenizer = AutoTokenizer.from_pretrained(MODEL)
ds = ds.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
            batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)), label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)          # the class with the highest score
    return {"macro_f1": f1_score(labels, preds, average="macro")}

args = TrainingArguments(
    output_dir="models/xlmr-sentiment",
    num_train_epochs=3,
    learning_rate=2e-5,                         # small: nudge, don't overwrite, what it knows
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,                # keep the epoch with the best val score
    metric_for_best_model="macro_f1",
    seed=42,
)

trainer = Trainer(model=model, args=args,
                  train_dataset=ds["train"], eval_dataset=ds["validation"],
                  processing_class=tokenizer, compute_metrics=compute_metrics)
trainer.train()
trainer.save_model("models/xlmr-sentiment/best")   # model + tokenizer, together
```

What maps to what you already know:

| Here | In the current project |
|---|---|
| `AutoTokenizer` | `TfidfVectorizer` |
| `AutoModelForSequenceClassification` | `LogisticRegression` |
| `trainer.train()` | `pipeline.fit()` |
| `trainer.save_model(...)` (saves model **and** tokenizer) | `joblib.dump(pipeline, ...)` |
| `eval_strategy="epoch"` + `load_best_model_at_end` | comparing configs on val |
| `max_length=128` | nothing: TF-IDF had no length limit. Longer comments get cut off. |

**The Pipeline rule, again.** A model loaded with a *different* tokenizer
gets the same silent garbage as Part 10.10's "pizza" column. Always load
both from the same folder.

**Using it:**

```python
from transformers import pipeline

clf = pipeline("text-classification", model="models/xlmr-sentiment/best", top_k=None)
clf(["delivery onek late 😒", "অসাধারণ সার্ভিস"])
# one list of {label, score} per comment, e.g. [{'label': 'negative', 'score': 0.9}, ...]
```

(Hugging Face's `pipeline` is a different thing from sklearn's `Pipeline`;
same word, unrelated code.)

**Neutral becomes a real class.** The 0.40–0.60 band retires. You can keep
the *idea* as an "uncertain" flag: if the highest class probability is below,
say, 0.5, show the comment as uncertain instead of trusting the label.

### 27.8 Speeding up labeling with an LLM (optional)

Labeling 5,000 comments by hand is slow. An LLM can **propose** labels, and
you **review** them. Reviewing is several times faster than labeling from
scratch.

Two rules:

1. **A human checks every label that goes into the gold test set.** If the
   test set is LLM-labeled, you are measuring "how often does my model agree
   with the LLM", not "how often is it right".
2. **Privacy is your decision.** This sends your commenters' words to an
   outside company. Check Meta's Platform Terms and your own privacy policy
   first, and strip author names (you only need the text).

A sketch using Claude through Anthropic's Python SDK (`uv add anthropic
pydantic`; the key goes in `.env` as `ANTHROPIC_API_KEY`, never in code):

```python
from typing import Literal

import anthropic
from pydantic import BaseModel


class Label(BaseModel):
    language: Literal["en", "bn", "banglish", "mixed"]
    sentiment: Literal["positive", "negative", "neutral"]
    sarcastic: bool
    reason: str                      # one short sentence; helps you review faster


RULES = """You label comments from a Bangladeshi shop's Facebook Page.
Comments may be English, Bangla script, Banglish (romanized Bangla), or mixed.
sentiment = what the writer really feels, not the surface words.
sarcastic = true only if they say the opposite of what they mean.
Sarcasm can be affectionate teasing; then sentiment can still be positive.
Questions with no feeling are neutral.
<paste your labeling examples from 27.4 here>"""

client = anthropic.Anthropic()       # reads ANTHROPIC_API_KEY


def suggest_label(comment: str, post: str = "") -> Label | None:
    response = client.messages.parse(
        model="claude-opus-5",
        max_tokens=4000,
        system=RULES,
        messages=[{"role": "user", "content": f"Post: {post}\n\nComment: {comment}"}],
        output_format=Label,         # the answer must match the Label class
    )
    if response.stop_reason == "refusal":
        return None                  # label this one by hand
    return response.parsed_output
```

`client.messages.parse(..., output_format=Label)` makes the API return JSON
that matches the `Label` class and turns it into a Python object, so you
never parse free text. `Literal[...]` means "only these exact values are
allowed".

Practical notes:
- **Measure the LLM on your gold set first.** Run it on the 1,000 gold
  comments and compare with your labels (macro-F1, kappa). That tells you
  how much review it needs.
- **Cheaper models exist** (`claude-sonnet-5`, `claude-haiku-4-5`). Switch
  only if they score close enough on the gold set.
- **Bulk jobs:** Anthropic's Message Batches API processes many requests
  asynchronously at a lower price. Good for labeling thousands of comments
  overnight.
- **Could the LLM *be* the classifier?** Yes (path C in 27.2). The
  `suggest_label` function is essentially that. The trade-off is ongoing
  cost, needing the internet, and the privacy question on every refresh.

### 27.9 Measuring "higher accuracy" honestly

**Use macro-F1, not accuracy.** Suppose 70% of your comments are positive. A
model that always says "positive" scores **70% accuracy** and is useless.
Its macro-F1 is terrible, because it gets 0 on negative and neutral. Macro-F1
treats each class as equally important:

```python
from sklearn.metrics import classification_report, f1_score

f1_score(y_true, y_pred, average="macro")
print(classification_report(y_true, y_pred, digits=3))   # per-class detail
```

**Report per language.** One overall number hides problems. A model can be
great on English and useless on Banglish. Fill a table like this for every
model you try, on the gold set:

| Model | Overall macro-F1 | en | bn | banglish | mixed | Sarcasm precision | Sarcasm recall |
|---|---|---|---|---|---|---|---|
| Current (Sentiment140 bigram-50k) | | | | | | n/a | n/a |
| A: char n-grams + Bangla data | | | | | | n/a | n/a |
| B: xlm-roberta-base fine-tuned | | | | | | | |
| B: banglabert fine-tuned | | | | | | | |
| C: LLM zero-shot | | | | | | | |

**The same exam rules as Part 9 apply.** Compare models on val. Touch the
gold set only to compare **finalists**, not after every tweak. If you keep
tuning until gold looks good, gold stops being a fair exam.

**Realistic expectations.** Published results for fine-tuned transformers on
Bangla social-media sentiment (e.g. SentNoB) are around 0.7 macro-F1, not
0.95. Social-media text is noisy and humans disagree. Your own
domain-specific data can push you above public benchmarks, and your
inter-annotator agreement (27.4) is the practical ceiling.

### 27.10 Wiring it into this repo (once a model wins)

The layering from Part 2 stays exactly as it is. `predict.py` is still the
only doorway between ML and the web app.

| File | Change |
|---|---|
| `dataset.py` | Loaders + label mappers for each new dataset; the new cleaning function (27.6.1); `detect_language()`. |
| `train.py` | Stays for the classical models. |
| `train_transformer.py` (new) | The fine-tuning script (27.7). Runs in Colab. Writes to a **versioned** folder like `models/xlmr-sentiment-2026-10-01/`, so a bad run can't overwrite a good one (the `train.py` trap). Writes per-language metrics into its `model_meta.json`. |
| `predict.py` | Loads the transformer once (keep the `lru_cache` idea). `classify()` keeps returning the same keys (`sentiment`, `score`, `confidence`) so the rest of the app doesn't break, plus new ones: `language`, `sarcastic`, `sarcasm_score`. |
| `db.py` | New columns on `comments`: `language TEXT`, `sarcastic INTEGER`, `model_version TEXT`. `CREATE TABLE IF NOT EXISTS` won't add columns to an existing table, so add them with `ALTER TABLE comments ADD COLUMN ...` (or delete `app.db`, Part 26). `model_version` records which model labeled each row. Reclassify old rows by setting `sentiment = NULL` (Part 25.1). |
| `app.py` / `analytics()` | Add sarcasm counts and a per-language breakdown to the JSON. |
| Frontend | A sarcasm tile and a language filter. Sarcasm is a **tag**, not a polarity, so don't add a fourth hue to the diverging color scale (see CLAUDE.md, chart colors). Use an icon or outline. If you do add a color, re-validate it for colorblind separation. |
| `pyproject.toml` | `torch` (CPU build) and `transformers`. The first install is a large download. |

### 27.11 Roadmap checklist

Do these in order; each step is useful on its own.

- [ ] **0. Gold set:** label ~1,000 real comments (27.4), check agreement,
      freeze it, measure the current model. *(No code changes.)*
- [ ] **1. Classical upgrade (learning step):** keep Bangla in `clean_text`,
      add SentNoB + BnSentMix via `dataset.py`, try `char_wb` n-grams.
      Compare on gold. *(Runs on this PC.)*
- [ ] **2. More of your own labels:** 2,000–5,000 comments, optionally with
      LLM suggestions (27.8) + human review.
- [ ] **3. Fine-tune** 2–3 transformer candidates for sentiment in Colab.
      Pick the winner on val, confirm on gold.
- [ ] **4. Sarcasm model:** Ben-Sarc + your `sarcastic` labels, with post
      context, threshold tuned on val.
- [ ] **5. Integrate** (27.10): `predict.py`, new DB columns, dashboard
      changes, reclassify.
- [ ] **6. Keep labeling:** a few hundred new comments each month,
      retrain occasionally. Language on your Page will drift.

### 27.12 Traps to avoid

- **Duplicate comments across splits.** Facebook comments repeat a lot
  (`price?`, `inbox please`, `nice`). If the same text lands in both train
  and test, scores look better than they are. Remove exact duplicates before
  splitting.
- **Splitting by comment instead of by post.** Comments under one post
  share topic and wording. Put all comments of a post in the **same** split
  (sklearn's `GroupShuffleSplit` with `groups=post_id`), or your test set
  quietly leaks from training.
- **Inconsistent "mixed" comments.** `valo product but delivery late` has
  both feelings. Decide once (e.g. "label the feeling the writer ends on",
  or add a `mixed` class) and write it into your rules. BnSentMix has a
  mixed class; map it deliberately.
- **Removing `না`.** The stop-word trap, in Bangla (27.6.1).
- **Tokenizer from a different folder than the model.** Silent garbage (27.7).
- **Trusting overall accuracy.** Use macro-F1, per language, and sarcasm
  precision/recall (27.9).
- **Unreviewed LLM labels in the gold set** (27.8).
- **Forgetting to restart `app.py`** after swapping models. Still cached
  (Part 24.4).

---

## The one-paragraph summary

`download_data.py` fetches 1.6M labeled tweets. `dataset.py` cleans them with
`clean_text` and splits them 70/15/15. `train.py` turns text into TF-IDF
numbers, fits three logistic-regression pipelines, picks the best on
validation, scores it once on test (80.2%), and saves it with joblib. That
file never changes on its own. `app.py` (Flask) uses `facebook.py` to fetch
comments, `db.py` to store them in SQLite (never twice), and `predict.py` to
load the saved pipeline, clean each new comment with the **same**
`clean_text`, get P(positive), and apply the 0.40–0.60 neutral band. The
Vue app (`main.js` → `App.vue` → seven components) calls the API through
`api.js`, keeps the results in reactive objects, and Vue plus Chart.js redraw
everything whenever those objects change.
