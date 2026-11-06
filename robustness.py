# makes modified versions of AI text: paraphrase, human_edit, shortened
# these are simple rule based edits, not a real paraphrasing model.
# for real paraphrases put them in data/modified_ai_text.csv instead
import random
import re

SYNONYMS = {
    "important": ["crucial", "key", "essential"],
    "significant": ["notable", "considerable", "major"],
    "significantly": ["notably", "considerably"],
    "utilize": ["use", "employ"],
    "numerous": ["many", "several"],
    "various": ["different", "several"],
    "provide": ["offer", "supply"],
    "provides": ["offers", "supplies"],
    "enhance": ["improve", "boost"],
    "enhances": ["improves", "boosts"],
    "ensure": ["make sure", "guarantee"],
    "demonstrate": ["show", "illustrate"],
    "demonstrated": ["shown", "illustrated"],
    "facilitate": ["help", "enable"],
    "facilitates": ["helps", "enables"],
    "approach": ["method", "strategy"],
    "approaches": ["methods", "strategies"],
    "crucial": ["vital", "critical"],
    "essential": ["necessary", "vital"],
    "overall": ["in general", "all things considered"],
    "additionally": ["also", "on top of that"],
    "moreover": ["also", "what is more"],
    "furthermore": ["besides", "in addition"],
    "however": ["but", "still"],
    "therefore": ["so", "thus"],
    "consequently": ["as a result", "so"],
    "in conclusion": ["to sum up", "all in all"],
    "for example": ["for instance", "say"],
    "transform": ["change", "reshape"],
    "transforming": ["changing", "reshaping"],
    "rapidly": ["quickly", "fast"],
    "challenges": ["difficulties", "problems"],
    "benefits": ["advantages", "gains"],
    "understand": ["grasp", "comprehend"],
    "understanding": ["grasping", "knowing"],
    "analyze": ["examine", "study"],
    "analysis": ["examination", "review"],
    "growing": ["increasing", "rising"],
    "widely": ["broadly", "commonly"],
    "help": ["assist", "support"],
    "helps": ["assists", "supports"],
    "complex": ["complicated", "intricate"],
    "effective": ["efficient", "useful"],
    "effectively": ["efficiently", "well"],
    "comprehensive": ["thorough", "wide-ranging"],
    "potential": ["possible", "likely"],
    "consider": ["think about", "weigh"],
    "consideration": ["thought", "attention"],
}

# one regex for all the words above, longest first so "in conclusion" matches before "in"
SYN_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in SYNONYMS), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

CONTRACTIONS = [
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
    ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"),
    ("cannot", "can't"), ("will not", "won't"), ("would not", "wouldn't"),
    ("has not", "hasn't"), ("have not", "haven't"),
    ("it is", "it's"), ("that is", "that's"), ("there is", "there's"),
    ("they are", "they're"), ("we are", "we're"),
]

FILLERS = ["Honestly,", "To be fair,", "I think", "Basically,", "Well,", "In my view,", "Frankly,"]

FILLER_ADVERBS = re.compile(
    r"(?<=[a-z,] )\b(very|really|quite|highly|extremely|particularly|also)\b\s+", flags=re.IGNORECASE
)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def match_case(original, replacement):
    # keep the capital letter if the old word had one
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def paraphrase_text(text, rng):
    def replace_word(match):
        word = match.group(0)
        if rng.random() < 0.7:
            return match_case(word, rng.choice(SYNONYMS[word.lower()]))
        return word

    out = SYN_PATTERN.sub(replace_word, text)

    # split some long sentences into two
    out = re.sub(r";\s+([a-z])", lambda m: ". " + m.group(1).upper(), out)

    def split_on_and(match):
        if rng.random() < 0.5:
            return ". In addition, " + match.group(1)
        return match.group(0)

    out = re.sub(r",\s+and\s+([a-z])", split_on_and, out)
    return out


def swap_two_letters(word, rng):
    # makes a small typo
    if len(word) < 5:
        return word
    i = rng.randint(1, len(word) - 3)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def human_edit_text(text, rng):
    out = text

    # contractions
    for long_form, short_form in CONTRACTIONS:
        pattern = re.compile(r"\b" + re.escape(long_form) + r"\b", flags=re.IGNORECASE)
        out = pattern.sub(
            lambda m, s=short_form: match_case(m.group(0), s) if rng.random() < 0.8 else m.group(0), out
        )

    # filler words at the start of some sentences
    new_sentences = []
    for sentence in split_sentences(out):
        if rng.random() < 0.25:
            filler = rng.choice(FILLERS)
            sentence = filler + " " + sentence[:1].lower() + sentence[1:]
        new_sentences.append(sentence)
    out = " ".join(new_sentences)

    # a few typos
    out = re.sub(
        r"\b[A-Za-z]{5,}\b",
        lambda m: swap_two_letters(m.group(0), rng) if rng.random() < 0.04 else m.group(0),
        out,
    )

    # drop some commas
    out = re.sub(r",(?=\s)", lambda m: "" if rng.random() < 0.15 else ",", out)
    return out


def shorten_text(text, rng):
    sentences = split_sentences(text)
    if len(sentences) >= 3:
        keep = [s for s in sentences if rng.random() < 0.6]
        if len(keep) < 2:
            keep = sentences[:2]
        if len(keep) == len(sentences):   # make sure at least one sentence is removed
            keep = keep[:-1]
    else:
        keep = sentences
    out = " ".join(keep)
    return FILLER_ADVERBS.sub("", out)


def make_modified_versions(text, seed):
    # returns {type: modified text}, skips the ones that did not change
    functions = {
        "paraphrase": paraphrase_text,
        "human_edit": human_edit_text,
        "shortened": shorten_text,
    }
    result = {}
    for kind, function in functions.items():
        rng = random.Random(f"{seed}-{kind}")
        modified = " ".join(function(text, rng).split())
        if modified and modified != text and len(modified.split()) >= 5:
            result[kind] = modified
    return result
