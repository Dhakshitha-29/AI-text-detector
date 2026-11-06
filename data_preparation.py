import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.config import ProjectError
from src.robustness import make_modified_versions

LABEL_ALIASES = {
    "human": "human", "human-written": "human", "human written": "human",
    "ai": "ai", "ai-generated": "ai", "ai generated": "ai", "machine": "ai", "llm": "ai",
}


def clean_text(text):
    # only remove weird characters and extra spaces
    # keep case and punctuation because the style features need them
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_labeled_csv(path):
    if not path.exists():
        raise ProjectError(
            f"File not found: {path}\n"
            "  put your csv there (columns: text,label) or run: python main.py demo-data"
        )
    try:
        df = pd.read_csv(path)
    except Exception as error:
        raise ProjectError(f"Could not read {path.name} ({error}). Header should be: text,label")

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ProjectError(
            f"{path.name} is missing column(s) {sorted(missing)}. Found: {list(df.columns)}. "
            "Header should be: text,label"
        )

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    if df["label"].isna().any():
        raise ProjectError(f"{path.name} has labels other than 'human' or 'ai'.")
    return df


def load_raw_data():
    human = load_labeled_csv(config.HUMAN_CSV)
    ai = load_labeled_csv(config.AI_CSV)
    df = pd.concat([human, ai], ignore_index=True)
    n_start = len(df)

    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= config.MIN_WORDS]
    n_after_short = len(df)

    # same text with two different labels -> remove it completely
    conflicting = df.groupby("text")["label"].transform("nunique") > 1
    df = df[~conflicting]
    df = df.drop_duplicates(subset="text")
    df = df.reset_index(drop=True)

    print(f"Loaded {n_start} rows, removed {n_start - n_after_short} empty/short and "
          f"{n_after_short - len(df)} duplicate/conflicting, {len(df)} left.")

    counts = df["label"].value_counts()
    for label in ("human", "ai"):
        if counts.get(label, 0) < 10:
            raise ProjectError(f"Need at least 10 '{label}' texts after cleaning, found {counts.get(label, 0)}.")

    df["group_id"] = range(len(df))
    return df


def split_documents(df):
    # split the original texts 70/15/15 (train/val/test), keeping the class ratio
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE
    )
    train, val = train_test_split(
        train_val, test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        stratify=train_val["label"], random_state=config.RANDOM_STATE,
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    df["variant"] = "original"
    return df


def modified_from_csv(ai_docs):
    # modified texts given by the user (for example from a real paraphrasing tool)
    if not config.MODIFIED_CSV.exists():
        return None
    try:
        mod = pd.read_csv(config.MODIFIED_CSV)
    except Exception as error:
        raise ProjectError(f"Could not read {config.MODIFIED_CSV.name} ({error}).")
    mod.columns = [str(c).strip().lower() for c in mod.columns]
    needed = {"original_text", "modified_text", "modification_type"}
    if needed - set(mod.columns):
        raise ProjectError(f"{config.MODIFIED_CSV.name} needs columns: original_text,modified_text,modification_type")

    lookup = dict(zip(ai_docs["text"], ai_docs["group_id"]))
    rows, skipped = [], 0
    for original, modified, kind in zip(mod["original_text"], mod["modified_text"], mod["modification_type"]):
        original, modified = clean_text(original), clean_text(modified)
        group_id = lookup.get(original)
        if group_id is None or not modified or modified == original or len(modified.split()) < config.MIN_WORDS:
            skipped += 1
            continue
        rows.append({"group_id": group_id, "text": modified, "label": "ai",
                     "variant": str(kind).strip().lower()})
    print(f"Modified texts from csv: {len(rows)} used, {skipped} skipped "
          "(original not found, unchanged or too short).")
    return pd.DataFrame(rows)


def modified_automatically(ai_docs):
    rows = []
    for _, doc in ai_docs.iterrows():
        versions = make_modified_versions(doc["text"], seed=config.RANDOM_STATE + int(doc["group_id"]))
        for kind, new_text in versions.items():
            rows.append({"group_id": doc["group_id"], "text": new_text, "label": "ai", "variant": kind})
    print(f"Modified texts generated by rules: {len(rows)}")
    return pd.DataFrame(rows)


def add_modified_texts(df):
    # modified texts are made AFTER the split and get the same split as the original text.
    # this way a text and its modified copy are never in train and test at the same time
    ai_docs = df[df["label"] == "ai"]
    modified = modified_from_csv(ai_docs)
    if modified is None:
        modified = modified_automatically(ai_docs)
    if modified.empty:
        raise ProjectError("No modified AI texts could be created. Check data/modified_ai_text.csv.")

    modified = modified.drop_duplicates(subset="text")
    modified = modified[~modified["text"].isin(set(df["text"]))]
    split_of = dict(zip(df["group_id"], df["split"]))
    modified["split"] = modified["group_id"].map(split_of)
    return pd.concat([df, modified], ignore_index=True)


def check_no_leakage(df):
    if (df.groupby("group_id")["split"].nunique() > 1).any():
        raise ProjectError("Leakage: a text and its modified copy are in different splits!")
    if (df.groupby("text")["split"].nunique() > 1).any():
        raise ProjectError("Leakage: the same text is in more than one split!")
    print("Leakage check passed: every text and its modified copies are in one split only.")


def print_class_distribution(df):
    originals = df[df["variant"] == "original"]
    table = pd.crosstab(originals["split"], originals["label"]).reindex(["train", "val", "test"])
    table["total"] = table.sum(axis=1)
    print("\nClass distribution (original texts):")
    print(table.to_string())
    modified = df[df["variant"] != "original"]
    print("\nModified AI texts per split and type:")
    print(pd.crosstab(modified["variant"], modified["split"]).to_string())
    ratio = originals["label"].value_counts(normalize=True)
    print(f"\nClass balance: human {ratio.get('human', 0):.1%} / ai {ratio.get('ai', 0):.1%}")


def prepare_dataset():
    config.ensure_dirs()
    df = load_raw_data()
    df = split_documents(df)
    df = add_modified_texts(df)
    check_no_leakage(df)
    print_class_distribution(df)
    df = df[["group_id", "text", "label", "variant", "split"]]
    df.to_csv(config.PROCESSED_CSV, index=False)
    print(f"\nSaved processed dataset to {config.PROCESSED_CSV}")
    return df


def load_processed():
    if not config.PROCESSED_CSV.exists():
        raise ProjectError(f"Processed dataset not found: {config.PROCESSED_CSV}\n  run: python main.py prepare")
    try:
        df = pd.read_csv(config.PROCESSED_CSV)
    except Exception as error:
        raise ProjectError(f"Could not read processed dataset ({error}). Run: python main.py prepare")
    needed = {"group_id", "text", "label", "variant", "split"}
    if needed - set(df.columns):
        raise ProjectError("Processed dataset has wrong columns. Run: python main.py prepare")
    return df


def select_training_rows(df, split, augmented):
    # augmented=False-- human + original AI only
    # augmented=True-- also include the modified AI texts
    part = df[df["split"] == split]
    if augmented:
        return part
    return part[part["variant"] == "original"]


def to_xy(part):
    # label 1 = AI, 0 = human
    return part["text"], (part["label"] == "ai").astype(int)
