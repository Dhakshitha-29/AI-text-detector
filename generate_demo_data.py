# makes a small fake dataset just to test the code. NOT real data, do not use for the report
import random

import pandas as pd

from src import config

TOPICS = [
    "artificial intelligence", "renewable energy", "online education", "climate change",
    "social media", "remote work", "public transport", "cyber security", "healthy eating",
    "space exploration", "electric cars", "cloud computing", "urban farming", "mobile banking",
    "video games", "recycling", "machine learning", "the internet of things", "tourism",
    "digital privacy", "smart cities", "robotics", "online shopping", "mental wellbeing",
    "quantum computing", "wearable devices", "open source software", "e-learning", "blockchain",
    "sustainable fashion",
]

# formal sentences (pretend AI text)
AI_TEMPLATES = [
    "{Topic} plays an important role in modern society and provides numerous benefits across various sectors.",
    "Furthermore, it is important to note that {topic} can significantly enhance efficiency and productivity.",
    "Moreover, researchers have demonstrated that {topic} offers a wide range of potential applications.",
    "In conclusion, {topic} represents a significant development that will continue to shape the future.",
    "Additionally, various approaches to {topic} help ensure effective and sustainable outcomes.",
    "However, it is essential to consider the challenges associated with {topic} in order to utilize it responsibly.",
    "Overall, understanding {topic} is crucial for individuals, organizations, and communities alike.",
    "For example, {topic} facilitates collaboration and enables numerous stakeholders to achieve their goals.",
    "Therefore, a comprehensive approach to {topic} is necessary to address complex challenges effectively.",
    "{Topic} is a complex and multifaceted subject that requires careful analysis and consideration.",
    "It is also worth mentioning that {topic} continues to evolve rapidly, and organizations must adapt accordingly.",
    "By leveraging {topic}, stakeholders can improve decision making, reduce costs, and enhance overall performance.",
    "Consequently, the growing importance of {topic} highlights the need for continuous learning and innovation.",
    "In addition, {topic} has the potential to transform traditional practices and create new opportunities.",
]

# casual sentences (pretend human text)
HUMAN_TEMPLATES = [
    "Honestly, I never really thought much about {topic} until last year.",
    "My cousin works with {topic} and she says it's way messier than people think.",
    "We tried it once. Total disaster.",
    "I don't know about you, but {topic} always seemed overhyped to me.",
    "Anyway, the thing about {topic} is that nobody agrees on what it even means!",
    "It took us three weeks, a lot of coffee, and two broken laptops to figure out {topic}.",
    "Why does everyone keep talking about {topic}? Beats me.",
    "Last Tuesday I read something about {topic} that honestly surprised me.",
    "There's 40 years of arguments about {topic}, and I've sat through most of them.",
    "Not sure it matters, but my teacher thought {topic} was the best part of the course.",
    "Sure, {topic} has its problems, but whatever, we'll manage.",
    "Half the time I just skim the notes on {topic} and hope for the best.",
    "A friend of mine called {topic} a waste of time; I'm not so convinced.",
    "Funny thing is, {topic} came up again at dinner and my dad had strong opinions.",
]


def _make_text(rng, own_templates, other_templates, noise=0.15):
    # 4 to 8 sentences about one topic; some come from the other style so it is not too easy
    topic = rng.choice(TOPICS)
    sentences = []
    for _ in range(rng.randint(4, 8)):
        pool = other_templates if rng.random() < noise else own_templates
        template = rng.choice(pool)
        sentences.append(template.format(topic=topic, Topic=topic[0].upper() + topic[1:]))
    return " ".join(sentences)


def generate_demo_dataset(n_per_class=400, seed=config.RANDOM_STATE):
    config.ensure_dirs()
    rng = random.Random(seed)
    human = {_make_text(rng, HUMAN_TEMPLATES, AI_TEMPLATES) for _ in range(n_per_class * 3)}
    ai = {_make_text(rng, AI_TEMPLATES, HUMAN_TEMPLATES) for _ in range(n_per_class * 3)}
    human = sorted(human - ai)[:n_per_class]           
    ai = sorted(ai - set(human))[:n_per_class]
    rng.shuffle(human)
    rng.shuffle(ai)
    pd.DataFrame({"text": human, "label": "human"}).to_csv(config.HUMAN_CSV, index=False)
    pd.DataFrame({"text": ai, "label": "ai"}).to_csv(config.AI_CSV, index=False)
    print(f"Synthetic DEMO data written: {len(human)} human + {len(ai)} ai texts")
    print(f"  {config.HUMAN_CSV}\n  {config.AI_CSV}")
    print("Note: this is fake template data, only for testing the code.")


if __name__ == "__main__":
    generate_demo_dataset()
