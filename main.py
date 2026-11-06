# run with:  python main.py            (menu)
#            python main.py demo-data | prepare | train | evaluate
#            python main.py predict "some text"
import sys

from src import config
from src.config import ProjectError

BANNER = """
============================================
AI-GENERATED TEXT DETECTION SYSTEM
============================================
1. Detect a text
2. Compare models
3. Run robustness experiment
4. Show evaluation metrics
5. Exit
"""


def read_multiline_text():
    # keeps reading lines until an empty line, so pasted paragraphs work
    print("Enter text (finish with an empty line):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return " ".join(lines)


def cmd_demo_data():
    from src.generate_demo_data import generate_demo_dataset
    generate_demo_dataset()


def cmd_prepare():
    from src.data_preparation import prepare_dataset
    prepare_dataset()


def cmd_train():
    from src.data_preparation import load_processed
    from src.train_baseline import train_and_save_baseline
    from src.train_proposed import train_and_save_proposed
    df = load_processed()
    train_and_save_baseline(df)
    train_and_save_proposed(df)


def cmd_evaluate():
    from src.data_preparation import load_processed
    from src.evaluate import print_model_comparison, print_robustness, run_experiments
    results = run_experiments(load_processed())
    print_model_comparison(results)
    print_robustness(results)


def cmd_predict(text):
    from src.predict import detect
    detect(text)


def menu():
    while True:
        print(BANNER)
        try:
            choice = input("Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        try:
            if choice == "1":
                cmd_predict(read_multiline_text())
            elif choice == "2":
                from src.data_preparation import load_processed
                from src.evaluate import load_or_run, print_model_comparison
                print_model_comparison(load_or_run(load_processed()))
            elif choice == "3":
                from src.data_preparation import load_processed
                from src.evaluate import load_or_run, print_robustness
                print_robustness(load_or_run(load_processed(), force=True))
            elif choice == "4":
                from src.evaluate import show_saved_metrics
                show_saved_metrics()
            elif choice == "5":
                print("Bye.")
                return
            else:
                print("Invalid choice. Please type a number from 1 to 5.")
        except ProjectError as error:
            print(f"\n[Error] {error}")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")


def main():
    args = sys.argv[1:]
    if not args:
        menu()
        return
    commands = {"demo-data": cmd_demo_data, "prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate}
    try:
        if args[0] in commands:
            commands[args[0]]()
        elif args[0] == "predict" and len(args) > 1:
            cmd_predict(" ".join(args[1:]))
        else:
            print(__doc__)
    except ProjectError as error:
        print(f"[Error] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
