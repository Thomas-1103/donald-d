import argparse

from .hf import DONALD_D


def main():
    parser = argparse.ArgumentParser(
        description="Visualize transformer hidden states"
    )

    parser.add_argument(
        "sentence",
        type=str,
        help="Input sentence",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model name",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="DONALD-D_visualisation.svg",
    )

    args = parser.parse_args()

    DONALD_D(
        args.sentence,
        model_name=args.model_name,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()