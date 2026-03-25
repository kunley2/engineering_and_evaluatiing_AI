import argparse

from config import Config
from pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['single_label', 'chained'],
        default=Config.DEFAULT_PIPELINE_MODE,
    )
    args = parser.parse_args()

    pipeline = Pipeline(mode=args.mode)
    pipeline.run()
