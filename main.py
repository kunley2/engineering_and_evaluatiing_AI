from config import Config
from pipeline import Pipeline


if __name__ == '__main__':
    pipeline = Pipeline(mode=Config.DEFAULT_PIPELINE_MODE)
    pipeline.run()
