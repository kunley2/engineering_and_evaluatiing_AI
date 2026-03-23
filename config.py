class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.20
    MIN_CLASS_COUNT = 3
    MAX_FEATURES = 2000
    MIN_DF = 2
    MAX_DF = 0.90
    DEFAULT_CSV_FILE = "AppGallery.csv"
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'
    TRANSLATION_MODEL = "facebook/m2m100_418M"

    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'