import argparse
from threading import Thread

import nltk
from platformdirs import user_data_dir

from threedont import Controller
from .app.state import Config, AppState

# Setup dependencies
def setup_nltk():
    nltk.download("wordnet", user_data_dir("threedont"))
    nltk.data.path.insert(0, user_data_dir("threedont"))


def main():
    nltk_thread = Thread(target=setup_nltk)
    nltk_thread.start()

    parser = argparse.ArgumentParser(description='3Dont')
    parser.add_argument('--test', action='store_true', help='Test the viewer')
    args = parser.parse_args()

    app_state = AppState("threedont")
    config = Config("threedont")

    controller = Controller(config, app_state)

    controller.run()
    print("Application stopped gracefully")
    nltk_thread.join()


if __name__ == '__main__':
    main()
