import argparse
from platformdirs import user_data_dir

from threedont import Controller
from .app.state import Config, AppState

# Setup dependencies
import nltk
nltk.download("wordnet", user_data_dir("threedont"))
nltk.data.path.insert(0, user_data_dir("threedont"))

def main():
    parser = argparse.ArgumentParser(description='3Dont')
    parser.add_argument('--test', action='store_true', help='Test the viewer')
    args = parser.parse_args()

    app_state = AppState("threedont")
    config = Config("threedont")

    controller = Controller(config, app_state)

    controller.run()
    print("Application stopped gracefully")


if __name__ == '__main__':
    main()
