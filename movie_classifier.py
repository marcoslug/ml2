import argparse
import logging

from ml2.predict import predict_genre

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--title", type=str, required=True, help="title of the movie")
    parser.add_argument("-d", "--description", type=str, required=True, help="short description of the movie")
    args = parser.parse_args()
    response = predict_genre(args.title, args.description)
    print(response)
