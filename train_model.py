import argparse
import logging

from ml2.train import train_from_file

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True,
                        help="path to the CSV file containing the training set")
    args = parser.parse_args()
    train_from_file(args.file)
