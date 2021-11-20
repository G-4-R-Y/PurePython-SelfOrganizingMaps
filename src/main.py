import argparse

from load_csv import Data
from SOM import SOM

# Basic usage:
# python main.py --data_dir [relative_path] --map_height [map_height] --map_width [map_width] --dist [0/1/2]

# Examples:
# python main.py --map_width 3 --map_height 1
# python main.py --data_dir ../data/lung-cancer/lung-cancer.data --output_dir ../data/outputs/lung_cancer_outputs.csv --header False --map_width 3 --map_height 1

def main(args):
    data = Data(args.data_dir, args.header, args.index, args.normalize) 
    model = SOM(int(args.map_width), int(args.map_height), data.num_cols, int(args.dist))
    model.fit(data)
    outputs = model.predict(data.features)
    data.save_full_table(args.header, args.index, args.output_dir, outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', \
                    default='../data/wine/wine.data')
    parser.add_argument('--output_dir', \
                    default='../data/outputs/output_table_wine.csv')
    parser.add_argument('--header', \
                    default='True', \
                    help="True/False for first row being a header")
    parser.add_argument('--index', \
                    default='True', \
                    help="True/False for first column being index column")
    parser.add_argument('--normalize', \
                    default='True', \
                    help="True/False")
    parser.add_argument('--map_height', \
                    default='8', \
                    help="Map height")
    parser.add_argument('--map_width', \
                    default='16', \
                    help="Map width")
    parser.add_argument('--dist', \
                    default='2', \
                    help="0 - Supreme | 1 - Manhattan | 2 - Euclidean")
    args = parser.parse_args()
    main(args)