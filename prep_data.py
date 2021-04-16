import argparse, shutil, os, os.path as osp
import znn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action', type=str,
        choices=['reprocess', 'extrema', 'fromscratch'],
        )
    args = parser.parse_args()

    if args.action == 'fromscratch':
        if osp.isdir('data'): shutil.rmtree('data')
        znn.dataset.make_npzs_signal()
        znn.dataset.make_npzs_bkg()
        znn.dataset.ZPrimeDataset('data/train')
        znn.dataset.ZPrimeDataset('data/test')

    elif args.action == 'reprocess':
        if osp.isdir('data/train/processed'): shutil.rmtree('data/train/processed')
        if osp.isdir('data/test/processed'): shutil.rmtree('data/test/processed')
        znn.dataset.ZPrimeDataset('data/train')
        znn.dataset.ZPrimeDataset('data/test')

    elif args.action == 'extrema':
        znn.dataset.ZPrimeDataset('data/train').extrema()


if __name__ == '__main__':
    main()