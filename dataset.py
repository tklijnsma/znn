import warnings
warnings.filterwarnings("ignore")

import os.path as osp, glob, numpy as np, sys, os, glob
import torch
from torch_geometric.data import (Data, Dataset)
import tqdm
import json
import uptools
import seutils
from math import pi

import random
random.seed(1001)



class ExtremaRecorder():
    '''
    Records the min and max of a set of values
    Doesn't store the values unless the standard deviation is requested too
    '''
    def __init__(self, do_std=True):
        self.min = 1e6
        self.max = -1e6
        self.mean = 0.
        self.n = 0
        self.do_std = do_std
        if self.do_std: self.values = np.array([])

    def update(self, values):
        self.min = min(self.min, np.min(values))
        self.max = max(self.max, np.max(values))
        n_new = self.n + len(values)
        self.mean = (self.mean * self.n + np.sum(values)) / n_new
        self.n = n_new
        if self.do_std: self.values = np.concatenate((self.values, values))

    def std(self):
        if self.do_std:
            return np.std(self.values)
        else:
            return 0.

    def __repr__(self):
        return (
            '{self.min:+7.3f} to {self.max:+7.3f}, mean={self.mean:+7.3f}{std} ({self.n})'
            .format(
                self=self,
                std='+-{:.3f}'.format(self.std()) if self.do_std else ''
                )
            )

    def hist(self, outfile):
        import matplotlib.pyplot as plt
        figure = plt.figure()
        ax = figure.gca()
        ax.hist(self.values, bins=25)
        plt.savefig(outfile, bbox_inches='tight')


class ZPrimeDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def download(self):
        pass
    
    @property
    def raw_file_names(self):
        if not hasattr(self, '_raw_file_names'):
            self._raw_file_names = [osp.relpath(f, self.raw_dir) for f in glob.iglob(self.raw_dir + '/*/*.npz')]
            self._raw_file_names.sort()
        return self._raw_file_names
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            self.processed_files = [ f'data_{i}.pt' for i in range(len(self.raw_file_names)) ]
            self.unshuffled_processed_files = self.processed_files[:]
            random.shuffle(self.processed_files)
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, i):
        # print('Loading', self.processed_dir+'/'+self.processed_files[i])
        data = torch.load(self.processed_dir+'/'+self.processed_files[i])
        return data

    def get_original_features(self, i):
        procfile = self.processed_files[i]
        i_unshuffled = self.unshuffled_processed_files.index(procfile)
        npzfile = self.raw_dir + '/' + self.raw_file_names[i_unshuffled]
        return np.load(npzfile)

    def npz_to_features(self, npz_file):
        d = np.load(npz_file)
        deta = d['eta'] - d['jet_eta']
        deta /= 2.

        dphi = d['phi'] - d['jet_phi']
        dphi %= 2.*pi # Map to 0..2pi range
        dphi[dphi > pi] = dphi[dphi > pi] - 2.*pi # Map >pi to -pi..0 range
        dphi /= 2. # Normalize to approximately -1..1 range
                   # (jet size is 1.5, but some things extend up to 2.)

        # Normalizations kind of guestimated so that 2 standard deviations are within 0..1
        pt = d['pt'] / 30.
        energy = d['energy'] / 100.

        return pt, deta, dphi, energy
    
    def process(self):
        for i, f in tqdm.tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            is_bkg = f.startswith('qcd')
            pt, deta, dphi, energy = self.npz_to_features(self.raw_dir + '/' + f)
            x = np.stack((pt,deta,dphi,energy)).T
            data = Data(
                x = torch.from_numpy(x),
                y = torch.LongTensor([0 if is_bkg else 1])
                )
            torch.save(data, self.processed_dir + f'/data_{i}.pt')

    def extrema(self):
        ext_pt = ExtremaRecorder()
        ext_eta = ExtremaRecorder()
        ext_phi = ExtremaRecorder()
        ext_energy = ExtremaRecorder()
        for i, f in tqdm.tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            is_bkg = f.startswith('qcd')
            pt, deta, dphi, energy = self.npz_to_features(self.raw_dir + '/' + f)
            ext_pt.update(pt)
            ext_eta.update(deta)
            ext_phi.update(dphi)
            ext_energy.update(energy)
        print('Max dims:')
        print('pt:    ', ext_pt)
        print('eta:   ', ext_eta)
        print('phi:   ', ext_phi)
        print('energy:', ext_energy)
        ext_pt.hist('extrema_pt.png')
        ext_eta.hist('extrema_eta.png')
        ext_phi.hist('extrema_phi.png')
        ext_energy.hist('extrema_energy.png')


def ntup_to_npz_signal(event, outfile):
    select_zjet = event[b'ak15GenJetsPackedNoNu_energyFromZ'].argmax()
    zjet = uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu', event, branches=[b'energyFromZ'])[select_zjet]
    if zjet.energyFromZ / zjet.energy < 0.01:
        # print('Skipping event: zjet.energyFromZ / zjet.energy = ', zjet.energyFromZ / zjet.energy)
        return False
    constituents = (
        uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu_constituents', event, b'isfromz')
        .flatten()
        .unflatten(event[b'ak15GenJetsPackedNoNu_nConstituents'])
        )[select_zjet]
    constituents = constituents.flatten()
    if not osp.isdir(osp.dirname(outfile)): os.makedirs(osp.dirname(outfile))
    np.savez(
        outfile,
        pt = constituents.pt,
        eta = constituents.eta,
        phi = constituents.phi,
        energy = constituents.energy,
        y = 1,
        jet_pt = zjet.pt,
        jet_eta = zjet.eta,
        jet_phi = zjet.phi,
        jet_energy = zjet.energy,        
        )
    return True

def ntup_to_npz_bkg(event, outfile):
    '''
    Just dumps the two leading jets to outfiles
    '''
    jets = uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu', event)
    constituents = (
        uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu_constituents', event)
        .flatten()
        .unflatten(event[b'ak15GenJetsPackedNoNu_nConstituents'])
        )
    for i in [0, 1]:
        # Do leading and subleading
        this_jet = jets[i]
        this_constituents = constituents[i].flatten()
        title = ['leading', 'subleading'][i]
        if not osp.isdir(osp.dirname(outfile)): os.makedirs(osp.dirname(outfile))
        np.savez(
            outfile.replace('.npz', '_'+ title + '.npz'),
            pt = this_constituents.pt,
            eta = this_constituents.eta,
            phi = this_constituents.phi,
            energy = this_constituents.energy,
            y = 0,
            jet_pt = this_jet.pt,
            jet_eta = this_jet.eta,
            jet_phi = this_jet.phi,
            jet_energy = this_jet.energy,
            )

def iter_arrays_qcd(N):
    xss = [0.23*7025.0, 620.6, 59.06, 18.21, 7.5, 0.6479, 0.08715, 0.005242, 0.0001349, 3.276]
    ntuple_path = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/boosted/ecfntuples/'
    samples = [
        'Feb15_qcd_BTV-RunIIFall18GS-00024_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00025_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00026_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00027_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00029_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00030_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00031_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00032_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00033_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00051_1_cfg',
        ]
    rootfiles = [ seutils.ls_wildcard(ntuple_path+s+'/*.root') for s in samples ]
    yield from uptools.iter_arrays_weighted(N, xss, rootfiles, treepath='gensubntupler/tree')

def iter_events_qcd(N):
    for arrays in iter_arrays_qcd(N):
        for i in range(uptools.numentries(arrays)):
            yield uptools.get_event(arrays, i)

def make_npzs_bkg(N=8000):
    train_outdir = 'data/train/raw/qcd'
    test_outdir = 'data/test/raw/qcd'
    for i_event, event in tqdm.tqdm(enumerate(iter_events_qcd(N)), total=N):
        outdir = test_outdir if i_event % 5 == 0 else train_outdir
        ntup_to_npz_bkg(event, outdir + f'{i_event}.npz')


def make_npzs_signal():
    directory = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/boosted/ecfntuples/Mar16_mz150_rinv0.1_mdark10'
    N = seutils.root.sum_dataset(directory, treepath='gensubntupler/tree')
    signal = seutils.ls_wildcard(directory + '/*.root')
    signal_name = osp.basename(osp.dirname(signal[0]))
    train_outdir = 'data/train/raw/' + signal_name
    test_outdir = 'data/test/raw/' + signal_name
    n_total = 0
    i_event_good = 0
    for i_event, event in tqdm.tqdm(
        enumerate(uptools.iter_events(signal, treepath='gensubntupler/tree', nmax=N)),
        total=N
        ):
        outdir = test_outdir if i_event_good % 5 == 0 else train_outdir
        succeeded = ntup_to_npz_signal(event, outdir + f'/{i_event_good}.npz')
        if succeeded: i_event_good += 1
        n_total += 1
    print(
        'Total events turned into npzs: {}/{}  ({} failures due to 0 Z-energy)'
        .format(i_event_good, n_total, n_total-i_event_good)
        )


def main():
    import argparse, shutil
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action', type=str,
        choices=['reprocess', 'extrema', 'fromscratch'],
        )
    args = parser.parse_args()

    if args.action == 'fromscratch':
        if osp.isdir('data'): shutil.rmtree('data')
        make_npzs_signal()
        make_npzs_bkg()
        ZPrimeDataset('data/train')
        ZPrimeDataset('data/test')

    elif args.action == 'reprocess':
        if osp.isdir('data/processed'): shutil.rmtree('data/processed')
        ZPrimeDataset('data/train')
        ZPrimeDataset('data/test')

    elif args.action == 'extrema':
        ZPrimeDataset('data/train').extrema()


if __name__ == '__main__':
    main()