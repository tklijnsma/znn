import os.path as osp, glob, numpy as np, sys, os, glob
import torch
from torch_geometric.data import (Data, Dataset)
import tqdm
import json
import uptools
import seutils

import random
random.seed(1001)

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
            random.shuffle(self.processed_files)
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, i):
        # print('Loading', self.processed_dir+'/'+self.processed_files[i])
        data = torch.load(self.processed_dir+'/'+self.processed_files[i])
        return data
    
    def process(self):
        max_pt = -1e6
        max_eta = -1e6
        max_phi = -1e6
        max_energy = -1e6
        for i, f in tqdm.tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            d = np.load(self.raw_dir + '/' + f)
            is_bkg = f.startswith('qcd')
            x = np.stack((
                torch.from_numpy(d['pt']),
                torch.from_numpy(d['eta']),
                torch.from_numpy(d['phi']),
                torch.from_numpy(d['energy']),
                )).T
            data = Data(
                x = torch.from_numpy(x),
                y = torch.tensor([0. if is_bkg else 1.])
                )
            torch.save(data, self.processed_dir + f'/data_{i}.pt')
            max_pt = max(np.max(np.abs(d['pt'])), max_pt)
            max_eta = max(np.max(np.abs(d['eta'])), max_eta)
            max_phi = max(np.max(np.abs(d['phi'])), max_phi)
            max_energy = max(np.max(np.abs(d['energy'])), max_energy)
        print('Max dims:')
        print('pt:', max_pt)
        print('eta:', max_eta)
        print('phi:', max_phi)
        print('energy:', max_energy)


def ntup_to_npz_signal(event, outfile):
    select_zjet = event[b'ak15GenJetsPackedNoNu_energyFromZ'].argmax()
    zjet = uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu', event, branches=[b'energyFromZ'])[select_zjet]
    if zjet.energyFromZ / zjet.energy < 0.01:
        print('Skipping event: zjet.energyFromZ / zjet.energy = ', zjet.energyFromZ / zjet.energy)
        return
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
        y = 1.
        )

def ntup_to_npz_bkg(event, outfile):
    '''
    Just dumps the two leading jets to outfiles
    '''
    all_constituents = (
        uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu_constituents', event)
        .flatten()
        .unflatten(event[b'ak15GenJetsPackedNoNu_nConstituents'])
        )
    for title, constituents in [ ('leading', all_constituents[0]), ('subleading', all_constituents[1]) ]:
        constituents = constituents.flatten()
        if not osp.isdir(osp.dirname(outfile)): os.makedirs(osp.dirname(outfile))
        np.savez(
            outfile.replace('.npz', '_'+ title + '.npz'),
            pt = constituents.pt,
            eta = constituents.eta,
            phi = constituents.phi,
            energy = constituents.energy,
            y = 0.
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

def make_npzs_bkg(N=5000):
    for i_event, event in tqdm.tqdm(enumerate(iter_events_qcd(N)), total=N):
        ntup_to_npz_bkg(event, f'data/raw/qcd/{i_event}.npz')


def make_npzs_signal(N=5000):
    signal = seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/boosted/ecfntuples/'
        'Feb23_mz150_rinv0.1_mdark10/*.root'
        )
    outdir = 'data/raw/' + osp.basename(osp.dirname(signal[0]))
    for i_event, event in tqdm.tqdm(
        enumerate(uptools.iter_events(signal, treepath='gensubntupler/tree', nmax=N)),
        total=N):
        ntup_to_npz_signal(event, outdir + f'/{i_event}.npz')

def main():
    make_npzs_signal()
    make_npzs_bkg()

if __name__ == '__main__':
    main()