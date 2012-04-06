#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
"""

from __future__ import division

__author__ = 'Maarten Versteegh'
__date__ = 'Fri Mar  2 15:34:27 2012'

from scikits.audiolab import wavread, wavwrite
import numpy as np
from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter
import os
import glob
import argparse

def rms(a):
    return np.sqrt(np.sum(a**2)/len(a))

def A_weighting_filter(fs):
    """construct an a-weighting filter at the specified samplerate
    from here: http://www.mathworks.com/matlabcentral/fileexchange/69
    """
    f1,f2,f3,f4,A1000 = 20.598997, 107.65265, 737.86223, 12194.217, 1.9997

    NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000/20)), 0, 0, 0, 0]
    DENs = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4) **2],
                       [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2], mode='full')
    DENs = np.convolve(np.convolve(DENs, [1, 2 * np.pi * f3], mode='full'),
                       [1, 2 * np.pi * f2], mode='full')

    return bilinear(NUMs, DENs, fs)

def A_weight(sig, fs):
    B, A = A_weighting_filter(fs)
    return lfilter(B,A,sig)

def analyze_dir(dirname):
    props = get_props_dir(dirname)
    for f in props:
        display_properties(f, props[f])

def display_props(props):
    for f in props:
        display_properties(f, props[f])

def get_props_dir(dirname):
    props = {}
    for f in glob.glob(os.path.join(dirname, '*.wav')):
        props[os.path.basename(f)] = analyze(*wavread(f))
    return props

def remove_DC_offset(sig):
    """DC offset fucks with the filters"""
    return sig - np.mean(sig)

def dB(level):
    return 20 * np.log10(level)

def analyze(sig, fs, enc):
    props = {}
    props['sig'] = sig
    props['fs'] = fs
    props['enc'] = enc
    props['dc'] = np.mean(sig)
    props['peak'] = np.max(np.abs(sig))
    props['rms'] = rms(sig)
    props['crest'] = props['peak']/props['rms']
    weighted_sig = A_weight(sig, fs)    
    props['weighted'] = rms(weighted_sig)
    return props

def display_properties(name, props):
    print '-'*20
    print 'Properties of %s' % name
    print 'Length:\t\t\t%.3fs' % (len(props['sig']) / props['fs'])
    print 'Sampling rate:\t\t%dkHz' % props['fs']
    print 'Encoding:\t\t%s' % props['enc']
    print 'Peak level:\t\t%.3f (%.3fdBFS)' % (props['peak'], dB(props['peak']))
    print 'RMS (unweighted):\t%.3f (%.3fdBFS)' % (props['rms'], dB(props['rms']))
    print 'RMS (A-weighted):\t%.3f (%.3fdBFS, %.3fdB)' % (props['weighted'], dB(props['weighted']), dB(props['weighted']/props['rms']))
    print 'Crest factor:\t\t%.3f (%.3fdB)' % (props['crest'], dB(props['crest']))
    print 'DC offset:\t\t%.3f' % props['dc']    
    print '-'*20

def match_dir(props, prop, outdir, ext):

    ref_prop = min(props.keys(), key=lambda x:props[x][prop])
    ref_peak = max(props.keys(),
                   key=lambda x:(props[x]['peak'] * props[ref_prop][prop] / props[x][prop]))
    for f in props:
        a = props[f]['sig'] * (props[ref_prop][prop] / (props[f][prop] * props[ref_peak]['peak']))
        bname = os.path.basename(f)
        wavwrite(a,
                 os.path.join(outdir, os.path.splitext(bname)[0] + ext + '.wav'),
                 props[f]['fs'],
                 props[f]['enc'])

def run():
    parser = argparse.ArgumentParser(prog='loudnessmatcher.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="""Match .wav files by loudness.
By default only displays loudness properties of files in directory.""",
                                     epilog="""Example usage:
                                     
$ ./loudness.py -s -m rms -o my_results

matches all the .wav files in the current directory and stores the output in "my_results/"
                                     """)
    parser.add_argument('-s','--silent',
                        action='store_true',
                        dest='silent',
                        help="don't display loudness properties of .wav files")
    parser.add_argument('-i',
                        nargs=1,
                        action='store',
                        default='.',
                        dest='indir', help='directory containing the .wav files to be analyzed')
    parser.add_argument('-o',
                        nargs=1,
                        action='store',
                        default=argparse.SUPPRESS,
                        dest='outdir',
                        help='destination of matched .wav files')
    parser.add_argument('-m','--match',
                        nargs=1,
                        action='store',
                        choices=['peak','rms','weighted'],
                        dest='match',
                        default=argparse.SUPPRESS,
                        help='match .wav files by a property. valid choices are "peak" (peak audio level), "rms" (root mean square in time domain) and "weighted" (a-weighted rms) [default]')
    parser.add_argument('-e', '--ext',
                        nargs=1,
                        action='store',
                        default='_matched',
                        dest='ext',
                        help='extension to add to matched filenames')
    options = vars(parser.parse_args())

    props = get_props_dir(options['indir'][0])
    if not 'outdir' in options:
        options['outdir'] = options['indir']
    if not os.path.exists(options['outdir'][0]):
            os.mkdir(options['outdir'][0])    
    if not options['silent']:
        display_props(props)
    if 'match' in options:
        match_dir(props, options['match'][0], options['outdir'][0], options['ext'][0])

if __name__ == '__main__':
    run()
