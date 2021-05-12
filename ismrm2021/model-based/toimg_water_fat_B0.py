#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Authors: 
# 2021 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
"""

import matplotlib.pyplot as plt

import os
import os.path
import sys

sys.path.insert(0, os.path.join(os.environ['TOOLBOX_PATH'], 'python'))

import cfl

import numpy as np

def toimg_water_fat_B0(cfl_file, png_file):

    R = np.squeeze(cfl.readcfl(cfl_file))

    water = np.absolute(R[:,:,0])
    fat = np.absolute(R[:,:,1])
    B0 = np.real(R[:,:,2])

    fig, axs = plt.subplots(figsize=(11,3), ncols=3, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0)

    axs[0].imshow(water, interpolation='none', cmap='gray')
    axs[1].imshow(fat, interpolation='none', cmap='gray')

    fig_B0 = axs[2].imshow(B0, interpolation='none', cmap='RdBu_r', vmin=-200, vmax=200)

    fig.colorbar(fig_B0, ax=axs, ticks=[-150,0,150], label='Hz', shrink=0.75)

    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[2].set_axis_off()

    plt.savefig(png_file, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":

    toimg_water_fat_B0(sys.argv[1], sys.argv[2])
