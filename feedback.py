#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from obci.analysis.balance.wii_read_manager import WBBReadManager
from obci.analysis.balance.wii_preprocessing import *
from obci.analysis.balance.wii_analysis import *

from obci.exps.ventures.analysis import analysis_baseline
from obci.exps.ventures.analysis import analysis_helper
 
#from obci.acquisition import acquisition_helper
import numpy as np
import matplotlib.patches as patches

BASELINE_FILE_PATH = 'dane/wii_dk/' #full path
BASELINE_FILE_NAME = 'new_tech_baseline_with_feedback_2016-03-02_15-14-50' #file name without extension
 
FILE_PATH = BASELINE_FILE_PATH  #full path
FILE_NAME = 'new_tech_sway_stay_with_feedback_2016-03-02_15-16-12' #file name without extension

py.style.use('ggplot')

class WiiSway(object):
    """
    Class to manage data from sway Wii Board experiment .
    """
    def __init__(self, filepath, baseline_name=BASELINE_FILE_NAME, filename=FILE_NAME):
        """
        *filepath* - path to data files
        *baseline_name* - baseline_file name
        """
        self.filepath = filepath
        self.baseline_name = baseline_name
        self.filename = filename
        
    
    def get_baseline_points(self):
        """ baseline area is a rectangle with parameters:
            center point: xc, yc,
            width:        2*xa,
            height:       2*yb.
        """
        xa, ya, xb, yb, xc, yc  = analysis_baseline.calculate(self.filepath, self.baseline_name, show=False)
        return xa, ya, xb, yb, xc, yc
        
    def read_file(self, tag_format='obci'):
        file_name = self.filepath + self.filename
        wbb_mgr = WBBReadManager(file_name+'.obci.xml', file_name+'.obci.raw', file_name + '.' + tag_format + '.tag')
        return wbb_mgr
    
    def extract_signal_fragments(self):
        "extract signal *filename* data for given condition (direction)"
        #load data from right sway task
        wbb_mgr = self.read_file()
        wbb_mgr.mgr = analysis_helper.set_first_timestamp(wbb_mgr.mgr) #adjusting tags to signal
        wbb_mgr.get_x()
        wbb_mgr.get_y()
        self.fs = analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS'))
        wbb_mgr.mgr.set_param('sampling_frequency', analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS')))
        #preprocessing
        wbb_mgr = wii_downsample_signal(wbb_mgr, factor=2, pre_filter=True, use_filtfilt=True)
        
        self.x = wbb_mgr.mgr.get_channel_samples('x')
        self.y = wbb_mgr.mgr.get_channel_samples('y')
        
        return self.x, self.y
    

    def max_sway(self):
        "maximal sway from given *direction* averaged over trials"
        max_sway, max_AP, max_ML = wii_max_sway_AP_MP(self.x, self.y)
        return max_sway, max_AP, max_ML
        
    def plot_movement(self, show=True):
        xa, _, _, yb, xc, yc = self.get_baseline_points()
        py.figure()
        time = np.linspace(0, len(self.x)/self.fs, len(self.x))
        ax1 = py.subplot(221)
        ax2 = py.subplot(223)
        ax3 = py.subplot(122)
        ax1.plot(time, self.x)
        ax1.set_ylabel('position COPx [cm]')
        ax1.set_title('COPx position')
        ax2.plot(time, self.y)
        ax2.set_ylabel('position COPy [cm]')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('COPy position')
        ax3.plot(self.x,self.y)
        ax3.set_ylabel('position COPy [cm]')
        ax3.set_xlabel('position COPx [cm]')
        ax3.set_title('COP position')
        ax3.add_patch(patches.Rectangle(
                     (xc, yc),   # (x,y)
                      2*xa,          # width
                      2*yb,          # height
                      ))
        py.tight_layout()
        #py.savefig('images/Feedback')
        if show:
            py.show()
        py.clf()

if __name__ == '__main__':
    wiisway = WiiSway(FILE_PATH)
    x, y = wiisway.extract_signal_fragments()
    print wiisway.max_sway()
    wiisway.plot_movement()

