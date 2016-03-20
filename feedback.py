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
import pprint
import pandas as pd
import ast

PERSON = 'dk'
BASELINE_FILE_PATH = 'dane/wii_' + PERSON + '/' #full path
FILE_PATH = BASELINE_FILE_PATH  #full path

if PERSON == 'dk':
    BASELINE_FILE_NAME = 'new_tech_baseline_with_feedback_2016-03-02_15-14-50' #file name without extension
    FILE_NAME = 'new_tech_sway_stay_with_feedback_2016-03-02_15-16-12' #file name without extension
else:
    BASELINE_FILE_NAME = 'new_tech_baseline_with_feedback_2016-03-02_15-02-45' #file name without extension
    FILE_NAME = 'new_tech_sway_stay_with_feedback_2016-03-02_15-05-01' #file name without extension
    

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
        self.XSCALE = 22.5
        self.YSCALE = 13.
        self.lim_X = 20
        self.lim_Y = 20
        
    
    def get_baseline_points(self):
        """ baseline area is a rectangle with parameters:
            center point: xc, yc,
            width:        2*xa,
            height:       2*yb.
        """
        xa, ya, xb, yb, xc, yc  = analysis_baseline.calculate(self.filepath, self.baseline_name, show=False)
        return xa, ya, xb, yb, xc, yc
        
    def read_file(self, tag_format='obci'):
        "read file with Wiiboard data"
        file_name = self.filepath + self.filename
        wbb_mgr = WBBReadManager(file_name+'.obci.xml', file_name+'.obci.raw', file_name + '.' + tag_format + '.tag')
        return wbb_mgr
    
    def extract_signal_fragments(self):
        "extract signal *filename* data for given condition (direction)"
        #load data from right sway task
        wbb_mgr = self.read_file(tag_format='game')
        wbb_mgr.mgr = analysis_helper.set_first_timestamp(wbb_mgr.mgr) #adjusting tags to signal
        wbb_mgr.get_x()
        wbb_mgr.get_y()
        self.fs = analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS'))
        wbb_mgr.mgr.set_param('sampling_frequency', analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS')))
        #preprocessing
        wbb_mgr = wii_downsample_signal(wbb_mgr, factor=2, pre_filter=True, use_filtfilt=True)
        smart_tags = wii_cut_fragments(wbb_mgr, start_tag_name='start_1'
                                       , end_tags_names='finish')#, \
        df = pd.DataFrame.from_dict( [(idx, int(i.get_end_tag()['desc']['type']),
                   ast.literal_eval(i.get_start_tag()['desc']['type'])['direction'],
                   ast.literal_eval(i.get_start_tag()['desc']['type'])['level'],
                   )
                   for idx, i in enumerate(smart_tags) if int(i.get_end_tag()['desc']['type'])])
        df.columns = ['idx', 'is_done','direction','level']
        directions = (df.tail(4)['direction'].tolist(), df.tail(4)['idx'].tolist())
        print df
        print directions
        self.x = [self.XSCALE*i.get_channel_samples('x') for ind,i in enumerate(smart_tags)]
        self.y = [self.YSCALE*i.get_channel_samples('y') for ind,i in enumerate(smart_tags)]
        return zip(self.x, self.y), directions
        
        
    

    def max_sway(self):
        "maximal sway from given *direction* averaged over trials"
        max_sway, max_AP, max_ML = wii_max_sway_AP_MP(self.x, self.y)
        return max_sway, max_AP, max_ML
        
    def plot_movement(self, x, y, name, show=False):
        xa, _, _, yb, xc, yc = self.get_baseline_points()
        py.figure()
        time = np.linspace(0, len(x)/self.fs, len(x))
        py.suptitle(name)
        ax1 = py.subplot(221)
        ax1.set_ylim((-self.lim_Y,self.lim_Y))
        ax2 = py.subplot(223)
        ax2.set_ylim((-self.lim_Y,self.lim_Y))
        ax3 = py.subplot(122)
        ax3.set_xlim((-self.lim_X,self.lim_X))
        ax3.set_ylim((-self.lim_Y,self.lim_Y))
        ax1.plot(time, x)
        ax1.set_ylabel('position COPx [cm]')
        ax1.set_title('COPx position')
        ax2.plot(time, y)
        ax2.set_ylabel('position COPy [cm]')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('COPy position')
        ax3.plot(x,y)
        ax3.set_ylabel('position COPy [cm]')
        ax3.set_xlabel('position COPx [cm]')
        ax3.set_title('COP position')
        ax3.add_patch(patches.Rectangle(
                     (self.XSCALE * xc, self.YSCALE * yc),   # (x,y)
                      2*xa*self.XSCALE,          # width
                      2*yb*self.YSCALE,          # height
                      ))
        py.tight_layout()
        py.savefig('images/'+'feedback_trial_%s_%s' % (name,PERSON))
        py.clf()
        if show:
            py.show()
            
class OnePlot(WiiSway):
    def plot_movement(self, signals, properties):
        xa, _, _, yb, xc, yc = self.get_baseline_points()
        fig=py.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for i  in range(len(properties[0])):
            x = signals[directions[1][i]][0]
            y = signals[directions[1][i]][1]
            ax.plot(x, y, 'r')
            ax.set_ylabel('position COPy [cm]')
            ax.set_xlabel('position COPx [cm]')
            ax.set_xlim((-self.lim_X, self.lim_X))
            ax.set_ylim((-self.lim_Y, self.lim_Y))
        ax.add_patch(patches.Rectangle(
                     (self.XSCALE * xc, self.YSCALE * yc),   # (x,y)
                      2*xa*self.XSCALE,          # width
                      2*yb*self.YSCALE,          # height
                      ))
        ax.set_title(PERSON)
        py.tight_layout()
        py.show()
        #py.savefig('images/'+'feedback_allDir_%s_%s' % (name,PERSON))
        

if __name__ == '__main__':
    wiisway = WiiSway(FILE_PATH)
    signals, directions= wiisway.extract_signal_fragments()
    pp = pprint.PrettyPrinter(indent=4) 
    pp.pprint(directions)
    
    many = 0
    if many:
        for i in range(len(directions[0])):
            x = signals[directions[1][i]][0]
            y = signals[directions[1][i]][1]
            wiisway.plot_movement(x,y, directions[0][i], show=False)
    else:
        wiisway = OnePlot(FILE_PATH)
        signals, directions= wiisway.extract_signal_fragments()
        wiisway.plot_movement(signals, directions)

        
    


