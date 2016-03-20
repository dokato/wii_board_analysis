#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from obci.analysis.balance.wii_read_manager import WBBReadManager
from obci.analysis.balance.wii_preprocessing import *
from obci.analysis.balance.wii_analysis import *

from obci.exps.ventures.analysis import analysis_baseline
from obci.exps.ventures.analysis import analysis_helper
 
#from obci.acquisition import acquisition_helper
import numpy as np
import matplotlib.pyplot as py
py.style.use('ggplot')

PERSON = 'mg'
FILE_PATH = 'dane/wii_' + PERSON + '/' #full path

if PERSON == 'dk':
    FILE_NAME_LEFT = 'wii_without_feedback_left_2016-03-02_15-13-46' #file name without extension 
    FILE_NAME_RIGHT = 'wii_without_feedback_right_2016-03-02_15-11-32' #file name without extension 
    FILE_NAME_DOWN = 'wii_without_feedback_down_2016-03-02_15-12-40' #file name without extension 
    FILE_NAME_UP = 'wii_without_feedback_up_2016-03-02_15-10-20' #file name without extension
else:
    FILE_NAME_LEFT = 'wii_without_feedback_left_2016-03-02_14-46-45' #file name without extension 
    FILE_NAME_RIGHT = 'wii_without_feedback_right_2016-03-02_14-44-03' #file name without extension 
    FILE_NAME_DOWN = 'wii_without_feedback_down_2016-03-02_14-45-29' #file name without extension 
    FILE_NAME_UP = 'wii_without_feedback_up_2016-03-02_14-40-47' #file name without extension

XSCALE = 22.5
YSCALE = 13.

def read_file(file_path, file_name, tag_format='obci'):
    file_name = file_path + file_name
    wbb_mgr = WBBReadManager(file_name+'.obci.xml', file_name+'.obci.raw', file_name + '.' + tag_format + '.tag')
    return wbb_mgr

class WiiSway(object):
    """
    Class to manage data from sway Wii Board experiment .
    """
    def __init__(self, filepath, tagnames=None):
        """
        *filepath* - path to data files
        *tagnames* - dictionary with tags for quick and long conditions
        """
        self.filepath = filepath
        if tagnames:
            self.tagnames = tagnames
        else:
            self.tags_labels = {'quick': ['szybkie_start', 'szybkie_stop'],
                           'long' : ['start', 'stop']}
                        
    
    def extract_signal_fragments(self, filename):
        "extract signal *filename* data for given condition (direction)"
        #load data from right sway task
        wbb_mgr = read_file(self.filepath, filename)
        wbb_mgr.get_x()
        wbb_mgr.get_y()
        self.fs = analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS'))
        wbb_mgr.mgr.set_param('sampling_frequency', analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS')))
        #preprocessing
        wbb_mgr = wii_downsample_signal(wbb_mgr, factor=2, pre_filter=True, use_filtfilt=True)

        #extract fragments from sway task (right)
        smart_tags_quick = wii_cut_fragments(wbb_mgr, start_tag_name=self.tags_labels['quick'][0],
                                                      end_tags_names=[self.tags_labels['quick'][1]])
        #sig_fragments_quick = [i.get_samples() for i in smart_tags_quick]
        x_quick = [XSCALE*i.get_channel_samples('x') for i in smart_tags_quick]
        y_quick = [YSCALE*i.get_channel_samples('y') for i in smart_tags_quick]
        #extract fragments from sway&stay task (right)
        smart_tags_long = wii_cut_fragments(wbb_mgr, start_tag_name=self.tags_labels['long'][0],
                                                     end_tags_names=[self.tags_labels['long'][0]])
        #sig_fragments_long = [i.get_samples() for i in smart_tags_long]
        x_long = [XSCALE*i.get_channel_samples('x') for i in smart_tags_quick]
        y_long = [YSCALE*i.get_channel_samples('y') for i in smart_tags_quick]
        self.N = len(x_quick)
        return wbb_mgr, x_quick, y_quick, x_long, y_long
    
    def add_right(self, filename):
        "add right condition"
        wbb, x_q, y_q, x_l, y_l = self.extract_signal_fragments(filename)        
        self.right_x_quick = x_q
        self.right_y_quick = y_q
        self.right_x_long  = x_l
        self.right_y_long  = y_l
        self.right_wbb_mgr = wbb

    def add_left(self, filename):
        "add left condition"
        wbb, x_q, y_q, x_l, y_l = self.extract_signal_fragments(filename)        
        self.left_x_quick = x_q
        self.left_y_quick = y_q
        self.left_x_long  = x_l
        self.left_y_long  = y_l
        self.left_wbb_mgr = wbb

    def add_up(self, filename):
        "add up condition"
        wbb, x_q, y_q, x_l, y_l = self.extract_signal_fragments(filename)        
        self.up_x_quick = x_q
        self.up_y_quick = y_q
        self.up_x_long  = x_l
        self.up_y_long  = y_l
        self.up_wbb_mgr = wbb

    def add_down(self, filename):
        "add down condition"
        wbb, x_q, y_q, x_l, y_l = self.extract_signal_fragments(filename)        
        self.down_x_quick = x_q
        self.down_y_quick = y_q
        self.down_x_long  = x_l
        self.down_y_long  = y_l
        self.down_wbb_mgr = wbb

    def max_sway_quick(self, direction):
        "maximal sway from given *direction*"
        sm_x = self.__dict__["{}_x_quick".format(direction)]
        sm_y = self.__dict__["{}_y_quick".format(direction)]
        x_sway = np.zeros(self.N)
        y_sway = np.zeros(self.N)
        sway = np.zeros(self.N)
        for i in range(self.N): 
            max_sway, max_AP, max_ML = wii_max_sway_AP_MP(sm_x[i], sm_y[i])
            x_sway[i] = max_AP
            y_sway[i] = max_ML
            sway[i] = max_sway
        self.__dict__["{}_best_idx".format(direction)] = np.argmax(sway)
        return np.max(sway)

    def plot_best_movement(self, direction, show=False):
        "Show or save COP pictures of data for given *direction*"
        idx = self.__dict__["{}_best_idx".format(direction)]
        sm_x = self.__dict__["{}_x_quick".format(direction)][idx]
        sm_y = self.__dict__["{}_y_quick".format(direction)][idx]
        py.figure()
        time = np.linspace(0, len(sm_x)/self.fs, len(sm_x))
        ax1 = py.subplot(221)
        ax2 = py.subplot(223)
        ax3 = py.subplot(122)
        ax1.plot(time, sm_x)
        ax1.set_ylabel('position COPx [cm]')
        ax1.set_title('COPx position')
        ax2.plot(time, sm_y)
        ax2.set_ylabel('position COPy [cm]')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('COPy position')
        ax3.plot(sm_x,sm_y)
        ax3.set_ylabel('position COPy [cm]')
        ax3.set_xlabel('position COPx [cm]')
        ax3.set_title('COP position')
        py.tight_layout()
        py.savefig('images/nofeedback_best_trial_' + direction)
        if show:
            py.show()
        py.clf()
    
    def plot_movement(self, direction, show=False):
        "Show or save COP pictures of data for given *direction*"
        sm_x = self.__dict__["{}_x_quick".format(direction)]
        sm_y = self.__dict__["{}_y_quick".format(direction)]
        py.figure()
        for i in range(self.N):
            time = np.linspace(0, len(sm_x[i])/self.fs, len(sm_x[i]))
            ax1 = py.subplot(221)
            ax2 = py.subplot(223)
            ax3 = py.subplot(122)
            ax1.plot(time, sm_x[i])
            ax1.set_ylabel('position COPx [cm]')
            ax1.set_title('COPx position')
            ax2.plot(time, sm_y[i])
            ax2.set_ylabel('position COPy [cm]')
            ax2.set_xlabel('Time [s]')
            ax2.set_title('COPy position')
            ax3.plot(sm_x[i],sm_y[i])
            ax3.set_ylabel('position COPy [cm]')
            ax3.set_xlabel('position COPx [cm]')
            ax3.set_title('COP position')
            py.tight_layout()
            py.savefig('images/' + direction + '_nofeedback_trial%d' % (i+1,))
            if show:
                py.show()
            py.clf()

if __name__ == '__main__':
    wiisway = WiiSway(FILE_PATH)
    wiisway.add_right(FILE_NAME_RIGHT)
    wiisway.add_left(FILE_NAME_LEFT)
    wiisway.add_down(FILE_NAME_DOWN)
    wiisway.add_up(FILE_NAME_UP)
    for i in ["left", "right", "up", "down"]:
        print("Max sway {}: {}".format(i, wiisway.max_sway_quick(i)))
        wiisway.plot_best_movement(i)
