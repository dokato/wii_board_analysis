#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from obci.analysis.balance.wii_read_manager import WBBReadManager
from obci.analysis.balance.wii_preprocessing import *
from obci.analysis.balance.wii_analysis import *

 
import matplotlib.pyplot as py
py.style.use('ggplot')

from obci.exps.ventures.analysis import analysis_baseline
from obci.exps.ventures.analysis import analysis_helper
 
#from obci.acquisition import acquisition_helper
from utils.wiiboard_utils import *

FILE_PATH = 'dane/wii_mg/' #full path
FILE_NAME = 'wii_baseline_2016-03-02_14-37-22' #file name without extension

FILE_PATHd = 'dane/wii_dk/' #full path
FILE_NAMEd = 'wii_baseline_2016-03-02_15-08-09' #file name without extension
XSCALE = 22.5
YSCALE = 13

def read_file(file_path, file_name, tag_format = 'obci'):
    "From *filepath* and *filename* it returns WBBReadManager object"
    file_name = file_path + file_name
    wbb_mgr = WBBReadManager(file_name+'.obci.xml', file_name+'.obci.raw', file_name + '.' + tag_format + '.tag')
    return wbb_mgr

def read_wiidata(filepath, filename, tags_labels=None, show=False, supt=True):
    """
    It reads *filepath* and *filename* wii board data for two conditions
    specified by a labels in a format:
        tags_labels = [(tag1 start, tag1 stop), (tag2 start, tag2 stop)]
    if *tags_labels* is None default values will be given.
    if *show* is True, pictures of paths are shown.
    returns (x1,y1, x2, y2, wbb_mgr)
    where *x* and *y* are positions for respective condition and *wbb_mgr*
    is signal class WBBReadManager.
    """
    wbb_mgr = read_file(filepath, filename)
    if not tags_labels:
        tags_labels = [('ss_start','ss_stop'),('ss_oczy_start','ss_oczy_stop')]

    #add two additional (x, y) channels (computed from sensor values)
    wbb_mgr.get_x()
    wbb_mgr.get_y()
 
    #estimate true sampling frequency value
    fs = analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS'))
    wbb_mgr.mgr.set_param('sampling_frequency', analysis_baseline.estimate_fs(wbb_mgr.mgr.get_channel_samples('TSS')))
 
    #preprocessing
    wbb_mgr = wii_downsample_signal(wbb_mgr, factor=2, pre_filter=True, use_filtfilt=True)
 
    #extract fragments from standing task with eyes open
    # first subject
    smart_tags = wii_cut_fragments(wbb_mgr, start_tag_name=tags_labels[0][0],
                                            end_tags_names=[tags_labels[0][1]])
    sm_x = smart_tags[0].get_channel_samples('x')*XSCALE
    sm_y = smart_tags[0].get_channel_samples('y')*YSCALE

    smart_tags = wii_cut_fragments(wbb_mgr, start_tag_name=tags_labels[1][0],
                                            end_tags_names=[tags_labels[1][1]])
    sm_x_oz = smart_tags[0].get_channel_samples('x')*XSCALE
    sm_y_oz = smart_tags[0].get_channel_samples('y')*YSCALE
    if show:
        print(wii_COP_path(wbb_mgr, sm_x, sm_y, plot=True))
        py.show()
        print(wii_COP_path(wbb_mgr, sm_x_oz, sm_y_oz, plot=True))
        py.show()
    return sm_x, sm_y, sm_x_oz, sm_y_oz, wbb_mgr
 
def wii_plot(sm_x, sm_y, sm_x_oz, sm_y_oz, n_bins=25, subject_name='', savepic=False):
    """
    From specified time series it plots histograms of standing deviations
    
    *subject_name* - name of file which will be printed as a suptitle
    *savepic* - saves figure in main directory
    """
    mn_x1, mn_y1 = min(sm_x), min(sm_y)
    mx_x1, mx_y1 = max(sm_x), max(sm_y)
    mn_x2, mn_y2 = min(sm_x), min(sm_y)
    mx_x2, mx_y2 = max(sm_x), max(sm_y)
    
    full_max_x = max(mx_x1, mx_x2)
    full_max_y = max(mx_y1, mx_y2)
    full_min_x = min(mn_x1, mn_x2)
    full_min_y = min(mn_y1, mn_y2)

    bins_own = np.array([np.linspace(full_min_x, full_max_x, n_bins),
                         np.linspace(full_min_y, full_max_y, n_bins)])
    
    fig = py.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    py.hist2d(sm_x,sm_y,bins=bins_own)
    py.xlim([full_min_x, full_max_x])
    py.ylim([full_min_y, full_max_y])
    py.title("Oczy otwarte")
    py.xlabel('Ox')
    py.ylabel('Oy')
    ax2 = fig.add_subplot(212)
    py.hist2d(sm_x_oz,sm_y_oz,bins=bins_own)
    py.xlim([full_min_x, full_max_x])
    py.ylim([full_min_y, full_max_y])
    py.title("Oczy zamkniete")
    py.xlabel('Ox')
    py.ylabel('Oy')
    position=fig.add_axes([0.93,0.1,0.02,0.35])
    py.colorbar(cax=position)
    if len(subject_name)>0:
        py.suptitle('subject: {}'.format(subject_name))
    if savepic:
        py.savefig('lochist_{}.png'.format(subject_name))
    py.show()

def calculate_coef_wii(sm_x, sm_y, wbb_mgr, title=''):
    """
    It prints out report of most common measures of postural steadiness:
    - maximal sway
    - mean COP
    - path length
    - RMS
    - The 95% confidence ellipse area
    - mean velocity
    - regions of localization
    """
    max_sway, max_AP, max_ML = wii_max_sway_AP_MP(sm_x, sm_y)
    mean_COP, mean_x_COP, mean_y_COP = wii_mean_COP_sway_AP_ML(sm_x, sm_y)
    path_length, path_length_x, path_length_y = wii_COP_path(wbb_mgr, sm_x, sm_y, plot=False)
    RMS, RMS_AP, RMS_ML = wii_RMS_AP_ML(sm_x, sm_y)
    e = wii_confidence_ellipse_area(sm_x, sm_y)
    mean_velocity, velocity_AP, velocity_ML = wii_mean_velocity(wbb_mgr, sm_x, sm_y)
    top_right, top_left, bottom_right, bottom_left = wii_get_percentages_values(wbb_mgr, sm_x, sm_y, plot=False)
    print(title)
    print('Maximal sway: {}; AP: {}; ML: {}'.\
                format(max_sway, max_AP, max_ML))
    print('Mean COP: {}; Mean X COP: {}; Mean Y COP: {}'.\
                format(mean_COP, mean_x_COP, mean_y_COP))
    print('Path length: {}; Path length X: {}; Path length Y: {}'.\
                format(path_length, path_length_x, path_length_y))
    print('RMS: {}; RMS AP: {}; RMS ML: {}'.\
                format(RMS, RMS_AP, RMS_ML))
    print('The 95% confidence ellipse area: {}'.format(e))
    print('mean_velocity: {}; AP: {}; ML: {}'.\
                format(mean_velocity, velocity_AP, velocity_ML))
    print('top_right: {}; top_left: {}; bottom_right: {};bottom_left: {}'.\
                format(top_right, top_left, bottom_right, bottom_left ))
    
def romberg_coeff(sm_x, sm_y, sm_x_oz, sm_y_oz, wbb_mgr, q=True):
    """
    From given coordinates vectors *x* and *y* for two consitions
    it returns Romberg coefficient which is ratio of path during eyes
    closed to eyes open condition.
    
    q - if False then silent. 
    """
    path_length, _, _ = wii_COP_path(wbb_mgr, sm_x, sm_y, plot=False)
    path_length_oz, _, _ = wii_COP_path(wbb_mgr, sm_x_oz, sm_y_oz, plot=False)
    
    if q:
        print('Romberg measure: {}'.format(path_length_oz/path_length))
    return path_length_oz/path_length

if __name__ == '__main__':
    #load data
    subject_name = FILE_PATH.split('_')[-1][:-1]
    sm_x, sm_y, sm_x_oz, sm_y_oz, wbb_mgr = read_wiidata(FILE_PATH, FILE_NAME, show=0)
    wii_plot(sm_x, sm_y, sm_x_oz, sm_y_oz, subject_name=subject_name)
    max_sway, max_AP, max_ML = wii_max_sway_AP_MP(sm_x, sm_y)
    calculate_coef_wii(sm_x, sm_y, wbb_mgr, title='Eyes open')
    calculate_coef_wii(sm_x_oz, sm_y_oz, wbb_mgr, title='Eyes closed')
    romberg_coeff(sm_x, sm_y, sm_x_oz, sm_y_oz, wbb_mgr)
