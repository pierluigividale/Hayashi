import numpy as np
import xarray as xr
# our local module:
import sjw_wavenumber_frequency_functions as wf
import matplotlib as mpl
import matplotlib.pyplot as plt

def wf_analysis(x, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs: 
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq

    z2 = wf.spacetime_power(x, **kwargs)
    z2avg = z2.mean(dim='component')
    z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0
    # the background is supposed to be derived from both symmetric & antisymmetric
    background = wf.smooth_wavefreq(z2avg, kern=wf.simple_smooth_kernel(), nsmooth=50, freq_name='frequency')
    # separate components
    z2_sym = z2[0,...]
    z2_asy = z2[1,...]
    # normalize
    nspec_sym = z2_sym / background 
    nspec_asy = z2_asy / background
    return nspec_sym, nspec_asy


def plot_normalized_symmetric_spectrum(s, ofil=None, titleS=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots(figsize=(14,12))
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 3.0, 16), cmap='Spectral_r',  extend='both')
    for ii in range(3,6):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)    
    ax.set_title(titleS)
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=300)


def plot_normalized_asymmetric_spectrum(s, ofil=None, titleA=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""

    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots(figsize=(14,12))
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 1.8, 16), cmap='Spectral_r', extend='both')
    for ii in range(0,3):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)
    ax.set_title(titleA)
    fig.colorbar(img)
    #fig.suptitle("Hayashi (1971) diagram (aka W-K, 1999)")
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=300)

#
# LOAD DATA, x = DataArray(time, lat, lon), e.g., daily mean precipitation
#
def get_data(filename, variablename):
    try: 
        ds = xr.open_dataset(filename)
    except ValueError:
        ds = xr.open_dataset(filename, decode_times=False)
    
    return ds[variablename]


if __name__ == "__main__":
    #
    # input file -- could make this a CLI argument
    #
    #fili = "OLR.12hr_2yrs.wheeler.nc" 
    #vari = "olr"
    #fili = "u-ch330_OLR_TROPICS_3hr.nc" 
    #vari = "toa_outgoing_longwave_flux"
    #fili = "n2560_RAL3p3_rlut_6h.nc"
    #fili = "n2560_RAL3p3_rlut_6h_CLIPPED.nc"
    #model = "N1280_GAL9"
    #model = "N1280_CoMA9"
    #model = "N2560_RAL3p3"
    #model = "um_CTC_km4p4_RAL3P3_n1280_GAL9"
    #fili = model+"_rlut_latlon_CLIPPED.nc"
    #vari = "rlut"
    #print(fili)
    model = "ERA5"
    fili = model+"_rlut_latlon.nc"
    vari = "ttr"
    
    #
    # Loading data ... example is very simple
    #
    data = get_data(fili, vari)  # returns OLR
    print(data.values)
    data.values=np.nan_to_num(data.values)

    #
    # Options ... right now these only go into wk.spacetime_power()
    #
    latBound = (-15,15)  # latitude bounds for analysis
#    spd      = 8    # SAMPLES PER DAY 3-hourly
    spd      = 4    # SAMPLES PER DAY 6-hourly
    nDayWin  = 96   # Wheeler-Kiladis [WK] temporal window length (days)
    nDaySkip = -65  # time (days) between temporal windows [segments]
                    # negative means there will be overlapping temporal segments
    twoMonthOverlap = 65
    opt      = {'segsize': nDayWin, 
                'noverlap': twoMonthOverlap, 
                'spd': spd, 
                'latitude_bounds': latBound, 
                'dosymmetries': True, 
                'rmvLowFrq':True}
    #            'rmvLowFrq':False}
    # in this example, the smoothing & normalization will happen and use defaults
    symComponent, asymComponent = wf_analysis(data, **opt)
    #
    # Plots ... sort of matching NCL, but not worrying much about customizing.
    #
    #outPlotName = "example_symmetric_plot_N1280_6hr.png"
    #outPlotName = "example_symmetric_plv3hr.png"
    outPlotName = "Symmetric_"+model+"_6hr.png"
    titleS=model+" Hayashi (1971) wave diagram: normalized Symmetric Component"
    plot_normalized_symmetric_spectrum(symComponent, outPlotName, titleS)

    #outPlotName = "example_asymmetric_plot_N1280_6hr.png"
    #outPlotName = "example_asymmetric_plv3hr.png"
    #outPlotName = "example_asymmetric_plot_N2560_6hr.png"
    outPlotName = "Asymmetric_"+model+"_6hr.png"
    titleA=model+" Hayashi (1971) wave diagram: normalized Anti-symmetric Component"
    plot_normalized_asymmetric_spectrum(asymComponent, outPlotName, titleA)
    
