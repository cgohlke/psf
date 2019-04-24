Point Spread Function calculations for fluorescence microscopy
==============================================================

Psf is a Python library to calculate Point Spread Functions (PSF) for
fluorescence microscopy.

This library is no longer actively developed.

:Authors:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_,
  Oliver Holub

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: 3-clause BSD

:Version: 2019.4.22

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.11 <https://www.numpy.org>`_
* `Matplotlib 2.2 <https://www.matplotlib.org>`_  (optional for plotting)
* A Python distutils compatible C compiler  (build)

Revisions
---------
2019.4.22
    Fix setup requirements.
    Fix compiler warning.
2019.1.1
    Update copyright year.

References
----------
(1) Electromagnetic diffraction in optical systems. II. Structure of the
    image field in an aplanatic system.
    B Richards and E Wolf. Proc R Soc Lond A, 253 (1274), 358-379, 1959.
(2) Focal volume optics and experimental artifacts in confocal fluorescence
    correlation spectroscopy.
    S T Hess, W W Webb. Biophys J (83) 2300-17, 2002.
(3) Electromagnetic description of image formation in confocal fluorescence
    microscopy.
    T D Viser, S H Wiersma. J Opt Soc Am A (11) 599-608, 1994.
(4) Photon counting histogram: one-photon excitation.
    B Huang, T D Perroud, R N Zare. Chem Phys Chem (5), 1523-31, 2004.
    Supporting information: Calculation of the observation volume profile.
(5) Gaussian approximations of fluorescence microscope point-spread function
    models.
    B Zhang, J Zerubia, J C Olivo-Marin. Appl. Optics (46) 1819-29, 2007.
(6) The SVI-wiki on 3D microscopy, deconvolution, visualization and analysis.
    https://svi.nl/NyquistRate
(7) Theory of Confocal Microscopy: Resolution and Contrast in Confocal
    Microscopy. http://www.olympusfluoview.com/theory/resolutionintro.html

Examples
--------
>>> import psf
>>> args = dict(shape=(32, 32), dims=(4, 4), ex_wavelen=488, em_wavelen=520,
...             num_aperture=1.2, refr_index=1.333,
...             pinhole_radius=0.55, pinhole_shape='round')
>>> obsvol = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
>>> print('%.5f, %.5f' % obsvol.sigma.ou)
2.58832, 1.37059
>>> obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
>>> obsvol[0, :3]
array([ 1.     ,  0.51071,  0.04397])
>>> # save the image plane to file
>>> obsvol.slice(0).tofile('_test_slice.bin')
>>> # save a full 3D PSF volume to file
>>> obsvol.volume().tofile('_test_volume.bin')

Refer to the psf_example.py file in the source distribution for more.
