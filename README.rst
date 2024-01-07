Point Spread Function calculations for fluorescence microscopy
==============================================================

Psf is a Python library to calculate Point Spread Functions (PSF) for
fluorescence microscopy.

The psf library is no longer actively developed.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.1.6

Quickstart
----------

Install the psf package and all dependencies from the
`Python Package Index <https://pypi.org/project/psf/>`_::

    python -m pip install -U psf[all]

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/psf>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.7, 3.12.1
- `NumPy <https://pypi.org/project/numpy/>`_ 1.26.3
- `Matplotlib <https://pypi.org/project/matplotlib/>`_  3.8.2
  (optional for plotting)

Revisions
---------

2024.1.6

- Change PSF.TYPES from dict to set (breaking).

2023.4.26

- Use enums.
- Derive Dimensions from UserDict.
- Add type hints.
- Convert to Google style docstrings.
- Drop support for Python 3.8 and numpy < 1.21 (NEP29).

2022.9.26

- Fix setup.py.

2022.9.12

- Remove support for Python 3.7 (NEP 29).
- Update metadata.

2021.6.6

- Remove support for Python 3.6 (NEP 29).

2020.1.1

- Remove support for Python 2.7 and 3.5.
- Update copyright.

2019.10.14

- Support Python 3.8.

2019.4.22

- Fix setup requirements.
- Fix compiler warning.

References
----------

1. Electromagnetic diffraction in optical systems. II. Structure of the
   image field in an aplanatic system.
   B Richards and E Wolf. Proc R Soc Lond A, 253 (1274), 358-379, 1959.
2. Focal volume optics and experimental artifacts in confocal fluorescence
   correlation spectroscopy.
   S T Hess, W W Webb. Biophys J (83) 2300-17, 2002.
3. Electromagnetic description of image formation in confocal fluorescence
   microscopy.
   T D Viser, S H Wiersma. J Opt Soc Am A (11) 599-608, 1994.
4. Photon counting histogram: one-photon excitation.
   B Huang, T D Perroud, R N Zare. Chem Phys Chem (5), 1523-31, 2004.
   Supporting information: Calculation of the observation volume profile.
5. Gaussian approximations of fluorescence microscope point-spread function
   models.
   B Zhang, J Zerubia, J C Olivo-Marin. Appl. Optics (46) 1819-29, 2007.
6. The SVI-wiki on 3D microscopy, deconvolution, visualization and analysis.
   https://svi.nl/NyquistRate

Examples
--------

>>> import psf
>>> args = dict(
...     shape=(32, 32),
...     dims=(4, 4),
...     ex_wavelen=488,
...     em_wavelen=520,
...     num_aperture=1.2,
...     refr_index=1.333,
...     pinhole_radius=0.55,
...     pinhole_shape='round'
... )
>>> obsvol = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
>>> obsvol.sigma.ou
(2.588..., 1.370...)
>>> obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
>>> print(obsvol, end='')
PSF
 ISOTROPIC|CONFOCAL
 shape: (32, 32) pixel
 dimensions: (4.00, 4.00) um, (55.64, 61.80) ou, (8.06, 8.06) au
 excitation wavelength: 488.0 nm
 emission wavelength: 520.0 nm
 numeric aperture: 1.20
 refractive index: 1.33
 half cone angle: 64.19 deg
 magnification: 1.00
 underfilling: 1.00
 pinhole radius: 0.550 um, 8.498 ou, 1.1086 au, 4.40 px
 computing time: ... ms
>>> obsvol[0, :3]
array([1.     , 0.51071, 0.04397])
>>> # write the image plane to file
>>> obsvol.slice(0).tofile('_test_slice.bin')
>>> # write a full 3D PSF volume to file
>>> obsvol.volume().tofile('_test_volume.bin')

Refer to `psf_example.py` in the source distribution for more examples.
