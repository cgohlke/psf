# psf.py

# Copyright (c) 2007-2026, Christoph Gohlke and Oliver Holub
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Point Spread Function calculations for fluorescence microscopy.

Psf is a Python library to calculate Point Spread Functions (PSF) for
fluorescence microscopy.

The psf library is no longer actively developed.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.1.18

Quickstart
----------

Install the psf package and all dependencies from the
`Python Package Index <https://pypi.org/project/psf/>`_::

    python -m pip install -U "psf[all]"

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/psf>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.11.9, 3.12.10, 3.13.11, 3.14.2 64-bit
- `NumPy <https://pypi.org/project/numpy/>`_ 2.4.1
- `Matplotlib <https://pypi.org/project/matplotlib/>`_  3.10.8
  (optional for plotting)

Revisions
---------

2026.1.18

- Use multi-phase initialization.
- Improve code quality.

2025.8.1

- Drop support for Python 3.10, support Python 3.14.

2025.1.1

- Improve type hints.
- Drop support for Python 3.9, support Python 3.13.

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.

2024.4.24

- Support NumPy 2.

2024.1.6

- …

Refer to the CHANGES file for older revisions.

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
...     pinhole_shape='round',
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

"""

from __future__ import annotations

__version__ = '2026.1.18'

__all__ = [
    'ANISOTROPIC',
    'CONFOCAL',
    'EMISSION',
    'EXCITATION',
    'GAUSSIAN',
    'GAUSSLORENTZ',
    'ISOTROPIC',
    'PARAXIAL',
    'PSF',
    'TWOPHOTON',
    'WIDEFIELD',
    'Dimensions',
    'Pinhole',
    'PinholeShape',
    'PsfType',
    '__version__',
    'imshow',
    'mirror_symmetry',
    'uv2zr',
    'zr2uv',
]

import contextlib
import enum
import math
import threading
import time
from collections import UserDict
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any, ClassVar, TypeVar

    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from numpy.typing import ArrayLike, NDArray

    DimensionT = TypeVar('DimensionT')
else:
    DimensionT = None


_psf: Any
try:
    from . import _psf  # type: ignore[no-redef, attr-defined]
except ImportError:
    import _psf  # type: ignore[no-redef]


class PsfType(enum.IntFlag):
    """Type of PSF defined by combining one model and one type."""

    ANISOTROPIC = 1
    """Anisotropic model (not implemented)."""
    ISOTROPIC = 2
    """Isotropic model."""
    GAUSSIAN = 4
    """Gaussian model."""
    GAUSSLORENTZ = 8
    """Gaussian Lorenzian model."""

    EXCITATION = 16
    """Excitation type."""
    EMISSION = 32
    """Emission type."""
    WIDEFIELD = 64
    """Widefield type."""
    CONFOCAL = 128
    """Confocal type."""
    TWOPHOTON = 256
    """Two-photon type."""

    PARAXIAL = 512
    """Border case for Gaussian approximation."""


class PinholeShape(enum.IntEnum):
    """Pinhole shapes."""

    ROUND = 0
    """Round pinhole."""
    SQUARE = 4
    """Square pinhole."""


ANISOTROPIC = PsfType.ANISOTROPIC
ISOTROPIC = PsfType.ISOTROPIC
GAUSSIAN = PsfType.GAUSSIAN
GAUSSLORENTZ = PsfType.GAUSSLORENTZ
EXCITATION = PsfType.EXCITATION
EMISSION = PsfType.EMISSION
WIDEFIELD = PsfType.WIDEFIELD
CONFOCAL = PsfType.CONFOCAL
TWOPHOTON = PsfType.TWOPHOTON
PARAXIAL = PsfType.PARAXIAL


class PSF:
    """Calculate point spread function of various types.

    Parameters:
        psftype, shape, num_aperture, refr_index, magnification, underfilling,\
        expsf, empsf, name:
            See class attributes.
        dims:
            Dimensions of data array in *micrometers*.
        ex_wavelen, em_wavelen:
            Excitation or emission wavelengths in *nanometers* if applicable.
        pinhole_radius:
            Outer radius of pinhole in *micrometers* in object space.
            This is the back-projected radius, i.e., the real physical radius
            of the pinhole divided by the magnification of the system.
        pinhole_shape:
            Pinhole shape, round or square.

    Notes:
        Calculations of the isotropic PSFs are based on the complex integration
        representation for the diffraction near the image plane proposed by
        Richards and Wolf [1-4].

        Gaussian approximations are calculated according to [5].

        Widefield calculations are used if the pinhole radius is larger than
        ~8 au.

        Models for polarized excitation or emission light (ANISOTROPIC) and the
        Gaussian-Lorentzian approximation (GAUSSLORENTZ) are not implemented.

    """

    TYPES: ClassVar[set[PsfType]] = {
        ISOTROPIC | EXCITATION,
        ISOTROPIC | EMISSION,
        ISOTROPIC | WIDEFIELD,
        ISOTROPIC | CONFOCAL,
        ISOTROPIC | TWOPHOTON,
        GAUSSIAN | EXCITATION,
        GAUSSIAN | EMISSION,
        GAUSSIAN | WIDEFIELD,  # == GAUSSIAN | EMISSION
        GAUSSIAN | CONFOCAL,
        GAUSSIAN | TWOPHOTON,
        GAUSSIAN | EXCITATION | PARAXIAL,
        GAUSSIAN | EMISSION | PARAXIAL,
        GAUSSIAN | WIDEFIELD | PARAXIAL,
        GAUSSIAN | CONFOCAL | PARAXIAL,
        GAUSSIAN | TWOPHOTON | PARAXIAL,
    }

    psftype: PsfType
    """Type of PSF."""

    name: str
    """Human readable label."""

    data: NDArray[numpy.float64]
    """PSF values in z,r space normalized to value at origin."""

    shape: tuple[int, int]
    """Size of data array in pixel."""

    dims: Dimensions[tuple[float, float]]
    """Dimensions of data array
    in px (pixel), um (micrometers), ou (optical units), and au (airy units).
    """

    ex_wavelen: float
    """Excitation wavelength in micrometers if applicable."""

    em_wavelen: float
    """Emission wavelength in micrometers if applicable."""

    num_aperture: float
    """Numerical aperture (NA) of objective."""

    refr_index: float
    """Index of refraction of sample medium."""

    magnification: float
    """Total magnification of optical system."""

    underfilling: float
    """Ratio of radius of objective back aperture to exp(-2) radius of
    excitation laser.
    """

    sigma: Dimensions[tuple[float, float]] | None
    """Gaussian sigmas if applicable
    in px (pixel), um (micrometers), ou (optical units), and au (airy units).
    """

    pinhole: Pinhole | None
    """Pinhole for confocal types."""

    expsf: PSF | None
    """Excitation PSF object for confocal types."""

    empsf: PSF | None
    """Emission PSF object for confocal types."""

    def __init__(
        self,
        psftype: PsfType,
        /,
        shape: tuple[int, int] = (256, 256),
        dims: tuple[float, float] = (4.0, 4.0),
        *,
        ex_wavelen: float = math.nan,
        em_wavelen: float = math.nan,
        num_aperture: float = 1.2,
        refr_index: float = 1.333,
        magnification: float = 1.0,
        underfilling: float = 1.0,
        pinhole_radius: float | None = None,
        pinhole_shape: PinholeShape | str = PinholeShape.ROUND,
        expsf: PSF | None = None,
        empsf: PSF | None = None,
        name: str | None = None,
    ):
        if psftype not in PSF.TYPES:
            msg = f'PSF type {psftype!r} is invalid or not supported'
            raise ValueError(msg)
        self.psftype = psftype
        self.name = str(name if name else enumstr(psftype))
        self.shape = int(shape[0]), int(shape[1])
        self.dims = Dimensions(px=shape, um=(float(dims[0]), float(dims[1])))

        self.ex_wavelen = ex_wavelen / 1e3 if ex_wavelen else math.nan
        self.em_wavelen = em_wavelen / 1e3 if em_wavelen else math.nan
        self.num_aperture = num_aperture
        self.refr_index = refr_index
        self.magnification = magnification
        self.underfilling = underfilling
        self.sigma = None
        self.pinhole = None
        self.expsf = expsf
        self.empsf = empsf

        if not (psftype & EXCITATION) and math.isnan(em_wavelen):
            msg = 'emission wavelength not specified'
            raise ValueError(msg)

        if not (psftype & EMISSION) and math.isnan(ex_wavelen):
            msg = 'excitation wavelength not specified'
            raise ValueError(msg)

        if psftype & CONFOCAL and pinhole_radius is None:
            msg = 'pinhole radius not specified'
            raise ValueError(msg)

        self.sinalpha = self.num_aperture / self.refr_index
        if self.sinalpha >= 1.0:
            msg = (
                f'quotient of the numeric aperture ({self.num_aperture:.1f}) '
                f'and refractive index ({self.refr_index:.1f}) is greater '
                f'than 1.0 ({self.sinalpha:.2f})'
            )
            raise ValueError(msg)

        if psftype & EMISSION:
            au = 1.22 * self.em_wavelen / self.num_aperture
            ou = zr2uv(
                self.dims.um,
                self.em_wavelen,
                self.sinalpha,
                self.refr_index,
                self.magnification,
            )
        else:
            au = 1.22 * self.ex_wavelen / self.num_aperture
            ou = zr2uv(
                self.dims.um,
                self.ex_wavelen,
                self.sinalpha,
                self.refr_index,
                1.0,
            )
        self.dims.update(
            ou=ou, au=(self.dims.um[0] / au, self.dims.um[1] / au)
        )

        if pinhole_radius is not None:
            self.pinhole = Pinhole(pinhole_radius, self.dims, pinhole_shape)

        clock = time.perf_counter
        start = clock()

        if psftype & GAUSSIAN:
            self.sigma = Dimensions(**self.dims)
            if self.underfilling != 1.0:
                msg = 'underfilling not supported in Gaussian approximation'
                raise NotImplementedError(msg)

            if psftype & EXCITATION or psftype & TWOPHOTON:
                widefield = True
                self.em_wavelen = math.nan
                self.magnification = math.nan
                self.pinh_radius = None
                lex = lem = self.ex_wavelen
                radius = 0.0
            elif psftype & EMISSION or psftype & WIDEFIELD:
                widefield = True
                self.ex_wavelen = math.nan
                self.magnification = math.nan
                lex = lem = self.em_wavelen
                radius = 0.0
            elif psftype & CONFOCAL:
                assert self.pinhole is not None
                radius = self.pinhole.radius.um
                if radius > 9.76 * self.ex_wavelen / self.num_aperture:
                    # use widefield approximation for pinholes > 8 AU
                    widefield = True
                    lex = lem = self.ex_wavelen
                else:
                    widefield = False
                    lex = self.ex_wavelen
                    lem = self.em_wavelen
                if self.pinhole.shape != PinholeShape.ROUND:
                    msg = 'Gaussian approximation only valid for round pinhole'
                    raise NotImplementedError(msg)

            paraxial = bool(psftype & PARAXIAL)
            self.sigma.um = _psf.gaussian_sigma(
                lex,
                lem,
                self.num_aperture,
                self.refr_index,
                radius,
                widefield,
                paraxial,
            )
            self.data = _psf.gaussian2d(self.dims.px, self.sigma.px)

        elif psftype & ISOTROPIC:
            if psftype & EXCITATION or psftype & TWOPHOTON:
                self.em_wavelen = math.nan
                self.magnification = math.nan
                self.data = _psf.psf(
                    0,
                    self.shape,
                    self.dims.ou,
                    1.0,
                    self.sinalpha,
                    self.underfilling,
                    1.0,
                    80,
                )
            elif psftype & EMISSION:
                self.ex_wavelen = math.nan
                self.underfilling = math.nan
                self.data = _psf.psf(
                    1,
                    self.shape,
                    self.dims.ou,
                    self.magnification,
                    self.sinalpha,
                    1.0,
                    1.0,
                    80,
                )
            elif psftype & CONFOCAL or psftype & WIDEFIELD:
                if em_wavelen < ex_wavelen:
                    msg = 'Excitation > Emission wavelength'
                    raise ValueError(msg)
                # start threads to calculate excitation and emission PSF
                threads = []
                if not (
                    self.expsf is not None
                    and self.expsf.psftype == ISOTROPIC | EXCITATION
                ):
                    threads.append(
                        (
                            'expsf',
                            PSFthread(
                                ISOTROPIC | EXCITATION,
                                shape,
                                dims,
                                ex_wavelen=ex_wavelen,
                                em_wavelen=math.nan,
                                num_aperture=num_aperture,
                                refr_index=refr_index,
                                magnification=1.0,
                                underfilling=underfilling,
                            ),
                        )
                    )
                if not (
                    self.empsf is not None
                    and self.empsf.psftype == ISOTROPIC | EMISSION
                ):
                    threads.append(
                        (
                            'empsf',
                            PSFthread(
                                ISOTROPIC | EMISSION,
                                shape,
                                dims,
                                ex_wavelen=math.nan,
                                em_wavelen=em_wavelen,
                                num_aperture=num_aperture,
                                refr_index=refr_index,
                                magnification=magnification,
                                underfilling=1.0,
                            ),
                        )
                    )
                for _a, t in threads:
                    t.start()
                for a, t in threads:
                    t.join()
                    setattr(self, a, t.psf)
                if self.expsf is None:
                    msg = 'Excitation PSF is None'
                    raise ValueError(msg)
                if self.empsf is None:
                    msg = 'Emission PSF is None'
                    raise ValueError(msg)
                if not self.expsf.iscompatible(self.empsf):
                    msg = 'Excitation and emission PSF not compatible'
                    raise ValueError(msg)
                if psftype & WIDEFIELD or (
                    self.pinhole is not None
                    and self.pinhole.radius.um
                    > 9.76 * self.ex_wavelen / self.num_aperture
                ):
                    # use widefield approximation for pinholes > 8 AU
                    self.data = _psf.obsvol(self.expsf.data, self.empsf.data)
                else:
                    assert self.pinhole is not None
                    self.data = _psf.obsvol(
                        self.expsf.data, self.empsf.data, self.pinhole.kernel()
                    )

        if psftype & TWOPHOTON:
            self.data *= self.data
        self.time = float(clock() - start) * 1e3

    def __getitem__(self, key: Any, /) -> NDArray[numpy.float64]:
        """Return value of data array at position."""
        return numpy.asarray(self.data[key])

    def __str__(self) -> str:
        s = [self.__class__.__name__, self.name]
        s.append(f'shape: ({self.dims.px[0]}, {self.dims.px[1]}) pixel')
        dims = self.dims.format(['um', 'ou', 'au'], ['%.2f', '%.2f', '%.2f'])
        s.append(f'dimensions: {dims}')
        if not math.isnan(self.ex_wavelen):
            s.append(f'excitation wavelength: {self.ex_wavelen * 1e3:.1f} nm')
        if not math.isnan(self.em_wavelen):
            s.append(f'emission wavelength: {self.em_wavelen * 1e3:.1f} nm')
        s.append(f'numeric aperture: {self.num_aperture:.2f}')
        s.append(f'refractive index: {self.refr_index:.2f}')
        angle = math.degrees(math.asin(self.sinalpha))
        s.append(f'half cone angle: {angle:.2f} deg')
        if not math.isnan(self.magnification):
            s.append(f'magnification: {self.magnification:.2f}')
        if not math.isnan(self.underfilling):
            s.append(f'underfilling: {self.underfilling:.2f}')
        if self.pinhole:
            pinhole = self.pinhole.radius.format(
                ['um', 'ou', 'au', 'px'], ['%.3f', '%.3f', '%.4f', '%.2f']
            )
            s.append(f'pinhole radius: {pinhole}')
        if self.sigma is not None:
            sigma = self.sigma.format(
                ['um', 'ou', 'au', 'px'], ['%.3f', '%.3f', '%.3f', '%.2f']
            )
            s.append(f'gauss sigma: {sigma}')
        s.append(f'computing time: {self.time:.2f} ms\n')
        return '\n '.join(s)

    def iscompatible(self, other: PSF, /) -> bool:
        """Return True if PSFs match dimensions and optical properties."""
        return (
            (self.dims.px[0] == other.dims.px[0])
            and (self.dims.px[1] == other.dims.px[1])
            and (self.dims.um[0] == other.dims.um[0])
            and (self.dims.um[1] == other.dims.um[1])
            and (self.num_aperture == other.num_aperture)
            and (self.refr_index == other.refr_index)
        )

    def slice(
        self, key: int | slice = slice(None), /
    ) -> NDArray[numpy.float64]:
        """Return z-slice of PSF with rotational symmetries applied."""
        return numpy.asarray(_psf.zr2zxy(self.data[key]))

    def volume(self) -> NDArray[numpy.float64]:
        """Return 3D volume of PSF with all symmetries applied.

        The shape of the returned array is
        `(2*self.shape[0]-1, 2*self.shape[1]-1, 2*self.shape[1]-1)`

        """
        return mirror_symmetry(_psf.zr2zxy(self.data))

    def imshow(
        self, subplot: Any = 111, **kwargs: Any
    ) -> tuple[Axes, AxesImage]:
        """Log-plot PSF image using matplotlib.pyplot. Return plot axis."""
        title = kwargs.get('title', self.name)
        aspect = (
            self.shape[1] / self.shape[0] * self.dims.um[0] / self.dims.um[1]
        )
        kwargs.update({'title': title, 'aspect': aspect})
        return imshow(subplot, self.data, **kwargs)


class PSFthread(threading.Thread):
    """Calculate point spread function in thread."""

    psf: PSF | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        threading.Thread.__init__(self)
        self.args = args
        self.kwargs = kwargs
        self.psf = None

    def run(self) -> None:
        self.psf = PSF(*self.args, **self.kwargs)


class Pinhole:
    """Pinhole object for confocal microscopy.

    Parameters:
        radius:
            Outer pinhole radius in micrometers in object space.
        dimensions:
            Dimensions of object space in 'px' (pixel), 'um' (micrometers),
            'ou' (optical units), and 'au' (airy units).
        shape:
            Shape of pinhole, round or square.

    Examples:
        >>> ph = Pinhole(0.1, dict(px=16, um=1.0), 'round')
        >>> ph
        Pinhole(1, Dimensions(px=1.6, um=0.1), 'ROUND')
        >>> ph.shape
        <PinholeShape.ROUND: 0>
        >>> ph.radius.px
        1.6
        >>> ph.kernel()
        array([[1.     , 1.6    , 0.6    ],
               [0.8    , 1.18579, 0.36393],
               [0.3    , 0.36393, 0.     ]])

    """

    shape: PinholeShape
    """Shape of pinhole, round or square."""

    radius: Dimensions[float]
    """Outer pinhole radius
    in px (pixel), um (micrometers), ou (optical units), and au (airy units).
    """

    _kernel: NDArray[numpy.float64] | None

    def __init__(
        self,
        radius: float,
        dimensions: Mapping[str, float | tuple[float, float]],
        shape: PinholeShape | str,
    ) -> None:
        self.shape = enumarg(PinholeShape, shape)  # type: ignore[assignment]
        with contextlib.suppress(TypeError, IndexError):
            dimensions = {
                k: v[1] for k, v in dimensions.items()  # type: ignore[index]
            }
        self.radius = Dimensions(**dimensions)
        self.radius.um = float(radius)
        self._kernel = None

    def kernel(self) -> NDArray[numpy.float64]:
        """Return convolution kernel for integration over the pinhole."""
        if self._kernel is None:
            self._kernel = _psf.pinhole_kernel(self.radius.px, self.shape)
        return self._kernel

    def __repr__(self) -> str:
        params = f'1, {self.radius}, {self.shape.name!r}'
        return f'{self.__class__.__name__}({params})'


class Dimensions(UserDict[str, DimensionT]):
    """Store dimensions in various units and perform linear conversions.

    Examples:
        >>> dim = Dimensions(px=100, um=2)
        >>> dim
        Dimensions(px=100, um=2)
        >>> dim(50, 'px', 'um')
        1.0
        >>> dim.px, dim.um
        (100, 2)
        >>> dim.px = 50
        >>> dim.um
        1.0
        >>> dim.format(('um', 'px'), ('%.2f', '%.1f'))
        '1.00 um, 50.0 px'
        >>> dim = Dimensions(px=(100, 200), um=(2, 8))
        >>> dim
        Dimensions(px=(100, 200), um=(2, 8))
        >>> dim((50, 50), 'px', 'um')
        (1.0, 2.0)
        >>> dim.ou = (1, 2)
        >>> dim.px
        (100, 200)
        >>> dim['px'] = (50, 100)
        >>> dim.ou
        (0.5, 1.0)

    """

    def __init__(
        self, adict: dict[str, Any] | None = None, /, **kwargs: Any
    ) -> None:
        data: dict[str, Any] = {}
        if adict is not None:
            data.update(adict)
        if kwargs:
            data.update(kwargs)
        self.__dict__['data'] = data  # avoid __setattr__

    def __call__(
        self, value: DimensionT, unit: str, newunit: str, /
    ) -> DimensionT:
        """Return value given in unit in another unit."""
        # TODO: how to type this correctly?
        dim = self.data[unit]
        new = self.data[newunit]
        try:
            return value * (new / dim)  # type: ignore[no-any-return, operator]
        except TypeError:
            return tuple(
                v * (o / u)
                for v, u, o in zip(
                    value, dim, new, strict=False
                )  # type: ignore[call-overload]
            )  # type: ignore[return-value]

    def __getattr__(self, unit: str, /) -> DimensionT:
        if unit == 'data':
            raise AttributeError
        return self.data[unit]

    def __setattr__(self, unit: str, value: DimensionT, /) -> None:
        """Add dimension or rescale all dimensions to new value."""
        if unit != 'data':
            self.__setitem__(unit, value)

    def __setitem__(self, unit: str, value: DimensionT, /) -> None:
        """Add dimension or rescale all dimensions to new value."""
        # TODO: how to type this correctly?
        data = self.data
        try:
            dim = data[unit]
        except KeyError:
            data[unit] = value
            return
        try:
            scale = value / dim  # type: ignore[operator]
            for k, v in data.items():
                data[k] = v * scale
        except TypeError:
            scale = tuple(
                v / d
                for v, d in zip(value, dim, strict=False)  # type: ignore[call-overload]
            )
            for k, v in data.items():
                data[k] = tuple(  # type: ignore[assignment]
                    v * s
                    for v, s in zip(
                        data[k], scale, strict=False
                    )  # type: ignore[call-overload]
                )

    def __repr__(self) -> str:
        params = ', '.join(f'{k}={v}' for k, v in self.data.items())
        return f'{self.__class__.__name__}({params})'

    def format(self, keys: Iterable[str], formatstr: Iterable[str], /) -> str:
        """Return formatted string."""
        s = []
        try:
            for k, f in zip(keys, formatstr, strict=False):
                s.append(f'{f % self[k]} {k}')
        except TypeError:
            for k, f in zip(keys, formatstr, strict=False):
                v = self[k]
                t = [f % i for i in v]  # type: ignore[attr-defined]
                s.append('({}) {}'.format(', '.join(t), k))
        return ', '.join(s)


def uv2zr(
    uv: tuple[float, float],
    /,
    wavelength: float,
    sinalpha: float,
    refr_index: float,
    magnification: float = 1.0,
) -> tuple[float, float]:
    """Return z,r in units of wavelength from u,v given in optical units.

    For excitation, magnification should be 1.

    Examples:
        >>> uv2zr((1, 1), 488, 0.9, 1.33)
        (72.094..., 64.885...)

    """
    a = wavelength / (2.0 * math.pi * sinalpha * refr_index * magnification)
    b = a / (sinalpha * magnification)
    return uv[0] * b, uv[1] * a


def zr2uv(
    zr: tuple[float, float],
    /,
    wavelength: float,
    sinalpha: float,
    refr_index: float,
    magnification: float = 1.0,
) -> tuple[float, float]:
    """Return u,v in optical units from z,r given in units of wavelength.

    For excitation, magnification should be 1.

    Examples:
        >>> zr2uv((1e3, 1e3), 488, 0.9, 1.33)
        (13.870..., 15.411...)

    """
    a = (2.0 * math.pi * refr_index * sinalpha * magnification) / wavelength
    b = a * sinalpha * magnification
    return zr[0] * b, zr[1] * a


def mirror_symmetry(data: ArrayLike, /) -> NDArray[numpy.float64]:
    """Apply mirror symmetry along one face in each dimension.

    The input array can be 1, 2, or 3-dimensional.
    The shape of the returned array is `2*data.shape-1` in each dimension.

    Examples:
        >>> mirror_symmetry([0, 1])
        array([1., 0., 1.])
        >>> mirror_symmetry([[0, 1], [0, 1]])
        array([[1., 0., 1.],
               [1., 0., 1.],
               [1., 0., 1.]])
        >>> mirror_symmetry(
        ...     [[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]]
        ... )[0]
        array([[1., 0., 1.],
               [1., 0., 1.],
               [1., 0., 1.]])

    """
    data = numpy.array(data)
    result = numpy.empty(tuple(2 * i - 1 for i in data.shape), numpy.float64)
    if data.ndim == 1:
        x = data.shape[0] - 1
        result[x:] = data
        result[:x] = data[-1:0:-1]
    elif data.ndim == 2:
        x, y = (i - 1 for i in data.shape)
        result[x:, y:] = data
        result[:x, y:] = data[-1:0:-1, :]
        result[:, :y] = result[:, -1:y:-1]
    elif data.ndim == 3:
        x, y, z = (i - 1 for i in data.shape)
        result[x:, y:, z:] = data
        result[:x, y:, z:] = data[-1:0:-1, :, :]
        result[:, :y, z:] = result[:, -1:y:-1, z:]
        result[:, :, :z] = result[:, :, -1:z:-1]
    else:
        msg = f'{data.ndim}-dimensional arrays not supported'
        raise NotImplementedError(msg)
    return result


def enumarg(enum: type[enum.IntEnum], arg: Any, /) -> enum.IntEnum:
    """Return enum member from its name or value."""
    try:
        return enum(arg)
    except Exception:
        try:
            return enum[arg.upper()]
        except Exception as exc:
            msg = f'invalid argument {arg!r}'
            raise ValueError(msg) from exc


def enumstr(enum: Any, /) -> str:
    """Return short string representation of Enum member.

    >>> enumstr(PsfType.ANISOTROPIC)
    'ANISOTROPIC'

    """
    name = enum.name
    if name is None:
        name = str(enum)
    return name  # type: ignore[no-any-return]


def imshow(
    subplot: Any,
    data: ArrayLike,
    /,
    *,
    title: str | None = None,
    sharex: Axes | None = None,
    sharey: Axes | None = None,
    vmin: float = -2.5,
    vmax: float = 0.0,
    cmap: Any = None,
    interpolation: str = 'lanczos',
    **kwargs: Any,
) -> tuple[Axes, AxesImage]:
    """Log-plot image using matplotlib.pyplot. Return plot axis and plot.

    Mirror symmetry is applied along the x and y axes.

    """
    import matplotlib
    from matplotlib import pyplot

    ax = pyplot.subplot(subplot, sharex=sharex, sharey=sharey, facecolor='k')
    if title:
        pyplot.title(title)
    cmap = matplotlib.colormaps.get_cmap('cubehelix' if cmap is None else cmap)
    try:
        # workaround: set alpha for i_bad
        cmap._init()
        cmap._lut[-1, -1] = 1.0
    except AttributeError:
        pass

    im = pyplot.imshow(
        mirror_symmetry(numpy.log10(data)),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        **kwargs,
    )
    pyplot.axis('off')
    return ax, im


if __name__ == '__main__':
    import doctest

    numpy.set_printoptions(suppress=True, precision=5)
    doctest.testmod(optionflags=doctest.ELLIPSIS)
