# psf_example.py

"""Point Spread Function example.

Demonstrate the use of the psf library for calculating point spread functions
for fluorescence microscopy.

"""

import numpy
from matplotlib import pyplot

import psf


def psf_example(cmap='hot', savebin=False, savetif=False, savevol=False,
                plot=True, **kwargs):
    """Calculate, save, and plot various point spread functions."""

    args = {
        'shape': (512, 512),  # number of samples in z and r direction
        'dims': (5.0, 5.0),   # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 520.0,  # emission wavelength in nanometers
        'num_aperture': 1.2,
        'refr_index': 1.333,
        'magnification': 1.0,
        'pinhole_radius': 0.05,  # in micrometers
        'pinhole_shape': 'square',
    }
    args.update(kwargs)

    obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
    expsf = obsvol.expsf
    empsf = obsvol.empsf
    gauss = gauss2 = psf.PSF(psf.GAUSSIAN | psf.EXCITATION, **args)

    print(expsf)
    print(empsf)
    print(obsvol)
    print(gauss)
    print(gauss2)

    if savebin:
        # save zr slices to BIN files
        empsf.data.tofile('empsf.bin')
        expsf.data.tofile('expsf.bin')
        gauss.data.tofile('gauss.bin')
        obsvol.data.tofile('obsvol.bin')

    if savetif:
        # save zr slices to TIFF files
        from tifffile import imsave

        imsave('empsf.tif', empsf.data)
        imsave('expsf.tif', expsf.data)
        imsave('gauss.tif', gauss.data)
        imsave('obsvol.tif', obsvol.data)

    if savevol:
        # save xyz volumes to files. Requires 32 GB for 512x512x512
        from tifffile import imsave

        imsave('empsf_vol.tif', empsf.volume())
        imsave('expsf_vol.tif', expsf.volume())
        imsave('gauss_vol.tif', gauss.volume())
        imsave('obsvol_vol.tif', obsvol.volume())

    if not plot:
        return

    # Log-plot xy, and rz slices
    pyplot.rc('font', family='sans-serif', weight='normal')
    pyplot.figure(dpi=96, figsize=(9.5, 5.0), frameon=True,
                  facecolor='w', edgecolor='w')
    pyplot.subplots_adjust(bottom=0.02, top=0.92, left=0.02, right=0.98,
                           hspace=0.01, wspace=0.01)

    ax = expsf.imshow(241, cmap=cmap)[0]
    empsf.imshow(242, sharex=ax, sharey=ax, cmap=cmap)
    obsvol.imshow(243, sharex=ax, sharey=ax, cmap=cmap)
    gauss.imshow(244, sharex=ax, sharey=ax, cmap=cmap)
    z = 0
    psf.imshow(245, data=expsf.slice(z), sharex=ax, cmap=cmap)
    psf.imshow(246, data=empsf.slice(z), sharex=ax, cmap=cmap)
    psf.imshow(247, data=obsvol.slice(z), sharex=ax, cmap=cmap)
    psf.imshow(248, data=gauss.slice(z), sharex=ax, cmap=cmap)

    # plot cross sections
    z = numpy.arange(0, gauss.dims.ou[0], gauss.dims.ou[0] / gauss.dims.px[0])
    r = numpy.arange(0, gauss.dims.ou[1], gauss.dims.ou[1] / gauss.dims.px[1])
    zr_max = 20.0
    pyplot.figure()
    pyplot.subplot(211)
    pyplot.title('PSF cross sections')
    pyplot.plot(r, expsf[0], 'r-', label=expsf.name + ' (r)')
    pyplot.plot(r, gauss2[0], 'r:', label='')
    # pyplot.plot(r, empsf.data[0], 'g--', label=empsf.name+' (r)')
    pyplot.plot(r, obsvol[0], 'b-', label=obsvol.name + ' (r)')
    pyplot.plot(r, gauss[0], 'b:', label="")
    pyplot.plot(z, expsf[:, 0], 'm-', label=expsf.name + ' (z)')
    pyplot.plot(z, gauss2[:, 0], 'm:', label='')
    # pyplot.plot(z, empsf.data[:,0], 'g--', label=empsf.name+' (z)')
    pyplot.plot(z, obsvol[:, 0], 'c-', label=obsvol.name + ' (z)')
    pyplot.plot(z, gauss[:, 0], 'c:', label='')
    pyplot.legend()
    pyplot.axis([0, zr_max, 0, 1])
    pyplot.subplot(212)
    pyplot.title('Residuals of gaussian approximation')
    pyplot.plot(r, expsf[0] - gauss2[0], 'r-', label=expsf.name + ' (r)')
    pyplot.plot(r, obsvol[0] - gauss[0], 'b-', label=obsvol.name + ' (r)')
    pyplot.plot(z, expsf[:, 0] - gauss2[:, 0], 'm-', label=expsf.name + ' (z)')
    pyplot.plot(z, obsvol[:, 0] - gauss[:, 0], 'c-',
                label=obsvol.name + ' (z)')
    pyplot.axis([0, zr_max, -0.1, 0.1])
    pyplot.tight_layout()

    pyplot.show()


if __name__ == '__main__':
    psf_example()
