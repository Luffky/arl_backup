from astropy.coordinates.sky_coordinate import SkyCoord
import numpy
from astropy.coordinates import CartesianRepresentation


def skycoord_to_lmn(pos: SkyCoord, phasecentre: SkyCoord):
    """
    Convert astropy sky coordinates into the l,m,n coordinate system
    relative to a phase centre.

    The l,m,n is a RHS coordinate system with
    * its origin on the sky sphere
    * m,n and the celestial north on the same plane
    * l,m a tangential plane of the sky sphere

    Note that this means that l increases east-wards
    """

    # Determine relative sky position
    todc = pos.transform_to(phasecentre.skyoffset_frame())
    dc = todc.represent_as(CartesianRepresentation)

    # Do coordinate transformation - astropy's relative coordinates do
    # not quite follow imaging conventions
    return dc.y.value, dc.z.value, dc.x.value - 1


def simulate_point(dist_uvw, l, m):
    """
    Simulate visibilities for unit amplitude point source at
    direction cosines (l,m) relative to the phase centre.

    This includes phase tracking to the centre of the field (hence the minus 1
    in the exponent.)

    Note that point source is delta function, therefore the
    FT relationship becomes an exponential, evaluated at
    (uvw.lmn)

    :param dist_uvw: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param l: horizontal direction cosine relative to phase tracking centre
    :param m: orthogonal directon cosine relative to phase tracking centre
    """

    # vector direction to source
    s = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])
    # complex valued Visibility data
    return numpy.exp(-2j * numpy.pi * numpy.dot(dist_uvw, s))


def uvw_to_xyz(uvw, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions relative to a sky position at
    :math:`(ha, dec)` to earth coordinates. Can be used for both
    antenna positions as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param uvw: :math:`(u,v,w)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre
    """

    u, v, w = numpy.hsplit(uvw, 3)

    # Two rotations:
    #  1. by 'dec-90' along the u axis
    #  2. by '-ha' along the z axis
    v0 = v * numpy.sin(dec) - w * numpy.cos(dec)
    z = v * numpy.cos(dec) + w * numpy.sin(dec)
    x = u * numpy.cos(ha) + v0 * numpy.sin(ha)
    y = -u * numpy.sin(ha) + v0 * numpy.cos(ha)

    return numpy.hstack([x, y, z])


def xyz_to_uvw(xyz, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions in earth coordinates to
    :math:`(u,v,w)` coordinates relative to astronomical source
    position :math:`(ha, dec)`. Can be used for both antenna positions
    as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre.
    """

    x, y, z = numpy.hsplit(xyz, 3)

    # Two rotations:
    #  1. by 'ha' along the z axis
    #  2. by '90-dec' along the u axis
    u = x * numpy.cos(ha) - y * numpy.sin(ha)
    v0 = x * numpy.sin(ha) + y * numpy.cos(ha)
    w = z * numpy.sin(dec) - v0 * numpy.cos(dec)
    v = z * numpy.cos(dec) + v0 * numpy.sin(dec)

    return numpy.hstack([u, v, w])