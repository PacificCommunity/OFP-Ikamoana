from math import acos, asin, pi, sin, tan


def dayLengthPISCES(jday: int, lat: float) -> float:
    """
    Compute the day length depending on latitude and the day. New
    function provided by Laurent Bopp as used in the PISCES model and
    used by SEAPODYM in 2020.

    Parameters
    ----------
    jday : int
        Day of the year.
    lat : float
        Latitude.

    Modification
    ------------
    original       : E. Maier-Reimer (GBC 1993)
	additions      : C. Le Quere (1999)
	modifications  : O. Aumont (2004)
    	Adapted to C      : P. Lehodey (2005)
        Adapted to Python : J. Lehodey (2021)

    Returns
    -------
    float
        The duration of the day (i.e. while the sun is shining) as a ratio in
        range [0,1].

    """

    rum = (jday - 80.0) / 365.25
    delta = sin(rum * pi * 2.0) * sin(pi * 23.5 / 180.0)
    codel = asin(delta)
    phi = lat * pi / 180.0

    argu = tan(codel) * tan(phi)
    argu = min(1.,argu)
    argu = max(-1.,argu)

    day_length = 24.0 - (2.0 * acos(argu) * 180.0 / pi / 15 )
    day_length = max(day_length,0.0)

    return day_length / 24.0
