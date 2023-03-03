import numpy as np
from datetime import datetime, timedelta, timezone

# Astrodynamic functions used in Porkchop plot
# Based on matlab sources in Orbital Mechanics for Engineering Students, Curtis
# https://www.elsevier.com/__data/assets/pdf_file/0009/915408/Appendix-D-MATLAB-Scripts.pdf
#
# Author: ravi_ram
#

# extract planet's J2000 orbital elements and centennial rates from Table.
# 
# planet_id      - 0 through 8, for Mercury through Pluto
# J2000_elements - 9 by 6 matrix of J2000 orbital elements for the nine
#                planets Mercury through Pluto. The columns of each 
#                row are: [a, e, i, RA, w_hat, L]
# cent_rates     - 9 by 6 matrix of the rates of change of the 
#                J2000_elements per Julian century (Cy). Using "dot"
#                for time derivative
# returns
#                orbital elements
def get_planet_ephemeris(planet_id, jd):
    J2000_elements = np.array((
    [0.38709927,  0.20563593,  7.00497902,  48.33076593,  77.45779628,  252.25032350],
    [0.72333566,  0.00677672,  3.39467605,  76.67984255, 131.60246718,  181.97909950], 
    [1.00000261,  0.01671123, -0.00001531,   0.0,        102.93768193,  100.46457166], 
    [1.52371034,  0.09339410,  1.84969142,  49.55953891, -23.94362959, 	-4.55343205],
    [5.20288700,  0.04838624,  1.30439695, 100.47390909,  14.72847983, 	34.39644501],
    [9.53667594,  0.05386179,  2.48599187, 113.66242448,  92.59887831, 	49.95424423],
    [19.18916464,  0.04725744,  0.77263783,  74.01692503, 170.95427630,  313.23810451],
    [30.06992276,  0.00859048,  1.77004347, 131.78422574,  44.96476227,  -55.12002969], 
    [39.48211675,  0.24882730, 17.14001206, 110.30393684, 224.06891629,  238.92903833]
    ))  
    J2000_coe = J2000_elements[planet_id]

    cent_rates = np.array(( 
    [0.00000037,  0.00001906, -0.00594749, -0.12534081,  0.16047689,  149472.67411175], 
    [0.00000390, -0.00004107, -0.00078890, -0.27769418,  0.00268329,  58517.81538729],  
    [0.00000562, -0.00004392, -0.01294668,  0.0,         0.32327364,   35999.37244981],  
    [0.0001847,   0.00007882, -0.00813131, -0.29257343,  0.44441088,   19140.30268499],  
    [-0.00011607, -0.00013253, -0.00183714,  0.20469106, 0.21252668,    3034.74612775], 
    [-0.00125060, -0.00050991,  0.00193609, -0.28867794, -0.41897216,    1222.49362201],
    [-0.00196176, -0.00004397, -0.00242939,  0.04240589,  0.40805281, 	428.48202785], 
    [0.00026291,  0.00005105,  0.00035372, -0.00508664, -0.32241464, 	218.45945325], 
    [-0.00031596,  0.00005170,  0.00004818, -0.01183482, -0.04062942, 	145.20780515]
    ))
    rates = cent_rates[planet_id]
    
    # Convert from AU to km:
    au             = 149597871; 
    J2000_coe[0]   = J2000_coe[0]*au;
    rates[0]       = rates[0]*au;

    # t0 - Julian centuries between J2000 and jd (equation 8.93a)
    t0     = (jd - 2451545)/36525
    # Equation 8.93b:
    elements = J2000_coe + rates*t0
    # return orbital elements
    return elements

# calculate the orbital elements and the state vector
# of a planet from the date (year, month, day) and
# universal time (hour, minute, second).
# planet_id - the planet number (0 - 8)
# jd        - julian day number of the date and time (equation 5.47)
# mu        - gravitational parameter of the sun (km^3/s^2)
def get_planet_state_vector(mu, planet_id, jd):
    # returns the inverse tangent (tan^-1) of the elements of X in degrees.
    atand = lambda x: np.rad2deg(np.arctan(x))

    # J2000 orbital elements 
    elements = get_planet_ephemeris(planet_id, jd)
    a      = elements[0]
    e      = elements[1]
    # angular momentum - equation 2.71:
    h      = np.sqrt(mu*a*(1 - e**2))   
    # reduce the angular elements to within the range 0 - 360 degrees
    incl   = elements[2];
    RA     = np.mod(elements[3],360);
    w_hat  = np.mod(elements[4],360);
    L      = np.mod(elements[5],360);
    w      = np.mod(w_hat - RA ,360);
    M      = np.mod(L - w_hat  ,360);
    # Algorithm 3.1 (for which M must be in radians)
    E      = kepler_E(e, np.radians(M)); # in rad
    # equation 3.13 (converting the result to degrees):
    TA     = np.mod(2*atand(np.sqrt((1 + e)/(1 - e))*np.tan(E/2)), 360)     
    coe    = [h, e, np.radians(RA), np.radians(incl), np.radians(w),
              np.radians(TA),  a,  w_hat,  L,  M,  E]
    # calculate state vectors from orbital elements - algorithm 4.5:
    [r, v] = sv_from_coe(coe, mu)
    # return orbital elements(coe), state vectors(r,v) and julian date
    return coe, r, v, jd

# Calculate the state vector from classical orbital elements
# Input:    h     - Specific angular momentum
#           e     - eccentricity
#           i     - orbital inclination
#           omega - right ascension of the ascending node
#           w     - argument of perigee
#           theta - true anomaly
#           mu    - gravitational parameter
#
# Output:   r     - The position vector
#           v     - The velocity vector
def sv_from_coe(coe, mu):
    # Create the rotation matrix about the x-axis (equation 4.32)
    def rot1(th):
        c, s = np.cos(th), np.sin(th)
        rot1 = np.array([ [1,  0, 0],
                          [0,  c, s],
                          [0, -s, c] ])
        return rot1
    # Create the rotation matrix about the z-axis (equation 4.34)
    def rot3(th):
        c, s = np.cos(th), np.sin(th)
        rot3 = np.array([ [ c, s, 0],
                          [-s, c, 0],
                          [ 0, 0, 1] ])
        return rot3    

    # unpack data
    h, e, omega, i, w, theta, *others = coe    
    # Calculate the position vector in the perifocal frame (equation 4.45)
    rp = (h**2/mu) * ( 1/(1+e*np.cos(theta)) ) * np.array([np.cos(theta), np.sin(theta), 0]).T
    # Calculate the velocity vector in the perifocal frame (equation 4.46)
    vp = (mu/h) * np.array([-np.sin(theta), e + np.cos(theta), 0]).T   
    # Calculate the transform matrix from perifocal to geocentric (equation 4.49)
    Q  = (rot3(w) @ rot1(i) @ rot3(omega)).T
    # Transform from perifocal to geocentric (equations 4.51 - r and v are column vectors)
    r = Q @ rp.T
    v = Q @ vp.T
    # return position, velocity
    return r, v

# This function uses Newton's method to solve Kepler's 
# equation  E - e*sin(E) = M  for the eccentric anomaly,
# given the eccentricity and the mean anomaly.
# 
# E  - eccentric anomaly (radians)
# e  - eccentricity, 
# M  - mean anomaly (radians),
def kepler_E(e, M):
    # Set an error tolerance:
    error = 1.e-8
    # select a starting value for E:
    if M < np.pi: E = M + e/2
    else:         E = M - e/2
    # iterate on Equation 3.17 until E is determined to within
    # the error tolerance:
    ratio = 1
    while np.abs(ratio) > error:
        ratio = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E = E - ratio
    # return eccentric anomaly
    return E

# planet_id - the planet number (0 - 8)
def get_planet_id(planet_name):
    # planets list  
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter',
               'Saturn', 'Uranus', 'Neptune', 'Pluto']
    try:
        # remove empty spaces and capitalizes the first character
        planet_name = planet_name.strip().capitalize()
        # find the index
        ind = planets.index(planet_name)
        return ind
    except ValueError:
        print( 'error: Planet \'', planet_name, '\' not in list.')        
        return -1
    
# gregorian date and time to julian day number
def greg_to_jd(year, month, day, hour=0, minute=0, second=0):
    # julian day number at 0 UT for any year between 1900 and 2100 using Equation 5.48
    j0 = 367*year - np.fix(7*(year + np.fix((month + 9)/12))/4)  + np.fix(275*month/9) + day + 1721013.5   
    # ut - universal time in fractions of a day
    ut     = (hour + minute/60 + second/3600)/24
    # jd - julian day number of the date and time (equation 5.47)
    jd     = j0 + ut
    # return julian day number
    return jd

# convert a decimal Julian Date to gregorian date and time
def jd_to_greg(jd):
    # convert jdn to gdt    
    jdn = int(jd)
    L = jdn + 68569
    N = int(4 * L / 146_097)
    L = L - int((146097 * N + 3) / 4)
    I = int(4000 * (L + 1) / 1_461_001)
    L = L - int(1461 * I / 4) + 31
    J = int(80 * L / 2447)
    day = L - int(2447 * J / 80)
    L = int(J / 11)
    month = J + 2 - 12 * L
    year = 100 * (N - 49) + I + L    
    # decimal processing
    offset = timedelta(days=(jd % 1), hours=+12)
    dt = datetime(year=year, month=month, day=day, tzinfo=timezone.utc)
    return dt + offset

# create date string from julian day number
def jd_str(jd):
    return jd_to_greg(np.float32(jd)).strftime('%d-%m-%Y')   

