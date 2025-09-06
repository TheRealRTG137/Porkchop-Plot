import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import datetime

#import astrodynamics as ad
import astrodynamics as ad 
import lambert as lb


# Porkchop Plot
# Each pair of departure/arrival dates defines a unique Earth to Mars transfer
# trajectory. For the purposes of these particular contour plots, each date pair
# is associated with an array of specific values for departure energy, and Mars
# arrival V∞ .
# The resulting contours are plotted in an Earth departure/Mars arrival mission
# space with a departure date coverage span of 180 days and an arrival date 
# coverage span of 400 days.
# 
# Since numerous events can cause a mission to launch at a date other than optimal,
# the plots include departure energies up to 50 km^2/sec^2 .
# Departure energies above 50 km^2/sec^2 were considered to be generally not of
# interest because of the large propulsive maneuvers they require.
# A 180-day span for departure date was used because it sufficiently covered the
# range of desired departure energies.
# 
# The 400-day coverage span for Mars arrival dates was used in order to display
# time-of-flight times ranging from 100 to 450 days.
#
# Author: ravi_ram
#

# global constants
mu = 1.32712428e11    # gravitational parameter of the sun (km^3/s^2)
#-------------------------------------------------------------------------------------------------------------------
# Section 1: # calculate c3 and delv values from lambert solution
#-------------------------------------------------------------------------------------------------------------------
def _get_lambert_estimates_(v_dep, v_arr, r1, r2, flight_time_secs,
                           orb_type, M, path):
    # multiple-solution for m>0 cases. not considered here
    # def solve(mu, r1, r2, t_sec, orb_type, path, m)
    v1_list, v2_list = lb.solve(mu, r1, r2, flight_time_secs, orb_type, path, M)
    v1, v2 = v1_list[0], v2_list[0]
    # Solve the problem
    #v1, v2 = lb.solve(mu, r1, r2, flight_time_secs, M=M, prograde=True, low_path=low_path)   
    # compute v_inf for departure and arrival (subtract planet velocities)
    v_inf_dep = np.linalg.norm(v_dep - v1) 
    v_inf_arr = np.linalg.norm(v_arr - v2)
    # characteristic energy. v_inf = orbital velocity when the
    # orbital distance tends to infinity.
    c3_dep = v_inf_dep**2
    c3_arr = v_inf_arr**2
    # ∆V = Vearth(t1) − VT(t1) + VMars (t2) − VT (t2)
    delv_total = v_inf_dep + v_inf_arr    
    #return c3_dep, delv_total
    return [c3_dep, delv_total]

##-------------------------------------------------------------------------------------------------------------------
# Section 1: # get plot data for a departure arrival combination
#------------------------------------------------------------------------------------------------------------------- 
# 
def _get_porkchop_plot_data_(dep_planet_name, jd_dep, arr_planet_name, jd_arr):
    # Departure and arrival planets id
    dep_planet_id = ad.get_planet_id(dep_planet_name)
    arr_planet_id = ad.get_planet_id(arr_planet_name)    
    # get state vector
    coe_dep, r_dep, v_dep, jd_d = ad.get_planet_state_vector(mu, dep_planet_id, jd_dep)
    coe_arr, r_arr, v_arr, jd_a = ad.get_planet_state_vector(mu, arr_planet_id, jd_arr)
    # jd string
    jd_dep_str, jd_arr_str = ad.jd_str(jd_dep), ad.jd_str(jd_arr)
    # time of flight, convert to seconds for lambert's function
    flight_time_days = (jd_arr - jd_dep)
    flight_time_secs = flight_time_days * (24.0 * 60.0 * 60.0)
    
    # lambert estimation (type-I and type-II)
    orb_type, M, low_path = 'prograde', 0.0, 'low'
    c3_and_delv_1 = _get_lambert_estimates_(v_dep, v_arr, r_dep, r_arr,
                                           flight_time_secs, orb_type, M, low_path)
    
    orb_type, M, low_path = 'prograde', 0.0, 'high'
    c3_and_delv_2 = _get_lambert_estimates_(v_dep, v_arr, r_dep, r_arr,
                                           flight_time_secs, orb_type, M, low_path)
    #return
    out = jd_dep_str, jd_arr_str, flight_time_days, \
          c3_and_delv_1, c3_and_delv_2, coe_dep, coe_arr
    return out
#-------------------------------------------------------------------------------------------------------------------
# 
# get porkchop plot contour data
def _generate_porkchop_plot_data_(dep_planet_name, jd_dep_list, arr_planet_name, jd_arr_list):
    # initialize contour lists
    jd_dep_str_list = np.empty(jd_dep_list.shape, dtype=object)
    jd_arr_str_list = np.empty(jd_arr_list.shape, dtype=object)
    # 2d array shape   
    contour_shape = (jd_arr_list.shape[0], jd_dep_list.shape[0])
    tof_days_list = np.zeros( contour_shape, dtype=np.float64)
    c3_dep_1_list = np.zeros( contour_shape, dtype=np.float64)
    c3_dep_2_list = np.zeros( contour_shape, dtype=np.float64)
    delv_t_1_list = np.zeros( contour_shape, dtype=np.float64)
    delv_t_2_list = np.zeros( contour_shape, dtype=np.float64)
    # generate result matrix
    rows, cols = tof_days_list.shape[0], tof_days_list.shape[1]
    for ix in range(0, rows):
        jd_arr_i = jd_arr_list[ix]
        for iy in range(0, cols):
            jd_dep_i = jd_dep_list[iy]
            # calculate plot data
            res = _get_porkchop_plot_data_(dep_planet_name, jd_dep_i, arr_planet_name, jd_arr_i)
            # unpack result
            jd_dep_str, jd_arr_str, flight_time_days, c3_and_delv_1, c3_and_delv_2, coe_dep, coe_arr = res
            c3_dep_1, delv_1_total = c3_and_delv_1
            c3_dep_2, delv_2_total = c3_and_delv_2
            # pack the list
            tof_days_list[ix][iy] = flight_time_days
            c3_dep_1_list[ix][iy] = c3_dep_1
            c3_dep_2_list[ix][iy] = c3_dep_2
            delv_t_1_list[ix][iy] = delv_1_total
            delv_t_2_list[ix][iy] = delv_2_total
            jd_arr_str_list[ix] = jd_arr_str
            jd_dep_str_list[iy] = jd_dep_str
        # end for iy
    #end for ix
    # return list
    out = jd_dep_str_list, jd_arr_str_list, tof_days_list, \
          c3_dep_1_list, c3_dep_2_list, \
          delv_t_1_list, delv_t_2_list
    return out
#-------------------------------------------------------------------------------------------------------------------
# 
# from the given departue, arrival dates, generate a valid
# list of axis dates for plotting
def _generate_date_matrix_(jd_dep, jd_arr):
    # days from dep and arr date
    #I AM CHANGING THIS
    jd_dep_end, jd_arr_end = 365*2, 1000*4 #160, 400 

    # check for date range validity
    val = jd_arr - (jd_dep + jd_dep_end)
    # minimum 90 days time of travel
    if val < 60:
        diff = str(round(val,2))         
        raise Exception("error: tof is %s days. change dep arr dates." % diff  )

    # grid resolution of the porkchop plot
    dt_dep, dt_arr = 10, 60   # 30 60
    # generate list of jd's
    jd_dep_list = np.array( list(range(int(jd_dep), int(jd_dep+jd_dep_end), int(dt_dep) )) )
    jd_arr_list = np.array( list(range(int(jd_arr), int(jd_arr+jd_arr_end), int(dt_arr) )) )
    # return date matrix
    return jd_dep_list, jd_arr_list
#-------------------------------------------------------------------------------------------------------------------
# 
# plot porkchop plot
def plot_porkchop(title, xlist, ylist,
                  xy_contour_data_1, xy_contour_data_2, clevels,
                  xy_tof_data, tlevels):
    
    def set_ticks(ax):
        # major grid
        x_tick_spacing, y_tick_spacing = 2, 5 #2 5
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
        
        # fontsize of major, minor ticks label
        ax.xaxis.set_tick_params(labelsize=7, rotation=90)
        ax.yaxis.set_tick_params(labelsize=7)
        
        ax.set_xlabel("Dep Date (dd-mm-yyyy)", fontsize=8)        
        ax.set_ylabel("Arr Date (dd-mm-yyyy)", fontsize=8)

        # Customize the major grid
        ax.grid(which='major', linestyle='dashdot', linewidth='0.5', color='gray')
        
        # Customize the minor grid
        ax.grid(which='minor', linestyle='dotted', linewidth='0.5', color='gray')
        
        # Turn on the minor ticks (minor grid)
        ax.minorticks_on()
        
        # Turn off the display of all ticks.
        ax.tick_params(which='both',
                        top='off',
                        left='off',
                        right='off',
                        bottom='off')
        # end function
        return
    #fig, ax = plt.subplots(figsize=(14,8))
    #ax.set_aspect('auto')
    #set_ticks(ax)
    #ax.set_title(title, fontsize=10)
    #plt.tight_layout()
    #plt.show()
    #return
                        
#-------------------------------------------------------------------------------------------------------------------
# 
    # countour text format ( include 'days')
    def tp_fmt(x):
        s = f"{x:.1f}"
        return rf"{s} days"
    
    # init figure    
    plt.figure(figsize=(12,9))

    # find the minimum value with the corresponding dep, arr dates
    # Type-I minimum
    # draw ∆V contours
    cp1 = plt.contour(xlist, ylist, xy_contour_data_1, clevels, cmap="rainbow")
    plt.clabel(cp1, inline=True, fontsize=7)

    # find the minimum value with the corresponding dep, arr dates
    # type-II
    # draw ∆V contours
    cp2 = plt.contour(xlist, ylist, xy_contour_data_2, clevels, cmap="rainbow")
    plt.clabel(cp2, inline=True, fontsize=7)
    
    # draw time-of-flight contours
    tp = plt.contour(xlist, ylist, xy_tof_data, tlevels, colors='k',linestyles=':')
    plt.clabel(tp, inline=True, fmt=tp_fmt, fontsize=7)
    
    # set title and x, y labels
    plt.title(title, fontdict={'fontsize':8})
    
    # set grids and ticks
    set_ticks(plt.gca())
    
    # set aspect ratio and layout
    plt.gca().set_aspect(1) #'auto')   
    plt.tight_layout()
    # show
    plt.show()
    return
#-------------------------------------------------------------------------------------------------------------------
# 
# main function to create the porkchop plot
def make_porkchop_plot(dep_planet_name, dep_date, arr_planet_name, arr_date, plot_type ='delv_plot'):
    # verify departure and arrival dates
    dt_dep, dt_arr = 0, 0 
    try:
        dt_dep = datetime.datetime.strptime(dep_date, '%d-%m-%Y')
        dt_arr = datetime.datetime.strptime(arr_date, '%d-%m-%Y')
    except ValueError:
        print ('error : wrong date format. [verify as dd-mm-yyyy]')
    
    # dep time
    d, m, y = dt_dep.day, dt_dep.month, dt_dep.year    
    jd_dep = ad.greg_to_jd(y, m, d)
    # arrival time
    d, m, y = dt_arr.day, dt_arr.month, dt_arr.year     
    jd_arr = ad.greg_to_jd(y, m, d)
    
    # date matrix
    res = _generate_date_matrix_(jd_dep, jd_arr)
    jd_dep_list, jd_arr_list = res

    # mu, dep_planet_name, jd_dep, arr_planet_name, jd_arr
    res = _generate_porkchop_plot_data_(dep_planet_name, jd_dep_list, arr_planet_name, jd_arr_list)   
    jd_dep_str_list, jd_arr_str_list, tof_days_list, c3_dep_1_list, c3_dep_2_list, delv_t_1_list, delv_t_2_list = res
    
    # contour levels    
    #c3_levels = [8, 9, 10, 12, 14, 15, 17, 18, 20, 22, 24, 30, 50]
    #t_levels  = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6500, 7000, 10000]

    c3_min, c3_max = np.nanmin([c3_dep_1_list, c3_dep_2_list]), np.nanmax([c3_dep_1_list, c3_dep_2_list])
    tof_min, tof_max = np.nanmin(tof_days_list), np.nanmax(tof_days_list)

    c3_levels = np.linspace(c3_min, c3_max, 12)   # 12 evenly spaced levels
    t_levels  = np.linspace(tof_min, tof_max, 10) # 10 evenly spaced levels

    
    # plot    
    if plot_type == 'delv_plot':
        title = 'Porkchop plot (∆V Total = ||$ΔV_{' +dep_planet_name+'}|| + ||ΔV_{'+ arr_planet_name +'}$||)' + '\n'
        plot_porkchop(title, jd_dep_str_list, jd_arr_str_list,
                         c3_dep_1_list, c3_dep_2_list, c3_levels,
                         tof_days_list, t_levels)
    elif plot_type == 'c3_plot':       
        title = 'Porkchop plot (C3-characteristic energy = $v_{id}^{2}$)' + '\n'
        plot_porkchop(title, jd_dep_str_list, jd_arr_str_list,
                         delv_t_1_list, delv_t_2_list, c3_levels,
                         tof_days_list, t_levels)
    else: raise Exception("error: plot types should be c3_plot or delv_plot")   
    # return    
    return 
#-------------------------------------------------------------------------------------------------------------------
# 

# main function
if __name__ == "__main__":
    # Mission space with nodal transfer.
    dep_planet = 'Earth'; dep_date = '25-12-2035'; #(format - dd-mm-yyyy)
    arr_planet = 'Neptune';  arr_date = '25-12-2045'; #(format - dd-mm-yyyy)
    
    # solve
    plot_values = 'delv_plot' # 'delv_plot' or 'c3_plot'
    make_porkchop_plot(dep_planet, dep_date, arr_planet, arr_date, plot_values)
#-------------------------------------------------------------------------------
