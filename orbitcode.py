# orbitcode.py
#
# Compute model orbital positions for Visual Binary
# Input parameters:
#    ELarr: array of orbital parameters
#           [period,Tperi,ecc,major,inc,W_cap,W_low]
#    time:  array containing times for which to compute position
# Output parameters:
#    theta_f: array for model PA
#    rho_f: array for model separation
#    flag_wa: set this flag if omega = omega_A (otherwise omega = omega_B)
#             This is necessary for simultaneous VB+SB fits.  SB orbits
#             define omega for primary, whereas VB orbits define omega 
#             for secondary. [, /flag_wa]
#
# based on Gail Schaefer's IDL code 
#
# EXAMPLE USAGE:
#    import numpy as np
#    from calc_vbfit import calc_vbfit
#    
#    # omega_B, angles in degrees, output in degrees
#    ELarr = np.array([35.3683, 54293.2078, 0.6476, 5.0, 140.64, 336.2, 325.18 - 180.])
#    time = np.array([58365.7858, 58744.8179])
#    
#    # calculate model positions
#    rho, theta = calc_vbfit(ELarr, time)
# ------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm
from scipy import interpolate
from scipy.misc import derivative as deriv
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def calc_vbfit(ELarr, time, flagwa=0):
	# Define orbital element parameters
	period = ELarr[0]
	Tperi  = ELarr[1]
	ecc    = ELarr[2]
	major  = ELarr[3]   # mas
	inc    = ELarr[4]   # RADIANS 
	W_cap  = ELarr[5]   # RADIANS # Omega_B
	W_low  = ELarr[6]   # RADIANS # omega_B
    
    # IF YOU INPUT WA, CONVERT TO WB
	if (flagwa == 1) and (W_low <= np.pi):
		W_low = W_low + np.pi
	if (flagwa == 1) and (W_low > np.pi):
		W_low = W_low - np.pi

	num = len(time)

	# calc_ei.pro
	# Determine the eccentric anomalies (Ei) as a function of time: 
	#  Ei = angular position of obs position projected up to a circular orbit
	#     mu(ti - T) = Ei - esin(Ei)  ... Kepler's Equation
	#     where mu = 360/P = 2*pi/P
	# Solve this transcendental equation through an iterative procedure.
	# written by Gail Shaefer
	Ei = np.zeros(num) 
	mu = 2 * np.pi / period		# ex - degrees / day
	
	for i in range(0, num):
		# Mi: mean anomoly - increases uniformly with time
		# zero at time Tperi, 2Pi each orbit
		# percent of orbital period that has passed, expressed as an angle	
		Mi = mu * (time[i] - Tperi) 	
		# deg / day     *     number of days   =   deg 

		# reduce to same epoch
		Mi = 2 * np.pi * (  (time[i] - Tperi)/period - int((time[i] - Tperi)/period) )
        
		# keep Mi between 0 and 2Pi
		if Mi < 0.0:
			Mi = Mi + 2*np.pi
		if Mi > 2*np.pi:
			Mi = Mi - 2*np.pi

		# solve_trans.pro
		# Solve transcendental equation of the form E - esinE = M.
		# Use iterative procedure to determine E.
		# Initial approximation: E_0 = M + esinM - e^2/2 sin(2M)
		# Improve solution by iterating the following to formulae:
		#	M_0 = E_0 - esin(E_0)
		#	E_1 = E_0 + (M - M_0)/(1 - ecos(E_0))
		#	(derivative of Kepler's equation)
		#
		# Method adapted from Heintz 1978 (p.34-35)
		# Results compared with point-&-click graphical method.  Iterative approach 
		# leads to exact solution that satisfies E - esinE = M.  Therefore, 
		# point-&-click method is subsequently removed from orbit fitting.
		#
		# INPUT:
		#	e: eccentricity
		#	M: mean anomaly   M= 2*Pi/P
		#
		# OUTPUT:
		#	EE: eccentric anomaly
		#
		# Created: 9 May 2002 by Gail Shaefer
		# 12 December 2007: add escape route in case routine doesn't converge

		# Initial approximation:
		# PYTHON SIN USES RADIANS
		EE = Mi + ecc*np.sin(Mi) + ecc**2/2*np.sin(2*Mi)

		EEi = 0.0	# parameter to hold initial value once enter while loop
		count = 0.0

		while ((abs(EE - EEi) > 0.00001) & (count < 10000)):
			EEi = EE
			Mi0 = EEi - ecc*np.sin(EEi)
			EE = EEi + (Mi - Mi0)/(1 - ecc*np.cos(EEi))
			count=count+1

		# return EE
		Eit = EE
			
		
		# keep Ei between 0 and 2Pi
		if Eit < 0.0:
			Eit = Eit + 2*np.pi
		if Eit > 2*np.pi:
			Eit = Eit - 2*np.pi

		Ei[i] = Eit	# radians
	
	# Determine true anomalies (nu)
	# nu = angle around from periastron
	nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(Ei/2.0))	# radians
	# keep nu between 0 and 2Pi
	for i in range(0, num):
		if nu[i] < 0.0: nu[i] = nu[i] + 2*np.pi

	# Determine radius vectors:
	rad = major*(1 - ecc*np.cos(Ei))

	# in order to check the value (correct quadrant) of nu:
	rad1 = major*(1-ecc**2)/(1 + ecc*np.cos(nu))

	# Determine (rho_f,theta_f) position from orbital elements for times of observation
	# (rad, nu) are the birds eye view coordinates, 
	#     but we need the coordinates projected onto the plane of the sky (rho, theta)
	theta_f = W_cap + np.arctan(np.tan(nu + W_low)*np.cos(inc))
	rho_f = rad*np.cos(nu + W_low)/np.cos(theta_f - W_cap)

	for i in range(0, num):
		# convert rad and rad1 from double precision to float
		# this will avoid round-off errors when making comparisons
		frad = rad
		frad1 = rad1

		#check that nu is in appropriate quadrant
		if frad[i] != frad1[i]:
			nu[i] = nu[i] + np.pi
		if nu[i] >= 2*np.pi:
			nu[i] = nu[i] - 2*np.pi


		#check that theta_f is in correct quadrant
		if rho_f[i] < 0:
			rho_f[i] = abs(rho_f[i])
			theta_f[i] = theta_f[i] + np.pi

		# theta between 0 and 2pi
		if theta_f[i] >= 2*np.pi:
			theta_f[i] = theta_f[i] - 2*np.pi
		if theta_f[i] < 0:
			theta_f[i] = theta_f[i] + 2*np.pi
	
	return(np.array(rho_f), np.array(theta_f))  # mas, radians

# ------------------------------------------------------------------------------------------

def calc_sb1fit(ELarr, time):
    # Define orbital element parameters
    period = ELarr[0]
    Tperi  = ELarr[1]
    ecc    = ELarr[2]
    K1     = ELarr[3]
    omega  = ELarr[4]   # RADIANS # omega_A
    Vsys   = ELarr[5]  

    num = len(time)
    
    # calc_ei.pro
    # Determine the eccentric anomalies (Ei) as a function of time: 
    #  Ei = angular position of obs position projected up to a circular orbit
    #     mu(ti - T) = Ei - esin(Ei)  ... Kepler's Equation
    #     where mu = 360/P = 2*pi/P
    # Solve this transcendental equation through an iterative procedure.
    # written by Gail Shaefer
    Ei = np.zeros(num) 
    mu = 2 * np.pi / period		# ex - degrees / day

    for i in range(0, num):
        # Mi: mean anomoly - increases uniformly with time
        # zero at time Tperi, 2Pi each orbit
        # percent of orbital period that has passed, expressed as an angle	
        Mi = mu * (time[i] - Tperi) 	
        # deg / day     *     number of days   =   deg 

        # reduce to same epoch
        Mi = 2 * np.pi * (  (time[i] - Tperi)/period - int((time[i] - Tperi)/period) )

        # keep Mi between 0 and 2Pi
        if Mi < 0.0:
            Mi = Mi + 2*np.pi
        if Mi > 2*np.pi:
            Mi = Mi - 2*np.pi

        # solve_trans.pro
        # Solve transcendental equation of the form E - esinE = M.
        # Use iterative procedure to determine E.
        # Initial approximation: E_0 = M + esinM - e^2/2 sin(2M)
        # Improve solution by iterating the following to formulae:
        #	M_0 = E_0 - esin(E_0)
        #	E_1 = E_0 + (M - M_0)/(1 - ecos(E_0))
        #	(derivative of Kepler's equation)
        #
        # Method adapted from Heintz 1978 (p.34-35)
        # Results compared with point-&-click graphical method.  Iterative approach 
        # leads to exact solution that satisfies E - esinE = M.  Therefore, 
        # point-&-click method is subsequently removed from orbit fitting.
        #
        # INPUT:
        #	e: eccentricity
        #	M: mean anomaly   M= 2*Pi/P
        #
        # OUTPUT:
        #	EE: eccentric anomaly
        #
        # Created: 9 May 2002 by Gail Shaefer
        # 12 December 2007: add escape route in case routine doesn't converge

        # Initial approximation:
        # PYTHON SIN USES RADIANS
        EE = Mi + ecc*np.sin(Mi) + ecc**2/2*np.sin(2*Mi)

        EEi = 0.0	# parameter to hold initial value once enter while loop
        count = 0.0

        while ((abs(EE - EEi) > 0.00001) & (count < 10000)):
            EEi = EE
            Mi0 = EEi - ecc*np.sin(EEi)
            EE = EEi + (Mi - Mi0)/(1 - ecc*np.cos(EEi))
            count=count+1

        # return EE
        Eit = EE


        # keep Ei between 0 and 2Pi
        if Eit < 0.0:
            Eit = Eit + 2*np.pi
        if Eit > 2*np.pi:
            Eit = Eit - 2*np.pi

        Ei[i] = Eit	# radians 
        
    # Determine true anomalies (nu)
    # nu = angle around from periastron
    nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(Ei/2.0))	# radians
    # keep nu between 0 and 2Pi
    for i in range(0, num):
        if nu[i] < 0.0: nu[i] = nu[i] + 2*np.pi

    # Check sign of nu
    cosnu = np.cos(nu)
    sinnu = np.sin(nu)
    actualcosnu = (np.cos(Ei) - ecc) / (1-ecc*np.cos(Ei))
    actualsinnu = np.sqrt(1-ecc**2)*np.sin(Ei) / (1-ecc*np.cos(Ei))
    for i in range(0,num):
        if cosnu[i]/abs(cosnu[i]) != actualcosnu[i]/abs(actualcosnu[i]):
            if sinnu[i]/abs(sinnu[i]) != actualsinnu[i]/abs(actualsinnu[i]):
                nu[i] = nu[i] + np.pi
        
    # calculate RV
    V1 = K1 * (ecc*np.cos(omega) + np.cos(nu+omega)) + Vsys
    return(V1)

# ------------------------------------------------------------------------------------------

def calc_sb2fit(ELarr, time):
    # Define orbital element parameters
    period = ELarr[0]
    Tperi  = ELarr[1]
    ecc    = ELarr[2]
    K1     = ELarr[3]
    K2     = ELarr[4]
    omega  = ELarr[5]   # RADIANS # omega_A
    Vsys   = ELarr[6]  

    num = len(time)
    
    # calc_ei.pro
    # Determine the eccentric anomalies (Ei) as a function of time: 
    #  Ei = angular position of obs position projected up to a circular orbit
    #     mu(ti - T) = Ei - esin(Ei)  ... Kepler's Equation
    #     where mu = 360/P = 2*pi/P
    # Solve this transcendental equation through an iterative procedure.
    # written by Gail Shaefer
    Ei = np.zeros(num) 
    mu = 2 * np.pi / period		# ex - degrees / day

    for i in range(0, num):
        # Mi: mean anomoly - increases uniformly with time
        # zero at time Tperi, 2Pi each orbit
        # percent of orbital period that has passed, expressed as an angle	
        Mi = mu * (time[i] - Tperi) 	
        # deg / day     *     number of days   =   deg 

        # reduce to same epoch
        Mi = 2 * np.pi * (  (time[i] - Tperi)/period - int((time[i] - Tperi)/period) )

        # keep Mi between 0 and 2Pi
        if Mi < 0.0:
            Mi = Mi + 2*np.pi
        if Mi > 2*np.pi:
            Mi = Mi - 2*np.pi

        # solve_trans.pro
        # Solve transcendental equation of the form E - esinE = M.
        # Use iterative procedure to determine E.
        # Initial approximation: E_0 = M + esinM - e^2/2 sin(2M)
        # Improve solution by iterating the following to formulae:
        #	M_0 = E_0 - esin(E_0)
        #	E_1 = E_0 + (M - M_0)/(1 - ecos(E_0))
        #	(derivative of Kepler's equation)
        #
        # Method adapted from Heintz 1978 (p.34-35)
        # Results compared with point-&-click graphical method.  Iterative approach 
        # leads to exact solution that satisfies E - esinE = M.  Therefore, 
        # point-&-click method is subsequently removed from orbit fitting.
        #
        # INPUT:
        #	e: eccentricity
        #	M: mean anomaly   M= 2*Pi/P
        #
        # OUTPUT:
        #	EE: eccentric anomaly
        #
        # Created: 9 May 2002 by Gail Shaefer
        # 12 December 2007: add escape route in case routine doesn't converge

        # Initial approximation:
        # PYTHON SIN USES RADIANS
        EE = Mi + ecc*np.sin(Mi) + ecc**2/2*np.sin(2*Mi)

        EEi = 0.0	# parameter to hold initial value once enter while loop
        count = 0.0

        while ((abs(EE - EEi) > 0.00001) & (count < 10000)):
            EEi = EE
            Mi0 = EEi - ecc*np.sin(EEi)
            EE = EEi + (Mi - Mi0)/(1 - ecc*np.cos(EEi))
            count=count+1

        # return EE
        Eit = EE


        # keep Ei between 0 and 2Pi
        if Eit < 0.0:
            Eit = Eit + 2*np.pi
        if Eit > 2*np.pi:
            Eit = Eit - 2*np.pi

        Ei[i] = Eit	# radians 
        
    # Determine true anomalies (nu)
    # nu = angle around from periastron
    nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(Ei/2.0))	# radians
    # keep nu between 0 and 2Pi
    for i in range(0, num):
        if nu[i] < 0.0: nu[i] = nu[i] + 2*np.pi

    # Check sign of nu
    cosnu = np.cos(nu)
    sinnu = np.sin(nu)
    actualcosnu = (np.cos(Ei) - ecc) / (1-ecc*np.cos(Ei))
    actualsinnu = np.sqrt(1-ecc**2)*np.sin(Ei) / (1-ecc*np.cos(Ei))
    for i in range(0,num):
        if cosnu[i]/abs(cosnu[i]) != actualcosnu[i]/abs(actualcosnu[i]):
            if sinnu[i]/abs(sinnu[i]) != actualsinnu[i]/abs(actualsinnu[i]):
                nu[i] = nu[i] + np.pi
        
    # calculate RV
    V1 = K1 * (ecc*np.cos(omega) + np.cos(nu+omega)) + Vsys
    V2 = -1 * K2 * (ecc*np.cos(omega) + np.cos(nu+omega)) + Vsys
    return(V1, V2)

# ------------------------------------------------------------------------------------------

def todcorfun(w,f,g1,g2,aa,dshift,nshift,pflag=False):
	
	# Wavelength grid parameters
	junk=open('binlog.txt', 'r')
	x = junk.read()
	x = x.strip().split()
	dl = float(x[1])	# ln(w) per pixel
	junk.close()
	velpix = dl*2.997925e5	# vel shift per pixel


	# Pixel shift 
	nx = nshift*2 + 1
	ny = nshift*2 + 1
	R  = np.zeros((nx, ny))	# CCF grid

	# Grid spacing in pixels
	dx = dshift
	dy = dshift

	# RMS of spectrum
	N = float(len(f))
	sigf = np.sqrt(np.sum(f**2.)/N)
	sig1 = np.sqrt(np.sum(g1**2.)/N)
	sig2 = np.sqrt(np.sum(g2**2.)/N)

	# Flux ratio = aa
	if aa <= 0:
		aa=1.
	ap = sig2/sig1 * aa		# a' in paper
	ag = np.zeros((nx, ny))		# calculated at each grid point


	# Pixel shifts
	lag = np.arange(nx)*dx - nshift*dx
	v = lag*velpix


	# Run CCF GRID
	for i in range(0, nx):
		for j in range(0, ny):
			s1 = int(lag[i])	# shift for g1
			s2 = int(lag[j])	# shift for g2
			
			#c1 =  np.correlate(f, np.roll(g1, s1)) / N / sigf / sig1	# CCF of template 1 wrt obs+lag 
			#c2 =  np.correlate(f, np.roll(g2, s2)) / N / sigf / sig2	# CCF of template 2 wrt obs+lag
			#c12 = np.correlate(g1, np.roll(g2, s2 - s1))/N/sig1/sig2	# CCF of template 1 wrt template 2
			
			ff = f - np.mean(f)
			gg = g1 - np.mean(g1)
			c1 = np.sum( ff * (np.roll(g1, s1) - np.mean(g1)) ) / np.sqrt(np.sum(ff**2)) / np.sqrt(np.sum((np.roll(g1, s1) - np.mean(g1))**2))   
			c2 = np.sum( ff * (np.roll(g2, s2) - np.mean(g2)) ) / np.sqrt(np.sum(ff**2)) / np.sqrt(np.sum((np.roll(g2, s2) - np.mean(g2))**2))   
			c12= np.sum( gg * (np.roll(g2, s2-s1) - np.mean(g2)) ) / np.sqrt(np.sum(gg**2)) / np.sqrt(np.sum( (np.roll(g2, s2-s1) - np.mean(g2))**2))   
			R[i,j] = (c1 + ap*c2) / np.sqrt(1. + 2.*ap*c12 + ap**2.) 	# eq A3  (CCF)
			ag[i,j] = (sig1/sig2) * (c1*c12 - c2) / (c2*c12 - c1) 		# eq A4  (flux ratio)

	# Find max
	rmax = np.amax(R)
	ind = np.unravel_index(np.argmax(R), (nx, nx))
	ix = ind[0] 	# index for best RV1
	iy = ind[1] 	# index for best RV2

	if ix <= 20 or ix >= nx -20 or iy <= 20 or iy >= nx -20:
		#print('  ** Too close to edge. Returning 0...')
		vt = [0,0]
		evt = [1000,1000]
		aa = 1.0
		return(vt, evt, aa, R, v)
	

	rp = R[:,iy]	# CCF vs RV1
	rs = R[ix,:]	# CCF vs RV2


	# Best fit RV 1
	c = np.polyfit(v[ix-20:ix+20], rp[ix-20:ix+20], 10)
	p1 = np.poly1d(c)
	dp1 = p1.deriv()
	d2p1 = dp1.deriv()
	vm1 = np.arange(400)*0.1 + v[ix] - 20
	rpm = p1(vm1)
	rpm2 = dp1(vm1)
	x = rpm2 > 0
	vx = vm1[max(np.where(x)[0])]

	# Best fit RV 2
	c = np.polyfit(v[iy-20:iy+20], rs[iy-20:iy+20], 10)
	p2 = np.poly1d(c)
	dp2 = p2.deriv()
	d2p2 = dp2.deriv()
	vm2 = np.arange(400)*0.1 + v[iy] - 20
	rsm = p2(vm2)
	rsm2 = dp2(vm2)
	x = rsm2 > 0
	vy = vm2[max(np.where(x)[0])]

	# Errors
	dxx = np.abs(d2p1(vx))
	dyy = np.abs(d2p2(vy))
	dvx =  1. /  np.sqrt( len(w) * dxx * rmax / (1.-rmax**2) )   *  velpix
	dvy =  1. /  np.sqrt( len(w) * dyy * rmax / (1.-rmax**2) )   *  velpix

	vt = [vx, vy]
	evt = [dvx, dvy]


	# PLOTS
	if pflag == True:	
		plt.figure(figsize=(12,6))
		plt.subplot(221)
		plt.plot(w,f)
		model = np.roll(g1, int(lag[ix]))*(1./(1.+aa)) + np.roll(g2, int(lag[iy]))*(1. - 1./(1.+aa))
		plt.plot(w,model)
		plt.subplot(222)
		plt.contourf( v, v, R)
		plt.plot([v[0],v[-1]], [v[0],v[-1]], color='black', linestyle='--')
		plt.xlabel(' RV 1 (km/s) ')
		plt.ylabel(' RV 2 (km/s) ')
		plt.plot([v[0],v[-1]], [0,0], color='black')
		plt.plot([0,0],[v[0], v[-1]], color='black')
		plt.plot([vx], [vy], 'o', color='black')
		plt.subplot(223)
		plt.plot(v, rp, 'o', ms=2)
		plt.plot(vm1, rpm)
		plt.xlabel(' RV 1 (km/s) ')
		plt.ylim(0,1)
		plt.subplot(224)
		plt.plot(v, rs, 'o', ms=2)
		plt.plot(vm2, rsm)
		plt.xlabel(' RV 2 (km/s) ')
		plt.ylim(0,1)
		plt.show()
		plt.close()

	# Bilinear interpolate the flux ratio
	f = interpolate.interp2d(v,v,ag, kind='cubic')
	aa = f(vx, vy)[0]


	return(vt, evt, aa, R, v)
    
# ------------------------------------------------------------------------------------------
    
def todcor1d(w,f,g1,aa,dshift,nshift,pflag):
    
    # Wavelength grid parameters
    junk=open('binlog.txt', 'r')
    x = junk.read()
    x = x.strip().split()
    dl = float(x[1])	# ln(w) per pixel
    junk.close()
    velpix = dl*2.997925e5	# vel shift per pixel
    
    # Pixel shift 
    nx = nshift*2 + 1
    R  = np.zeros(nx)	# CCF grid
    
    # Grid spacing in pixels
    dx = dshift
    
    # RMS of spectrum
    N = float(len(f))
    sigf = np.sqrt(np.sum(f**2.)/N)
    sig1 = np.sqrt(np.sum(g1**2.)/N)
    
    # Pixel shifts
    lag = np.arange(nx)*dx - nshift*dx
    v = lag*velpix
    
    # Run CCF GRID
    # R[i] = sum(  (x[i] - xmean) * (y[i] - ymean)  ) / sqrt(sum((x[i] - xmean)^2)) / sqrt(sum((y[i] - ymean)^2))
    
    for i in range(0,nx):
        s1 = int(lag[i])	# shift for g1
        ff = np.nan_to_num(f - 1)        #np.mean(f)
        gg = np.nan_to_num(np.roll(g1, s1) - 1)        # np.mean(g1) 
        numer = np.sum( ff * gg )
        denom = np.sqrt(np.sum(ff**2)) * np.sqrt(np.sum(gg**2))   
        R[i] = numer / denom         
        
    return(R, v)

# ------------------------------------------------------------------------------------------

def calc_deriv_vb_ell(ELarr, elfix, mfit, time, xp_d, yp_d, err_maj, err_min, err_pa, flag_wa=0):
    #calc_deriv_vb_ell, ELarr, elfix, mfit, time, xp_d, yp_d, err_maj, err_min, err_pa, theta_f, rho_f, xp_f, yp_f, chimat, colmat, flag_wa = flag_wa
    # Define orbital element parameters
    period= ELarr[0]
    Tperi = ELarr[1]
    ecc   = ELarr[2]
    major = ELarr[3]
    inc   = ELarr[4]  # RADIANS   
    W_cap = ELarr[5]  # RADIANS   # default = Omega_B
    W_low = ELarr[6]  # RADIANS   # default = omega_B
    
    # flagwa=1 if input is wA not WB
    if (flag_wa == 1) and (W_low <= np.pi):
        W_low = W_low + np.pi
    if (flag_wa == 1) and (W_low > np.pi):
        W_low = W_low - np.pi
        
    #print(W_cap)

    num = len(time)
    # calc_ei.pro
    # Determine the eccentric anomalies (Ei) as a function of time: 
    #  Ei = angular position of obs position projected up to a circular orbit
    #     mu(ti - T) = Ei - esin(Ei)  ... Kepler's Equation
    #     where mu = 360/P = 2*pi/P
    # Solve this transcendental equation through an iterative procedure.
    # written by Gail Shaefer
    Ei = np.zeros(num) 
    mu = 2 * np.pi / period		# ex - degrees / day

    for i in range(0, num):
    	# Mi: mean anomoly - increases uniformly with time
    	# zero at time Tperi, 2Pi each orbit
    	# percent of orbital period that has passed, expressed as an angle	
    	Mi = mu * (time[i] - Tperi) 	
    	# deg / day     *     number of days   =   deg 

    	# reduce to same epoch
    	Mi = 2 * np.pi * (  (time[i] - Tperi)/period - int((time[i] - Tperi)/period) )

    	# keep Mi between 0 and 2Pi
    	if Mi < 0.0:
    		Mi = Mi + 2*np.pi
    	if Mi > 2*np.pi:
    		Mi = Mi - 2*np.pi


    	# solve_trans.pro
    	# Solve transcendental equation of the form E - esinE = M.
    	# Use iterative procedure to determine E.
    	# Initial approximation: E_0 = M + esinM - e^2/2 sin(2M)
    	# Improve solution by iterating the following to formulae:
    	#	M_0 = E_0 - esin(E_0)
    	#	E_1 = E_0 + (M - M_0)/(1 - ecos(E_0))
    	#	(derivative of Kepler's equation)
    	#
    	# Method adapted from Heintz 1978 (p.34-35)
    	# Results compared with point-&-click graphical method.  Iterative approach 
    	# leads to exact solution that satisfies E - esinE = M.  Therefore, 
    	# point-&-click method is subsequently removed from orbit fitting.
    	#
    	# INPUT:
    	#	e: eccentricity
    	#	M: mean anomaly   M= 2*Pi/P
    	#
    	# OUTPUT:
    	#	EE: eccentric anomaly
    	#
    	# Created: 9 May 2002 by Gail Shaefer
    	# 12 December 2007: add escape route in case routine doesn't converge

    	# Initial approximation:
    	# PYTHON SIN USES RADIANS
    	EE = Mi + ecc*np.sin(Mi) + ecc**2/2*np.sin(2*Mi)

    	EEi = 0.0	# parameter to hold initial value once enter while loop

    	count = 0.0

    	while ((abs(EE - EEi) > 0.000001) & (count < 10000)):
    		EEi = EE
    		Mi0 = EEi - ecc*np.sin(EEi)
    		EE = EEi + (Mi - Mi0)/(1 - ecc*np.cos(EEi))
    		count=count+1

    	# return EE
    	Eit = EE
		
	
    	# keep Ei between 0 and 2Pi
    	if Eit < 0.0:
    		Eit = Eit + 2*np.pi
    	if Eit > 2*np.pi:
    		Eit = Eit - 2*np.pi

    	Ei[i] = Eit	# radians

    # Determine true anomalies (nu)
    # nu = angle around from periastron
    nu = 2*np.arctan(    np.sqrt((1+ecc)/(1-ecc))   *   np.tan(Ei/2.0)    )	# radians
    
    # Determine radius vectors:
    rad = major*(1 - ecc*np.cos(Ei))

    # in order to check the value (correct quadrant) of nu:
    rad1 = major*(1-ecc**2)/(1 + ecc*np.cos(nu))

    # Determine (rho_f,theta_f) position from orbital elements for times of observation
    # (rad, nu) are the birds eye view coordinates, 
    #     but we need the coordinates projected onto the plane of the sky (rho, theta)
    theta_f = W_cap + np.arctan(np.tan(nu + W_low)*np.cos(inc))
    rho_f = rad*np.cos(nu + W_low)/np.cos(theta_f - W_cap)
    
    # Compute fit values xp_f and yp_f
    xp_f = rho_f*np.cos(theta_f - err_pa)
    yp_f = rho_f*np.sin(theta_f - err_pa)
    

    for i in range(0, num):
    	# convert rad and rad1 from double precision to float
    	# this will avoid round-off errors when making comparisons
    	frad = rad
    	frad1 = rad1

    	#check that nu is in appropriate quadrant
    	if frad[i] != frad1[i]:
    		nu[i] = nu[i] + np.pi
    	if nu[i] >= 2*np.pi:
    		nu[i] = nu[i] - 2*np.pi

    	# keep nu between 0 and 2Pi
    	if nu[i] < 0.0:
    		nu[i] = nu[i] + 2*np.pi

    	#check that theta_f is in correct quadrant
    	if rho_f[i] < 0:
    		rho_f[i] = abs(rho_f[i])
    		theta_f[i] = theta_f[i] + np.pi

    	# theta between 0 and 2pi
    	if theta_f[i] >= 2*np.pi:
    		theta_f[i] = theta_f[i] - 2*np.pi
    	if theta_f[i] < 0:
    		theta_f[i] = theta_f[i] + 2*np.pi


    # Calculate derivatives evaluated at initial conditions 
    #	drho/del = (dx/dP,dx/dT,dx/de,dx/da,dx/di,dx/dW,dx/dw)
    #	dtheta/del = (dy/dP,dy/dT,dy/de,dy/da,dy/di,dy/dW,dy/dw)
    # where (x,y) -> (rho,theta) and
    # del = (P,T,e,a,i,Omega,omega)

    #initialize derivative arrays 
    dxp_del = np.zeros((mfit,num))	#7: number of orbital elements
    dyp_del = np.zeros((mfit,num))	#num: number of data points
    chimat = np.zeros((mfit,mfit))+0.0001 	#matrix for minimizing chi**2
    colmat = np.zeros(mfit)+0.0001	    #column matrix for minimizing chi**2
    
    delta_el = np.zeros(mfit)		#solution to matrix equation
    timediff=np.zeros(num)

    for i in range(0, num):
    	# reduce time difference to same epoch
    	# (fraction of period covered at time t)
        fracP = (time[i] - Tperi)/period - np.round((time[i] - Tperi)/period)
        timediff[i] = fracP

    dEi_dP = -2*np.pi*(time - Tperi)/period**2/(1. - ecc*np.cos(Ei))
    dEi_dT = -2*np.pi/period/(1. - ecc*np.cos(Ei))
    dEi_de = np.sin(Ei)/(1. - ecc*np.cos(Ei))

    dnu_dP = np.sqrt(1. - ecc**2)/(1. - ecc*np.cos(Ei))*dEi_dP
    dnu_dT = np.sqrt(1. - ecc**2)/(1. - ecc*np.cos(Ei))*dEi_dT
    dnu_de = ((1. - ecc**2)*dEi_de + np.sin(Ei))/np.sqrt(1. - ecc**2)/(1. - ecc*np.cos(Ei))
    dtheta_dP = np.cos(inc)*(np.cos(theta_f - W_cap)/np.cos(nu + W_low))**2*dnu_dP
    dtheta_dT = np.cos(inc)*(np.cos(theta_f - W_cap)/np.cos(nu + W_low))**2*dnu_dT
    dtheta_de = np.cos(inc)*(np.cos(theta_f - W_cap)/np.cos(nu + W_low))**2*dnu_de
    dtheta_da = np.zeros(num)
    dtheta_di = -1*np.tan(inc)*np.cos(theta_f - W_cap)*np.sin(theta_f - W_cap)
    dtheta_dWc = np.zeros(num) + 1.
    dtheta_dwl = np.cos(inc)*(np.cos(theta_f - W_cap)/np.cos(nu + W_low))**2

    drho_dP = rho_f*(ecc*np.sin(Ei)/(1. - ecc*np.cos(Ei))*dEi_dP \
    		- np.tan(theta_f - W_cap)/np.cos(inc)*dnu_dP \
    		+ np.tan(theta_f - W_cap)*dtheta_dP)
    drho_dT = rho_f*(ecc*np.sin(Ei)/(1. - ecc*np.cos(Ei))*dEi_dT \
    		- np.tan(theta_f - W_cap)/np.cos(inc)*dnu_dT \
    		+ np.tan(theta_f - W_cap)*dtheta_dT)
    drho_de = rho_f*((-np.cos(Ei) + ecc*np.sin(Ei)*dEi_de)/(1. - ecc*np.cos(Ei)) \
    		- np.tan(theta_f - W_cap)/np.cos(inc)*dnu_de \
    		+ np.tan(theta_f - W_cap)*dtheta_de)
    drho_da = rho_f/major
    drho_di = rho_f*np.tan(theta_f - W_cap)*dtheta_di
    drho_dWc = np.zeros(num)
    drho_dwl = rho_f*np.tan(theta_f - W_cap)*(-1./np.cos(inc) + dtheta_dwl)

    #dxp_del = (dx_dP,dx_dT,dx_de,dx_da,dx_di,dx_dW,dx_dw)
    #dyp_del = (dy_dP,dy_dT,dy_de,dy_da,dy_di,dy_dW,dy_dw)


    k=0
    if (elfix[0] != 0):
    	dxp_del[k,:] = np.cos(theta_f - err_pa)*drho_dP - rho_f*np.sin(theta_f - err_pa)*dtheta_dP
    	k=k+1
    
    if (elfix[1] != 0):
    	dxp_del[k,:] = np.cos(theta_f - err_pa)*drho_dT  - rho_f*np.sin(theta_f - err_pa)*dtheta_dT
    	k=k+1
    
    if (elfix[2] != 0):
    	dxp_del[k,:] = np.cos(theta_f - err_pa)*drho_de  - rho_f*np.sin(theta_f - err_pa)*dtheta_de
    	k=k+1
    
    if (elfix[3] != 0):
    	dxp_del[k,:] = np.cos(theta_f - err_pa)*drho_da  - rho_f*np.sin(theta_f - err_pa)*dtheta_da
    	k=k+1
    
    if (elfix[4] != 0):
    	dxp_del[k,:] = np.cos(theta_f - err_pa)*drho_di - rho_f*np.sin(theta_f - err_pa)*dtheta_di
    	k=k+1
    
    if (elfix[5] != 0):
    	dxp_del[k,:] = np.cos(theta_f - err_pa)*drho_dWc - rho_f*np.sin(theta_f - err_pa)*dtheta_dWc
    	k=k+1
    
    if (elfix[6] != 0):
    	dxp_del[k,:] = np.cos(theta_f - err_pa)*drho_dwl - rho_f*np.sin(theta_f - err_pa)*dtheta_dwl
    	k=k+1


    k=0
    if (elfix[0] != 0):
    	dyp_del[k,:] = np.sin(theta_f - err_pa)*drho_dP + rho_f*np.cos(theta_f - err_pa)*dtheta_dP
    	k=k+1
    
    if (elfix[1] != 0):
    	dyp_del[k,:] = np.sin(theta_f - err_pa)*drho_dT + rho_f*np.cos(theta_f - err_pa)*dtheta_dT
    	k=k+1
    
    if (elfix[2] != 0):
    	dyp_del[k,:] = np.sin(theta_f - err_pa)*drho_de + rho_f*np.cos(theta_f - err_pa)*dtheta_de
    	k=k+1
    
    if (elfix[3] != 0):
    	dyp_del[k,:] = np.sin(theta_f - err_pa)*drho_da + rho_f*np.cos(theta_f - err_pa)*dtheta_da
    	k=k+1
    
    if (elfix[4] != 0):
    	dyp_del[k,:] = np.sin(theta_f - err_pa)*drho_di + rho_f*np.cos(theta_f - err_pa)*dtheta_di
    	k=k+1
    
    if (elfix[5] != 0):
    	dyp_del[k,:] = np.sin(theta_f - err_pa)*drho_dWc + rho_f*np.cos(theta_f - err_pa)*dtheta_dWc
    	k=k+1
    
    if (elfix[6] != 0):
    	dyp_del[k,:] = np.sin(theta_f - err_pa)*drho_dwl + rho_f*np.cos(theta_f - err_pa)*dtheta_dwl
    	k=k+1
    
    

    # Set up matrix for minimizing chi squared
    # Set up column matrix too
    # (col) = (chimat)(delta_el)
    #chimat => alpha,  colmat=>beta
    diff_xp = np.zeros(num)	  # array for [rho(data) - rho(fit)]
    diff_yp = np.zeros(num)     # array for [theta(data) - theta(fit)]


    for k in range(0, num):
    	# weight derivative for each data point by corresponding 
    	# measurement error
    	diff_xp[k] = xp_d[k] - xp_f[k]
    	diff_yp[k] = yp_d[k] - yp_f[k]

    	for i in range (0, mfit):    		
            for j in range(0, mfit):
                chimat[i,j] = chimat[i,j] + dxp_del[i,k]*dxp_del[j,k]/err_maj[k]**2 + dyp_del[i,k]*dyp_del[j,k]/err_min[k]**2
            colmat[i] = colmat[i] + diff_xp[k]*dxp_del[i,k]/err_maj[k]**2 + diff_yp[k]*dyp_del[i,k]/err_min[k]**2

    
    return(theta_f, rho_f, xp_f, yp_f, chimat, colmat)

# ------------------------------------------------------------------------------------------

def calc_deriv_sb1(ELarr, elfix, mfit, time, V1_d, err_V1):
    # Define orbital element parameters
    period = ELarr[0]
    Tperi  = ELarr[1]
    ecc    = ELarr[2]
    K1     = ELarr[3]
    omega  = ELarr[4]   # RADIANS # omega_A  *** 
    Vsys   = ELarr[5]  
    num = len(time)


        
    # calc_ei.pro
    Ei = np.zeros(num) 
    mu = 2 * np.pi / period		# ex - degrees / day

    for i in range(0, num):
        # Mi: mean anomoly - increases uniformly with time
        # zero at time Tperi, 2Pi each orbit
        # percent of orbital period that has passed, expressed as an angle	
        Mi = mu * (time[i] - Tperi) 	
        # deg / day     *     number of days   =   deg 

        # reduce to same epoch
        Mi = 2 * np.pi * (  (time[i] - Tperi)/period - int((time[i] - Tperi)/period) )

        # keep Mi between 0 and 2Pi
        if Mi < 0.0:
            Mi = Mi + 2*np.pi
        if Mi > 2*np.pi:
            Mi = Mi - 2*np.pi

        # solve_trans.pro
        EE = Mi + ecc*np.sin(Mi) + ecc**2/2*np.sin(2*Mi)

        EEi = 0.0	# parameter to hold initial value once enter while loop
        count = 0.0

        while ((abs(EE - EEi) > 0.00001) & (count < 10000)):
            EEi = EE
            Mi0 = EEi - ecc*np.sin(EEi)
            EE = EEi + (Mi - Mi0)/(1 - ecc*np.cos(EEi))
            count=count+1

        # return EE
        Eit = EE


        # keep Ei between 0 and 2Pi
        if Eit < 0.0:
            Eit = Eit + 2*np.pi
        if Eit > 2*np.pi:
            Eit = Eit - 2*np.pi

        Ei[i] = Eit	# radians 
        
    # Determine true anomalies (nu)
    # nu = angle around from periastron
    nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(Ei/2.0))	# radians
    # keep nu between 0 and 2Pi
    for i in range(0, num):
        if nu[i] < 0.0: nu[i] = nu[i] + 2*np.pi

    # Check sign of nu
    cosnu = np.cos(nu)
    sinnu = np.sin(nu)
    actualcosnu = (np.cos(Ei) - ecc) / (1-ecc*np.cos(Ei))
    actualsinnu = np.sqrt(1-ecc**2)*np.sin(Ei) / (1-ecc*np.cos(Ei))
    for i in range(0,num):
        if cosnu[i]/abs(cosnu[i]) != actualcosnu[i]/abs(actualcosnu[i]):
            if sinnu[i]/abs(sinnu[i]) != actualsinnu[i]/abs(actualsinnu[i]):
                nu[i] = nu[i] + np.pi

    # calculate RV
    V1_f = K1 * (ecc*np.cos(omega) + np.cos(nu+omega)) 
    
    
    # Calculate derivatives evaluated at initial conditions 
    # 	dx/dEL = (dx/dP,dx/dT,dx/de,dx/dK1,dx/dK2,dx/dw.dx/dVsys)
    # 	dy/dEL = (dy/dP,dy/dT,dy/de,dy/dK2,dy/dK2,dy/dw,dy/dVsys)
    # where (x,y) -> (V1,V2) and
    # dEL = (P,T,e,K1,K2,omega,Vsys)
    dV1_del  = np.zeros((mfit,num))	# 7: number of orbital elements
    chimat   = np.zeros((mfit,mfit)) 	# matrix for minimizing chi^2
    colmat   = np.zeros(mfit)		# column matrix for minimizing chi^2
    delta_el = np.zeros(mfit)		# solution to matrix equation
    timediff = np.zeros(num)
    
    for i in range(0, num):
    	# reduce time difference to same epoch
    	# (fraction of period covered at time t)
    	fracP = (time[i] - Tperi)/period - int((time[i] - Tperi)/period)
    	#(remove effects of positive and negative time -> arbitrary zero-point
    	# for Tperi...... fracP goes from 0.0 to 1.0)
    	timediff[i] = fracP
    
    dEi_dP = -2*np.pi*(time - Tperi)/period**2/(1.0 - ecc*np.cos(Ei))
    dEi_dT = -2*np.pi/period/(1.0 - ecc*np.cos(Ei))
    dEi_de = np.sin(Ei)/(1.0 - ecc*np.cos(Ei))
    
    dnu_dP = np.sqrt(1.0 - ecc**2)/(1.0 - ecc*np.cos(Ei))*dEi_dP
    dnu_dT = np.sqrt(1.0 - ecc**2)/(1.0 - ecc*np.cos(Ei))*dEi_dT
    dnu_de =((1.0 - ecc**2)*dEi_de + np.sin(Ei))/np.sqrt(1.0 - ecc**2)/(1.0 - ecc*np.cos(Ei))

    dV1_dP = -K1*np.sin(nu + omega)*dnu_dP
    dV1_dT = -K1*np.sin(nu + omega)*dnu_dT
    dV1_de =  K1*(np.cos(omega) - np.sin(nu + omega)*dnu_de)
    dV1_dw = -K1*(ecc*np.sin(omega) + np.sin(nu + omega))
    dV1_dK1 = V1_f/K1
    dV1_dK2 = np.zeros(num)
    dV1_dVsys = np.zeros(num) + 1.0

    
    k=0
    if (elfix[0] != 0):
    	dV1_del[k,:] = dV1_dP
    	k=k+1
    
    if (elfix[1] != 0):
    	dV1_del[k,:] = dV1_dT
    	k=k+1
    
    if (elfix[2] != 0):
    	dV1_del[k,:] = dV1_de
    	k=k+1
    
    if (elfix[3] != 0):
    	dV1_del[k,:] = dV1_dK1
    	k=k+1
    
    if (elfix[4] != 0):
    	dV1_del[k,:] = dV1_dw
    	k=k+1
    
    if (elfix[5] != 0):
    	dV1_del[k,:] = dV1_dVsys
    	k=k+1
        
    # Set up matrix for minimizing chi squared
    # Set up column matrix too
    # (col) = (chimat)(delta_el)
    diff_V1 = np.zeros(num)

    # correct velocities for Vsys
    V1_f = V1_f + Vsys
    
    for k in range(0, num):
    	# weight derivative for each data point by corresponding 
    	# measurement error

    	diff_V1[k] = (V1_d[k]-V1_f[k])

    	# IDL -- the ## operator performs typical matrix multiplication
    	#        the * multiplies individual elements (no summing)

    	for i in range(0, mfit):
    		for j in range(0, mfit):
    			chimat[i,j] = chimat[i,j] + dV1_del[i,k]*dV1_del[j,k]/err_V1[k]**2 
    		colmat[i] = colmat[i] + diff_V1[k]*dV1_del[i,k]/err_V1[k]**2 

    return(V1_f, chimat, colmat) 
                   
# ------------------------------------------------------------------------------------------

def calc_deriv_sb2(ELarr, elfix, mfit, time, V1_d, V2_d, err_V1, err_V2):
    # Define orbital element parameters
    period = ELarr[0]
    Tperi  = ELarr[1]
    ecc    = ELarr[2]
    K1     = ELarr[3]
    K2     = ELarr[4]
    omega  = ELarr[5]   # RADIANS # omega_A  *** 
    Vsys   = ELarr[6]  
    num = len(time)


        
    # calc_ei.pro
    # Determine the eccentric anomalies (Ei) as a function of time: 
    #  Ei = angular position of obs position projected up to a circular orbit
    #     mu(ti - T) = Ei - esin(Ei)  ... Kepler's Equation
    #     where mu = 360/P = 2*pi/P
    # Solve this transcendental equation through an iterative procedure.
    # written by Gail Shaefer
    Ei = np.zeros(num) 
    mu = 2 * np.pi / period		# ex - degrees / day

    for i in range(0, num):
        # Mi: mean anomoly - increases uniformly with time
        # zero at time Tperi, 2Pi each orbit
        # percent of orbital period that has passed, expressed as an angle	
        Mi = mu * (time[i] - Tperi) 	
        # deg / day     *     number of days   =   deg 

        # reduce to same epoch
        Mi = 2 * np.pi * (  (time[i] - Tperi)/period - int((time[i] - Tperi)/period) )

        # keep Mi between 0 and 2Pi
        if Mi < 0.0:
            Mi = Mi + 2*np.pi
        if Mi > 2*np.pi:
            Mi = Mi - 2*np.pi

        # solve_trans.pro
        # Solve transcendental equation of the form E - esinE = M.
        # Use iterative procedure to determine E.
        # Initial approximation: E_0 = M + esinM - e^2/2 sin(2M)
        # Improve solution by iterating the following to formulae:
        #	M_0 = E_0 - esin(E_0)
        #	E_1 = E_0 + (M - M_0)/(1 - ecos(E_0))
        #	(derivative of Kepler's equation)
        #
        # Method adapted from Heintz 1978 (p.34-35)
        # Results compared with point-&-click graphical method.  Iterative approach 
        # leads to exact solution that satisfies E - esinE = M.  Therefore, 
        # point-&-click method is subsequently removed from orbit fitting.
        #
        # INPUT:
        #	e: eccentricity
        #	M: mean anomaly   M= 2*Pi/P
        #
        # OUTPUT:
        #	EE: eccentric anomaly
        #
        # Created: 9 May 2002 by Gail Shaefer
        # 12 December 2007: add escape route in case routine doesn't converge

        # Initial approximation:
        # PYTHON SIN USES RADIANS
        EE = Mi + ecc*np.sin(Mi) + ecc**2/2*np.sin(2*Mi)

        EEi = 0.0	# parameter to hold initial value once enter while loop
        count = 0.0

        while ((abs(EE - EEi) > 0.00001) & (count < 10000)):
            EEi = EE
            Mi0 = EEi - ecc*np.sin(EEi)
            EE = EEi + (Mi - Mi0)/(1 - ecc*np.cos(EEi))
            count=count+1

        # return EE
        Eit = EE


        # keep Ei between 0 and 2Pi
        if Eit < 0.0:
            Eit = Eit + 2*np.pi
        if Eit > 2*np.pi:
            Eit = Eit - 2*np.pi

        Ei[i] = Eit	# radians 
        
    # Determine true anomalies (nu)
    # nu = angle around from periastron
    nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(Ei/2.0))	# radians
    # keep nu between 0 and 2Pi
    for i in range(0, num):
        if nu[i] < 0.0: nu[i] = nu[i] + 2*np.pi

    # Check sign of nu
    cosnu = np.cos(nu)
    sinnu = np.sin(nu)
    actualcosnu = (np.cos(Ei) - ecc) / (1-ecc*np.cos(Ei))
    actualsinnu = np.sqrt(1-ecc**2)*np.sin(Ei) / (1-ecc*np.cos(Ei))
    for i in range(0,num):
        if cosnu[i]/abs(cosnu[i]) != actualcosnu[i]/abs(actualcosnu[i]):
            if sinnu[i]/abs(sinnu[i]) != actualsinnu[i]/abs(actualsinnu[i]):
                nu[i] = nu[i] + np.pi

    # calculate RV
    V1_f = K1 * (ecc*np.cos(omega) + np.cos(nu+omega)) 
    V2_f = -1 * K2 * (ecc*np.cos(omega) + np.cos(nu+omega))    
    
    
    # Calculate derivatives evaluated at initial conditions 
    # 	dx/dEL = (dx/dP,dx/dT,dx/de,dx/dK1,dx/dK2,dx/dw.dx/dVsys)
    # 	dy/dEL = (dy/dP,dy/dT,dy/de,dy/dK2,dy/dK2,dy/dw,dy/dVsys)
    # where (x,y) -> (V1,V2) and
    # dEL = (P,T,e,K1,K2,omega,Vsys)
    dV1_del  = np.zeros((mfit,num))	# 7: number of orbital elements
    dV2_del  = np.zeros((mfit,num))	# num: number of data points
    chimat   = np.zeros((mfit,mfit)) 	# matrix for minimizing chi^2
    colmat   = np.zeros(mfit)		# column matrix for minimizing chi^2
    delta_el = np.zeros(mfit)		# solution to matrix equation
    timediff = np.zeros(num)
    
    for i in range(0, num):
    	# reduce time difference to same epoch
    	# (fraction of period covered at time t)
    	fracP = (time[i] - Tperi)/period - int((time[i] - Tperi)/period)
    	#(remove effects of positive and negative time -> arbitrary zero-point
    	# for Tperi...... fracP goes from 0.0 to 1.0)
    	timediff[i] = fracP
    
    dEi_dP = -2*np.pi*(time - Tperi)/period**2/(1.0 - ecc*np.cos(Ei))
    dEi_dT = -2*np.pi/period/(1.0 - ecc*np.cos(Ei))
    dEi_de = np.sin(Ei)/(1.0 - ecc*np.cos(Ei))
    
    dnu_dP = np.sqrt(1.0 - ecc**2)/(1.0 - ecc*np.cos(Ei))*dEi_dP
    dnu_dT = np.sqrt(1.0 - ecc**2)/(1.0 - ecc*np.cos(Ei))*dEi_dT
    dnu_de =((1.0 - ecc**2)*dEi_de + np.sin(Ei))/np.sqrt(1.0 - ecc**2)/(1.0 - ecc*np.cos(Ei))

    dV1_dP = -K1*np.sin(nu + omega)*dnu_dP
    dV1_dT = -K1*np.sin(nu + omega)*dnu_dT
    dV1_de =  K1*(np.cos(omega) - np.sin(nu + omega)*dnu_de)
    dV1_dw = -K1*(ecc*np.sin(omega) + np.sin(nu + omega))
    dV1_dK1 = V1_f/K1
    dV1_dK2 = np.zeros(num)
    dV1_dVsys = np.zeros(num) + 1.0

    dV2_dP =  K2*np.sin(nu + omega)*dnu_dP
    dV2_dT =  K2*np.sin(nu + omega)*dnu_dT
    dV2_de = -K2*(np.cos(omega) - np.sin(nu + omega)*dnu_de)
    dV2_dw =  K2*(ecc*np.sin(omega) + np.sin(nu + omega))
    dV2_dK2 = V2_f/K2
    dV2_dK1 = np.zeros(num)
    dV2_dVsys = np.zeros(num) + 1.0
    
    k=0
    if (elfix[0] != 0):
    	dV1_del[k,:] = dV1_dP
    	k=k+1
    
    if (elfix[1] != 0):
    	dV1_del[k,:] = dV1_dT
    	k=k+1
    
    if (elfix[2] != 0):
    	dV1_del[k,:] = dV1_de
    	k=k+1
    
    if (elfix[3] != 0):
    	dV1_del[k,:] = dV1_dK1
    	k=k+1
    
    if (elfix[4] != 0):
    	dV1_del[k,:] = dV1_dK2
    	k=k+1
    
    if (elfix[5] != 0):
    	dV1_del[k,:] = dV1_dw
    	k=k+1
    
    if (elfix[6] != 0):
    	dV1_del[k,:] = dV1_dVsys
    	k=k+1
        
    k=0
    if (elfix[0] != 0):
    	dV2_del[k,:] = dV2_dP
    	k=k+1
    
    if (elfix[1] != 0):
    	dV2_del[k,:] = dV2_dT
    	k=k+1
    
    if (elfix[2] != 0):
    	dV2_del[k,:] = dV2_de
    	k=k+1
    
    if (elfix[3] != 0):
    	dV2_del[k,:] = dV2_dK1
    	k=k+1
    
    if (elfix[4] != 0):
    	dV2_del[k,:] = dV2_dK2
    	k=k+1
    
    if (elfix[5] != 0):
    	dV2_del[k,:] = dV2_dw
    	k=k+1
    
    if (elfix[6] != 0):
    	dV2_del[k,:] = dV2_dVsys
    	k=k+1    
    
    
    # Set up matrix for minimizing chi squared
    # Set up column matrix too
    # (col) = (chimat)(delta_el)
    diff_V1 = np.zeros(num)
    diff_V2 = np.zeros(num)

    # correct velocities for Vsys
    V1_f = V1_f + Vsys
    V2_f = V2_f + Vsys
    
    for k in range(0, num):
    	# weight derivative for each data point by corresponding 
    	# measurement error

    	diff_V1[k] = (V1_d[k]-V1_f[k])
    	diff_V2[k] = (V2_d[k]-V2_f[k])

    	# IDL -- the ## operator performs typical matrix multiplication
    	#        the * multiplies individual elements (no summing)

    	for i in range(0, mfit):
    		for j in range(0, mfit):
    			chimat[i,j] = chimat[i,j] + dV1_del[i,k]*dV1_del[j,k]/err_V1[k]**2 + dV2_del[i,k]*dV2_del[j,k]/err_V2[k]**2
    		colmat[i] = colmat[i] + diff_V1[k]*dV1_del[i,k]/err_V1[k]**2 + diff_V2[k]*dV2_del[i,k]/err_V2[k]**2

    return(V1_f, V2_f, chimat, colmat) 
                   
# ------------------------------------------------------------------------------------------

def newt_raph_ell(param, elfix, time, rho, theta, dmajor, dminor, theta_err):
    # observations
    num = len(time)
    
    # convert theta and theta_err to radians
    theta = theta*np.pi/180.
    theta_err = theta_err*np.pi/180.
    
    # convert data points to x and y coordinates
    xarr = rho * np.cos(theta)	# x coordinate = DEC
    yarr = rho * np.sin(theta)	# y coordinate = RA
    
    # project separation into components along the direction of the error ellipse
    xparr = rho * np.cos(theta - theta_err)  # x' (major)
    yparr = rho * np.sin(theta - theta_err)  # y' (minor) 
    
    
    # Obtain values for P,T,e,a,i,Omega,omega 
    period= param[0]
    Tperi = param[1]
    ecc = 	param[2]
    major = param[3]
    inc = 	param[4]
    W_cap = param[5]
    w_low = param[6]
    
    #convert i, Omega, omega to radians
    inc = inc*np.pi/180.
    W_cap = W_cap*np.pi/180.
    w_low = w_low*np.pi/180.
    
    # determine indices of elements for which to vary    
    nEl = len(elfix)
    k = len(np.where(elfix == 0)[0])
    mfit = nEl - k 		# number of elements to improve
    elvar = np.where(elfix != 0)[0]
    El    = np.array([period,Tperi,ecc,major,inc,W_cap,w_low])
    ELadj = np.array([period,Tperi,ecc,major,inc,W_cap,w_low])
    elLabel = ['P','T','e','a','i','OmegaB','omegaB']
    
    
    
    # plot model and data of initial guess
    rho_mod, theta_mod = calc_vbfit(El, time)	# mas, degrees
    xxfit = rho_mod * np.cos(theta_mod)	#  DEC
    yyfit = rho_mod * np.sin(theta_mod)	#  RA
    tnum = 1000.0
    tstep = period/tnum
    tmin = Tperi
    tarr = np.arange(tnum)*tstep + tmin
    rho_mod, theta_mod = calc_vbfit(El, tarr)   
    xmod = rho_mod * np.cos(theta_mod)	
    ymod = rho_mod * np.sin(theta_mod)	
    
    
    # Plot orbit
    fig = plt.figure(figsize=(5,4))
    plt.plot([0], [0], marker='o', color='red', ms=3)
    plt.plot(yarr, xarr, linestyle=' ', color='blue', marker='o', ms=6) 
    plt.plot(ymod, xmod, color='k')
    for i in range(0, num):
        plt.plot([yarr[i], yyfit[i]],[xarr[i], xxfit[i]], color='black', alpha=0.5) 
    xr = np.ceil(np.max([xarr,yarr]))*1.2
    plt.xlim(xr, -1*xr)
    plt.ylim(-1*xr, xr)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$\\Delta$RA (mas)' )
    plt.ylabel('$\\Delta$DEC (mas)')
    plt.title('Visual Orbit - Initial Guess')
    plt.savefig('vbfit1_initial_guess.png')
    plt.close()
    
    
    # ---------------------------------------------------------------------------------------------
    # LOOP THROUGH FIT
    count = 0
    delta_chi = 1.0		# set initially to begin loop
    lam = 0.01   	# Marquardt method to ensure convergence
        
    while((delta_chi > 0.0001)):
        # Determine errors in orbital elements
        # set up matrices for error determination
        # Invert matrix through Gauss Jordan elimination (Numerical Recipes in C)

        # calculate covariance matrix and column matrix
        theta_f, rho_f, xpfit, ypfit, alpha, beta = calc_deriv_vb_ell(El, elfix, mfit, time, xparr, yparr, dmajor, dminor, theta_err)

        # Determine chi squared
        # convert data points to x and y coordinates
        xfit = rho_f * np.cos(theta_f)	#  DEC
        yfit = rho_f * np.sin(theta_f)	#  RA
        chi2old = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2)
        
        #determine errors:
        #  & adjust alpha matrix by Marquardt parameter lambda
        invmat = np.linalg.inv(alpha)
        ELerr = np.zeros(mfit)
        for i in range(0, mfit):
            invmat = abs(invmat)
            ELerr[i] = np.sqrt(invmat[i,i])
            alpha[i,i] = alpha[i,i]*(1.0 + lam)

        # Make adjustments for next iteration
        if (mfit == 1):
        	delta_el = [beta/alpha]
        delta_el  = np.linalg.solve(alpha, beta)
        
        for i in range(0, mfit):
            ELadj[elvar[i]] = El[elvar[i]] + delta_el[i]
        k=0
        
        # Make sure ecc > 0, Make sure angles 0-360 or 0-180
        if ELadj[2] < 0: ELadj[2] = 0.
        while ELadj[4] > np.pi: ELadj[4] = ELadj[4] - np.pi
        while ELadj[4] < 0: ELadj[4] = ELadj[4] + np.pi
        while ELadj[5] > 2*np.pi: ELadj[5] = ELadj[5] - 2*np.pi
        while ELadj[5] < 0: ELadj[5] = 2*np.pi - abs(ELadj[5])
                
        # Determine chi squared of new params
        theta_f, rho_f, xpfit, ypfit, alpha, beta = calc_deriv_vb_ell(ELadj, elfix, mfit, time, xparr, yparr, dmajor, dminor, theta_err)
        chi2new = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2)
                
        #   print,"chi2 of next modification:",chi2new
        if chi2new < chi2old:
            El = ELadj * 1.0 
            chi2 = chi2new
            delta_chi = chi2old - chi2new
            lam = lam/10.0
            print(count, El, chi2)
        else:
            lam = lam*10.0
            delta_chi = 1.0
            
        # Make sure angles 0-360 or 0-180
        if El[2] < 0: El[2] = 0.
        while El[4] > np.pi: El[4] = El[4] - np.pi
        while El[4] < 0: El[4] = El[4] + np.pi
        while El[5] > 2*np.pi: El[5] = El[5] - 2*np.pi
        while El[5] < 0: El[5] = 2*np.pi - abs(El[5])
                
        #; Inclination wrap around ( stay within -180 to +180)
        #;junk = el(4)
        #;if junk ge !pi then while junk gt !pi do junk = junk - !pi 
        #;if junk lt 0 then while junk lt 0 do junk = junk + !pi
        #;el(4) = junk
        #
        #; Omega wrap around
        #if el(5) lt 0. then el(5) = 2*!pi - abs(el(5))
        #if el(5) ge 2.*!pi then el(5) = el(5) - 2.*!pi
        
        
        count = count+1
        
        #print(count, El, chi2)
        
        #print('%3i %10i %10i %10i' % (count, chi2old, chi2new, chi2))
        
        # do not exit unless lam is less than 0.001
        if (lam > 0.001) & (delta_chi < 0.001):
            delta_chi = 1.0

    lam = 0.0
    
    # ---------------------------------------------------------------------------------------------
    # GET BEST FIT MODEL
    period = El[0]
    Tperi  = El[1]
    ecc    = El[2]
    major  = El[3]
    inc    = El[4]
    W_cap  = El[5]
    w_low  = El[6]
    
    # we have square error bars, so pretend ellipse has PA=0 and dmaj = dDEC...
    theta_f, rho_f, xpfit, ypfit, alpha, beta = calc_deriv_vb_ell(El, elfix, mfit, time, xparr, yparr,  dmajor, dminor, theta_err)
    invmat = np.linalg.inv(alpha)

    #determine errors:
    chi2_final = np.sum((xparr-xpfit)**2/dmajor**2 + (yparr-ypfit)**2/dminor**2)
    
    # convert data points to x and y coordinates
    xfit = rho_f * np.cos(theta_f)	# x coordinate
    yfit = rho_f * np.sin(theta_f)	# y coordinate
    
    # assume equal uncertainties in x and y:
    chi2_eq = np.sum((xarr-xfit)**2/dmajor**2 + (yarr-yfit)**2/dmajor**2)
    
    # degrees of freedom
    dof = 2*num - mfit
    if dof == 0: dof=1
    
    # reduced chi squared:
    chi2red = abs(chi2_eq / dof)
    ELerr = El * 0.
    for ii in range(0,mfit):
        i = elvar[ii]
        invmat = abs(invmat)
        ELerr[i] = np.sqrt(chi2red)*np.sqrt(invmat[ii,ii])

    
    # create full model orbit
    tnum=100    
    tstep = period/tnum
    tmin = Tperi
    tarr = np.linspace(Tperi, Tperi+period, tnum)  
    rho_mod, theta_mod = calc_vbfit(El, tarr)   
    xmod = rho_mod * np.cos(theta_mod)
    ymod = rho_mod * np.sin(theta_mod)
 
 
    # ---------------------------------------------------------------------------------------------
    # Results
    best = El * 1.
    error = ELerr * 1.
    print('%s %10.5f' % ('   reduced chi2    ', chi2red))
    
    # Save all variables for later
    # X = DEC, Y = RA
    ra = yarr	# observed
    dec = xarr
    rafit = yfit	# model at observed
    decfit = xfit
    ramod = ymod	# full model
    decmod = xmod
    #np.savez('.vb_el.sav',ra=ra, dec=dec, rafit=rafit, decfit=decfit, ramod=ramod, decmod=decmod)
    return(best, error, rafit, decfit, ramod, decmod)

# ------------------------------------------------------------------------------------------

def fit_orbit_vbsb1_ell(vb_file, sb_file, param, parfit):
    # Read in VB data 
    # readcol, vb_file, time_vb, theta, dtheta, rho, drho
    #print('  Reading in observations...')
    #print(' ' )
    tab  = ascii.read(vb_file,  data_start=0)
    time_vb = np.array(tab['col1'])
    theta   = np.array(tab['col2'])
    rho     = np.array(tab['col3'])
    dmajor  = np.array(tab['col4'])
    dminor  = np.array(tab['col5'])
    theta_err= np.array(tab['col6'])
    vb_num = len(time_vb)
    
    # convert theta to radians
    theta = theta*np.pi/180.
    theta_err = theta_err*np.pi/180.
    
    # convert data points to x and y coordinates IN COUTEAU NOTATION.  
    #    (theta is measured from Y axis, not X axis)
    xarr = rho * np.cos(theta)	    # x coordinate = DEC, bc PA=0=N
    yarr = rho * np.sin(theta)	    # y coordinate = RA

    # project separation into components along the direction of the error ellipse
    xparr = rho * np.cos(theta - theta_err)  	# x' (major)
    yparr = rho * np.sin(theta - theta_err)  	# y' (minor) 


    # Read in SB data
    # readcol, sb2_file, time_sb2, V1_sb2, dv1_sb2, v2_sb2, dv2_sb2
    tab  = ascii.read(sb_file,  data_start=0)
    time_sb2 = np.array(tab['col1'])
    v1_sb2   = np.array(tab['col2'])
    dv1_sb2  = np.array(tab['col3'])
    sb2_num = len(time_sb2)
    
    # Initial guess for param
    period= param[0]
    Tperi = param[1]
    ecc = 	param[2]
    major = param[3]
    inc = 	param[4]
    W_cap = param[5]
    w_low = param[6]
    k1 = 	param[7]
    Vsys = 	param[9]
    
    # Fix = 0, float = 1 
    f0 = parfit[0]
    f1 = parfit[1]
    f2 = parfit[2]
    f3 = parfit[3]
    f4 = parfit[4]
    f5 = parfit[5]
    f6 = parfit[6]
    f7 = parfit[7]
    f8 = parfit[9]
    
    #-----------------------------------------------
    # Order of orbital element arrays
    # do not set here yet (need to get angles into radians)
    # EL_vb = [period,Tperi,ecc,major,inc,W_cap,w_low]
    #             0    1     2    3    4    5     6
    # EL_sb2 = [period,Tperi,ecc,K1,K2,w_low,Vsys]
    #              0    1     2   7  8   6     9
    # EL_sb1 = [period,Tperi,ecc,K1,w_low,Vsys]
    #              0    1     2   7   6     9
    
    elfix=np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8])    # parfit array
    nEl = len(elfix)

    elfix_vb=np.array([f0,f1,f2,f3,f4,f5,f6])
    nEl_vb = len(elfix_vb)    # number of VB elements = 7

    elfix_sb2=np.array([f0,f1,f2,f7,f6,f8])
    nEl_sb2 = len(elfix_sb2)  # number of SB2 elements = 6

    # convert i, Omega, omega to radians
    inc = inc*np.pi/180.0
    W_cap = W_cap*np.pi/180.0
    w_low = w_low*np.pi/180.0
    
    # number of elements to improve
    k=0
    for i in range(0, nEl): 
        if elfix[i] == 0: k=k+1
    mfit = nEl - k 			   # number of elements to vary    

    k=0
    for i in range(0, nEl_vb): 
        if elfix_vb[i] == 0: k=k+1
    mfit_vb = nEl_vb - k 	    # number of VB elements to vary	

    k=0
    for i in range(0, nEl_sb2): 
        if elfix_sb2[i] == 0: k=k+1
    mfit_sb2 = nEl_sb2 - k 	    # number of SB elements to vary
    
    #print(' # VB elements to vary = ', mfit_vb)
    #print(' # SB elements to vary = ', mfit_sb2)
    
    El = np.array([period,Tperi,ecc,major,inc,W_cap,w_low,k1,Vsys])
    El_vb = np.array([period,Tperi,ecc,major,inc,W_cap,w_low])
    El_sb2 = np.array([period,Tperi,ecc,k1,w_low,Vsys])

    Eladj = np.copy(El)
    elLabel = ['P','T','e','a(mas)','i','Omega_B','omega_A','K1(km/s)','Vsys']

    # determine which indices of full set are being varied
    vb_subset = np.zeros(mfit_vb, dtype=int)    # 3,4,5,6
    sb2_subset = np.zeros(mfit_sb2, dtype=int)  # 6,7,8

    # determine subarray of indices that are VB, SB1, and SB2 parameters
    vb_par = np.zeros(nEl_vb, dtype=int)   
    sb2_par = np.zeros(nEl_sb2, dtype=int) 

    # determine indices of elements for which to vary
    el_subset = np.where(elfix != 0)[0]
    Elvar = np.copy(El[el_subset])

    k=0

    # loop through each VB element and check if vary?
    for i in range(0, nEl_vb):
        ind = np.where(El_vb[i] == El)[0][0]
        vb_par[i] = ind
        if elfix_vb[i] != 0:
            ind = np.where(El_vb[i] == Elvar)[0][0]
            vb_subset[k] = ind
            k=k+1
            
    k=0
    for i in range(0, nEl_sb2):
        ind = np.where(El_sb2[i] == El)[0][0]
        sb2_par[i] = ind
        if elfix_sb2[i] != 0:
            ind = np.where(El_sb2[i] == Elvar)[0][0]
            sb2_subset[k] = ind
            k=k+1
        
    #-----------------------------------------------
    
    # FITTING PROCEDURE
    print('  Running LM fit...')
    print('')

    # LOOP THROUGH FIT
    count = 0
    delta_chi = 1.0		# set initially to begin loop
    lam = 0.0001   	# Marquardt method to ensure convergence   0.001
        
    while((delta_chi > 0.0001)):
        # Determine errors in orbital elements
        # set up matrices for error determination
        # Invert matrix through Gauss Jordan elimination (Numerical Recipes in C)

        # calculate covariance matrix and column matrix
        theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(El_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
        V1_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb1(El_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, dv1_sb2)        
        #print(alpha_vb)
        #print(alpha_sb2)
        #print('')

        # Determine chi squared
        # convert data points to x and y coordinates
        xfit = rho_f * np.cos(theta_f)
        yfit = rho_f * np.sin(theta_f)
        
        chi2old = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 )
        #print(count, El, chi2old)
       
        
	    # combine SB1 and VB matrices...
        #
	    # initialize full alpha,beta arrays
        alpha = np.zeros((mfit,mfit))
        beta  = np.zeros(mfit)
        beta[vb_subset ] = beta[vb_subset ] + beta_vb
        beta[sb2_subset] = beta[sb2_subset] + beta_sb2
        
        #print(alpha)
        
        #print(vb_subset)
        
        for i in range(0, mfit_vb):
            for j in range(0, mfit_vb):
                alpha[vb_subset[i],vb_subset[j]] = alpha[vb_subset[i],vb_subset[j]] + alpha_vb[i,j]
                #print(alpha[vb_subset[i],vb_subset[j]])
        #quit()
        for i in range(0, mfit_sb2):
            for j in range(0, mfit_sb2):
                alpha[sb2_subset[i],sb2_subset[j]] = alpha[sb2_subset[i],sb2_subset[j]] + alpha_sb2[i,j]
        invmat = np.linalg.inv(alpha)       
        
        #determine errors:
        #  & adjust alpha matrix by Marquardt parameter lambda
        ELerr = np.zeros(mfit)
        for i in range(0, mfit):
            invmat = abs(invmat)
            ELerr[i] = np.sqrt(invmat[i,i])
            alpha[i,i] = alpha[i,i]*(1.0 + lam)
        
        # Make adjustments for next iteration
        Eladj = np.copy(El)
        if (mfit == 1):
        	delta_el = [beta/alpha]
        else:
            delta_el  = np.linalg.solve(alpha, beta)
                        
        Eladj[el_subset] = np.copy(Eladj[el_subset]) + delta_el
 
        # Make sure ecc > 0, Make sure angles 0-360 or 0-180
        if Eladj[2] < 0: Eladj[2] = 0.   # ecc
        if Eladj[3] < 0:                 # a 
            Eladj[3] = abs(Eladj[3])
            Eladj[5] = Eladj[5] + np.pi
        while Eladj[4] > np.pi: Eladj[4] = Eladj[4] - np.pi    # inc
        while Eladj[4] < 0: Eladj[4] = Eladj[4] + np.pi
        while Eladj[5] > 2*np.pi: Eladj[5] = Eladj[5] - 2*np.pi  #OmB
        while Eladj[5] < 0: Eladj[5] = 2*np.pi - abs(Eladj[5])

        ELadj_vb  = np.copy(Eladj[vb_par])
        ELadj_sb2 = np.copy(Eladj[sb2_par])         
                       
        # Determine chi squared of new params
        theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(ELadj_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
        V1_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb1(ELadj_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, dv1_sb2)        
        xfit = rho_f * np.cos(theta_f)	# y coordinate
        yfit = rho_f * np.sin(theta_f)	# x coordinate
        chi2new = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 )
        
        
        #   print,"chi2 of next modification:",chi2new
        if chi2new < chi2old:
            El = Eladj * 1.0 
            El_vb = ELadj_vb * 1.0 
            El_sb2 = ELadj_sb2 * 1.0 
            chi2 = chi2new * 1.0 
            delta_chi = chi2old - chi2new
            lam = lam/10.0
            print(El, chi2)
            
        else:
            lam = lam*10.0
            delta_chi = 1.0
            
        count = count+1
        
        #print('%3i %10i %10i %10i' % (count, chi2old, chi2new, chi2))
        
        # do not exit unless lam is less than 0.001
        if (lam > 0.001) & (delta_chi < 0.001):
            delta_chi = 1.0

    lam = 0.0
    
    
    # Determine final error matrix:
    theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(El_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
    V1_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb1(El_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, dv1_sb2)        
    
    # alpha,beta arrays
    alpha = np.zeros((mfit,mfit))
    beta  = np.zeros(mfit)
    beta[vb_subset ] = beta[vb_subset ] + beta_vb
    beta[sb2_subset] = beta[sb2_subset] + beta_sb2
    
    for i in range(0, mfit_vb):
        for j in range(0, mfit_vb):
            alpha[vb_subset[i],vb_subset[j]] = alpha[vb_subset[i],vb_subset[j]] + alpha_vb[i,j]
    for i in range(0, mfit_sb2):
        for j in range(0, mfit_sb2):
            alpha[sb2_subset[i],sb2_subset[j]] = alpha[sb2_subset[i],sb2_subset[j]] + alpha_sb2[i,j]
    invmat = np.linalg.inv(alpha)       
    
    #determine errors:
    ELerr = np.zeros(mfit)
    for i in range(0, mfit):
        invmat = abs(invmat)
        ELerr[i] = np.sqrt(invmat[i,i])
        alpha[i,i] = alpha[i,i]*(1.0 + lam)
        
    
    # reduced chi squared:
    chi2 = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 )
    chi2_vb = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2)
    chi2_sb = np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 )
    dof = 2.0*vb_num + 2.0*sb2_num - mfit
    chi2red = chi2/dof
    chi2red_all= chi2red
    chi2red_vb = chi2_vb/(2.0*vb_num - mfit_vb) 
    chi2red_sb = chi2_sb/(2.0*sb2_num - mfit_sb2) 
    
    Elerr_fit = np.zeros(mfit)
    for i in range(0,mfit):
         Elerr_fit[i] = np.sqrt(chi2red)*np.sqrt(invmat[i,i])
    
    
    
    # reformat EL
    Elbest = np.zeros(9)
    Elbest[0] = El[0]	# Period
    Elbest[1] = El[1]	# T0
    Elbest[2] = El[2]	# ecc
    Elbest[3] = El[3]	# a
    Elbest[4] = El[4]	# inc
    Elbest[5] = El[5]	# Big w
    Elbest[6] = El[6]	# little w
    Elbest[7] = El[7]	# k1
    Elbest[8] = El[8]	# v0
    

    ## Error array
    Elerr = np.zeros(9)
    for i in range(0, mfit):
        Elerr[el_subset[i]] = Elerr_fit[i]
    #ELerr[0] = np.sqrt(chi2red_all)* np.sqrt(invmat[0,0])	# Period
    #ELerr[1] = np.sqrt(chi2red_all)* np.sqrt(invmat[1,1])	# T0
    #ELerr[2] = np.sqrt(chi2red_all)* np.sqrt(invmat[2,2])	# ecc
    #ELerr[3] = np.sqrt(chi2red_vb) * np.sqrt(invmat[3,3])	# a
    #ELerr[4] = np.sqrt(chi2red_vb) * np.sqrt(invmat[4,4])	# inc
    #ELerr[5] = np.sqrt(chi2red_vb) * np.sqrt(invmat[5,5])	# Big w
    #ELerr[6] = np.sqrt(chi2red_all)* np.sqrt(invmat[6,6])	# little w
    #ELerr[7] = np.sqrt(chi2red_sb) * np.sqrt(invmat[7,7])	# k1
    #ELerr[8] = 0.0	# k2
    #ELerr[9] = np.sqrt(chi2red_sb) * np.sqrt(invmat[8,8])	# v0

    
    return(Elbest, ELerr, chi2red_all, chi2red_vb, chi2red_sb)

# ------------------------------------------------------------------------------------------

def fit_orbit_vbsb2_ell_OLD(vb_file, sb_file, param, parfit):
    # This version can handle elliptical uncertainties for visual orbits.  
    # Minimize chi2 based on projections along the direction of error ellipse (x',y').
    # Compute the orbital elements through a Newton-Raphson technique.
    # Fits simultaneously for VB + SB2 orbits
    # Written by Gail Schaefer
    # Calls the following routines:
    #     calc_deriv_vb.pro
    #     calc_deriv_sb2.pro
    #  x   calc_Ei.pro
    #  x   solve_trans.pro
    #  x   calc_sb2fit.pro - for plotting model orbits
    #  x   calc_vbfit.pro - for plotting model orbits
    # 
    
    # Read in VB data 
    # readcol, vb_file, time_vb, theta, dtheta, rho, drho
    print('  Reading in observations...')
    print(' ' )
    tab  = ascii.read(vb_file,  data_start=0)
    time_vb  = np.array(tab['col1'])
    theta    = np.array(tab['col2'])
    rho      = np.array(tab['col3'])
    dmajor   = np.array(tab['col4'])
    dminor   = np.array(tab['col5'])
    theta_err= np.array(tab['col6'])
    vb_num = len(time_vb)
    
    # convert theta to radians
    theta = theta*np.pi/180.
    theta_err = theta_err*np.pi/180.
    
    # convert data points to x and y coordinates IN COUTEAU NOTATION.  
    #    (theta is measured from Y axis, not X axis)
    xarr = rho * np.cos(theta)	    # x coordinate = DEC, bc PA=0=N
    yarr = rho * np.sin(theta)	    # y coordinate = RA

    # project separation into components along the direction of the error ellipse
    xparr = rho * np.cos(theta - theta_err)  	# x' (major)
    yparr = rho * np.sin(theta - theta_err)  	# y' (minor) 

    # Read in SB data
    # readcol, sb2_file, time_sb2, V1_sb2, dv1_sb2, v2_sb2, dv2_sb2
    tab  = ascii.read(sb_file,  data_start=0)
    time_sb2 = np.array(tab['col1'])
    v1_sb2   = np.array(tab['col2'])
    dv1_sb2  = np.array(tab['col3'])
    v2_sb2   = np.array(tab['col4'])
    dv2_sb2  = np.array(tab['col5'])
    sb2_num = len(time_sb2)
    
    # Initial guess for param
    period= param[0]
    Tperi = param[1]
    ecc = 	param[2]
    major = param[3]
    inc = 	param[4]
    W_cap = param[5]
    w_low = param[6]    #  omega_A    for vbsb2 fits! will use flagwa later
    k1 = 	param[7]
    k2 = 	param[8]
    Vsys = 	param[9]
    
    # Fix = 0, float = 1 
    f0 = parfit[0]
    f1 = parfit[1]
    f2 = parfit[2]
    f3 = parfit[3]
    f4 = parfit[4]
    f5 = parfit[5]
    f6 = parfit[6]
    f7 = parfit[7]
    f8 = parfit[8]
    f9 = parfit[9]
    
    #-----------------------------------------------
    # Order of orbital element arrays
    # do not set here yet (need to get angles into radians)
    # EL_vb = [period,Tperi,ecc,major,inc,W_cap,w_low]
    #             0    1     2    3    4    5     6
    # EL_sb2 = [period,Tperi,ecc,K1,K2,w_low,Vsys]
    #              0    1     2   7  8   6     9
    # EL_sb1 = [period,Tperi,ecc,K1,w_low,Vsys]
    #              0    1     2   7   6     9
    
    elfix=np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9])
    nEl = len(elfix)

    elfix_vb=np.array([f0,f1,f2,f3,f4,f5,f6])
    nEl_vb = len(elfix_vb)

    elfix_sb2=np.array([f0,f1,f2,f7,f8,f6,f9])
    nEl_sb2 = len(elfix_sb2)

    # convert i, Omega, omega to radians. 
    inc = inc*np.pi/180.0
    W_cap = W_cap*np.pi/180.0
    w_low = w_low*np.pi/180.0
    
    # number of elements to improve
    k=0
    for i in range(0, nEl): 
        if elfix[i] == 0: k=k+1
    mfit = nEl - k 			    

    k=0
    for i in range(0, nEl_vb): 
        if elfix_vb[i] == 0: k=k+1
    mfit_vb = nEl_vb - k 		

    k=0
    for i in range(0, nEl_sb2): 
        if elfix_sb2[i] == 0: k=k+1
    mfit_sb2 = nEl_sb2 - k 	
    
    El = np.array([period,Tperi,ecc,major,inc,W_cap,w_low,k1,k2,Vsys])
    El_vb = np.array([period,Tperi,ecc,major,inc,W_cap,w_low])
    El_sb2 = np.array([period,Tperi,ecc,k1,k2,w_low,Vsys])

    Eladj = np.copy(El)
    elLabel = ['P','T','e','a(mas)','i','Omega_B','omega_A','K1(km/s)','K2(km/s)','Vsys']

    # determine which indices of full set are being varied
    vb_subset = np.zeros(mfit_vb, dtype=int)
    sb2_subset = np.zeros(mfit_sb2, dtype=int)

    # determine subarray of indices that are VB, SB1, and SB2 parameters
    vb_par = np.zeros(nEl_vb, dtype=int)
    sb2_par = np.zeros(nEl_sb2, dtype=int)

    # determine indices of elements for which to vary
    el_subset = np.where(elfix > 0)[0]
    Elvar = El[el_subset]

    k=0
    for i in range(0, nEl_vb):
        ind = np.where(El_vb[i] == El)[0][0]
        vb_par[i] = ind
        if elfix_vb[i] != 0:
            ind = np.where(El_vb[i] == Elvar)[0][0]
            vb_subset[k] = ind
            k=k+1
    k=0
    for i in range(0, nEl_sb2):
        ind = np.where(El_sb2[i] == El)[0][0]
        sb2_par[i] = ind
        if elfix_sb2[i] != 0:
            ind = np.where(El_sb2[i] == Elvar)[0][0]
            sb2_subset[k] = ind
            k=k+1

    
    #-----------------------------------------------
    # Plot initial guess orbit
    # calculate model for VB
        
    # flag_wa makes w = wA not wB
    tnum=100    
    tarr = np.linspace(El[1], El[1]+El[0], tnum)  
    rho_mod, theta_mod = calc_vbfit(El_vb, time_vb, flagwa=1)
    xmod = rho_mod * np.cos(theta_mod)	# x coordinate
    ymod = rho_mod * np.sin(theta_mod)	# y coordinate
    # calculate model for SB2
    phobs = (time_sb2 - Tperi) / period % 1
    phmod = (tarr - Tperi) / period % 1
    rvmod1, rvmod2 = calc_sb2fit(El_sb2, time_sb2)
    # Plot orbit
    print('  Plotting initial guess...')
    print('')
    fig = plt.figure(figsize=(10,4))
    
    ax1 = fig.add_axes([0.57, 0.10, 0.40, 0.80])   
    plt.plot([0], [0], marker='o', color='red', ms=3)
    plt.plot(yarr, xarr, linestyle=' ', color='blue', marker='o', ms=6) 
    #for i in range(0, vb_num):
    #    #  width = total major axis,  height = total minor axis,  angle = deg ccw
    #    e1 = Ellipse(xy=(yarr[i], xarr[i]), width=dmajor[i]*2, height=dminor[i]*2, angle=90-theta_err[i]*180/np.pi, fill=False, color='blue', zorder=5)
    #    ax1.add_patch(e1)
    xr = np.ceil(np.max([xarr,yarr]))
    plt.xlim(xr, -1*xr)
    plt.ylim(-1*xr, xr)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$\\Delta$RA (mas)' )
    plt.ylabel('$\\Delta$DEC (mas)')
    plt.title('Visual Orbit')
    
    ax2 = fig.add_axes([0.07, 0.10, 0.40, 0.80])  
    plt.errorbar(phobs, v1_sb2, dv1_sb2, linestyle=' ', color='red', capsize=0.1)
    plt.plot(phobs, v1_sb2, marker='o', color='red', linestyle=' ',  ms=6)
    plt.errorbar(phobs, v2_sb2, dv2_sb2, linestyle=' ', color='blue', capsize=0.1)
    plt.plot(phobs, v2_sb2, marker='o', color='blue', linestyle=' ',  ms=6)
    plt.xlabel('Phase')
    plt.ylabel('RV (km/s)')
    plt.title('Spectroscopic Orbit')
    plt.ylim(-150, 150)
    
    ax1.plot(ymod, xmod, color='k', marker='x', linestyle=' ', ms=6)
    ax2.plot(phobs, rvmod1, color='k', marker='x', linestyle=' ', ms=6)
    ax2.plot(phobs, rvmod2, color='k', marker='x', linestyle=' ', ms=6)
    plt.savefig('vbsb2_initial_guess.png')
    plt.close()
    
        
    #-----------------------------------------------
    
    # FITTING PROCEDURE
    print('  Running LM fit...')
    print('')

    # LOOP THROUGH FIT
    count = 0
    delta_chi = 1.0		# set initially to begin loop
    lam = 0.001   	# Marquardt method to ensure convergence
        
    while((delta_chi > 0.0001)):   

        # Determine errors in orbital elements
        # set up matrices for error determination
        # Invert matrix through Gauss Jordan elimination (Numerical Recipes in C)

        # calculate covariance matrix and column matrix
        theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(El_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
        V1_f_sb2, V2_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb2(El_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, v2_sb2, dv1_sb2, dv2_sb2)        

        # Determine chi squared
        # convert data points to x and y coordinates
        xfit = rho_f * np.cos(theta_f)
        yfit = rho_f * np.sin(theta_f)
        chi2old = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
        
	    # combine SB1 and VB matrices...
        #
	    # initialize full alpha,beta arrays
        alpha = np.zeros((mfit,mfit))
        beta  = np.zeros(mfit)
        beta[vb_subset ] = beta[vb_subset ] + beta_vb
        beta[sb2_subset] = beta[sb2_subset] + beta_sb2
        
        for i in range(0, mfit_vb):
            for j in range(0, mfit_vb):
                alpha[vb_subset[i],vb_subset[j]] = alpha[vb_subset[i],vb_subset[j]] + alpha_vb[i,j]
        for i in range(0, mfit_sb2):
            for j in range(0, mfit_sb2):
                alpha[sb2_subset[i],sb2_subset[j]] = alpha[sb2_subset[i],sb2_subset[j]] + alpha_sb2[i,j]
        invmat = np.linalg.inv(alpha)       
        
        #determine errors:
        #  & adjust alpha matrix by Marquardt parameter lambda
        ELerr = np.zeros(mfit)
        for i in range(0, mfit):
            invmat = abs(invmat)
            ELerr[i] = np.sqrt(invmat[i,i])
            alpha[i,i] = alpha[i,i]*(1.0 + lam)
        
        # Make adjustments for next iteration
        Eladj = np.copy(El)
        if (mfit == 1):
        	delta_el = [beta/alpha]
        else:
            delta_el  = np.linalg.solve(alpha, beta)
                        
        Eladj[el_subset] = np.copy(Eladj[el_subset]) + delta_el
 
        # Make sure ecc > 0, Make sure angles 0-360 or 0-180
        if Eladj[2] < 0: Eladj[2] = 0.   # ecc
        if Eladj[3] < 0:                 # a 
            Eladj[3] = abs(Eladj[3])
            Eladj[5] = Eladj[5] + np.pi
        while Eladj[4] > np.pi: Eladj[4] = Eladj[4] - np.pi    # inc
        while Eladj[4] < 0: Eladj[4] = Eladj[4] + np.pi
        while Eladj[5] > 2*np.pi: Eladj[5] = Eladj[5] - 2*np.pi  #OmB
        while Eladj[5] < 0: Eladj[5] = 2*np.pi - abs(Eladj[5])

        ELadj_vb  = np.copy(Eladj[vb_par])
        ELadj_sb2 = np.copy(Eladj[sb2_par])         
                       
        # Determine chi squared of new params
        theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(ELadj_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
        V1_f_sb2, V2_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb2(ELadj_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, v2_sb2, dv1_sb2, dv2_sb2)        
        xfit = rho_f * np.cos(theta_f)	# y coordinate
        yfit = rho_f * np.sin(theta_f)	# x coordinate
        chi2new = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
        
        
        #   print,"chi2 of next modification:",chi2new
        if chi2new < chi2old:
            El = Eladj * 1.0 
            El_vb = ELadj_vb * 1.0 
            El_sb2 = ELadj_sb2 * 1.0 
            chi2 = chi2new * 1.0 
            delta_chi = chi2old - chi2new
            lam = lam/10.0
            #print(El, np.array([chi2]))
            

            # Plot orbit
            fig = plt.figure(figsize=(10,4))

            ax1 = fig.add_axes([0.57, 0.10, 0.40, 0.80])   
            plt.plot([0], [0], marker='o', color='red', ms=3)
            plt.plot(yarr, xarr, linestyle=' ', color='blue', marker='o', ms=3) 
            #for i in range(0, vb_num):
            #    #  width = total major axis,  height = total minor axis,  angle = deg ccw
            #    e1 = Ellipse(xy=(yarr[i], xarr[i]), width=dmajor[i]*2, height=dminor[i]*2, angle=90-theta_err[i]*180/np.pi, fill=False, color='blue', zorder=5)
            #    ax1.add_patch(e1)
            xr = np.ceil(np.max([xarr,yarr]))
            plt.xlim(xr, -1*xr)
            plt.ylim(-1*xr, xr)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('$\\Delta$RA (mas)' )
            plt.ylabel('$\\Delta$DEC (mas)')
            plt.title('Visual Orbit')

            ax2 = fig.add_axes([0.07, 0.10, 0.40, 0.80])  
            plt.errorbar(phobs, v1_sb2, dv1_sb2, linestyle=' ', color='red', capsize=0.1)
            plt.plot(phobs, v1_sb2, marker='o', color='red', linestyle=' ',  ms=2)
            plt.errorbar(phobs, v2_sb2, dv2_sb2, linestyle=' ', color='blue', capsize=0.1)
            plt.plot(phobs, v2_sb2, marker='o', color='blue', linestyle=' ',  ms=2)
            plt.xlabel('Phase')
            plt.ylabel('RV (km/s)')
            plt.title('Spectroscopic Orbit')
            plt.ylim(-150, 150)

            ax1.plot(yfit, xfit, color='purple', marker='x', linestyle=' ', ms=3)
            ax2.plot(phobs, V1_f_sb2, color='purple', marker='x', linestyle=' ', ms=3)
            ax2.plot(phobs, V2_f_sb2, color='purple', marker='x', linestyle=' ', ms=3)
            plt.savefig('vbsb2_final_fit.png')
            plt.close()
    
            
             
        else:
            lam = lam*10.0
            delta_chi = 1.0
            
        count = count+1
        
        
        print(count, El, chi2)
        #print('%3i %10i %10i %10i' % (count, chi2old, chi2new, chi2))
        
        # do not exit unless lam is less than 0.001
        if (lam > 0.001) & (delta_chi < 0.001):
            delta_chi = 1.0

    lam = 0.0
    
    
    # Determine final error matrix:
    theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(El_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
    V1_f_sb2, V2_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb2(El_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, v2_sb2, dv1_sb2, dv2_sb2)        
    
    
    
    # alpha,beta arrays
    alpha = np.zeros((mfit,mfit))
    beta  = np.zeros(mfit)
    beta[vb_subset ] = beta[vb_subset ] + beta_vb
    beta[sb2_subset] = beta[sb2_subset] + beta_sb2
    
    for i in range(0, mfit_vb):
        for j in range(0, mfit_vb):
            alpha[vb_subset[i],vb_subset[j]] = alpha[vb_subset[i],vb_subset[j]] + alpha_vb[i,j]
    for i in range(0, mfit_sb2):
        for j in range(0, mfit_sb2):
            alpha[sb2_subset[i],sb2_subset[j]] = alpha[sb2_subset[i],sb2_subset[j]] + alpha_sb2[i,j]
    invmat = np.linalg.inv(alpha)       
    
    #determine errors:
    ELerr = np.zeros(mfit)
    for i in range(0, mfit):
        invmat = abs(invmat)
        ELerr[i] = np.sqrt(invmat[i,i])
        alpha[i,i] = alpha[i,i]*(1.0 + lam)
        
    
    # reduced chi squared:
    chi2 = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
    chi2_vb = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2)
    chi2_sb = np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
    dof = 2.0*vb_num + 2.0*sb2_num - mfit
    chi2red = chi2/dof
    chi2red_all= chi2red
    chi2red_vb = chi2_vb/(2.0*vb_num - mfit_vb) 
    chi2red_sb = chi2_sb/(2.0*sb2_num - mfit_sb2) 
    
    print('')
    print('  reduced chi2 = ', chi2red)
    
    ## Error array
    #ELerr = np.zeros(10)
    #ELerr[0] = np.sqrt(chi2red_all)* np.sqrt(invmat[0,0])	# Period
    #ELerr[1] = np.sqrt(chi2red_all)* np.sqrt(invmat[1,1])	# T0
    #ELerr[2] = np.sqrt(chi2red_all)* np.sqrt(invmat[2,2])	# ecc
    #ELerr[3] = np.sqrt(chi2red_vb) * np.sqrt(invmat[3,3])	# a
    #ELerr[4] = np.sqrt(chi2red_vb) * np.sqrt(invmat[4,4])	# inc
    #ELerr[5] = np.sqrt(chi2red_vb) * np.sqrt(invmat[5,5])	# Big w
    #ELerr[6] = np.sqrt(chi2red_all)* np.sqrt(invmat[6,6])	# little w
    #ELerr[7] = np.sqrt(chi2red_sb) * np.sqrt(invmat[7,7])	# k1
    #ELerr[8] = np.sqrt(chi2red_sb) * np.sqrt(invmat[8,8])	# k2
    #ELerr[9] = np.sqrt(chi2red_sb) * np.sqrt(invmat[9,9])	# v0
    

    Elerr_fit = np.zeros(mfit)
    for i in range(0,mfit):
         Elerr_fit[i] = np.sqrt(chi2red)*np.sqrt(invmat[i,i])
    
    # reformat EL
    Elbest = np.zeros(10)
    Elbest[0] = El[0]	# Period
    Elbest[1] = El[1]	# T0
    Elbest[2] = El[2]	# ecc
    Elbest[3] = El[3]	# a
    Elbest[4] = El[4]	# inc
    Elbest[5] = El[5]	# Big w
    Elbest[6] = El[6]	# little w
    Elbest[7] = El[7]	# k1
    Elbest[8] = El[8]	# k2
    Elbest[9] = El[9]	# v0
    
    # Error array
    Elerr = np.zeros(10)
    for i in range(0, mfit):
        Elerr[el_subset[i]] = Elerr_fit[i]
    
    #for i in range(0, 10):
    #    print('%10s %10.3f %10.3f' % (elLabel[i], Elbest[i], Elerr[i]))
    
    return(El, ELerr, chi2red_all, chi2red_vb, chi2red_sb)


def fit_orbit_vbsb2_ell(time_vb, theta, rho, dmajor, dminor, theta_err, time_sb2, v1_sb2, dv1_sb2, v2_sb2, dv2_sb2, param, parfit):
    # This version can handle elliptical uncertainties for visual orbits.  
    # Minimize chi2 based on projections along the direction of error ellipse (x',y').
    # Compute the orbital elements through a Newton-Raphson technique.
    # Fits simultaneously for VB + SB2 orbits
    # Written by Gail Schaefer,  translated to python by Katie Lester
    # Calls the following functions from "orbitcode.py":
    #     calc_deriv_vb
    #     calc_deriv_sb2

    
    ## Read in VB data 
    ## readcol, vb_file, time_vb, theta, dtheta, rho, drho
    #print('  Reading in observations...')
    #print(' ' )
    #tab  = ascii.read(vb_file,  data_start=0)
    #time_vb  = np.array(tab['col1'])
    #theta    = np.array(tab['col2'])
    #rho      = np.array(tab['col3'])
    #dmajor   = np.array(tab['col4'])
    #dminor   = np.array(tab['col5'])
    #theta_err= np.array(tab['col6'])
    vb_num = len(time_vb)
    
    # convert theta to radians
    theta = theta*np.pi/180.
    theta_err = theta_err*np.pi/180.
    
    # convert data points to x and y coordinates IN COUTEAU NOTATION.  
    #    (theta is measured from Y axis, not X axis)
    xarr = rho * np.cos(theta)	    # x coordinate = DEC, bc PA=0=N
    yarr = rho * np.sin(theta)	    # y coordinate = RA

    # project separation into components along the direction of the error ellipse
    xparr = rho * np.cos(theta - theta_err)  	# x' (major)
    yparr = rho * np.sin(theta - theta_err)  	# y' (minor) 

    ## Read in SB data
    ## readcol, sb2_file, time_sb2, V1_sb2, dv1_sb2, v2_sb2, dv2_sb2
    #tab  = ascii.read(sb_file,  data_start=0)
    #time_sb2 = np.array(tab['col1'])
    #v1_sb2   = np.array(tab['col2'])
    #dv1_sb2  = np.array(tab['col3'])
    #v2_sb2   = np.array(tab['col4'])
    #dv2_sb2  = np.array(tab['col5'])
    sb2_num = len(time_sb2)
    
    # Initial guess for param
    period= param[0]
    Tperi = param[1]
    ecc =   param[2]
    major = param[3]
    inc =   param[4]
    W_cap = param[5]
    w_low = param[6]    #  omega_A    for vbsb2 fits! will use flagwa later
    k1 =    param[7]
    k2 =    param[8]
    Vsys =  param[9]
    
    
    # Fix = 0, float = 1 
    f0 = parfit[0]
    f1 = parfit[1]
    f2 = parfit[2]
    f3 = parfit[3]
    f4 = parfit[4]
    f5 = parfit[5]
    f6 = parfit[6]
    f7 = parfit[7]
    f8 = parfit[8]
    f9 = parfit[9]
    
    #-----------------------------------------------
    # Order of orbital element arrays
    # do not set here yet (need to get angles into radians)
    # EL_vb = [period,Tperi,ecc,major,inc,W_cap,w_low]
    #             0    1     2    3    4    5     6
    # EL_sb2 = [period,Tperi,ecc,K1,K2,w_low,Vsys]
    #              0    1     2   7  8   6     9
    # EL_sb1 = [period,Tperi,ecc,K1,w_low,Vsys]
    #              0    1     2   7   6     9
    
    elfix=np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9])
    nEl = len(elfix)

    elfix_vb=np.array([f0,f1,f2,f3,f4,f5,f6])
    nEl_vb = len(elfix_vb)

    elfix_sb2=np.array([f0,f1,f2,f7,f8,f6,f9])
    nEl_sb2 = len(elfix_sb2)

    # convert i, Omega, omega to radians. 
    inc = inc*np.pi/180.0
    W_cap = W_cap*np.pi/180.0
    w_low = w_low*np.pi/180.0
    
    # number of elements to improve
    k=0
    for i in range(0, nEl): 
        if elfix[i] == 0: k=k+1
    mfit = nEl - k     

    k=0
    for i in range(0, nEl_vb): 
        if elfix_vb[i] == 0: k=k+1
    mfit_vb = nEl_vb - k 

    k=0
    for i in range(0, nEl_sb2): 
        if elfix_sb2[i] == 0: k=k+1
    mfit_sb2 = nEl_sb2 - k 
    
    El = np.array([period,Tperi,ecc,major,inc,W_cap,w_low,k1,k2,Vsys])
    El_vb = np.array([period,Tperi,ecc,major,inc,W_cap,w_low])
    El_sb2 = np.array([period,Tperi,ecc,k1,k2,w_low,Vsys])
        
    Eladj = np.copy(El)
    elLabel = ['P','T','e','a(mas)','i','Omega_B','omega_A','K1(km/s)','K2(km/s)','Vsys']

    # determine which indices of full set are being varied
    vb_subset = np.zeros(mfit_vb, dtype=int)
    sb2_subset = np.zeros(mfit_sb2, dtype=int)

    # determine subarray of indices that are VB, SB1, and SB2 parameters
    vb_par = np.zeros(nEl_vb, dtype=int)
    sb2_par = np.zeros(nEl_sb2, dtype=int)

    # determine indices of elements for which to vary
    el_subset = np.where(elfix > 0)[0]
    Elvar = El[el_subset]

    k=0
    for i in range(0, nEl_vb):
        ind = np.where(El_vb[i] == El)[0][0]
        vb_par[i] = ind
        if elfix_vb[i] != 0:
            ind = np.where(El_vb[i] == Elvar)[0][0]
            vb_subset[k] = ind
            k=k+1
    k=0
    for i in range(0, nEl_sb2):
        ind = np.where(El_sb2[i] == El)[0][0]
        sb2_par[i] = ind
        if elfix_sb2[i] != 0:
            ind = np.where(El_sb2[i] == Elvar)[0][0]
            sb2_subset[k] = ind
            k=k+1

    
    #-----------------------------------------------
    # Plot initial guess orbit
    # calculate model for VB
        
    # flag_wa makes w = wA not wB
    tnum=100    
    tarr = np.linspace(El[1], El[1]+El[0], tnum)  
    rho_mod, theta_mod = calc_vbfit(El_vb, time_vb, flagwa=1)
    xmod = rho_mod * np.cos(theta_mod)	# x coordinate
    ymod = rho_mod * np.sin(theta_mod)	# y coordinate
    # calculate model for SB2
    phobs = (time_sb2 - Tperi) / period % 1
    phmod = (tarr - Tperi) / period % 1
    rvmod1, rvmod2 = calc_sb2fit(El_sb2, time_sb2)
    # Plot orbit
    print('  Plotting initial guess...')
    print('')
    fig = plt.figure(figsize=(10,4))
    
    ax1 = fig.add_axes([0.57, 0.10, 0.40, 0.80])   
    plt.plot([0], [0], marker='o', color='red', ms=3)
    plt.plot(yarr, xarr, linestyle=' ', color='blue', marker='o', ms=6) 
    for i in range(0, vb_num):
        plt.plot([yarr[i], ymod[i]],[xarr[i], xmod[i]], color='black', alpha=0.5) 
    #    #  width = total major axis,  height = total minor axis,  angle = deg ccw
    #    e1 = Ellipse(xy=(yarr[i], xarr[i]), width=dmajor[i]*2, height=dminor[i]*2, angle=90-theta_err[i]*180/np.pi, fill=False, color='blue', zorder=5)
    #    ax1.add_patch(e1)
    xr = np.ceil(np.max([xarr,yarr]))*1.2
    plt.xlim(xr, -1*xr)
    plt.ylim(-1*xr, xr)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$\\Delta$RA (mas)' )
    plt.ylabel('$\\Delta$DEC (mas)')
    plt.title('Visual Orbit')
    
    ax2 = fig.add_axes([0.07, 0.10, 0.40, 0.80])  
    plt.errorbar(phobs, v1_sb2, dv1_sb2, linestyle=' ', color='red', capsize=0.1)
    plt.plot(phobs, v1_sb2, marker='o', color='red', linestyle=' ',  ms=6)
    plt.errorbar(phobs, v2_sb2, dv2_sb2, linestyle=' ', color='blue', capsize=0.1)
    plt.plot(phobs, v2_sb2, marker='o', color='blue', linestyle=' ',  ms=6)
    plt.xlabel('Phase')
    plt.ylabel('RV (km/s)')
    plt.title('Spectroscopic Orbit')
    plt.ylim(-150, 150)
    
    ax1.plot(ymod, xmod, color='k', marker='x', linestyle=' ', ms=6)
    ax2.plot(phobs, rvmod1, color='k', marker='x', linestyle=' ', ms=6)
    ax2.plot(phobs, rvmod2, color='k', marker='x', linestyle=' ', ms=6)
    #plt.savefig('fit1_initial_guess.png')
    plt.show()
    plt.close()
    
    ans = 'y'
    ans = input('  Continue? [y] ')
    if ans != 'y':
        return(None)
    print('')
    
        
    #-----------------------------------------------
    
    # FITTING PROCEDURE
    print('  Running LM fit...')
    print('')

    # LOOP THROUGH FIT
    count = 0
    delta_chi = 1.0  # set initially to begin loop
    lam = 0.001      # Marquardt method to ensure convergence
        
    while((delta_chi > 0.0001)):   

        # Determine errors in orbital elements
        # set up matrices for error determination
        # Invert matrix through Gauss Jordan elimination (Numerical Recipes in C)

        # calculate covariance matrix and column matrix
        theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(El_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
        V1_f_sb2, V2_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb2(El_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, v2_sb2, dv1_sb2, dv2_sb2)        

        # Determine chi squared
        # convert data points to x and y coordinates
        xfit = rho_f * np.cos(theta_f)
        yfit = rho_f * np.sin(theta_f)
        chi2old = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
        
        # combine SB1 and VB matrices...
        #
        # initialize full alpha,beta arrays
        alpha = np.zeros((mfit,mfit))
        beta  = np.zeros(mfit)
        beta[vb_subset ] = beta[vb_subset ] + beta_vb
        beta[sb2_subset] = beta[sb2_subset] + beta_sb2
        
        for i in range(0, mfit_vb):
            for j in range(0, mfit_vb):
                alpha[vb_subset[i],vb_subset[j]] = alpha[vb_subset[i],vb_subset[j]] + alpha_vb[i,j]
        for i in range(0, mfit_sb2):
            for j in range(0, mfit_sb2):
                alpha[sb2_subset[i],sb2_subset[j]] = alpha[sb2_subset[i],sb2_subset[j]] + alpha_sb2[i,j]
        invmat = np.linalg.inv(alpha)       
        
        #determine errors:
        #  & adjust alpha matrix by Marquardt parameter lambda
        ELerr = np.zeros(mfit)
        for i in range(0, mfit):
            invmat = abs(invmat)
            ELerr[i] = np.sqrt(invmat[i,i])
            alpha[i,i] = alpha[i,i]*(1.0 + lam)
        
        # Make adjustments for next iteration
        Eladj = np.copy(El)
        if (mfit == 1):
            delta_el = [beta/alpha]
        else:
            delta_el  = np.linalg.solve(alpha, beta)
                        
        Eladj[el_subset] = np.copy(Eladj[el_subset]) + delta_el
 
        # Make sure ecc > 0, Make sure angles 0-360 or 0-180
        if Eladj[2] < 0: Eladj[2] = 0.   # ecc
        if Eladj[3] < 0:                 # a 
            Eladj[3] = abs(Eladj[3])
            Eladj[5] = Eladj[5] + np.pi
        while Eladj[4] > np.pi: Eladj[4] = Eladj[4] - np.pi    # inc
        while Eladj[4] < 0: Eladj[4] = Eladj[4] + np.pi
        while Eladj[5] > 2*np.pi: Eladj[5] = Eladj[5] - 2*np.pi  #OmB
        while Eladj[5] < 0: Eladj[5] = 2*np.pi - abs(Eladj[5])

        ELadj_vb  = np.copy(Eladj[vb_par])
        ELadj_sb2 = np.copy(Eladj[sb2_par])         
                       
        # Determine chi squared of new params
        theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(ELadj_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
        V1_f_sb2, V2_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb2(ELadj_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, v2_sb2, dv1_sb2, dv2_sb2)        
        xfit = rho_f * np.cos(theta_f)	# y coordinate
        yfit = rho_f * np.sin(theta_f)	# x coordinate
        chi2new = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
        
        
        #   print,"chi2 of next modification:",chi2new
        if chi2new < chi2old:
            El = Eladj * 1.0 
            El_vb = ELadj_vb * 1.0 
            El_sb2 = ELadj_sb2 * 1.0 
            chi2 = chi2new * 1.0 
            delta_chi = chi2old - chi2new
            lam = lam/10.0
            
             
        else:
            lam = lam*10.0
            delta_chi = 1.0
            
        count = count+1
        
        
        print(count, El, chi2)
        #print('%3i %10i %10i %10i' % (count, chi2old, chi2new, chi2))
        
        # do not exit unless lam is less than 0.001
        if (lam > 0.001) & (delta_chi < 0.001):
            delta_chi = 1.0
            
    print('')
    print('  Plotting final solution...')
    # Plot orbit
    fig = plt.figure(figsize=(10,4))
    
    ax1 = fig.add_axes([0.57, 0.10, 0.40, 0.80])   
    plt.plot([0], [0], marker='o', color='red', ms=3)
    plt.plot(yarr, xarr, linestyle=' ', color='blue', marker='o', ms=3) 
    #for i in range(0, vb_num):
    #    #  width = total major axis,  height = total minor axis,  angle = deg ccw
    #    e1 = Ellipse(xy=(yarr[i], xarr[i]), width=dmajor[i]*2, height=dminor[i]*2, angle=90-theta_err[i]*180/np.pi, fill=False, color='blue', zorder=5)
    #    ax1.add_patch(e1)
    xr = np.ceil(np.max([xarr,yarr]))
    plt.xlim(xr, -1*xr)
    plt.ylim(-1*xr, xr)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$\\Delta$RA (mas)' )
    plt.ylabel('$\\Delta$DEC (mas)')
    plt.title('Visual Orbit')
    
    ax2 = fig.add_axes([0.07, 0.10, 0.40, 0.80])  
    plt.errorbar(phobs, v1_sb2, dv1_sb2, linestyle=' ', color='red', capsize=0.1)
    plt.plot(phobs, v1_sb2, marker='o', color='red', linestyle=' ',  ms=2)
    plt.errorbar(phobs, v2_sb2, dv2_sb2, linestyle=' ', color='blue', capsize=0.1)
    plt.plot(phobs, v2_sb2, marker='o', color='blue', linestyle=' ',  ms=2)
    plt.xlabel('Phase')
    plt.ylabel('RV (km/s)')
    plt.title('Spectroscopic Orbit')
    plt.ylim(-150, 150)
    
    ax1.plot(yfit, xfit, color='k', marker='x', linestyle=' ', ms=5)
    ax2.plot(phobs, V1_f_sb2, color='k', marker='x', linestyle=' ', ms=5)
    ax2.plot(phobs, V2_f_sb2, color='k', marker='x', linestyle=' ', ms=5)
    #plt.savefig('fit2_final_solution.png')
    plt.show()
    plt.close()
            
    lam = 0.0
    
    
    # Determine final error matrix:
    theta_f, rho_f, xpfit, ypfit, alpha_vb, beta_vb = calc_deriv_vb_ell(El_vb, elfix_vb, mfit_vb, time_vb, xparr, yparr, dmajor, dminor, theta_err, flag_wa=1)
    V1_f_sb2, V2_f_sb2, alpha_sb2, beta_sb2 = calc_deriv_sb2(El_sb2, elfix_sb2, mfit_sb2, time_sb2, v1_sb2, v2_sb2, dv1_sb2, dv2_sb2)        
    
    
    
    # alpha,beta arrays
    alpha = np.zeros((mfit,mfit))
    beta  = np.zeros(mfit)
    beta[vb_subset ] = beta[vb_subset ] + beta_vb
    beta[sb2_subset] = beta[sb2_subset] + beta_sb2
    
    for i in range(0, mfit_vb):
        for j in range(0, mfit_vb):
            alpha[vb_subset[i],vb_subset[j]] = alpha[vb_subset[i],vb_subset[j]] + alpha_vb[i,j]
    for i in range(0, mfit_sb2):
        for j in range(0, mfit_sb2):
            alpha[sb2_subset[i],sb2_subset[j]] = alpha[sb2_subset[i],sb2_subset[j]] + alpha_sb2[i,j]
    invmat = np.linalg.inv(alpha)       
    
    #determine errors:
    ELerr = np.zeros(mfit)
    for i in range(0, mfit):
        invmat = abs(invmat)
        ELerr[i] = np.sqrt(invmat[i,i])
        alpha[i,i] = alpha[i,i]*(1.0 + lam)
        
    
    # reduced chi squared:
    chi2 = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2) + np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
    chi2_vb = np.sum((xparr - xpfit)**2/dmajor**2 + (yparr - ypfit)**2/dminor**2)
    chi2_sb = np.sum((v1_sb2 - V1_f_sb2)**2/dv1_sb2**2 + (v2_sb2 - V2_f_sb2)**2/dv2_sb2**2)
    dof = 2.0*vb_num + 2.0*sb2_num - mfit
    chi2red = chi2/dof
    chi2red_all= chi2red
    chi2red_vb = chi2_vb/(2.0*vb_num - mfit_vb) 
    chi2red_sb = chi2_sb/(2.0*sb2_num - mfit_sb2) 
    
    print('')
    print('  reduced chi2 = ', chi2red)
    
    ## Error array
    #ELerr = np.zeros(10)
    #ELerr[0] = np.sqrt(chi2red_all)* np.sqrt(invmat[0,0])	# Period
    #ELerr[1] = np.sqrt(chi2red_all)* np.sqrt(invmat[1,1])	# T0
    #ELerr[2] = np.sqrt(chi2red_all)* np.sqrt(invmat[2,2])	# ecc
    #ELerr[3] = np.sqrt(chi2red_vb) * np.sqrt(invmat[3,3])	# a
    #ELerr[4] = np.sqrt(chi2red_vb) * np.sqrt(invmat[4,4])	# inc
    #ELerr[5] = np.sqrt(chi2red_vb) * np.sqrt(invmat[5,5])	# Big w
    #ELerr[6] = np.sqrt(chi2red_all)* np.sqrt(invmat[6,6])	# little w
    #ELerr[7] = np.sqrt(chi2red_sb) * np.sqrt(invmat[7,7])	# k1
    #ELerr[8] = np.sqrt(chi2red_sb) * np.sqrt(invmat[8,8])	# k2
    #ELerr[9] = np.sqrt(chi2red_sb) * np.sqrt(invmat[9,9])	# v0
    

    Elerr_fit = np.zeros(mfit)
    for i in range(0,mfit):
         Elerr_fit[i] = np.sqrt(chi2red)*np.sqrt(invmat[i,i])
    
    # reformat EL
    Elbest = np.zeros(10)
    Elbest[0] = El[0]  # Period
    Elbest[1] = El[1]  # T0
    Elbest[2] = El[2]  # ecc
    Elbest[3] = El[3]  # a
    Elbest[4] = El[4]  # inc
    Elbest[5] = El[5]  # Big w
    Elbest[6] = El[6]  # little w
    Elbest[7] = El[7]  # k1
    Elbest[8] = El[8]  # k2
    Elbest[9] = El[9]  # v0
    
    # Error array
    Elerr = np.zeros(10)
    for i in range(0, mfit):
        Elerr[el_subset[i]] = Elerr_fit[i]
    
    #for i in range(0, 10):
    #    print('%10s %10.3f %10.3f' % (elLabel[i], Elbest[i], Elerr[i]))
    
    return(El, ELerr, chi2red_all, chi2red_vb, chi2red_sb)






