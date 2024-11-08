import synth_ecg_gen.utils.vcg as vcg
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


# matrix to convert from VCG to 12-lead ECG
# See https://onlinelibrary.wiley.com/doi/10.1002/clc.1980.3.2.87
DowerMatrix = np.array([[0.632, 0.235, -0.397, -0.434, 0.515, -0.081, -0.515, 0.044, 0.882, 1.213, 1.125, 0.831],
         [-0.235, 1.066, 1.301, -0.415, -0.768, 1.184, 0.157, 0.164, 0.098, 0.127, 0.127, 0.076],
         [0.059, -0.132, -0.191, 0.037, 0.125, -0.162, -0.917, -1.387, -1.277, -0.601, -0.086, 0.230]])

# dictionary for lead order in output matrix after dower transformation
Dower_lead_map = {'I':0, 'II':1, 'III':2, 'aVR':3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11}

# rotation matrices. Input in degrees
def Rx(x) : 

	th_x = x*np.pi/180.

	return np.array([[1, 0, 0],
      				 [0, np.cos(th_x), -np.sin(th_x)],
      				 [0, np.sin(th_x), np.cos(th_x)]])

def Ry(y) : 

	th_y = y*np.pi/180.

	return np.array([[np.cos(th_y), 0, np.sin(th_y)],
    				 [0, 1, 0],
    				 [-np.sin(th_y), 0, np.cos(th_y)]])

def Rz(z) : 

	th_z = z*np.pi/180.

	return np.array([[np.cos(th_z), -np.sin(th_z), 0],
    				 [np.sin(th_z), np.cos(th_z), 0],
    				 [0, 0, 1]])


#TODO don't take first 10 seconds, or at least exclude the first cardiac cycle
# solve with default parameters
def default_vcg_solve(HR, fs=512, duration=10) :

	v0 = np.array([0,.3,.3,.3])

	tspan = np.linspace(0, duration, int(duration*fs))

	f = vcg.VCG(HR).call

	sol = solve_ivp(f, [tspan[0], tspan[-1]], v0, t_eval=tspan)

	return sol.t, sol.y.T[:,1:]

# solve input ode object
def solve_vcg_object(vcg_ode, fs=512, duration=10, v0=np.array([0,.3,.3,.3])) :

	tspan = np.linspace(0, duration, int(duration*fs))

	sol = solve_ivp(vcg_ode.call, [tspan[0], tspan[-1]], v0, t_eval=tspan)

	return sol.t, sol.y.T[:,1:]


def convert_vcg_to_12lead(vcg) :

	return vcg @ DowerMatrix


def rotate_vcg(vcg, th_x=0, th_y=0, th_z=0) :

	return (Rx(th_x) @ Ry(th_y) @ Rz(th_z) @ vcg.T).T


# plot standard 12-lead ECG. Input must be 10s, 12 leads
# n_squares is how many .5 mV blocks should be used per plot. 
def plot12(ecg, fs=512, n_squares=8) :

	# make sure the signal is 10 seconds long
	assert len(ecg) == fs*10

	quarter_wform = int(fs*2.5)
	qtime = np.linspace(0,2.5,quarter_wform)

	fig, axs = plt.subplots(4,4, figsize=[22,16])

	ax_strip = plt.subplot(4,1,4)

	time = np.linspace(0,10,10*fs)

	axs[0,0].plot(time, ecg[:,Dower_lead_map['I']], 'k')
	axs[1,0].plot(time, ecg[:,Dower_lead_map['II']], 'k')
	axs[2,0].plot(time, ecg[:,Dower_lead_map['III']], 'k')

	axs[0,1].plot(time[:3*quarter_wform], ecg[quarter_wform:,Dower_lead_map['aVR']], 'k')
	axs[1,1].plot(time[:3*quarter_wform], ecg[quarter_wform:,Dower_lead_map['aVL']], 'k')
	axs[2,1].plot(time[:3*quarter_wform], ecg[quarter_wform:,Dower_lead_map['aVF']], 'k')

	axs[0,2].plot(time[:2*quarter_wform], ecg[2*quarter_wform:,Dower_lead_map['V1']], 'k')
	axs[1,2].plot(time[:2*quarter_wform], ecg[2*quarter_wform:,Dower_lead_map['V2']], 'k')
	axs[2,2].plot(time[:2*quarter_wform], ecg[2*quarter_wform:,Dower_lead_map['V3']], 'k')

	axs[0,3].plot(time[:quarter_wform], ecg[3*quarter_wform:,Dower_lead_map['V4']], 'k')
	axs[1,3].plot(time[:quarter_wform], ecg[3*quarter_wform:,Dower_lead_map['V5']], 'k')
	axs[2,3].plot(time[:quarter_wform], ecg[3*quarter_wform:,Dower_lead_map['V6']], 'k')

	ax_strip.plot(time, ecg[:,Dower_lead_map['II']], 'k')


	majorgrid_y = np.linspace(-n_squares/2, n_squares/2, n_squares+1)
	minorgrid_y = np.linspace(-n_squares/2, n_squares/2, n_squares*5+1)

	majorgrid_x = np.linspace(0, 10, int(10/.2))
	minorgrid_x = np.linspace(0, 10, int(10/.04))


	i = 0
	for a in axs.flatten() :

		majorgrid_x = np.linspace(0.1*((i%4)==3) + 0.1*((i%4)==1), 10 + 0.1*((i%4)==3) + 0.1*((i%4)==1), int(10/.2)+1)
		minorgrid_x = np.linspace(0.0 + 0.02*((i%4)==3) + 0.02*((i%4)==1), 10.0 + 0.02*((i%4)==3) + 0.02*((i%4)==1), int(10/.04)+1)


		a.hlines(majorgrid_y, xmin=-1000, xmax=len(ecg)*1.1, colors='k', alpha=0.5, linewidth=0.5)
		a.hlines(minorgrid_y, xmin=-1000, xmax=len(ecg)*1.1, colors='k', alpha=0.5, linewidth=0.3)

		a.vlines(majorgrid_x, ymin=-1000, ymax=1000, colors='k', alpha=0.5, linewidth=0.5)
		a.vlines(minorgrid_x, ymin=-1000, ymax=1000, colors='k', alpha=0.5, linewidth=0.3)

		a.set_yticks([])
		a.set_xticks([])
		a.axis("off")
		a.set_xlim(left=0, right=2.5)
		a.set_ylim(bottom=-n_squares/2, top=n_squares/2)

		i += 1


	majorgrid_y = np.linspace(-n_squares/2, n_squares/2, n_squares+1)
	minorgrid_y = np.linspace(-n_squares/2, n_squares/2, n_squares*5+1)

	majorgrid_x = np.linspace(0, 10, int(10/.2)+1)
	minorgrid_x = np.linspace(0, 10, int(10/.04)+1)

	ax_strip.set_yticks([])
	ax_strip.set_xticks([])

	ax_strip.spines["right"].set_visible(False)
	ax_strip.spines["left"].set_visible(False)
	ax_strip.spines["top"].set_visible(False)
	ax_strip.spines["bottom"].set_visible(False)

	ax_strip.set_xlim(left=0, right=10)
	ax_strip.set_ylim(bottom=-n_squares/2, top=n_squares/2)
	ax_strip.hlines(majorgrid_y, xmin=-1000, xmax=len(ecg)*1.1, colors='k', alpha=0.5, linewidth=0.5)
	ax_strip.hlines(minorgrid_y, xmin=-1000, xmax=len(ecg)*1.1, colors='k', alpha=0.5, linewidth=0.3)
	ax_strip.vlines(majorgrid_x, ymin=-1000, ymax=1000, colors='k', alpha=0.5, linewidth=0.5)
	ax_strip.vlines(minorgrid_x, ymin=-1000, ymax=1000, colors='k', alpha=0.5, linewidth=0.3)

	plt.subplots_adjust(wspace=0, hspace=0)


