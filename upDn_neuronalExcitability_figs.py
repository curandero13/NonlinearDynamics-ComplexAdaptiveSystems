from upDn_baseCode import *
import matplotlib 
matplotlib.use('macosx')

eCharge=1.60217733e-19 # Coulombs; 
kBoltzmann=1.38065812e-20 #mJ/K
zeroT=273.15 #deg Kelvin
TCelcius = 36
v_T = kBoltzmann * (zeroT + TCelcius)/ eCharge
C_m = 20.0; vTCm= v_T * C_m
print(r'v_T=%g, C_m=%g, v_T C_m= %g'%(v_T,C_m, vTCm))

voltages = {'u_w':-10.0/v_T, 'u_m': -20.0/v_T,'u_U':60.0/v_T, 'u_D':-90.0/v_T, 'u_UD':-80.0/v_T}
biases = {'b_w':0.65, 'b_U':0.5, 'b_D':0.3, 'b_UD':0.1, 'g_m':4, 'g_w':2.4}
rates = {'r_w':0.25, 'a_F': 0*100 / vTCm, 'r_U':1, 'r_D':1, 'r_UD':1e-4, 'a_U': 4, 'a_D': 6, 'a_UD':800}
numerics = {'timeMin': -0.0, 'timeMax':300.0, 'timeStep':1/40.0, 'ic': np.array([0.0001, -60.0/v_T]),\
            'uMin':-100/v_T,'uMax':40/v_T, 'wMin':0,'wMax':1, 'uStep':0.1/v_T,'wStep':0.01}
p = {'v_T': v_T, 'C_m':C_m, 'vTCm': v_T * C_m, 'kappa_w':0.35}
p= {**p, **voltages, **biases, **rates, **numerics}
#
upDn = UD(params= p, variables=['u','w'])
upDn.uRange = np.arange(upDn.pars['uMin'], upDn.pars['uMax'], upDn.pars['uStep'])
upDn.wRange = np.arange(upDn.pars['wMin'], upDn.pars['wMax'], upDn.pars['wStep'])

# -------------
# Fig 1. Two kinds of partitions of the phase plane 
# -------------
f = pl.figure(figsize=(11,4)); pl.ioff(); 
ax = list(); rows = 1; cols= 1;
for n in range(rows*cols):
    ax.append(f.add_subplot(rows,cols,n+1))  
    upDn.phasePlane(ax=ax[n], W = np.linspace(0,1,50), U = np.linspace(-90,30,200)/26.64, \
        wNullLabel=r'$\partial_t w = 0$', vNullLabel=r'$\partial_t w = 0$', plotNullClines=1)
pl.ion(); pl.draw(); pl.show()



# -------------
# Fig 2. Dynamics from different initial conditions
# -------------

upDn.pars['u_m']= -20.0/v_T; upDn.pars['g_m']= 4;  
upDn.pars['u_w']= -10.0/v_T; upDn.pars['g_w']= 2.4; upDn.pars['b_w'] = 0.65; upDn.pars['kappa_w']=0.35; upDn.pars['r_w'] = 0.25
upDn.pars['b_D']= 0.3; #upDn.pars['b_UD']= 0.8
upDn.pars['a_F'] = 0 / vTCm; # 70 pA en rheobase 
upDn.pars['a_U'] = 4.; upDn.pars['a_D'] = 6; upDn.pars['a_UD'] = 800
upDn.pars['timeMax'] = 100.0; upDn.pars['timeStep']=1/70.0; 
upDn.pars['ic'] = np.array([-50.0/v_T, 0.001])
#
vics = np.arange(-90,-49,10)/upDn.pars['v_T']
nIcs = len(vics)
orbits = list()
for v0 in vics:
    upDn.pars['ic'][0] = v0
    orbits.append(upDn.getDynamics())
    
f = pl.figure(figsize=(11,4)); pl.ioff(); 
ax = list(); rows = 1; cols= 3;
for n in range(rows*cols):
    ax.append(f.add_subplot(rows,cols,n+1))  
for n in range(nIcs):
    ax[0].plot(orbits[nIcs-1-n]['timeSamples'], orbits[nIcs-1-n]['vOrbit'],lw=(n+1)/2, alpha=0.3+(n+1)/(3+nIcs), label =r'$(t, v)$')
    ax[1].plot(orbits[nIcs-1-n]['wOrbit'], orbits[nIcs-1-n]['vOrbit'],lw=(n+1)/2, alpha=0.3+(n+1)/(3+nIcs), label =r'$(w,v)$')
    ax[2].plot(orbits[nIcs-1-n]['dvdt'], orbits[nIcs-1-n]['vOrbit'],lw=(n+1)/2, alpha=0.3+(n+1)/(3+nIcs), label =r'$( \partial_t v, v)$')
    ax[0].set_xlim(-1,20)
pl.ion(); pl.draw(); pl.show()

