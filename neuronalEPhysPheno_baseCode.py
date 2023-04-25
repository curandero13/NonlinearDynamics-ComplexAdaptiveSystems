import numpy as np
import numpy.linalg as la
from numpy.random import uniform
import matplotlib.pylab as pl
import sympy as sy
from sympy.plotting import plot, PlotGrid, plot3d
from sympy.utilities.iterables import flatten
import time
import pickle
 
# ----------------------
# Auxiliary functions
# ----------------------
eCharge=1.60217733e-19 # Coulombs
kBoltzmann=1.38065812e-20 #mJ/K
zeroT=273.15 #deg Kelvin
TCelcius = 36
v_T = kBoltzmann * (zeroT + TCelcius)/ eCharge
#print('v_T = %g mV'% v_T)

# ----------------------
# Auxiliary functions
# ----------------------
def delDictItem(d,key):
    r = d.copy()
    del(r[key])
    return r

def dictMerge(dict1, dict2):
    return(dict2.update(dict1))

def biasedExp(x,a,b=1):
    return sy.exp(b * (x-a))

def sigmoid(x,a,n=1):
    xn = x**n
    return xn /(xn + a**n)

def save_object(obj,fName):
    try:
        with open(fName+".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
 
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

# ---------------------------------
# Current injection function
# ---------------------------------
def UpTopDn(t,upStart=200.0,upStop=400, dnStart=600.0,dnStop=800.0,rampAmp=1.0):
    slope_up = rampAmp/(upStop-upStart)
    int_up = -slope_up*upStart
    slope_dn = -rampAmp/(dnStop-dnStart)
    int_dn= rampAmp-slope_dn*dnStart
    c1=np.int16((upStart<t)&(t<=upStop))
    c2=np.int16((upStop<t)&(t<=dnStart))
    c3=np.int16((t>dnStart)&(t<=dnStop))
    y=c1*(slope_up*t +int_up)+ c2*rampAmp+ c3*(slope_dn*t +int_dn)
    return y

# ---------------------------------
# Spike trains
# ---------------------------------
def spikeInds(dv,dvdtThresh=100):
    i = np.where(dv>dvdtThresh)[0]-1;
    di= np.where(i[1:]-i[:-1]>1)[0]+1;
    si = list()
    si.append(i[0])
    for n in range(len(di)):
        si.append(i[di][n])
    return si

def calcISI(spikeTimes):
    isis= np.zeros(len(spikeTimes))
    isis[1:] = (spikeTimes[1:]-spikeTimes[:-1])
    return isis

def calcIFR(spikeTimes):
    ifrs= np.zeros(len(spikeTimes))
    ifrs[1:] =1/ (spikeTimes[1:]-spikeTimes[:-1])
    return ifrs

# ----------------------
# Numerics
# ----------------------
def calcDiffQuotients(t,x):
    nn = len(x)
    d = np.zeros(nn)
    if (len(t) == nn):
        d[1:] = (x[1:]-x[0:])/(t[1:]-t[0:])
    return d

def secant_method(f, x0, x1, tol=1e-5, n=0):
    n += 1 # increment counter
    y0, y1 = f(x0), f(x1) # calculate function values at endpoints
    xn = x1 - y1 * ((x1 - x0) / (y1 - y0)) # calculate next root approximation
    if -tol < y1 < tol: # check tolerance condition
        return xn, n
    # recursive call with updated interval
    return secant_method(f, x1, xn, n=n)

def RK2_autonomousStep(f, U, stepSize):
    k = stepSize * f(U) / 2
    return U + stepSize * f(U + k)

def RK2_nonautonomousStep(f, U, p, stepSize):
    k = stepSize * f(U,p) / 2
    return U + stepSize * f( U + k, p)

# ------------------------
# Vector fields
# ------------------------
def field2D(f,p):
    rango_x = np.linspace(p['xMin'],p['xMax'],p['xStepSize'])
    rango_y = np.linspace(p['yMin'],p['yMax'],p['yStepSize'])
    xVec,yVec = np.meshgrid(rango_x,rango_y)
    dx,dy = f(np.array([xVec,yVec]),p)
    return xVec,yVec,dx,dy

# ----------------------
# Bifurcation
# ----------------------
def fixedPoint2D_typeMarker(eigenValues,ms=5,color_nodo='orange',color_foco='blue'):
    pf=dict()
    l1,l2 = eigenValues
    r1,r2 = np.real(eigenValues)
    i1,i2 = np.imag(eigenValues)
    pf['eigVal_1'] = l1; pf['eigVal_2']=l2; pf['r1']=r1; pf['r2']=r2; pf['i1']=i1; pf['i2']=i2;
    pf['transp'] = 1; pf['mfc']= 'white'; pf['ms']= ms
    if (i1*i2<0): 
        pf['type']='focus';pf['mec']= color_foco; pf['mfc'] = color_foco
        if r1>0: 
            pf['localDyn']='repeller'; pf['marker']='o'; pf['mfc'] ='white'; 
        elif r1<0:
            pf['localDyn']='attractor'; pf['marker']='o'
        else:
            pf['localDyn']='centre'; pf['marker']='o'; pf['mfc']= 'cyan'; pf['ms'] = 1.5*ms
    else: 
        pf['type']='node'; pf['mec']=color_nodo; pf['mfc'] =color_nodo; 
        if (r1*r2) < 0:
            pf['localDyn'] ='saddle'; pf['marker']='x'; pf['ms'] = 2*ms
        elif (r1*r2>0):
            if r1>0: pf['localDyn']='repeller'; pf['marker']='o'; pf['mfc'] ='white' 
            if r1<0: pf['localDyn']='attractor'; pf['marker']='o' 
        elif (r1*r2 == 0) | (r1*r2<1e-15): 
            pf['localDyn']='degen'; pf['marker']='o'; pf['mfc']= 'lightgray'; pf['mec']= 'cyan'; pf['ms']=1.5*ms
    return pf

def findBifurcationsCod1_FPL(fpsL, parName='a_F'):
    lostAttract_v = list(); gainAttract_v = list()
    lostAttract_par =list(); gainAttract_par = list()
    bifurcation_v = list(); bifurcation_par = list(); bifurcation_change = list()
    # Fixed points are stored in the order given by the v-values. 
    lastDyn=fpsL['fpTypes'][0]['localDyn'] 
    lastType = fpsL['fpTypes'][0]['type']
    for m in range(1,fpsL['nFPs']):
        uStar = fpsL['fps'][m][0]
        par = fpsL[parName][m] 
        dyn = fpsL['fpTypes'][m]['localDyn']
        ty = fpsL['fpTypes'][m]['type']
        if (lastDyn == dyn) & (lastType == ty): continue
        else: 
            change = '%s %s -> %s %s'%(lastDyn,lastType, dyn, ty)
            vLoc = 'v = %g mV'% vStar
            print('\n FP changed from %s at %s \n '%(change,vLoc))
            bifurcation_v.append(vStar)
            bifurcation_par.append(par)
            bifurcation_change.append(change)
            if (lastDyn != dyn): # change in attractivity
                if (lastDyn == 'attractor') & (dyn != 'attractor' ):
                    print('\n ... FP lost attractivity \n ')
                    lostAttract_v.append(vStar)
                    lostAttract_par.append(fpsL[parName][m])                    
                elif (lastDyn != 'attractor') & (dyn == 'attractor' ): 
                    print('\n ... FP became an attractor \n ')
                    gainAttract_v.append(vStar)
                    gainAttract_par.append(fpsL[parName][m])
            elif (lastType != ty): # change in type
                print('\n ... FP type changed')
        lastDyn = dyn
        lastType = ty
    print('Found %d bifurcations involving the type or stability of the fixed points'%len(bifurcation_v))
    return {'vLoc':bifurcation_v, parName : bifurcation_par, 'change':bifurcation_change, \
            'lostAttract_v':lostAttract_v, 'lostAttract_par':lostAttract_par, 'gainAttract_v':gainAttract_v, 'gainAttract_par':gainAttract_par}


# ----------------------
# Symbolic 2D system 
# ----------------------
class system2D: 
    
    def __init__(self, params, variables=('x','y')):
        self.vars = variables
        self.pars = params.copy();
        self.pars_orig = params.copy()
        self.dict2symbols(params)
        self.createStateVars(variables)
        self.pars['stepSize'] = self.pars['timeStep']
        #self.defineFunctions()
        #self.updateFunctions()
        return
    
    def createStateVars(self,variables):
        print("Setting %s and %s as state variables"%(variables[0],variables[1]))
        str1 = "self.%s, self.%s = sy.symbols(%s)"%(variables[0],variables[1],variables)
        print(str1)
        return exec(str1)
    
    def resetParameters(self):
        print('Resetting dictionary to the original', self.pars_orig); self.pars = self.pars_orig.copy(); 
        return 
        
    def dict2symbols(self,di):
        for k in di.keys():
            exec("self.%s = sy.Symbol('%s')"%(k,k))
        return
    
    def nuFu_expr(self, expression, variables):
        str1 = "self.%s_ = sy.lambdify(%s, self.%s.subs(self.pars), 'numpy')"%(expression, variables, expression)
        #print(str1)
        return exec(str1)
    
    def trayectory_Autonomous(self,f):
        self.timeSamples = np.arange(self.pars['timeMin'], self.pars['timeMax'], self.pars['timeStep'])
        self.nSteps = len(self.timeSamples)            
        U = np.zeros((self.nSteps, np.prod(np.shape(self.pars['ic']))),"float64")
        U[0] = self.pars['ic']
        for i in range(self.nSteps-1):
            U[i+1] = RK2_autonomousStep(f, U = U[i], stepSize = self.pars['timeStep'])
        return U.transpose()
    
    def orbitsFromICs(self, ics):
        nIcs = len(ics)
        orbits = list()
        for n in range(nIcs):
            self.pars['ic'] = ics[n]
            orbits.append(self.getDynamics())
        return orbits
            
    def subsJacobian(self):
        self.sysJacobian =  self.evolution.subs(self.pars).jacobian(self.vars)
        return 
        
    def eigvaluesFromFP(self,fp):
        self.subsJacobian()
        self.nuFu_expr(expression= 'sysJacobian', variables=self.vars)
        return la.eigvals(self.sysJacobian_(*fp))
    
    def parameterFromFP(self, fp, parName):
        xStar,yStar = fp; #print('Fixed point: ',fp)
        pp = self.pars.copy(); pp.pop(parName); 
        fpExpr = self.fpExpr.subs(pp).subs( [(self.vars[0],xStar),(self.vars[1],yStar)])
        #print(fpExpr)
        return np.float64(sy.solve(fpExpr, parName)[0])
    
    def fpType(self,eigenValues, ms=5, nodeColor='orange',focusColor='blue'):
        return fixedPoint2D_typeMarker(eigenValues, ms, nodeColor, focusColor)

    def cod1FPTypes(self, fps, parName, ms=5, nodeColor='blue', focusColor='gray'):
        """Inputs:
        fps ~ list of fixed points (each entry is an ordered pair, tuple, or array)
        """
        nfp = len(fps); print('Processing %d fixed points'%nfp); 
        tStart = time.process_time()
        pp = self.pars.copy(); pp.pop(parName); cod1 = list();
        parValues = list(); evs = list(); fpTypes=list()
        for n in range(nfp): 
            parValues.append(self.parameterFromFP(fps[n],parName))
            self.pars[parName] = parValues[n]
            self.subsJacobian()
            evs.append(self.eigvaluesFromFP(fps[n]))
            #print('%s = %g'%( parName, parValues[n]))
            fpTypes.append(self.fpType(evs[n], ms, nodeColor, focusColor))
        cod1={ 'fps':fps, parName: parValues, 'evs':evs, 'fpTypes':fpTypes, 'nFPs':nfp, 'parameter':parName}
        #print(cod1)
        print('Took %d seconds to calculate the fixed point list for %s'%(time.process_time()-tStart, parName))
        return cod1
    
    def cod1SecondParameterVariation(self, secParName, secParVals, fps, mainParName = 'a_F'):
        self.updateFunctions()
        #
        nLists = len(secParVals)
        fpsList = list()
        for n in range(nLists):
            self.pars[secParName] = secParVals[n]
            fpsList.append (self.cod1FPTypes(fps, parName=mainParName, ms=5, nodeColor='blue', focusColor='gray'))
        return fpsList

    def bifurcationDiagram_Cod1(self, ax, cod1, coordinate=0, fpScaleFactor=1, parScaleFactor=1, xLabel='', yLabel=r'$v_{*}$ (mV)'):
        for n in range(cod1['nFPs']):
              fpt= cod1['fpTypes'][n]
              ax.plot(cod1[cod1['parameter']][n]*parScaleFactor, cod1['fps'][n][coordinate]*fpScaleFactor, fpt['marker'], \
                    markerfacecolor=fpt['mfc'], markeredgecolor=fpt['mec'])
        ax.set_xlabel(cod1['parameter']); 
        ax.set_ylabel(yLabel)
        ax.set_xlabel(xLabel)
        return ax

    def bifurcCod1_secondParameterVariation(self, axList, fpsTypeList, secParName, \
                fpScaleFactor, parScaleFactor, coordinate=0, xLabel='', yLabel=''):
        self.updateFunctions()
        #
        nPanels = len(axList)
        nFPList = len(fpsTypeList)
        if nPanels==nFPList: 
            print('Found %d lists of fixed points with their types'%nPanels)
            for n in range(nPanels):
                axList[n]=self.bifurcationDiagram_Cod1(axList[n], cod1= fpsTypeList[n], coordinate=coordinate, \
                    fpScaleFactor=fpScaleFactor, parScaleFactor=parScaleFactor, xLabel='', yLabel=yLabel) 
        else: 
            for n in range(nFPList):
                axList[0]=self.bifurcationDiagram_Cod1(axList[0], cod1= fpsTypeList[n], coordinate=coordinate, \
                    fpScaleFactor=fpScaleFactor, parScaleFactor=parScaleFactor, xLabel='', yLabel=yLabel) 

        axList[-1].set_xlabel(xLabel)
        return axList

class UD(system2D):
    def __init__(self, params,variables):
        super().__init__(params,variables)
        self.defineFunctions()
        #self.updateFunctions()
        return
    
    def defineFunctions(self):
        print("Defining all functions. Notice v is u normalized by v_T")
        self.e_m = biasedExp( x=self.u, a=self.u_m, b=self.g_m)
        self.e_w = biasedExp( x=self.u, a=self.u_w, b=self.g_w)
        self.e_U = biasedExp( x=self.u, a=self.u_U, b=1)
        self.e_D = biasedExp( x=self.u, a=self.u_D, b=1)
        self.e_UD = biasedExp( x=self.u, a=self.u_UD, b=1)
        self.alpha_w = self.r_w * (self.e_w ** self.b_w)
        self.beta_w = self.r_w * (self.e_w ** (self.b_w-1))
        self.alpha_U = self.r_U * (self.e_U ** self.b_U)
        self.beta_U = self.r_U * (self.e_U ** (self.b_U-1))
        self.alpha_D = self.r_D * (self.e_D ** self.b_D)
        self.beta_D = self.r_D * (self.e_D ** (self.b_D-1))
        self.alpha_UD = self.r_UD * (self.e_UD ** self.b_UD)
        self.beta_UD = self.r_UD * (self.e_UD ** (self.b_UD-1))
        self.tau_w = 1/ (self.alpha_w + self.beta_w)
        self.m_inf = self.e_m / ( 1 + self.e_m)
        self.w_inf =  self.e_w / (1 + self.e_w)
        self.u_w_inf = (self.u_w - sy.log(1/self.w - 1))/self.g_w
        self.J_U = self.a_U * self.m_inf * (1-self.w) * (self.alpha_U - self.beta_U)
        self.J_D = self.a_D * self.w * (self.alpha_D - self.beta_D)
        self.J_UD = self.a_UD * (self.alpha_UD - self.beta_UD)
        self.du = self.a_F - self.J_U - self.J_D - self.J_UD
        self.a_F_inf =  (self.J_U + self.J_D + self.J_UD).subs({'w':self.w_inf})
        self.dw = (self.w**self.kappa_w) * (self.alpha_w - (self.alpha_w + self.beta_w)*self.w )
        self.w_vNull = (self.a_F - self.J_UD - self.a_U * self.m_inf * (self.alpha_U - self.beta_U)) \
        / (self.a_D * (self.alpha_D - self.beta_D) - self.a_U * self.m_inf * (self.alpha_U - self.beta_U))
        self.evolution = sy.Matrix([self.du, self.dw])
        self.N_U = self.pars['vTCm'] * self.a_U / self.r_U
        self.N_D = self.pars['vTCm'] * self.a_D / self.r_D
        self.N_UD = self.pars['vTCm'] * self.a_UD / self.r_UD
        #self.fpExpr = self.du-self.dw
        self.fpExpr = self.du.subs({'w':self.w_inf})
        return
    
    def updateFunctions(self):
        self.nuFu_expr(expression='dw',variables=['u','w'])
        self.nuFu_expr(expression='du',variables=['u','w'])
        self.nuFu_expr(expression='m_inf',variables=['u'])
        self.nuFu_expr(expression='w_inf',variables=['u'])
        self.nuFu_expr(expression='w_vNull',variables=['u'])
        self.nuFu_expr(expression='J_UD',variables=['u'])
        self.nuFu_expr(expression='J_U',variables=['u','w'])
        self.nuFu_expr(expression='J_D',variables=['u','w'])
        self.nuFu_expr(expression='fpExpr',variables=['u'])
        self.nuFu_expr(expression='a_F_inf',variables=['u'])
        return 
    
    def plot_a_F_Inf(self,ax,W = np.linspace(0,1,50), V = np.linspace(-60,30,250), aFMin=-10, aFMax = 500, aFLabel=r'$a_{F\infty}$'):
        U = V/self.pars['v_T']
        aF = self.a_F_inf(U)
        ax.plot(U,aF,label= aFLabel)
        ax.set_xlim(V.min(),V.max())
        ax.set_ylim(aFMin,aFMax)
        ax.set_xlabel(r'$a_{F \infty}$')
        ax.set_ylabel(r'$v$ (mV)')
        return ax,aF
    
    def phasePlane(self, ax, W = np.linspace(0,1,50), V = np.linspace(-60,30,250), wNullLabel='', vNullLabel='',plotNullClines=1):
        U = V/self.pars['v_T']
        self.nuFu_expr(expression='dw',variables=['u','w'])
        self.nuFu_expr(expression='du',variables=['u','w'])
        self.nuFu_expr(expression='w_inf',variables=['u'])
        self.nuFu_expr(expression='w_vNull',variables=['u'])
        Wgrid,Ugrid = np.meshgrid(W,U)
        dU = self.du_(Ugrid, Wgrid); 
        dW = self.dw_(Ugrid, Wgrid); 
        speed = np.sqrt(dU**2 + dW**2)
        lw = 5*speed / speed.max()
        ax.streamplot(W, U*self.pars['v_T'], dW, dU*self.pars['v_T'], density=0.8, color = 'gray', linewidth=lw)
        if plotNullClines>0:
            w_wNull = self.w_inf_(U); w_vNull = self.w_vNull_(U)
            ax.plot(w_vNull, U*self.pars['v_T'], '-', lw=2,   color = 'green', alpha=0.75, label=vNullLabel);
            ax.plot(w_wNull, U*self.pars['v_T'], '-', lw=2, color = 'orange', alpha=0.75, label=wNullLabel)
        ax.set_xlim(W.min(),W.max())
        ax.set_ylim(U.min()*self.pars['v_T'],U.max()*self.pars['v_T'])
        #ax.legend(loc='lower right')
        return ax
       
    def duw_(self,Z):
        return np.array([self.du_(*Z),self.dw_(*Z)])

    def trayectory_nonAutonomous(self, parNames=[], parVals=[]):
        '''
        parNames and parVals must be lists of the same length. 
        Each element in parVals must have the same length as the timeSample vector
        '''     
        self.timeSamples = np.arange(self.pars['timeMin'],self.pars['timeMax'],self.pars['timeStep'])
        self.nSteps = len(self.timeSamples)            
        nPars = len(parNames)

        diNP = self.pars.copy()
        for n in range(nPars):
            diNP = delDictItem(diNP,parNames[n])

        self.duPars_ = sy.lambdify(self.vars+parNames, self.du.subs(diNP), 'numpy')
        self.dwPars_ = sy.lambdify(self.vars+parNames, self.dw.subs(diNP), 'numpy')
        U = np.zeros((self.nSteps, np.prod(np.shape(self.pars['ic']))),"float64")
        U[0]=self.pars['ic'] 
        fPars = lambda Z : np.array([self.duPars_( *Z), self.dwPars_( *Z)])
        for nn in range(self.nSteps-1):
            pars = np.array([parVals[n][nn] for n in range(nPars)])
            UPars = np.hstack( [U[nn], pars])
            k = self.pars['timeStep'] * fPars(UPars) / 2
            kPars = np.hstack( [k, np.zeros(nPars)])
            U[nn+1] = U[nn] + self.pars['timeStep'] * fPars( UPars + kPars)
        #self.updateFunctions()
        return U.transpose()  
                   
    def iClampSquareStims(self, iLevels, timeStimStart, timeStimStop):
        nLevels = len(iLevels)
        iAmps = list()
        self.timeSamples = np.arange(self.pars['timeMin'],self.pars['timeMax'],self.pars['timeStep'])
        self.nSteps = len(self.timeSamples)
        a = np.int64(np.ceil( (timeStimStart-self.pars['timeMin'])/self.pars['timeStep']))
        b = np.int64(np.floor( (timeStimStop-self.pars['timeMin'])/self.pars['timeStep']))
        for n in range(nLevels):
            iAmps.append(np.zeros(self.nSteps))
            iAmps[n][a:b] = iLevels[n]
        return iAmps

    def iClamp(self, iList):
        nCommands = len(iList)
        vOrbits = list()  #; wOrbits = list(); 
        stimValue = list()
        for n in range(nCommands):
            vOrbit, wOrbit = self.trayectory_nonAutonomous(parNames=['a_F'], parVals=[iList[n]])
            vOrbits.append(vOrbit); #wOrbits.append(wOrbit); 
        return vOrbits #, wOrbits

    def steadyStateFromIC(self, timeMax = 1000):
        self.pars['timeMax'] = timeMax
        self.updateFunctions() 
        u,w = upDn.trayectory_Autonomous(upDn.duw_)
        self.pars['ic'] = np.array([u[-1],w[-1]])
        print('Found steady state near (%g,%g)'%(u[-1]*self.pars['v_T'],w[-1]))
        return upDn.pars['ic']

    def getDynamics(self, parNames=[], parVals=[]):
        self.updateFunctions()
        if len(parNames)==0: 
            self.uOrbit, self.wOrbit = self.trayectory_Autonomous(self.duw_)
        else: 
            self.uOrbit, self.wOrbit = self.trayectory_nonAutonomous(parNames, parVals)
        self.vOrbit = self.uOrbit * self.pars['v_T']
        self.dvdt = np.zeros(len(self.vOrbit))
        self.dvdt[1:] = (self.vOrbit[1:]-self.vOrbit[:-1])/self.pars['timeStep']
        print('Max dv/dt = %g V/s'% self.dvdt.max())
        self.v_w_inf = self.pars['v_T'] * (self.pars['u_w'] - np.log(1/self.wOrbit -1)/self.pars['g_w'])
        self.I_U = self.pars['vTCm'] * self.J_U_(self.uOrbit,self.wOrbit)  
        self.I_D = self.pars['vTCm'] * self.J_D_(self.uOrbit,self.wOrbit) 
        self.I_UD = self.pars['vTCm'] * self.J_UD_(self.uOrbit)
        return {'wOrbit':self.wOrbit, 'vOrbit':self.vOrbit, 'dvdt':self.dvdt, 'v_w_inf':self.v_w_inf, \
                'I_U': self.I_U, 'I_D':self.I_D, 'I_UD':self.I_UD, 'timeSamples':self.timeSamples}

    def plotDynamicProfile(self,ax, vMin = -80, vMax = 40, iMin=-20, iMax=70, wMin=0, wMax=1): 
        #
        print(type(self.u),type(self.w))
        ax[0].plot(self.timeSamples, self.vOrbit, color='black', label=r'$(t,v)$')
        ax[0].plot(self.timeSamples, self.pars['v_T']*self.v_w_inf, '--', color='black', label=r'$(t,w_{\inf}^{-1}(w))$')
        ax[0].set_xlabel(r'time (ms)'); ax[0].set_ylabel(r'mV')
        ax[0].set_ylim(vMin,vMax); ax[0].legend()
        ax[1].plot(self.dvdt, self.uOrbit, color='black', label=r'$(\partial_t v, v)$')
        ax[1].plot(self.dvdt, self.pars['v_T']*self.v_w_inf, '--', color='black', label=r'$(\partial_t v, w_{\inf}^{-1}(w))$')
        ax[2].plot(self.wOrbit, self.uOrbit, color='black', label=r'$(w, v)$')
        #ax[2].plot(self.wOrbit, self.pars['v_T']*self.u_w_inf, '--', color='black', label=r'$(w, w_{\inf}^{-1}(w))$')
        self.phasePlane(ax[2], V=np.linspace(vMin,vMax,300)/v_T, W=np.linspace(wMin,wMax,200))
        for nn in range(3): ax[nn].set_ylim(vMin,vMax);
        ax[3].plot(self.timeSamples, self.I_U, color='green', label=r'$I_{U}$')
        ax[3].plot(self.timeSamples, self.I_D, color='orange', label=r'$I_{D}$')
        ax[3].set_xlabel(r'time (ms)'); ax[3].set_ylabel(r'pA');
        ax[6].plot(self.timeSamples, self.I_UD, color='blue', label=r'$I_{UD}$')
        ax[6].set_xlabel(r'time (ms)'); #ax[6].set_ylabel(r'pA'); 
        ax[6].set_ylim(iMin,iMax); #ax[6].set_ylabel(r'mV'); 
        ax[7].plot(self.dvdt, self.I_UD, color='blue', label=r'$I_{UD}$')
        ax[4].plot(self.dvdt, self.I_U, color='green', label=r'$I_{U}$')
        ax[4].plot(self.dvdt, self.I_D, color='orange', label=r'$I_{D}$')
        ax[8].plot(self.wOrbit, self.I_UD, color='blue', label=r'$(w,I_{UD})$')
        ax[5].plot(self.wOrbit, self.I_U, color='green', label=r'$(w,I_{U})$')
        ax[5].plot(self.wOrbit, self.I_D, color='orange', label=r'$(w,I_{UD})$')
        ax[7].set_xlabel(r'$\partial_t v(t)$ (V/s)'); 
        ax[7].set_ylabel(r'pA'); ax[7].set_ylim(iMin,iMax);
        for nn in range(6,9): ax[nn].set_ylim(iMin,iMax);
        #pl.ion(); pl.draw()
        for nn in range(9):  ax[nn].legend()
        return ax



