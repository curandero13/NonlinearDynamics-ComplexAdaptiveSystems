import numpy as np
import numpy.linalg as la
from numpy.random import uniform
import matplotlib.pylab as pl
import sympy as sy
from sympy.plotting import plot, PlotGrid, plot3d
from sympy.utilities.iterables import flatten
import time

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

def secant_method(f, x0, x1, tol=1e-5, n=0):
    n += 1 # increment counter
    y0, y1 = f(x0), f(x1) # calculate function values at endpoints
    xn = x1 - y1 * ((x1 - x0) / (y1 - y0)) # calculate next root approximation
    if -tol < y1 < tol: # check tolerance condition
        return xn, n
    # recursive call with updated interval
    return secant_method(f, x1, xn, n=n)

# ----------------------
# Numerics
# ----------------------
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
        pf['type']='focus';pf['mec']=color_foco;
        if r1>0: 
            pf['localDyn']='repeller'; pf['marker']='o'; pf['mfc'] ='white'; 
        elif r1<0:
            pf['localDyn']='attractor'; pf['marker']='o'; pf['mfc'] =color_foco
        else:
            pf['localDyn']='centre'; pf['marker']='o'; pf['mfc']= 'gray'; pf['mec']= 'cyan'; pf['ms']=1.5*ms
    else: 
        pf['type']='node'; pf['mec']=color_nodo; 
        if (r1*r2) < 0:
            pf['localDyn'] ='saddle'; pf['marker']='x'; pf['mfc']=color_nodo; pf['ms'] = 2*ms
        elif (r1*r2>0):
            if r1>0: pf['localDyn']='repeller'; pf['marker']='o'; pf['mfc'] ='white' 
            if r1<0: pf['localDyn']='attractor'; pf['marker']='o'; pf['mfc'] =color_nodo 
        elif (r1*r2 == 0) | (r1*r2<1e-15): 
            pf['localDyn']='degen'; pf['marker']='o'; pf['mfc']= 'lightgray'; pf['mec']= 'cyan'; pf['ms']=1.5*ms
    return pf

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
    
    def subsJacobian(self):
        self.sysJacobian =  self.evolution.subs(self.pars).jacobian(self.vars)
        return 
        
    def eigvaluesFromFP(self,fp):
        self.subsJacobian()
        self.nuFu_expr(expression= 'sysJacobian', variables=self.vars)
        return la.eigvals(self.sysJacobian_(*fp))
    
    def parameterFromFP(self, fp, parName):
        vStar,wStar = fp; #print('Fixed point: ',fp)
        pp = self.pars.copy(); pp.pop(parName); 
        #print(self.J_inf.subs(pp).subs({'w':wStar, 'v':vStar}))
        fpEq = sy.Eq( self.ssEquation.subs(pp).subs( [(self.vars[0],vStar),(self.vars[1],wStar)], 0))
        return np.float64(sy.solve(fpEq, parName)[0])
    
    def fpType(self,eigenValues, ms=5, nodeColor='orange',focusColor='blue'):
        return fixedPoint2D_typeMarker(eigenValues, ms, nodeColor, focusColor)

    def cod1FPs(self, fps, parName, ms=5, nodeColor='blue', focusColor='gray'):
        """Inputs:
        fps ~ list of fixed points (each entry is an ordered pair, tuple, or array)
        """
        nfp = len(fps)
        tStart = time.process_time()
        pp = dict(p); pp.pop(parName); cod1 = list();
        fps = list(); parValues = list(); evs = list(); fpTypes=list()
        for n in range(nfp): 
            parValues.append(self.parameterFromFP(fps[n],parName))
            self.pars[parName] = parValues[n]
            self.nuFu_Jacobian_subsPars()
            evs.append(self.eigvaluesFromFP(fps[n]))
            #print('(parameter, ev)=(%g,%g)'%(pv,ev))
            fpTypes.append(self.fpType(evs[n], ms, nodeColor, focusColor))
        cod1={ 'fps':fps, 'parName': parValues, 'evs':evs, 'fpTypes':fpTypes, 'nFPs':nfp}
        print('Took %d seconds to calculate the fixed point list for %s'%(time.process_time()-tStart, parName))
        return cod1
    
    def bifurcationDiagram_Cod1(self, ax, fps, coordinate, parName, nodeColor='gray', focusColor='gray', yLabel=''):
        cod1 = self.cod1FPs(fps, parName = parName, ms=3, color_nodo=nodeColor,color_foco=focusColor)
        for n in range(cod1['nFPs']):
              fpt= cod1['fpTypes'][n]
              ax.plot(cod1['parName'][n], cod1['fps'][n][coordinate],fpt['marker'], \
                    markerfacecolor=fpt['mfc'], markeredgecolor=fpt['mec'])
        ax.set_xlabel(parName); 
        ax.set_ylabel(yLabel)
        self.pars = self.pars_orig
        return cod1  

