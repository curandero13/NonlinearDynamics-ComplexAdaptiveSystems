from symbolicSystem_2D import *

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
        self.dw = (self.w**self.kappa_w) * (self.alpha_w - (self.alpha_w + self.beta_w)*self.w )
        self.w_vNull = (self.a_F - self.J_UD - self.a_U * self.m_inf * (self.alpha_U - self.beta_U)) \
        / (self.a_D * (self.alpha_D - self.beta_D) - self.a_U * self.m_inf * (self.alpha_U - self.beta_U))
        self.evolution = sy.Matrix([self.dw,self.du])
        self.ssEquation = self.du.subs({'w': self.w_inf})
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
        self.nuFu_expr(expression='ssEquation',variables=['u'])
        return 
    
    def phasePlane(self, ax, W = np.linspace(0,1,50), U = np.linspace(-60,30,200)/26.64, wNullLabel='', vNullLabel='',plotNullClines=1):
        self.nuFu_expr(expression='dw',variables=['u','w'])
        self.nuFu_expr(expression='du',variables=['u','w'])
        self.nuFu_expr(expression='w_inf',variables=['u'])
        self.nuFu_expr(expression='w_vNull',variables=['u'])
        Wgrid,Ugrid = np.meshgrid(W,U)
        dU = self.du_(Ugrid, Wgrid); 
        dW = self.dw_(Ugrid, Wgrid); 
        ax.streamplot(W, U*self.pars['v_T'], dW, dU*self.pars['v_T'], color = 'gray', linewidth=1)
        if plotNullClines>0:
            w_wNull = self.w_inf_(U); w_vNull = self.w_vNull_(U)
            ax.plot(w_vNull, U*self.pars['v_T'], lw=2,   color = 'green', alpha=0.75);
            ax.plot(w_wNull, U*self.pars['v_T'], lw=2, color = 'orange', alpha=0.75)
        ax.set_xlim(W.min(),W.max())
        ax.set_ylim(U.min()*self.pars['v_T'],U.max()*self.pars['v_T'])
        return ax
    
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
    
        self.dwPars_ = sy.lambdify(self.vars + parNames, self.dw.subs(diNP), 'numpy')
        self.duPars_ = sy.lambdify(self.vars + parNames, self.du.subs(diNP), 'numpy')
        fPars = lambda Zpars : np.array([self.duPars_(*Zpars), self.dwPars_(*Zpars)])
        Z = np.zeros((self.nSteps, np.prod(np.shape(self.pars['ic']))),"float64")
        Z[0]=self.pars['ic'] 
        for nn in range(self.nSteps-1):
            pars = np.array([parVals[n][nn] for n in range(nPars)])
            ZPars = np.hstack( [Z[nn], pars])
            k = self.pars['timeStep'] * fPars(ZPars) / 2
            kPars = np.hstack( [k, pars])
            Z[nn+1] = Z[nn] + self.pars['timeStep'] * fPars( ZPars + kPars)
        self.updateFunctions()
        return U.transpose()            
            
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
        return {'wOrbit':self.wOrbit, 'vOrbit':self.vOrbit, 'dvdt':self.dvdt, 'v_w_inf':self.v_w_inf, 'I_U': self.I_U, 'I_D':self.I_D, 'I_UD':self.I_UD,\
            'timeSamples':self.timeSamples}
    
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


