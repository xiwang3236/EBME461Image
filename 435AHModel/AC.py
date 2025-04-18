import numpy as np, math, matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ------------------------------------------------------
# 1. Constants, geometry, channels
# ------------------------------------------------------
R, T, Ffar = 8.314, 310.0, 96485.0

A_sp, A_tj, A_PC = 6e-4, 6e-7, 2.6e-5     # m²
L, H             = 1e-5, 6e-3             # m
W_c, W_p         = A_sp*L, A_PC*H         # m³
Ks, Kp, sigma    = 2e-11, 2e-10, 1.0

Pi = dict(PUMP=6e-6, NKCC=1e-6, AEs=4e-6, AEp=0.4e-6,
          NBCs=6e-6, NBCp=1e-6, NHE=3.4e-6,
          K_s=5e-8, K_p=3e-7, Cl_p=6e-8, TJ=6e-6)

k1, k_m1 = 5.3, 1.0                  # H2CO3 ⇌ HCO3+H   (1/mM)
kd, kh   = 4.96e5, 1.45e3            # hydration / dehydration (1/s)

z  = {'Na':1,'K':1,'Cl':-1,'HCO3':-1,'H':1,'CO2':0,'H2CO3':0}
zX = -1.5

# ------------------------------------------------------
# 2. Flux helpers
# ------------------------------------------------------
llog = lambda x: math.log(max(x, 1e-12))

def ghk_flux(A,P,z_i,Vm,Vk,Cm,Ck):
    phi = Ffar*(Vm-Vk)/(R*T)
    if abs(phi) < 1e-8:
        return A*P*(Cm-Ck)
    phi = max(min(phi,50),-50)
    den = 1.0-math.exp(-z_i*phi)
    den = math.copysign(max(abs(den),1e-12),den)
    num = z_i*phi*Cm - Ck*math.exp(-z_i*phi)
    return A*P*num/den

def nkcc_flux(A,P,Na_c,K_c,Cl_c,Na_s,K_s,Cl_s):
    return A*P*llog((K_c*Na_c*Cl_c**2)/(K_s*Na_s*Cl_s**2))

def nbc_flux(A,P,Na_c,Hc,Na_x,Hx,Vc,Vx):
    phi = Ffar*(Vc-Vx)/(R*T)
    phi = max(min(phi,50),-50)
    return A*P*(llog((Na_c*Hc**2)/(Na_x*Hx**2)) - phi)

def ae_flux(A,P,Cl_c,Hx,Cl_x,Hc):
    return A*P*llog((Cl_c*Hx)/(Cl_x*Hc))

def nhe_flux(A,P,Hs,Na_c,Hc,Na_s):
    return A*P*llog((Hs*Na_c)/(Hc*Na_s))

def pump_flux(A,P,Na_c,K_p):
    KNa = 0.2*(1+Na_c/8.33)
    KK  = 0.1*(1+K_p /18.5)
    return A*P*(Na_c/(KNa+Na_c))**3 * (K_p/(KK+K_p))**2

# ------------------------------------------------------
# 3. Residuals (18 algebraic equations)
# ------------------------------------------------------
def residuals(x, kd_local, kh_local):
    Cc, Cp = x[:7], x[7:14]
    Cx, Q  = x[14], x[15]
    Vc, Vp = x[16], x[17]
    Cs = np.array([150,5,130,25,10**(-7.42), 0.03, 0.005])
    Na,K_,Cl,HCO3,H_,CO2,H2 = range(7)

    ghk = lambda A,P,z_,Vm,Vk,Cm,Ck: ghk_flux(A,P,z_,Vm,Vk,Cm,Ck)
    nk  = lambda *a: nkcc_flux(A_sp,Pi['NKCC'],*a)
    nbc = lambda tag,*a: nbc_flux(A_sp,Pi[tag],*a)
    ae  = lambda tag,*a: ae_flux (A_sp,Pi[tag],*a)
    nhe = lambda *a: nhe_flux(A_sp,Pi['NHE'],*a)

    # ---- Na⁺ ----
    J_sc_Na = nk(Cc[Na],Cc[K_],Cc[Cl], Cs[Na],Cs[K_],Cs[Cl]) \
            + nbc('NBCs', Cc[Na],Cc[HCO3], Cs[Na],Cs[HCO3], Vc,0) \
            - nhe(Cs[H_], Cc[Na], Cc[H_], Cs[Na])

    J_cp_Na = 3*pump_flux(A_sp,Pi['PUMP'], Cc[Na],Cp[K_]) \
            + nbc('NBCp', Cc[Na],Cc[HCO3], Cp[Na],Cp[HCO3], Vc,Vp)

    J_sp_Na = ghk(A_tj,Pi['TJ'], z['Na'], 0,Vp, Cs[Na],Cp[Na])

    # ---- K⁺ ----
    J_sc_K = nk(Cc[Na],Cc[K_],Cc[Cl], Cs[Na],Cs[K_],Cs[Cl]) \
           - ghk(A_sp,Pi['K_s'], z['K'], Vc,0, Cc[K_],Cs[K_])

    J_cp_K = -2*pump_flux(A_sp,Pi['PUMP'], Cc[Na],Cp[K_]) \
           + ghk(A_sp,Pi['K_p'], z['K'], Vc,Vp, Cc[K_],Cp[K_])

    J_sp_K = ghk(A_tj,Pi['TJ'], z['K'], 0,Vp, Cs[K_],Cp[K_])

    # ---- Cl⁻ ----
    J_AEs = ae('AEs', Cc[Cl], Cs[HCO3], Cs[Cl], Cc[HCO3])
    J_AEp = ae('AEp', Cc[Cl], Cp[HCO3], Cp[Cl], Cc[HCO3])

    J_sc_Cl = 2*nk(Cc[Na],Cc[K_],Cc[Cl], Cs[Na],Cs[K_],Cs[Cl]) - J_AEs
    J_cp_Cl = J_AEp + ghk(A_sp,Pi['Cl_p'], z['Cl'], Vc,Vp, Cc[Cl],Cp[Cl])
    J_sp_Cl = ghk(A_tj,Pi['TJ'], z['Cl'], 0,Vp, Cs[Cl],Cp[Cl])

    # ---- HCO₃⁻ ----
    J_sc_HCO3 = 2*nbc('NBCs', Cc[Na],Cc[HCO3], Cs[Na],Cs[HCO3], Vc,0) + J_AEs
    J_cp_HCO3 = 2*nbc('NBCp', Cc[Na],Cc[HCO3], Cp[Na],Cp[HCO3], Vc,Vp) - J_AEp
    J_sp_HCO3 = ghk(A_tj,Pi['TJ'], z['HCO3'], 0,Vp, Cs[HCO3],Cp[HCO3])

    # ---- CO₂ & H₂CO₃ ----
    P_CO2_s, P_CO2_p = 1.5e-3, 1.5e-2
    P_H2_s , P_H2_p  = 1.28e-5,1.28e-4
    J_sc_CO2 = P_CO2_s*A_sp*(Cs[CO2]-Cc[CO2])
    J_cp_CO2 = P_CO2_p*A_sp*(Cc[CO2]-Cp[CO2])
    J_sp_CO2 = P_CO2_s*A_tj*(Cs[CO2]-Cp[CO2])
    J_sc_H2  = P_H2_s *A_sp*(Cs[H2] -Cc[H2])
    J_cp_H2  = P_H2_p *A_sp*(Cc[H2]-Cp[H2])
    J_sp_H2  = P_H2_s *A_tj*(Cs[H2]-Cp[H2])

    # reactions
    Rc4 = (k_m1*Cc[CO2]-k1*Cc[HCO3]*Cc[H_])*W_c
    Rp4 = (k_m1*Cp[CO2]-k1*Cp[HCO3]*Cp[H_])*W_p
    Rc6 = (kd_local*Cc[H2]-kh_local*Cc[CO2])*W_c
    Rp6 = (kd_local*Cp[H2]-kh_local*Cp[CO2])*W_p
    Rc7, Rp7 = -Rc6, -Rp6

    # water fluxes
    Q_sc = -A_sp*Ks*sigma*R*T*(np.sum(Cs) - np.sum(Cc) - Cx)
    Q_cp = -A_sp*Kp*sigma*R*T*(np.sum(Cc) - np.sum(Cp))
    Q_sp = -A_tj*Ks*sigma*R*T*(np.sum(Cs) - np.sum(Cp))

    F = np.zeros(18)
    F[:6]  = [J_sc_Na-J_cp_Na,
              J_cp_Na+J_sp_Na-Q*Cp[Na],
              J_sc_K-J_cp_K,
              J_cp_K+J_sp_K-Q*Cp[K_],
              J_sc_Cl-J_cp_Cl,
              J_cp_Cl+J_sp_Cl-Q*Cp[Cl]]
    F[6:12]= [J_sc_HCO3-J_cp_HCO3+Rc4,
              J_cp_HCO3+J_sp_HCO3-Q*Cp[HCO3]+Rp4,
              J_sc_CO2-J_cp_CO2+Rc6,
              J_cp_CO2+J_sp_CO2-Q*Cp[CO2]+Rp6,
              J_sc_H2-J_cp_H2+Rc7,
              J_cp_H2+J_sp_H2-Q*Cp[H2]+Rp7]
    F[12] = sum(z[k]*Cc[i] for i,k in enumerate(z)) + zX*Cx
    F[13] = sum(z[k]*Cp[i] for i,k in enumerate(z))
    F[14],F[15] = Q_sc-Q_cp, Q_cp+Q_sp-Q
    F[16] = Cc[H2]-k1/k_m1*Cc[HCO3]*Cc[H_]
    F[17] = Cp[H2]-k1/k_m1*Cp[HCO3]*Cp[H_]
    return F

# ------------------------------------------------------
# 4. Initial guess and solve
# ------------------------------------------------------
Cc0 = [17.8,154,45,26.9,3.5e-8,0.03,0.005]
Cp0 = [151.8,4.3,127,28.6,3.2e-8,0.03,0.005]
charge = sum(z[k]*Cc0[i] for i,k in enumerate(z))
CX0    = -charge/zX
init = np.array(Cc0+Cp0+[CX0,3e-11,-0.075,-0.001])

sol_CA = fsolve(lambda x: residuals(x,kd,kh),           init,
                xtol=1e-10,maxfev=20000,factor=0.1)
sol_no = fsolve(lambda x: residuals(x,kd*1e-6,kh*1e-6), init,
                xtol=1e-10,maxfev=20000,factor=0.1)

print(f"Water flow Q (CA active):    {sol_CA[15]:.2e}  m³/s")
print(f"Water flow Q (CA inhibited): {sol_no[15]:.2e}  m³/s")

# ------------------------------------------------------
# 5. Channel‑flux bar plot
# ------------------------------------------------------
def channel_fluxes(sol):
    Cc,Cp,Vc,Vp = sol[:7],sol[7:14],sol[16],sol[17]
    Na,K_,Cl,HCO3,H_,CO2,H2 = range(7); Cs=[150,5,130,25,10**(-7.42),0.03,0.005]
    ghk_l = lambda P,z_,Vm,Vk,Cm,Ck: ghk_flux(A_sp,P,z_,Vm,Vk,Cm,Ck)
    nk    = lambda : nkcc_flux(A_sp,Pi['NKCC'],Cc[Na],Cc[K_],Cc[Cl],Cs[0],Cs[1],Cs[2])
    nbc_s = lambda : nbc_flux(A_sp,Pi['NBCs'],Cc[Na],Cc[HCO3],Cs[0],Cs[3],Vc,0)
    nbc_p = lambda : nbc_flux(A_sp,Pi['NBCp'],Cc[Na],Cc[HCO3],Cp[Na],Cp[HCO3],Vc,Vp)
    ae_p  = lambda : ae_flux (A_sp,Pi['AEp'],Cc[Cl],Cp[HCO3],Cp[Cl],Cc[HCO3])
    ae_s  = lambda : ae_flux (A_sp,Pi['AEs'],Cc[Cl],Cs[3],Cs[2],Cc[HCO3])
    nhe_  = lambda : nhe_flux(A_sp,Pi['NHE'],Cs[4],Cc[Na],Cc[H_],Cs[0])
    K_cs  = lambda : -ghk_l(Pi['K_s'], z['K'],  Vc,0,  Cc[K_],Cs[1])
    K_cp  = lambda :  ghk_l(Pi['K_p'], z['K'],  Vc,Vp,Cc[K_],Cp[K_])
    Cl_cp = lambda :  ghk_l(Pi['Cl_p'],z['Cl'], Vc,Vp,Cc[Cl],Cp[Cl])
    return np.array([ pump_flux(A_sp,Pi['PUMP'],Cc[Na],Cp[K_]),
                     -nk(), -nbc_s(), nbc_p(), ae_p(), ae_s(),
                     nhe_(), K_cs(), K_cp(), Cl_cp() ])

labels = ['PUMP','NKCC','NBCs','NBCp','AEp','AEs','NHE',
          'K$_{cs}$','K$_{cp}$','Cl$_{cp}$']
J_CA, J_no = channel_fluxes(sol_CA), channel_fluxes(sol_no)

x = np.arange(len(labels)); w = 0.35
plt.bar(x-w/2, J_CA, width=w, label='CA active')
plt.bar(x+w/2, J_no, width=w, label='CA inhibited')
plt.xticks(x, labels, rotation=45)
plt.ylabel('Flux (mol m$^{-2}$ s$^{-1}$)\npositive = out of cell')
plt.legend(); plt.grid(True); plt.title('Channel fluxes: CA vs no‑CA')
plt.tight_layout(); plt.show()
