import numpy as np
import matplotlib.pyplot as plt


from scipy.interpolate import interp1d
from scipy.optimize import curve_fit #Use non-linear least squares to fit a function, f, to data.
from scipy.integrate import solve_ivp, quad
from scipy.linalg import eig # Solve an ordinary or generalized eigenvalue problem of a square matrix.

####################################################################################################
def progressbar(current_value, total_value, bar_lengh, progress_char): 
    """
    Barra de progreso
    """
    percentage = int((current_value/total_value)*100)                                                # Percent Completed Calculation 
    progress = int((bar_lengh * current_value ) / total_value)                                       # Progress Done Calculation 
    loadbar = "Progress: [{:{len}}]{}%".format(progress*progress_char,percentage, len = bar_lengh)    # Progress Bar String
    print(loadbar, end='\r')
####################################################################################################
# Sistema de ecuaciones de lineales de primer orden. Obtenias mediante reducción. 

def system(r, Vector, arg):
    """
    Sistema de ecuaciones
    [sigma, sigma', u, u'] -> [sigma_0, sigma_1, u_0, u_1]
    
    phi ==> sigma_0 (perfil del campo)  
    r ==> coordenada radial
    u ==> energía u = E-V(r), con V(r) el potencial gravitacional. 
    
    Nota: Ver ecuaciones (32a) y (32b) de 2402.07998v1. 
    """
    sigma_0, sigma_1, u_0, u_1, E_0 = Vector 
    Lambda,  = arg

    if r > 0:
        f0 = sigma_1
        f1 = Lambda*sigma_0**3-2*sigma_1/r-u_0*sigma_0
        f2 = u_1
        f3 = -sigma_0**2-2*u_1/r
        dE = -2*np.pi*(sigma_1**2+Lambda*sigma_0**4)*r**2

    else:
        f0 = sigma_1
        f1 = (Lambda*sigma_0**3-u_0*sigma_0)/3
        f2 = u_1
        f3 = -sigma_0**2/3
        dE = E_0

        
    return [f0, f1, f2, f3, dE]
####################################################################################################
# Calculo del valor de la energía y de la Masa

def energ(r, sigma_0, V0):
    """
    Energia y Masa
    
    V0 ==> el valor inicial del potencial (valor de frontera del potencial) 
    """
    sigF = interp1d(r, sigma_0, kind='quadratic') #Una interpolacion de sigma que usaremos la la integración posterior
    Af = lambda r: r*sigF(r)**2  #Para calcular el valor de la Energía (De donde ontenemos esta expresión?)
    Bf = lambda r: r**(2)*sigF(r)**2  # Para calcular el valor de la Masa
     # Ver ecuaciones 6ca y 6cb de https://arxiv.org/pdf/2302.00717
    rmin = r[0]
    rfin = r[-1]

    Energia = V0 - quad(Af, rmin, rfin)[0]   # energía: (2c^2 m)/Lambda  -> Lambda=4pi m^3/Mp^2
    Masa = quad(Bf, rmin, rfin)[0]  # masa: c*hb/(G*m*Lambda^(1/2)) Ver expresion 44a 
    return Energia, Masa

####################################################################################################
#Shooting para obtener el valor de E_n para el estado excitado n (numero de nodos)
# dado una f_0 inicial o de frontera

def Freq_solveG2(sigma_0, u_max, u_min, Lambda, rmax_, rmin_, nodos, df0=0, du0=0,
                met='RK45', Rtol=1e-09, Atol=1e-10):
    """
    Orden de las variables U = phi, dphi, w, dw,
    """
    
    #print('Finding a profile with ', nodos, 'nodes')
    # IMPORTANT: it is not possible to find two event at same time
    # rmax ==> se escoje lo suficientemente grande para que siempre pare el código por el límite de precisión numérica.
    # Events
    arg = [Lambda, ]
    def Sig(r, U, arg): return U[0]
    def dSig(r, U, arg): return U[1]
    Sig.direction = 0
    dSig.direction = 0
    while True:
        u0_ = (u_max+u_min)/2
        U0 = [sigma_0, df0, u0_, du0, 0]
        sol_ = solve_ivp(system, [rmin_, rmax_], U0, events=(Sig, dSig),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol)
                          # 'DOP853''LSODA'
        #print(u0_, abs((u_max-u_min)/2))
        # Cualcula la solucion para la energía u0 que hemos sugerido
        if sol_.t_events[1].size == nodos+1 and sol_.t_events[0].size == nodos:
            #t_events[1].size cuando las derivadas son cero (da el valor de r y f1 en ese punto)
            #t_events[0].size cuando f0 es cero (da el valor de r y f0 en ese punto). 
            # si el numero de derivadas es n+1 y si el numero de ceros es igual al numero de nodos
            # entonces encontramos el valor buscado u_0
            print('Found', u0_)
            return u0_, rmax_, sol_.t_events[0]
        elif sol_.t_events[1].size > nodos+1:  # una vez por nodo
             # si hay mas nodos reducir la energía
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                u_max = u0_
                rTemp_ = sol_.t_events[0][-1]
            else:  # si pasa por cero más veces que 2*nodos se aumenta la w, sino se disminuye
                u_min = u0_
                rTemp_ = sol_.t_events[1][-1]
        elif sol_.t_events[1].size <= nodos+1:
            # si hay menos nodos aumentar la energía
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                u_max = u0_
                rTemp_ = sol_.t_events[0][-1]
            else:
                u_min = u0_
                rTemp_ = sol_.t_events[1][-1]

        # checking the lim freq.
        if abs((u_max-u_min)/2) <= 1e-14: #1e-14
            print('Maxima precisión alcanzada', 'u0', u0_, 'radio', rTemp_)
            return u0_, rTemp_, sol_.t_events[0]
        
####################################################################################################
def profilesFromSolut(datos, rmin=0, Nptos=2000, inf=True):
    """
    Usando una solución
    """
    # [i, rTemp, j, 0, posNodos, met, Rtol, Atol, U0])
    sigma_0, rTemp, Lambda, nodos, posNodos, met , Rtol, Atol, U0 = datos
    
    #Porque escribir posNodos, Qué función tiene está variable?
    # boundary conditions
    V0 = [sigma_0, 0., U0, 0., 0]  # sigma, dsigma, u, du, E_0
    rspan = np.linspace(rmin, rTemp, Nptos)
    arg = [Lambda, ]

    sol2 = solve_ivp(system, [rmin, rTemp], V0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)

    Ec = sol2.y[2][-1]  # energía u = E - Uf
    #masa = -(sol2.y[2][-1]-Ec)*sol2.t[-1]  # M = -Uf*r
    
    # calculando energía y masa por la integral
    Energia, Masa = energ(sol2.t, sol2.y[0], U0) 

    if inf:
        print(r'masa=', Masa)
        print('')
        print(r'energía= ', Ec, r'energíaInt= ', Energia)
        print('')
        print(r'Lambda', Lambda)
        print('')
        #print(r'gamma', gamma)
    #en, Mas, rD, sD, dsD, uD, duD, cer0, nodos, posNodos, Lambda
    
    return Energia, Masa, sol2.t, sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3],sol2.y[4], nodos, posNodos, Lambda

####################################################################################################

def extend(rD, sD, dsD, uD, duD, Ext, Np=1000, inf=False, ptos=400):
    """
    Extendiendo solución del fondo
    """
    # Parámetros
    def parametrosS(r, S):
        yr1, yr2 = S[-2], S[-1]
        r1, r2 = r[-2], r[-1]

        k = np.real(np.log(np.abs(yr1*r1/(yr2*(r2)))))
        s = np.exp(-k*r1)/r1
        C = yr1/s
        return C, k

    #def parametrosS2(r, S, En, M, ptos):
    #    def expDec(x, c1):
    #        k = np.sqrt(-En)
    #        sig = c1*np.exp(-k*x)/x**(1-M/(2*k))
    #        return sig

    #    popt, pcov = curve_fit(expDec, r[-ptos:], S[-ptos:])
    #    return popt

    # funciones asíntóticas
    def sigm(r, C, k):
        y = C*np.exp(-k*r)/r
        dy = -(C*np.exp(-k*r)*(1+k*r))/r**2
        return y, dy

    #def sigm2(r, C, En, M):
    #    k = np.sqrt(-En)
    #    y = C*np.exp(-k*r)/r**(1-M/(2*k))
    #    dy = C*np.exp(-k*r)*r**(-2+M/(2*k))*(M-2*k*(1+k*r))/(2*k)
    #    return y, dy

    def Up(r, A, B):
        y = A+B/r
        dy = -B/r**2
        return y, dy

    rad = np.linspace(rD[-1], rD[-1]+Ext, Np)

    # calculando parámetros
    En, Mas = energ(rD, sD, uD[0])
    Ap, k = parametrosS(rD, sD)
    #Ap = parametrosS2(rD, sD, En, Mas, ptos=ptos)
    
    # uniendo datos
    sExt, dsExt = sigm(rad, Ap, k)
    #sExt, dsExt = sigm2(rad, Ap, En, Mas)
    uExt, duExt = Up(rad, En, Mas)

    rDnew = np.concatenate((rD[:-1], rad), axis=None)
    sDnew = np.concatenate((sD[:-1], sExt), axis=None)
    dsDnew = np.concatenate((dsD[:-1], dsExt), axis=None)
    uDnew = np.concatenate((uD[:-1], uExt), axis=None)
    duDnew = np.concatenate((duD[:-1], duExt), axis=None)

    fsN = interp1d(rDnew, sDnew, kind='quadratic') # quadratic
    fprof = lambda x: x**2*fsN(x)**2
    masa = quad(fprof, rDnew[0], rDnew[-1])[0]
    
    # checking
    if inf:
        print('checking ')
        print('Energia: ', En, ' ', uExt[-1]) #, ' ', k**2)
        print('Masa: ', Mas,  ' ', masa)

    return rDnew, sDnew, dsDnew, uDnew, duDnew, [masa, En, sD[0]]

####################################################################################################

# Util ()
def find_nearest(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    #print(idx)
    return (array[idx], idx)

def M99vsR99 (DatosFiltrados): 

        # Integrando
        rtake = -160
        rmin = 0
        k = 0
        radSigmMasa, Sig0R99M99EnergyT = [], []
        for dataPerf in DatosFiltrados:
            # Resolviendo Numericamente
            #sigma_0, rTemp, Lambda, nodos, posNodos, met, Rtol, Atol, U0 = dataPerf
            en, Mas, rD, sD, dsD, uD, duD, E_0, nodos, posNodos, Lambda = profilesFromSolut(dataPerf, inf=False)

            # Extendiendo
            Ext = (sD[0]*rD[-1])+70
            Np = int(Ext/2)
            rDnew, sDnew, dsDnew, uDnew, duDnew, datosEquiv = extend(rD[:rtake], sD[:rtake], dsD[:rtake], uD[:rtake], duD[:rtake],
                                                                        Ext, Np, inf=False)

            # Interpolando perfil numerico
            sigmaPerf = interp1d(rDnew, sDnew)
            Intdensidad = lambda r: (r**2)*sigmaPerf(r)**2

            # Masa perfil
            MasaProf = []
            for i in rDnew:
                temp = quad(Intdensidad, rDnew[0], i)[0]
                MasaProf.append(temp)

            radSigmMasa.append([rDnew, sDnew, MasaProf])  # salvando r, sigma(r), M(r)

            # R99
            MTotal = MasaProf[-1]
            valM, indx = find_nearest(MasaProf, 0.99*MTotal)
            R99, M99 = rDnew[indx], MasaProf[indx]
            print('Mas', Mas,'Mtotal', MTotal, 'M99', M99, 'Lambda', Lambda)

            # E- valor
            #energyPro = lambda r: r*sigmaPerf(r)**2
            #temp = quad(energyPro, rDnew[0], rDnew[-1])[0]
            #Energy = uD[0] - temp   # energia barra
            #print(en, Energy)

            Sig0R99M99EnergyT.append([sDnew[0],R99, M99])  # salvando sigma0, R99, M99, E
            print(k)
            k += 1
            
        return Sig0R99M99EnergyT

        # Salvando
        #np.save('radSigmMasan1Lm1.npy', np.array(radSigmMasa, dtype=object)) # save
        #np.save('Sig0R99M99Energyn1Lm1.npy', np.array(Sig0R99M99EnergyT, dtype=object)) # save

####################################################################################################

#[0.0001, 20, 0, 1, array([10.20695393]), 'DOP853', 1e-13, 1e-15, 0.094734375]


####################################################################################################

def Segundo_filtro (l1,l0,lN1):
    
    RL0 = []
    ML0 = []
    RL1 = []
    ML1 = []
    RL_1 = []
    ML_1 = []

    for i, sublista in enumerate(l0):

        if sublista[1] < 40:

                    RL0.append(sublista[1])
                    ML0.append(4*(np.pi)*sublista[2])
        else: 
            continue 

    for i, sublista in enumerate(lN1):

        if sublista[1] < 40:

                    RL_1.append(sublista[1])
                    ML_1.append(4*(np.pi)*sublista[2])
        else: 
            continue 

    for i, sublista in enumerate(l1):

        if sublista[1] < 40:

                    RL1.append(sublista[1])
                    ML1.append(4*(np.pi)*sublista[2])
        else: 
            continue 
    
    return RL0, ML0, RL1, ML1 ,RL_1, ML_1


## Función para filtar malos datos
# datos.append([sigma_0, rTemp, LAmbda, nodos, posNodos, met, Rtol, Atol, U0])

####################################################################################################
def Filtracion (Datos, inf=False):   

    datos_filtro_1 = []
    datos_filtro_0 = []
    datos_filtro_N1 = []
    
    met='RK45'
    Rtol = 1e-8 ## 1e-13
    Atol = 1e-9 


    for i in Datos:  

                U0 = [i[0], 0, i[8], 0, 0]

                #print(i[2], type)
                if i[2] == 1:  
                    arg = [i[2]]
                    sol_ = solve_ivp(system, [0, i[1]], U0, args=(arg,), method=met,  rtol=Rtol, atol=Atol)

                    if sol_.y[0][-1] > 1: 
                                continue
                    else: 

                        if inf:
                            plt.plot(sol_.t, sol_.y[0])

                        datos_filtro_1.append(i)

                elif i[2] == 0:  
                    #Lambda, gamma = arg
                    arg = [i[2]]

                    sol_ = solve_ivp(system, [0, i[1]], U0, args=(arg,), method=met,  rtol=Rtol, atol=Atol)

                    if sol_.y[0][-1] > 1: 
                                continue
                    else: 
                        if inf:
                            plt.plot(sol_.t, sol_.y[0])

                        datos_filtro_0.append(i)


                elif i[2] == -1: 

                    arg = [i[2]]
                    sol_ = solve_ivp(system, [0, i[1]], U0, args=(arg,), method=met,  rtol=Rtol, atol=Atol)

                    if sol_.y[0][-1] > 1: 
                                continue
                    else: 

                        if inf:
                            plt.plot(sol_.t, sol_.y[0])

                        datos_filtro_N1.append(i)
                        
    return datos_filtro_1, datos_filtro_0, datos_filtro_N1 


    plt.show()
    #plt.savefig('Perfiles_varios')