import numpy as np 
import matplotlib.pyplot as plt
from optbayesexpt import OptBayesExpt, MeasurementSimulator
import time

# this is the model of our experiment
def M_p(tau_p, gamma_p, gamma_m):
    #Args:
    #    gamma (plus and minus) = decay rates
    #    tau (plus and minus) = time after preparing state

    G = np.sqrt(gamma_p**2 + gamma_m**2 - gamma_p * gamma_m)
    Beta_p = gamma_p + gamma_m + G
    Beta_m = gamma_p + gamma_m - G
    return (1/(2*G))*((G + gamma_p)*np.e**(-Beta_p * tau_p) + (G - gamma_p)*np.e**(-Beta_m * tau_p))

def M_m(tau_m, gamma_p, gamma_m):
    #Args:
    #    gamma (plus and minus) = decay rates
    #    tau (plus and minus) = time after preparing state

    G = np.sqrt(gamma_p**2 + gamma_m**2 - gamma_p * gamma_m)
    Beta_p = gamma_p + gamma_m + G
    Beta_m = gamma_p + gamma_m - G
    return (1/(2*G))*((G + gamma_m)*np.e**(-Beta_p * tau_m) + (G - gamma_m)*np.e**(-Beta_m * tau_m))

def model_m_p(sets, pars, cons):
    #unpack model setting and parameters
    tau, = sets
    gamma_p, gamma_m = pars

    return M_p(tau, gamma_p, gamma_m)

def model_m_m(sets, pars, cons):
    #unpack model setting and parameters
    tau, = sets
    gamma_p, gamma_m = pars

    return M_m(tau, gamma_p, gamma_m)


tau_domain = np.linspace(0, 10**4, 1001)
sets = (tau_domain,)

n_samples = 10000
gamma_p = np.random.uniform(10e-4, 10e-2, n_samples)
gamma_m = np.random.uniform(10e-4, 10e-2, n_samples)
parameters = (gamma_p, gamma_m)

constants = ()

tauOBE_p = OptBayesExpt(model_m_p, sets, parameters, constants, scale=False)
tauOBE_m = OptBayesExpt(model_m_m, sets, parameters, constants, scale=False)


#Measurement Simulation
true_pars = (3e-3, 1e-3)
noise_level = 0.1

simulation_p = MeasurementSimulator(model_m_p, true_pars, constants, noise_level=noise_level)
simulation_m = MeasurementSimulator(model_m_m, true_pars, constants, noise_level=noise_level)

#Experiment Demo

def find_tau():
    global pickiness
    global optimum
    #global noiselevel
    
    n_measure = 200
    taudata = np.zeros(n_measure) 
    Mdata = np.zeros(n_measure)

    print('Mplus')
    for i in np.arange(n_measure):
        if optimum:
            tauset = tauOBE_p.opt_setting()
        else:
            tauset = tauOBE_p.good_setting(pickiness=pickiness)
        M_meas = simulation_p.simdata(tauset)

        taudata[i] = tauset[0]
        Mdata[i] = M_meas 
        measurement_record = (tauset, M_meas, noise_level)
        tauOBE_p.pdf_update(measurement_record)
        print(tauOBE_p.mean())

    plt.plot(tau_domain, M_p(tau_domain, tauOBE_p.mean()[0], tauOBE_p.mean()[1]), 'green')

    print('MMINUS')
    for i in np.arange(n_measure):
        if optimum:
            tauset = tauOBE_m.opt_setting()
        else:
            tauset = tauOBE_m.good_setting(pickiness=pickiness)
        M_meas = simulation_m.simdata(tauset)

        taudata[i] = tauset[0]
        Mdata[i] = M_meas 
        measurement_record = (tauset, M_meas, noise_level)
        tauOBE_m.pdf_update(measurement_record)
        print(tauOBE_m.mean())



    
pickiness = 5
optimum = True
#optimum = False

start_time = time.time()
find_tau()  
print((time.time()-start_time))
#test_domain = (tau_domain, )
#print(tau_domain)
#M_true = simulation.simdata(test_domain, noise_level = 0)
#print(M_true)
#plt.plot(tau_domain, M_true)
#plt.xscale('log')
#plt.show()

plt.plot(tau_domain, M_p(tau_domain, true_pars[0], true_pars[1]), 'red')
plt.plot(tau_domain, M_m(tau_domain, true_pars[0], true_pars[1]), 'blue')
plt.xscale('log')
plt.show()
