import numpy as np
import scipy as sp
import pandas as pd
from scipy.integrate import solve_ivp
import warnings


class Model(object):
    """! The learning model base class.
    Defines the base class for the learning models used in predictions.
    """
    def __call__(self, init_point, para, time_range):
        """! The learning model base class calling method.
        @param init_point
        @param para 
        @param time_range
        """
        ''' forward function '''
        warnings.warn('Model call method does not implement')
        raise NotImplementedError

'''
Inherit the Model Object:
class xxx(Model):
    def __init__(self):
        pass
    def __call__(self, init_point, para, time_range):
        pass
'''

class Learner_SuEIR(Model):
    """! Class for SuEIR (Susceptible, Unreported, Exposed, Infected, Recovered) model learner.
    """
    def __init__(self, N, E_0, I_0, R_0, a, decay, bias=0.005):
        """! Initializer for Learner_SuEIR.
        @param N  Total population
        @param E_0  Initial exposed population
        @param I_0  Initial infected population
        @param R_0  Initial recovered population
        @param a  Learning rate parameter (starting rate)
        @param decay  Learning rate parameter (responsible for progressively lowering the rate)
        @param bias  A numerical value possibly used in calculating the fatality/removed ratio, by default 0.005
        """
        self.N = N
        self.E_0 = E_0
        self.I_0 = I_0
        self.R_0 = R_0
        self.a = a
        self.decay = decay
        # Fatality/removed ratio
        self.FRratio = a * \
            np.minimum(np.exp(-decay * (np.arange(1000) + 0)), 1)+bias
        self.pop_in = 0
        self.pop = N*5
        self.bias=1000000

        self.initial_N = N
        self.initial_pop_in = self.pop_in
        self.initial_bias=1000000

    def __call__(self, size, params, init, lag=0):
        """! Calling method for the learner class.
        @param size  The ending time value for solve_ivp
        @param params  List of statistical parameters for change in each compartment of people. Beta, gamma, sigma, mu
        @param init  Initial values of the amounts of people in each compartment
        @param lag  Parameter used in calculations. By default 0
        @return  Tuple of predicted susceptible, exposed, infected, recovered, confirmed and fatality cases
        """

        beta, gamma, sigma, mu = params

        def calc_grad(t, y):
            """! Function that calculates the population amounts at different times.
            @param t  Time
            @param y  A list of the division of the population. Susceptible, exposed, infected, recovered
            @return  A list of population compartment sizes according to time.
            """
            S, E, I, _ = y

            # new population joining the susceptible population
            new_pop_in = self.pop_in*(self.pop-self.N)*(np.exp(-0.03*np.maximum(0, t-self.bias))+0.05) 
            return [new_pop_in-beta*S*(E+I)/self.N, beta*S*(E+I)/self.N-sigma*E, mu*E-gamma*I, gamma*I]

        # Uses solve_ivp function to get an array of arrays of the population sizes at set time intervals.
        # Solution has the [S,E,I,R] values at different times.
        solution = solve_ivp(
            calc_grad, [0, size], init, t_eval=np.arange(0, size, 1))

        # Removed people per day
        temp_r_perday = np.diff(solution.y[3])
        # Estimated fatality rate per day
        temp_F_perday = temp_r_perday * \
            self.FRratio[lag:len(temp_r_perday)+lag]
        # Since the -1 day info is not accessible, we treat the death of day 0 as exactly the R_0
        # which means no recoveries before day 0. Then calculated cumulative sum of fatalities.
        temp_F = np.empty(len(temp_F_perday) + 1)
        np.cumsum(temp_F_perday, out=temp_F[1:])
        temp_F[0] = 0
        temp_F += solution.y[3][0]

        # Note that I is the active cases instead of the cumulative confirmed cases
        # Confirmed = I + R, death is prior estimated
        # return pred_S, pred_E, pred_I, pred_R, pred_confirmed, pred_fatality
        return solution.y[0], solution.y[1], solution.y[2], solution.y[3], solution.y[2] + solution.y[3], temp_F

    def reset(self):
        """! Reset function for learner class
        """
        self.N = self.initial_N
        self.pop_in = self.initial_pop_in
        self.bias = self.initial_bias

class Learner_SEIR(Model):
    """! Class for SEIR (Susceptible, Exposed, Infected, Recovered) model learner.
    """
    def __init__(self, N, E_0, I_0, R_0, a, decay, bias=0):
        """! Initializer for Learner_SEIR.
        @param N  Total population
        @param E_0  Initial exposed population
        @param I_0  Initial infected population
        @param R_0  Initial recovered population
        @param a  Learning rate parameter (starting rate)
        @param decay  Learning rate parameter (responsible for progressively lowering the rate)
        @param bias  A numerical value possibly used in calculating the fatality/removed ratio, by default 0.005
        """
        self.N = N
        self.E_0 = E_0
        self.I_0 = I_0
        self.R_0 = R_0
        self.a = a
        # Fatality/removed ratio
        self.FRratio = a * \
            np.minimum(np.exp(-decay * (np.arange(1000) + bias)), 1) # decaying death date

    def __call__(self, size, params, init, lag = 0):
        """! Calling method for the learner class.
        @param size  The ending time value for solve_ivp
        @param params  List of statistical parameters for change in each compartment of people. Beta, gamma, sigma
        @param init  Initial values of the amounts of people in each compartment
        @return  Tuple of predicted susceptible, exposed, infected, recovered, confirmed and fatality cases
        """
        beta, gamma, sigma = params
        S_0, E_0, I_0, R_0 = init

        def calc_grad(t, y):
            """! Function that calculates the population amounts at different times.
            @param t  Time
            @param y  A list of the division of the population. Susceptible, exposed, infected, recovered
            @return  A list of population compartment sizes according to time.
            """
            S, E, I, _ = y
            return [-beta*S*I/self.N, beta*S*I/self.N-sigma*E, sigma*E-gamma*I, gamma*I]

        # Uses solve_ivp function to get an array of arrays of the population sizes at set time intervals.
        # Solution has the [S,E,I,R] values at different times.
        solution = solve_ivp(
            calc_grad, [0, size], init, t_eval=np.arange(0, size, 1))

        # Removed people per day
        temp_r_perday = np.diff(solution.y[3])
        # Estimated fatality rate per day                      
        temp_F_perday = temp_r_perday * \
            self.FRratio[lag:len(temp_r_perday)+lag]
        # Since the -1 day info is not accessible, we treat the death of day 0 as exactly the R_0
        # which means no recoveries before day 0. Then calculated cumulative sum of fatalities.
        temp_F = np.empty(len(temp_F_perday) + 1)
        np.cumsum(temp_F_perday, out=temp_F[1:])
        temp_F[0] = 0
        temp_F += solution.y[3][0]

        # Note that I is the active cases instead of the cumulative confirmed
        # Confirmed = I + R, death is prior estimated
        # return pred_S, pred_E, pred_I, pred_R, pred_confirmed, pred_fatality
        return solution.y[0], solution.y[1], solution.y[2], solution.y[3], solution.y[2] + solution.y[3], temp_F


class Learner_SuEIR_H(Model):
    """! Class for SuEIR_H (Susceptible, Unreported, Exposed, Infected, Recovered, Hospitalized) model learner.
    
    Hospitalization Prediction modified from SuEIR model: by removing exponential estimation
    dS = -beta S(E + I) / N
    dE = beta S(E + I) / N - sigma E
    dI = mu E - gamma I
    dR = gamma I
    ---
    dH = alpha I - rho H
    dD = theta H
    ---
    pred_S, pred_E, pred_I, pred_R, pred_confirm, pred_fatality, pred_hospital as
    S, E, I, R, I + R, D, H
    """

    def __init__(self, N, E_0):
        """! Initializer for Learner_SuEIR_H.
        @param N  Total population
        @param E_0  Initial exposed population
        """
        self.N = N
        self.E_0 = E_0

    def __call__(self, size, params, init):
        """! Calling method for the learner class.
        @param size  The ending time value for solve_ivp
        @param params  List of statistical parameters for change in each compartment of people. Beta, gamma, sigma, mu, aplha, rho, theta
        @param init  Initial values of the amounts of people in each compartment
        @return  Tuple of predicted susceptible, exposed, infected, recovered, confirmed, fatality and hospitalized cases
        """
        # alpha, rho, theta is added by hospitalization
        beta, gamma, sigma, mu, alpha, rho, theta = params

        def calc_grad(t, y):
            """! Function that calculates the population amounts at different times.
            @param t  Time
            @param y  A list of the division of the population. Susceptible, exposed, infected, recovered, hospitalized
            @return  A list of population compartment sizes according to time.
            """
            S, E, I, _, H, _ = y
            return [-beta*S*(E+I)/self.N,  # dS
                    beta*S*(E+I)/self.N-sigma*E,  # dE
                    mu*E-gamma*I, gamma*I,  # dI and dR
                    alpha * I - rho * H, theta * H]  # dH and dD
        
        # Uses solve_ivp function to get an array of arrays of the population sizes at set time intervals.
        # Solution has the [S,E,I,R,H,D] values at different times.
        solution = solve_ivp(
            calc_grad, [0, size], init, t_eval=np.arange(0, size, 1)).y

        # return pred_S, pred_E, pred_I, pred_R, pred_confirm, pred_fatality, pred_hospital
        return solution[0], solution[1], solution[2], solution[3], solution[2] + solution[3], solution[5], solution[4]


if __name__ == '__main__':
    # m = xxx()
    # m(None, None, 1)

    pass
