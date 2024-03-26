import numpy as np
import pandas as pd
import json
import argparse
# According to 'us' -package changelog, DC_STATEHOOD env variable should be set truthy before (FIRST) import
import os
os.environ['DC_STATEHOOD'] = '1'
import us

from model import Learner_SuEIR
from data import JHU_US, JHU_global, NYTimes
from rolling_train_modified import rolling_train, rolling_prediction, loss
from util import get_start_date, write_val_to_json
from matplotlib import pyplot as plt

# Import hard coded dates, decays and "a" values.
import prediction_data as pdata

parser = argparse.ArgumentParser(description='validation of prediction performance for all states')
parser.add_argument('--END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--VAL_END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--level', default = "state",
                    help='state, nation or county')
parser.add_argument('--state', default = "default",
                    help='state')
parser.add_argument('--nation', default = "default",
                    help='nation')
parser.add_argument('--county', default = "default",
                    help='county')
parser.add_argument('--dataset', default = "NYtimes",
                    help='nytimes')
parser.add_argument('--popin', type=float, default = 0,
                    help='popin')
args = parser.parse_args()

print(args)

# severe_state = ["Florida"]  

def validation_loss(model, init, params_all, train_data, val_data, new_sus, pop_in):
    """! The function calculates validation loss of the model with the given parameters.
    @param model The SuEIR epidemic model.
    @param init A python list containing initial parameters S0, I0, E0, R0.
    @param params_all Parameters gained by training the model.
    @param train_data Training data used to calculate predicted output.
    @param val_data Validation data used to calculate validation loss by comparing it to the prediction data.
    @param new_sus Amount of new suspectible individuals.
    @param pop_in Flag used to calculate the amount of population joining the suspecitible population. Gives realism to calculations.
    @return Float value representing the validation loss of the model.
    """

    val_data_confirm, val_data_fatality = val_data[0], val_data[1]
    val_size = len(val_data_confirm)

    pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, new_sus, pred_range=val_size, pop_in=pop_in)
    # print(pred_fatality, val_data_fatality)
    # return 0.5*loss(np.maximum(0, pred_fatality-val_data_fatality[0]), np.maximum(0, val_data_fatality-val_data_fatality[0]), smoothing=10) \
    #  + loss(np.maximum(0, pred_confirm-val_data_confirm[0]), np.maximum(0,val_data_confirm-val_data_confirm[0]), smoothing=10)
    
    return  0.5*loss(pred_confirm, val_data_confirm, smoothing=0.1) + loss(pred_fatality, val_data_fatality, smoothing=0.1)

def get_county_list(cc_limit=200, pop_limit=50000):
    """! The function creates a list of counties from the wanted dataset.
    @param cc_limit The lower limit for confirmed cases for a county.
    @param pop_limit A lower limit for population in a county.
    @return A python list containing a list of valid counties.
    """
    
    non_county_list = ["Puerto Rico", "American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess", "Grand Princess"]
    data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')
    with open("data/county_pop.json", 'r') as f:
        County_Pop = json.load(f)
    county_list = []
    for region in County_Pop.keys():
        county, state = region.split("_")
        # print(state)
        if County_Pop[region][0]>=pop_limit and state not in non_county_list:        
            train_data = data.get("2020-03-22", args.END_DATE, state, county)
            confirm, death = train_data[0], train_data[1]
            start_date = get_start_date(train_data)
            if len(death) > 0 and np.max(death) >=0 and np.max(confirm) > cc_limit and start_date < "2020-05-10" and county != "Lassen":
                county_list += [region]

    return county_list

def get_region_list():
    """! The function creates a list of regions, nations or counties depending on input parameters. It also creates the names for the directories where the validation files are written.
    @return A dictionary containing parameters, which depend on whether a state, county or nation was chosen.
    """

    state = 0
    County_Pop = 0
    Nation_Pop = 0
    # initial the dataloader, get region list 
    # get the directory of output validation files
    if args.level == "state":
        # data = NYTimes(level='states')
        data = NYTimes(level='states') if args.dataset == "NYtimes" else JHU_US(level='states')
        nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands", "Northern Mariana Islands"]
        region_list = [state for state in data.state_list if state not in nonstate_list]
        # region_list = mid_dates_state.keys()
        # print(data.state_list)
        mid_dates = pdata.mid_dates_state
        write_dir = "val_results_state/" + args.dataset + "_" 
        if args.state != "default":
            region_list = [args.state]  
            # region_list = ["New York", "California", "Illinois", "North Carolina", "Florida", "Texas", "Georgia", "Arizona", "South Carolina", "Alabama"]
            # region_list = ["New York", "California"]
            write_dir = "val_results_state/test" + args.dataset + "_"
        
    elif args.level == "county":
        state = "California"
        # data = NYTimes(level='counties')
        data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')
        # region_list = mid_dates_county.keys()
        # region_list = ["Cochise_Arizona"]
        mid_dates = pdata.mid_dates_county
        with open("data/county_pop.json", 'r') as f:
            County_Pop = json.load(f)       

        if args.state != "default" and args.county != "default":
            region_list = [args.county + "_" + args.state] 
            # region_list = ["New York_New York", "Los Angeles_California", "Dallas_Texas"] 
            write_dir = "val_results_county/test" + args.dataset + "_"

        else:
            region_list = get_county_list(cc_limit=2000, pop_limit=10)
            print("# feasible counties:", len(region_list))
            write_dir = "val_results_county/" + args.dataset + "_"

    elif args.level == "nation":
        data = JHU_global()
        region_list = pdata.START_nation.keys()
        mid_dates = pdata.mid_dates_nation
        write_dir = "val_results_world/" + args.dataset + "_" 

        if args.nation != "default":
            region_list = [args.nation] 
            write_dir = "val_results_world/test" + args.dataset + "_" 

        with open("data/world_pop.json", 'r') as f:
            Nation_Pop = json.load(f)

    return {'region_list': region_list, 'mid_dates': mid_dates, 'write_dir': write_dir, 'data': data, 'state': state, 'County_Pop': County_Pop, 'Nation_Pop': Nation_Pop}

def generate_parameters(region, param_dict):
    """! The function creates a dictionary of variables, such as the a and decay parameters, to use when generating validation results.
    @param region The current region (state, county or nation) which is used to generate validation results.
    @param param_dict A dictionary that contains needed parameters. Result of refactoring.
    @return A dictionary containing variables, which depend on whether a state, county or nation was chosen.
    """

    # Initialize full_data.
    full_data = 0

    nation = 0
    state = param_dict['state']
    mid_dates = param_dict['mid_dates']
    data = param_dict['data']
    if args.level == "state":
        state = str(region)
        df_Population = pd.read_csv('data/us_population.csv')
        print(state)
        Pop=df_Population[df_Population['STATE']==state]["Population"].to_numpy()[0]
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state),100)
        if state in mid_dates.keys():
            second_start_date = mid_dates[state]
            train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
            reopen_flag = True
        else:
            second_start_date = "2020-08-30"
            train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
            reopen_flag = False

        if state in mid_dates.keys():
            resurge_start_date = pdata.mid_dates_state_resurge[state] if state in pdata.mid_dates_state_resurge.keys() else "2020-09-15"
            train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
                data.get(resurge_start_date, args.END_DATE, state)]
            full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
                data.get(resurge_start_date, args.VAL_END_DATE, state)]

        val_data = data.get(args.END_DATE, args.VAL_END_DATE, state)
        if state in pdata.decay_state.keys():
            a, decay = pdata.decay_state[state][0], pdata.decay_state[state][1]
        else:
            a, decay = 0.7, 0.3          
        # will rewrite it using json
        pop_in = 1/400
        if state == "California":
            pop_in = 0.01
            
    elif args.level == "county":
        county, state = region.split("_")
        region = county + ", " + state
        key = county + "_" + state

        County_Pop = param_dict['County_Pop']
        Pop=County_Pop[key][0]
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state, county))
        if state=="California" and county in mid_dates.keys():
            second_start_date = mid_dates[county]
            reopen_flag = True
        elif state in pdata.mid_dates_state.keys() and not (state == "Arkansas" or state == "Montana"):
            second_start_date = pdata.mid_dates_state[state]
            reopen_flag = True
        else:
            second_start_date = "2020-08-30"
            reopen_flag = False

        if start_date < "2020-05-10":
            train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, args.END_DATE, state, county)]
        else:
            train_data = [data.get(start_date, args.END_DATE, state, county)]
        val_data = data.get(args.END_DATE, args.VAL_END_DATE, state, county)
        if state in pdata.decay_state.keys():
            a, decay = pdata.decay_state[state][0], pdata.decay_state[state][1]
        else:
            a, decay = 0.7, 0.32
        if county in pdata.north_cal and state=="California":
            decay = 0.03
        pop_in = 1/400

        if state in pdata.mid_dates_state.keys():
            resurge_start_date = pdata.mid_dates_state_resurge[state] if state in pdata.mid_dates_state_resurge.keys() else "2020-09-15"
            train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
                data.get(resurge_start_date, args.END_DATE, state, county)]
            full_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
                data.get(resurge_start_date, args.VAL_END_DATE, state, county)]

    elif args.level == "nation":
        nation = str(region)
        Nation_Pop = param_dict['Nation_Pop']
        Pop = Nation_Pop["United States"] if nation == "US" else Nation_Pop[nation]
        if nation in pdata.mid_dates_nation.keys():
            second_start_date = mid_dates[nation]
            reopen_flag = True
        elif nation == "Turkey":
            second_start_date = "2020-06-07"
            reopen_flag = False
        else:
            second_start_date = "2020-07-30"
            reopen_flag = False
        pop_in = 1/2000 if nation == "Germany" else 1/400

        start_date = pdata.START_nation[nation]
        train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, args.END_DATE, nation)]
        full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, args.END_DATE, nation)]

        if nation=="US":
            resurge_start_date = "2020-09-15"
            train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, resurge_start_date, nation), data.get(resurge_start_date, args.END_DATE, nation)]
            full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, resurge_start_date, nation), data.get(resurge_start_date, args.END_DATE, nation)]

        val_data = data.get(args.END_DATE, args.VAL_END_DATE, nation)
        a, decay = pdata.FR_nation[nation]

    return {'a': a, 'decay': decay, 'pop_in': pop_in, 'Pop': Pop, 'state': state,
             'train_data': train_data, 'reopen_flag': reopen_flag, 'val_data': val_data,
             'full_data': full_data, 'start_date': start_date, 'second_start_date': second_start_date, 'nation': nation}

def generate_validation_results(parameters, params_allregion, region):
    """! The function fills the params_allregion dictionary with validation results per region.
    @param parameters A dictionary that contains needed parameters. Result of refactoring.
    @param params_allregion A dictionary that will contain all the validation results from all the wanted regions after modification.
    @param region The current region (state, county or nation) which is used to generate validation results.
    @return The modified params_allregion dictionary.
    """

    pop_in = parameters['pop_in']
    state = parameters['state']
    train_data = parameters['train_data']
    reopen_flag = parameters['reopen_flag']
    val_data = parameters['val_data']
    full_data = parameters['full_data']

    mean_increase = 0
    if len(train_data)>1:
        last_confirm, last_fatality = train_data[-1][0], train_data[-1][1]
        daily_confirm = np.diff(last_confirm)
        mean_increase = np.median(daily_confirm[-7:] - daily_confirm[-14:-7])/2 + np.median(daily_confirm[-14:-7] - daily_confirm[-21:-14])/2
        # if mean_increase<1.1:
        #     pop_in = 1/5000
        if not reopen_flag or args.level == "county":
            if np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1:
                pop_in = 1/5000
            elif mean_increase < np.mean(daily_confirm[-7:])/40:
                pop_in = 1/5000
            elif mean_increase > np.mean(daily_confirm[-7:])/10 and np.mean(daily_confirm[-7:])>60:
                pop_in = 1/500
            else:
                pop_in = 1/1000
        if args.level=="state" and reopen_flag and (np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1):
            pop_in = 1/500
            if state == "California":
                pop_in = 0.01
        if args.level == "nation" and ( region=="Canada"):
            pop_in = 1/5000
        if args.level != "nation" and (state == "New York"):
            pop_in = 1/5000
        if args.level == "nation" and (region == "Iran"):
            pop_in =  1/1000 
        if args.level == "nation" and (region == "US"):
            pop_in = 1/400
        if args.popin >0:
            pop_in = args.popin
    
    print("region: ", region, " start date: ", parameters['start_date'], " mid date: ", parameters['second_start_date'],
        " end date: ", args.END_DATE, " Validation end date: ", args.VAL_END_DATE, "mean increase: ", mean_increase, pop_in )    

    # print(train_data)
    # candidate choices of N and E_0, here r = N/E_0
    Ns = np.asarray([0.2])*parameters['Pop']
    rs = np.asarray([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 120, 150, 200, 400])
    if args.level == "county":
        rs = np.asarray([30,  40, 50, 60, 70, 80,  90, 100, 120, 150, 200, 400])
        # rs = np.asarray([200])

    if args.level == "nation":

        if region == "South Africa" :
            rs *= 4
        if region == "India" or region == "Qatar":
            rs *= 4
        if region == "Argentina" :
            rs *= 4

    A_inv, I_inv, R_inv, loss_list0, loss_list1, params_list, learner_list, I_list = [],[],[],[],[],[],[],[]
        
    val_log = []
    min_val_loss = 10 #used for finding the minimum validation loss
    for N in Ns:
        for r in rs:
            E_0 = N/r

            # In order to simulate the reopen, we assume at the second stage, there are N new suspectible individuals
            new_sus = 0 if reopen_flag else 0
            if args.level == "state" or args.level == "county":
                bias = 0.025 if reopen_flag or (state=="Louisiana" or state=="Washington" or state == "North Carolina" or state == "Mississippi") else 0.005
                if state == "Arizona" or state == "Alabama" or state == "Florida" or state=="Indiana" or state=="Wisconsin" or state == "Hawaii" or state == "California" or state=="Texas" or state=="Illinois":
                    bias = 0.01
                if state == "Arkansas" or state == "Iowa" or state == "Minnesota" or state == "Louisiana" \
                    or state == "Nevada" or state == "Kansas" or state=="Kentucky" or state == "Tennessee" or state == "West Virginia":
                    bias = 0.05
            if args.level == "nation":

                bias = 0.02 if reopen_flag else 0.01
                nation = parameters['nation']
                if nation == "Germany":
                    bias = 0.02
                if nation == "US":
                    bias = 0.02
            data_confirm, data_fatality = train_data[0][0], train_data[0][1]
            # print (bias)
            model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=parameters['a'], decay=parameters['decay'], bias=bias)

            # At the initialization we assume that there is not recovered cases.
            init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]
            print(init)
            # train the model using the candidate N and E_0, then compute the validation loss
            params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)
            val_loss = validation_loss(model, init, params_all, train_data, val_data, new_sus, pop_in=pop_in)

            for params in params_all:
                beta, gamma, sigma, mu = params
                # we cannot allow mu>sigma otherwise the model is not valid
                if mu>sigma:
                    val_loss = 1e6

            # using the model to forecast the fatality and confirmed cases in the next 100 days, 
            # output max_daily, last confirm and last fatality for validation
            pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, new_sus, pop_in=pop_in, pred_range=100, daily_smooth=True)
            max_daily_confirm = np.max(np.diff(pred_confirm))
            pred_confirm_last, pred_fatality_last = pred_confirm[-1], pred_fatality[-1]
            # print(np.diff(pred_fatality))
            # print(sigma/mu)
            #prevent the model from explosion
            if pred_confirm_last >  8*train_data[-1][0][-1] or  np.diff(pred_confirm)[-1]>=np.diff(pred_confirm)[-2]:
                val_loss = 1e8

            # record the information for validation
            val_log += [[N, E_0] + [val_loss] + [pred_confirm_last] + [pred_fatality_last] + [max_daily_confirm] + loss_all  ]

            # plot the daily inc confirm cases
            confirm = train_data[0][0][0:-1].tolist() + train_data[-1][0][0:-1].tolist() + pred_confirm.tolist()
            true_confirm =  train_data[0][0][0:-1].tolist() + train_data[-1][0][0:-1].tolist() + val_data[0][0:-1].tolist()

            deaths = train_data[0][1][0:-1].tolist() + train_data[-1][1][0:-1].tolist() + pred_fatality.tolist()
            true_deaths =  train_data[0][1][0:-1].tolist() + train_data[-1][1][0:-1].tolist() + val_data[1][0:-1].tolist()
            #When the smallest validation loss yet is found, the plots are overwritten
            if val_loss < min_val_loss:
                plt.figure()
                plt.plot(confirm)
                plt.plot(true_confirm)
                plt.savefig("figure_"+args.level+"/daily_increase.pdf")
                plt.close()
                plt.figure()
                plt.plot(deaths)
                plt.plot(true_deaths)
                plt.savefig("figure_"+args.level+"/daily_increase_death.pdf")
                plt.close()
            min_val_loss = np.minimum(val_loss, min_val_loss)
            # print(val_loss)

    params_allregion[region] = val_log
    print (np.asarray(val_log))
    best_log = np.array(val_log)[np.argmin(np.array(val_log)[:,2]),:]
    print("Best Val loss: ", best_log[2], " Last CC: ", best_log[3], " Last FC: ", best_log[4], " Max inc Confirm: ", best_log[5] )

    return params_allregion

def generate_validation_files():
    """! The function creates the validation results for each region, state or county, and saves them in json-files.
    """
        
    region_list_dict = get_region_list()
    write_directory = region_list_dict['write_dir']
    region_list = region_list_dict['region_list']
    parameters_from_all_regions = {}

    for region in region_list:

        # generate training data, validation data
        # get the population
        # get the start date, and second start date
        # get the parameters a and decay
        
        parameters = generate_parameters(region, region_list_dict)
        
        print(len(parameters['train_data']))
        
        parameters_from_all_regions = generate_validation_results(parameters, parameters_from_all_regions, region)

    # write all validation results into files
    write_file_name_all = write_directory + "val_params_" + "END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
    write_file_name_best = write_directory + "val_params_best_" + "END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE

    write_val_to_json(parameters_from_all_regions, write_file_name_all, write_file_name_best)

# The main functionality of the file happens here. At the end validation parameters, such as validation loss, for each region
# (or state, depends on chosen level parameters) is saved to two files, which are then used to generate predictions in the generate_predictions.py -file.
# This process is described as "The passed inputs go through the ’Validation File Generator’ which generates a parameter validation set.
# Once the parameter validation set is created, it is passed to the ’Estimation Phase’ which in turn generates the predicted data file." in the linked
# master's thesis.
if __name__ == '__main__':
    generate_validation_files()
