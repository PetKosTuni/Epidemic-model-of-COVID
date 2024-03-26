import numpy as np
import pandas as pd
import json
import argparse
# According to 'us' -package changelog, DC_STATEHOOD env variable should be set truthy before (FIRST) import
import os
os.environ['DC_STATEHOOD'] = '1'
import us

from util import get_start_date, state2fips
from model import Learner_SuEIR
from data import JHU_US, JHU_global
from rolling_train_modified import rolling_train, rolling_prediction
from datetime import timedelta

# Import hard coded dates, decays and "a" values.
import prediction_data as pdata

parser = argparse.ArgumentParser(description='validation of prediction performance for all states')
parser.add_argument('--END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--VAL_END_DATE', default = "default",
                    help='end date for training models')
args = parser.parse_args()
PRED_START_DATE = args.VAL_END_DATE


print(args)


# Create filename from prediction start date. 
pred_start_date = "2021-09-19"
write_file_name = pred_start_date + "-UCLA-SuEIR_state.csv"

# Create list containing 200 Saturdays starting from 2021-09-25.
Sat_list = [(pd.to_datetime("2021-09-25") + timedelta(days=i*7)).strftime("%Y-%m-%d") for i in range(200)]

# Fetch data for US states from JHU.
data = JHU_US(level="states")

# Create list for US states and exclude non-state entities listed in nonstate_list.
nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands", "Northern Mariana Islands", "vermont"]
state_list = ["US"]+[state for state in data.state_list if state not in nonstate_list]


prediction_range = 100
frame = []
for state in state_list:
    # Get data for US ("US" was set to the first element to the state_list).
    if state == "US":
        nation = state

        # Fetch data for US from JHU.
        data = JHU_global()

        # Get middle dates for nations.
        region_list = pdata.mid_dates_nation.keys()
        mid_dates = pdata.mid_dates_nation
        
        # Get start dates for US.
        second_start_date = mid_dates[nation]
        start_date = pdata.START_nation[nation]

        # Get data from US for training and full result.
        train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, args.END_DATE, nation)]
        full_data = [data.get('2020-03-22', second_start_date, nation), data.get(second_start_date, PRED_START_DATE, nation)]

        # "state" is still US...
        if state=="US":
            # Override previously initialized train_data and full_data with data including resurged start date.
            resurge_start_date = "2020-09-15"
            train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, resurge_start_date, nation), data.get(resurge_start_date, args.END_DATE, nation)]
            full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, resurge_start_date, nation), data.get(resurge_start_date, PRED_START_DATE, nation)]

        # Get a value and decay for US.
        a, decay = pdata.FR_nation[nation]
        reopen_flag = True 

        # Open json file with name based on validation end date and end date arguments given to the script.
        json_file_name = "val_results_world/test" + "JHU" + "_val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
        with open(json_file_name, 'r') as f:           
            NE0_region = json.load(f)

        # Get N and E_0 from opened file and set pop_in to 1/400.
        N, E_0 = NE0_region[state][0], NE0_region[state][1]
        pop_in = 1/400

    # Going through invidual states.
    else:
        # Get data for US states from JHU.
        data = JHU_US(level="states")

        # Set start date and middle dates list.
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state),100)
        mid_dates = pdata.mid_dates_state

        # If state is in middle dates list.
        if state in mid_dates.keys():
            # Set its second start date.
            second_start_date = mid_dates[state]
            reopen_flag = True

        # Else set 2020-08-30 as second start date.
        else:
            second_start_date = "2020-08-30" 
            reopen_flag = False

        # Get data from state for training and full result.
        train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
        full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, PRED_START_DATE, state)]

        # If state is in middle dates list.
        if state in pdata.mid_dates_state.keys():
            # Set resurged start dates.
            resurge_start_date = pdata.mid_dates_state_resurge[state] if state in pdata.mid_dates_state_resurge.keys() else "2020-09-15"

            # Override previously initialized train_data and full_data with data including resurged start date.
            train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
             data.get(resurge_start_date, args.END_DATE, state)]
            full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
             data.get(resurge_start_date, PRED_START_DATE, state)]

        # If state is in decay_state.
        if state in pdata.decay_state.keys():
            # Set its a value and decay from decay_state.
            a, decay = pdata.decay_state[state][0], pdata.decay_state[state][1]

        # Else set pretermined a value and decay.
        else:
            a, decay = 0.7, 0.3

    	# Open json file with name based on validation end date and end date arguments given to the script.
        json_file_name = "val_results_state/" + "JHU" + "_val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
        with open(json_file_name, 'r') as f:           
            NE0_region = json.load(f)

        # Get N and E_0 from opened file and set pop_in to 1/400.
        N, E_0 = NE0_region[state][0], NE0_region[state][1]
        pop_in = 1/400

        # If state is California or New York, set pop_in to 0.01.
        if state == "California" or state == "New York":
            pop_in = 0.01

    # Get last confirmed cases and last fatalities from training data.
    last_confirm, last_fatality = train_data[-1][0], train_data[-1][1]

    # Calculate daily and mean values from confirmed cases.
    daily_confirm = np.diff(last_confirm)
    mean_increase = np.median(daily_confirm[-7:] - daily_confirm[-14:-7])/2 + np.median(daily_confirm[-14:-7] - daily_confirm[-21:-14])/2

    # If state or nation was not in middle dates list.
    if not reopen_flag:
        # Set pop_in value depending on the amount of daily confirmed cases.
        if np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1:
            pop_in = 1/5000

        elif mean_increase < np.mean(daily_confirm[-7:])/40:
            pop_in = 1/5000

        elif mean_increase > np.mean(daily_confirm[-7:])/10 and np.mean(daily_confirm[-7:])>60:
            pop_in = 1/500

        else:
            pop_in = 1/1000

    # Set pop_in to 1/500 if state was in middle dates list and exceeds given requirements.
    if reopen_flag and (np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1):
        pop_in = 1/500

    # If state was US, set pop_in to 1/400.
    if state == "US":
        pop_in = 1/400

    # Print state, training end date, prediction start date and middle date.
    print("state: ", state, " training end date: ", args.END_DATE, " prediction start date: ", PRED_START_DATE, " mid date: ", second_start_date)  

    # Initialize new_sus. Useless if.
    new_sus = 0 if reopen_flag else 0

    # Add bias for states. If not state, set bias to 0.02.
    if state != "US":
        # State bias.
        bias = 0.025 if reopen_flag or (state=="Louisiana" or state=="Washington" or state == "North Carolina" or state == "Mississippi") else 0.005
        if state == "Arizona" or state == "Alabama" or state == "Florida" or state=="Indiana" or state=="Wisconsin" or state == "Hawaii" or state == "California" or state=="Texas" or state=="Illinois":
            bias = 0.01
        if state == "Arkansas" or state == "Iowa" or state == "Minnesota" or state == "Louisiana" \
         or state == "Nevada" or state == "Kansas" or state=="Kentucky" or state == "Tennessee" or state == "West Virginia":
            bias = 0.05
    
    # US bias.
    else:
        bias = 0.02

    # Get confimed cases and fatalities from training data.
    data_confirm, data_fatality = train_data[0][0], train_data[0][1]

    # Create model using Learner_SuEIR.
    model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay,bias=bias)
    init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]
    
    # Get params_all list and loss_all.
    params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)

    # Get true loss and true prediction.
    loss_true = [NE0_region[state][-2], NE0_region[state][-1]]
    pred_true = rolling_prediction(model, init, params_all, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

    confirm = full_data[0][0][0:-1].tolist() + full_data[1][0][0:-1].tolist() + pred_true[0].tolist()

    # Print state, training loss, maximum death cases, maximum confirmed cases and popin.
    print ("region: ", state, " training loss: ",  \
        loss_all, loss_true, " maximum death cases: ", int(pred_true[1][-1]), " maximum confirmed cases: ", int(pred_true[0][-1]), "popin", pop_in) 

    # Get intervals for parameters.
    interval1 = np.linspace(0.7, 1.0, num=12)
    interval2 = np.linspace(1.3, 1.0, num=12)
    
    # Get parameters from trained model.
    params = params_all[1] if len(params_all)==2 else params_all[2]

    # Initialize parameters and prediction results.
    A_inv, I_inv, R_inv=[],[],[]
    prediction_list=[]

    # Iterate over parameter intervals.
    for index_cof in range(12):
        # Get beta, gamma, sigma and mu lists.
        beta_list = np.asarray([interval1[index_cof], interval2[index_cof]])*params[0]
        gamma_list = np.asarray([interval1[index_cof], interval2[index_cof]])*params[1]
        sigma_list = np.asarray([interval1[index_cof], interval2[index_cof]])*params[2]
        mu_list = np.asarray([interval1[index_cof], interval2[index_cof]])*params[3]

        param_list=[]
        for beta0 in beta_list:
            for gamma0 in gamma_list:
                for sigma0 in sigma_list:
                    for mu0 in mu_list: 
                        # Create temporary parameter without if there are 2 parameters in params_all list.
                        if len(params_all) == 2:
                            temp_param = [params_all[0]] + [np.asarray([beta0, gamma0, sigma0, mu0])]

                        # Else create temporary parameter with params_all[1] added.
                        else:
                            temp_param = [params_all[0]] + [params_all[1]] + [np.asarray([beta0, gamma0, sigma0, mu0])]
                        
                        # Create temporary prediction using rolling_prediction.
                        temp_pred = rolling_prediction(model, init, temp_param, full_data, new_sus, pred_range=100, pop_in=pop_in, daily_smooth=True)

                        # Add temporary prediction to prediction list.
                        prediction_list += [temp_pred]

    # Add true prediction to the prediction list.
    prediction_list += [pred_true]

    # Separate prediction components into lists for infections, recoveries and active cases.
    for _pred in prediction_list:
        I_inv += [_pred[0]]
        R_inv += [_pred[1]]
        A_inv += [_pred[2]]       

    
    # Convert lists into NumPy arrays.
    I_inv=np.asarray(I_inv)
    R_inv=np.asarray(R_inv)
    A_inv=np.asarray(A_inv)   

    # Date strings for the prediction range.
    dates = [(pd.to_datetime(PRED_START_DATE) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(prediction_range)]

    # Define quantiles for cases and deaths.
    case_quantiles = [0.025, 0.100, 0.250, 0.500 ,0.750 ,0.900 ,0.975, "NA"]
    death_quantiles = [0.010, 0.025, 0.050, 0.100 ,0.150 ,0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500  \
            ,0.550, 0.600 ,0.650 ,0.700 ,0.750, 0.800, 0.850 ,0.900 ,0.950, 0.975, 0.990, "NA"]


    # pred_start_date = PRED_START_DATE        # Possibly needs to be uncommented?

    # Iterate over death quantiles.
    for quantile in death_quantiles:
        # Select week indices corresponding to Saturdays.
        week_inds = [i for i in range(len(dates)) if dates[i] in Sat_list]

        # Get R components that are non-negative.
        R_inv_wk = np.maximum(R_inv[:,week_inds], 1e-13)

        # Week numbers.
        diff_dates = [str(i+1) + " wk ahead cum death" for i in range(len(week_inds))]

        # Get week dates.
        wk_dates = [Sat_list[i] for i in range(len(week_inds))]

        # Add prediction data.
        pred_data = {}
        pred_data["location_name"] = state
        pred_data["forecast_date"] = pred_start_date
        pred_data["target"] = diff_dates
        pred_data["target_end_date"] = wk_dates
        pred_data["location"] = state2fips(state)
        pred_data["type"] = "point" if quantile=="NA" else "quantile"
        pred_data["quantile"] = quantile
        pred_data["value"] = R_inv_wk[-1,:].tolist() if quantile=="NA" else np.percentile(R_inv_wk,quantile*100,axis=0).tolist()
        
        # Convert prediction data to DataFrame and add it to frame list.
        df = pd.DataFrame(pred_data)
        frame.append(df)

        # Prepare data for incidence deaths.
        death_lastsat = full_data[-1][1][-1]
        diffR_wk = np.zeros(R_inv_wk.shape)
        diffR_wk[:, 1:] = np.diff(R_inv_wk)
        diffR_wk[:, 0] = R_inv_wk[:,0]-death_lastsat

        # Get diffR_wk values that are non-negative.
        diffR_wk = np.maximum(diffR_wk, 0)

        # Week numbers.
        diff_dates = [str(i+1) + " wk ahead inc death" for i in range(len(week_inds))]

        # Copy prediction data and set "target" and "value".
        pred_data_inc = pred_data.copy()
        pred_data_inc["target"] = diff_dates
        pred_data_inc["value"] = diffR_wk[-1,:].tolist() if quantile=="NA" else np.percentile(diffR_wk,quantile*100,axis=0).tolist()
        
        # Convert prediction data to DataFrame and add it to frame list.
        df = pd.DataFrame(pred_data_inc)
        frame.append(df)

    # Iterate over case quantiles.
    for quantile in case_quantiles:
        # Select week indices corresponding to Saturdays.
        week_inds = [i for i in range(len(dates)) if dates[i] in Sat_list and i<62]

        # Get I components that are non-negative.
        I_inv_wk = I_inv[:,week_inds]

        # Week numbers.
        wk_dates = [Sat_list[i] for i in range(len(week_inds))]


        death_lastsat = full_data[-1][0][-1]

        # Calculate differences in infected cases for each week. 
        diffI_wk = np.zeros(I_inv_wk.shape)
        diffI_wk[:, 1:] = np.diff(I_inv_wk)
        diffI_wk[:, 0] = I_inv_wk[:,0] - death_lastsat

        # Week numbers.
        diff_dates = [str(i+1) + " wk ahead inc case" for i in range(len(week_inds))]

        # Add prediction data.
        pred_data_case = {}
        pred_data_case["location_name"] = state
        pred_data_case["forecast_date"] = pred_start_date
        pred_data_case["target"] = diff_dates
        pred_data_case["target_end_date"] = wk_dates
        pred_data_case["location"] = state2fips(state)
        pred_data_case["type"] = "point" if quantile=="NA" else "quantile"
        pred_data_case["quantile"] = quantile
        pred_data_case["value"] = diffI_wk[-1,:].tolist() if quantile=="NA" else np.percentile(diffI_wk,quantile*100,axis=0).tolist()
        
        # Convert prediction data to DataFrame and add it to frame list.
        df = pd.DataFrame(pred_data_case)
        frame.append(df)

# Combine all dataframes from frame list to a single DataFrame.
results = pd.concat(frame).reset_index(drop=True)

# Convert result DataFrame to CSV file.
results.to_csv(write_file_name, index=False)