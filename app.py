from flask import Flask, render_template, request
import datetime
import numpy as np
import pandas as pd
import json
app = Flask(__name__)


@app.route('/',methods = ['POST', 'GET'])
def index():
    states_list = ['andhra pradesh', 'assam', 'bihar', 'chhattisgarh', 'gujarat', 'karnataka', 'madhya pradesh', 'maharashtra', 'odisha', 'rajasthan', 'tamil nadu', 'telangana', 'uttar pradesh', 'west bengal']
    img_filenames = [
    'Black Gram.jpeg',
    'Dry chillies.jpeg', 'Gram.jpeg', 'Horse-gram.jpeg',
    'Jowar.jpeg', 'Linseed.jpeg', 'Maize.jpeg','Mung beans.jpeg', 
    'Pigeon peas(Toor Dal).jpeg', 'Ragi.jpeg', 'Rice.jpeg',
    'Sesame Seed.jpeg','Sunflower.jpeg'
    ]
    img_filenames_tab1 = img_filenames[:8]
    img_filenames_tab2 = img_filenames[8:] 
    if request.method == 'GET':

    # Split the list into two arrays of 8 each
        getReturnDict = {
            'states':states_list,
            'img_filenames_tab1':img_filenames_tab1,
            'img_filenames_tab2':img_filenames_tab2
        }
        return render_template('index.html',**getReturnDict)
    else:
        states_list = ['andhra pradesh', 'assam', 'bihar', 'chhattisgarh', 'gujarat', 'karnataka', 'madhya pradesh', 'maharashtra', 'odisha', 'rajasthan', 'tamil nadu', 'telangana', 'uttar pradesh', 'west bengal'] 
        rainfall_inp = float(request.form.get('rainfall'))
        N_inp = float(request.form.get('n'))
        P_inp = float(request.form.get('p'))
        K_inp = float(request.form.get('k'))
        pH_inp = float(request.form.get('ph'))
        temperature_inp = float(request.form.get('temperature'))
        state_inp = str(request.form.get('state_name'))
        date_inp = datetime.datetime.strptime(str(request.form.get('date')), "%Y-%m-%d")
        month_inp = int(date_inp.month)
        #Load Data Used for best model
        df_Model3_Data = pd.read_csv('Crop_Yield_Combined_Model3_NoOutl.csv')
        #extract feature names 
        featureColumnNames = df_Model3_Data.columns[:-1]
        #create new Datafram with just features;ready to input prediction data
        df_Model3_ToUse = pd.DataFrame(columns=featureColumnNames)
        state_inp_lower = "State_Name_" + state_inp.lower()
        user_inp_dict = {}
        user_inp_dict['rainfall'] = rainfall_inp
        user_inp_dict['N'] = N_inp
        user_inp_dict['P'] = P_inp
        user_inp_dict['K'] = K_inp
        user_inp_dict['pH'] = pH_inp
        user_inp_dict['temperature'] = temperature_inp
        user_inp_dict[state_inp_lower] = 1

        #enter crop_type(which season crop) based on month
        enterCropTye(user_inp_dict,month_inp)
        print(user_inp_dict)
        #Enter user inputs into df_Model4_ToUse
        user_inp_df = pd.DataFrame([user_inp_dict])
        df_Model3_ToUse = pd.concat([df_Model3_ToUse,user_inp_df],ignore_index=True)
        #Enter use inputs to predict yield for each crop
        crop_columns = [col for col in df_Model3_ToUse.columns if col.startswith('Crop_') and not col.startswith('Crop_Type')]

        #add rows corresponding to number of crops
        df_Model3_ToUse = pd.concat([df_Model3_ToUse]*(len(crop_columns)),ignore_index=True)
        for index in range(len(crop_columns)):
            df_Model3_ToUse.at[index,crop_columns[index]] = 1
        df_Model3_ToUse.fillna(0,inplace=True)
        print(df_Model3_ToUse.shape)

        # Loading beta values from json file
        bestModelWeightsJson = open('model3(best).json')
        bestModelWeights = json.load(bestModelWeightsJson)
        beta_values = bestModelWeights['beta']
        # Convert lists back to pandas Series
        means_columns = bestModelWeights['means']['columns']
        means_values = bestModelWeights['means']['values']
        model3_means = pd.Series(means_values, index=means_columns)
        # Convert lists back to pandas Series
        stds_columns = bestModelWeights['stds']['columns']
        stds_values = bestModelWeights['stds']['values']
        model3_stds = pd.Series(stds_values, index=stds_columns)
        columns_to_normalize = ['rainfall','temperature','N', 'P', 'K', 'pH']

        #initiate predicting
        pred = predict_linreg(df_Model3_ToUse, beta_values,model3_means,model3_stds,columns_to_normalize)
        # Flatten the array
        pred_flat = pred.flatten()
        # Combine crop_columns and pred_flat into a dictionary
        prediction_dict = dict(zip(crop_columns, pred_flat))
        sorted_prediction = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))
        rounded_prediction = rounded_prediction = {key: max(round(value, 1), 0) for key, value in sorted_prediction.items()}
        toReturnDict = {
            'states':states_list,
            'img_filenames_tab1':img_filenames_tab1,
            'img_filenames_tab2':img_filenames_tab2,
            'rainfall': str(rainfall_inp),
            'n': str(N_inp),
            'p': str(P_inp),
            'k': str(K_inp),
            'ph': str(pH_inp),
            'date': date_inp.strftime("%Y-%m-%d"),
            'temperature': str(temperature_inp),
            'prediction_dict': rounded_prediction

        }
        return render_template('index.html',**toReturnDict)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)


#---------------------Functions start---------------------
#To get Crop Type
def enterCropTye(user_inp_dict,month_inp):
    crop_types = ['Crop_Type_kharif', 'Crop_Type_rabi', 'Crop_Type_summer', 'Crop_Type_whole year']
    if month_inp in [3,4,5]:
        user_inp_dict['Crop_Type_kharif'] = 1
        user_inp_dict['Crop_Type_whole year'] = 1
    elif month_inp in [12,1,2]:
        user_inp_dict['Crop_Type_rabi'] = 1
        user_inp_dict['Crop_Type_whole year'] = 1
    elif month_inp in [6,7,8]:
        user_inp_dict['Crop_Type_summer'] = 1
        user_inp_dict['Crop_Type_whole year'] = 1
    else:
        user_inp_dict['Crop_Type_whole year'] = 1
def calc_linreg(X, beta):
    return np.matmul(X,beta)
def normalize_z(dfin, columns_to_normalize=None, columns_means=None, columns_stds=None):
    if columns_means is None:
        columns_means = dfin.mean(axis=0)
    if columns_stds is None:
        columns_stds = dfin.std(axis=0)
    
    if columns_to_normalize is None:
        # If columns_to_normalize is not specified, normalize all columns
        columns_to_normalize = dfin.columns

    # Normalize selected columns using z-score formula
    dfout = dfin.copy()  # Create a copy to avoid modifying the original DataFrame
    dfout[columns_to_normalize] = (dfin[columns_to_normalize] - columns_means[columns_to_normalize]) / columns_stds[columns_to_normalize]
    
    return dfout, columns_means, columns_stds
def prepare_feature(df_feature):
    if isinstance(df_feature, pd.DataFrame):
        df_feature_np_array = df_feature.to_numpy()
    else:
        df_feature_np_array = df_feature
    df_feature_np_array = np.insert(df_feature_np_array,0,1,axis=1)
    return df_feature_np_array
def predict_linreg(df_feature, beta, means=None, stds=None, columns_to_normalize=None):
    if columns_to_normalize is None:
        # If columns_to_normalize is not specified, normalize all columns
        columns_to_normalize = df_feature.columns
    if means is None or stds is None:
        df_feature_normalized, means, stds = normalize_z(df_feature,columns_to_normalize)
    else:
        df_feature_normalized,_,_ = normalize_z(df_feature,columns_to_normalize,means,stds)
    df_feature_normalized = prepare_feature(df_feature_normalized)
    y_pred_array = calc_linreg(df_feature_normalized, beta)
    return y_pred_array

        #-----------------functions end---------------------
