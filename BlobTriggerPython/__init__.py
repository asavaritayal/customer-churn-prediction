import logging
import azure.functions as func

def main(myblob):
    
    init()
    formatted_data = prep(myblob)
    prediction = predict(formatted_data)

    if(prediction == "1.0"):
        logging.info("Customer is likely to churn")
    else:
        logging.info("Customer is happy :)")


def init():
    
    from sklearn.externals import joblib
    global model

    model = joblib.load('Predict/model.pkl')


def prep(raw_data):

    from io import StringIO
    import pandas

    data = pandas.read_csv(StringIO(raw_data.read().decode("utf-8")))
    data_frame = pandas.DataFrame(data, columns=['age' , 'annualincome' , 'calldroprate' , 'callfailurerate' , 'callingnum' , 'customerid' , 'customersuspended' , 'education' , 'gender' , 'homeowner' , 'maritalstatus' , 'monthlybilledamount' , 'noadditionallines' , 'numberofcomplaints' , 'numberofmonthunpaid' , 'numdayscontractequipmentplanexpiring' , 'occupation' , 'penaltytoswitch' , 'state' , 'totalminsusedinlastmonth' , 'unpaidbalance' , 'usesinternetservice' , 'usesvoiceservice' , 'percentagecalloutsidenetwork' , 'totalcallduration' , 'avgcallduration' , 'churn' , 'year' , 'month'])
    
    return data_frame



def predict(input_df):

    import json
    import pandas

    input_df_encoded = input_df

    input_df_encoded = input_df_encoded.drop('year', 1)
    input_df_encoded = input_df_encoded.drop('month', 1)
    input_df_encoded = input_df_encoded.drop('churn', 1)
    
    columns_encoded = ['age', 'annualincome', 'calldroprate', 'callfailurerate', 'callingnum',
       'customerid', 'monthlybilledamount', 'numberofcomplaints',
       'numberofmonthunpaid', 'numdayscontractequipmentplanexpiring',
       'penaltytoswitch', 'totalminsusedinlastmonth', 'unpaidbalance',
       'percentagecalloutsidenetwork', 'totalcallduration', 'avgcallduration',
       'churn', 'customersuspended_No', 'customersuspended_Yes',
       'education_Bachelor or equivalent', 'education_High School or below',
       'education_Master or equivalent', 'education_PhD or equivalent',
       'gender_Female', 'gender_Male', 'homeowner_No', 'homeowner_Yes',
       'maritalstatus_Married', 'maritalstatus_Single', 'noadditionallines_\\N',
       'occupation_Non-technology Related Job', 'occupation_Others',
       'occupation_Technology Related Job', 'state_AK', 'state_AL', 'state_AR',
       'state_AZ', 'state_CA', 'state_CO', 'state_CT', 'state_DE', 'state_FL',
       'state_GA', 'state_HI', 'state_IA', 'state_ID', 'state_IL', 'state_IN',
       'state_KS', 'state_KY', 'state_LA', 'state_MA', 'state_MD', 'state_ME',
       'state_MI', 'state_MN', 'state_MO', 'state_MS', 'state_MT', 'state_NC',
       'state_ND', 'state_NE', 'state_NH', 'state_NJ', 'state_NM', 'state_NV',
       'state_NY', 'state_OH', 'state_OK', 'state_OR', 'state_PA', 'state_RI',
       'state_SC', 'state_SD', 'state_TN', 'state_TX', 'state_UT', 'state_VA',
       'state_VT', 'state_WA', 'state_WI', 'state_WV', 'state_WY',
       'usesinternetservice_No', 'usesinternetservice_Yes',
       'usesvoiceservice_No', 'usesvoiceservice_Yes']
    
    for column_encoded in columns_encoded:
        if not column_encoded in input_df.columns:
            input_df_encoded[column_encoded] = 0

    columns_to_encode = ['customersuspended', 'education', 'gender', 'homeowner', 'maritalstatus', 'noadditionallines', 'occupation', 'state', 'usesinternetservice', 'usesvoiceservice']
    for column_to_encode in columns_to_encode:
        dummies = pandas.get_dummies(input_df[column_to_encode])
        one_hot_col_names = []
        for col_name in list(dummies.columns):
            one_hot_col_names.append(column_to_encode + '_' + col_name)
            input_df_encoded[column_to_encode + '_' + col_name] = 1
        input_df_encoded = input_df_encoded.drop(column_to_encode, 1)
    
    pred = model.predict(input_df_encoded)
    return json.dumps(str(pred[0]))