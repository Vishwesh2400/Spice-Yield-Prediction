from flask import Flask, render_template,request
#from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from datetime import datetime
import crops
import random
import pickle
from sklearn.ensemble import RandomForestRegressor

# import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

#cors = CORS(app, resources={r"/ticker": {"origins": "http://localhost:port"}})

commodity_dict = {
    "black_pepper": ["black_pepper","Black_Pepper.csv","black_pepper.pkl"],
    "cardamom": ["cardamom","Cardamom.csv","cardamom.pkl"],
    "chilly": ["chilly","Chilly.csv","chilly.pkl"],
    "coriander": ["coriander","Coriander.csv","coriander.pkl"],
    "cumin": ["cumin","Cummin.csv","cumin.pkl"],
    "garlic": ["garlic","Garlic.csv","garlic.pkl"],
    "ginger": ["ginger","Ginger.csv","ginger.pkl"],
    "rapeseed_mustard": ["rapeseed_mustard","Rape Seed & Mustard.csv","rapeseed_mustard.pkl"],
    "turmeric": ["turmeric","Turmeric.csv","turmeric.pkl"]
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
base = {
    "Black_pepper": 899,
    "Cardamom": 1600,
    "Chilly": 80,
    "Coriander": 88,
    "Cumin": 220,
    "Garlic": 150,
    "Ginger": 70,
    "Rapeseed_mustard": 1399,
    "Turmeric": 350

}
commodity_list = []


class Commodity:

    def __init__(self, cn,csv_name,model_file):
        self.cropname = cn
        self.name = csv_name
        self.dataset = pd.read_csv(csv_name)
        self.model = model_file
        self.X = self.dataset.iloc[:, :-1].values
        self.Y = self.dataset.iloc[:, 3].values

        #from sklearn.model_selection import train_test_split
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

        # Fitting decision tree regression to dataset
        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7,18)
        #self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor = pickle.load(open(self.model,'rb'))
        self.regressor.fit(self.X, self.Y)
        #y_pred_tree = self.regressor.predict(X_test)
        # fsa=np.array([float(1),2019,45]).reshape(1,3)
        # fask=regressor_tree.predict(fsa)

    def getPredictedValue(self, value):
        if value[1]>=2019:
            fsa = np.array(value).reshape(1, 3)
            print(" ",self.regressor.predict(fsa))
            return self.regressor.predict(fsa)[0]
        else:
            c=self.X[:,0:2]
            x=[]
            for i in c:
                x.append(i.tolist())
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(0,len(x)):
                if x[i]==fsa:
                    ind=i
                    break
            #print(index, " ",ind)
            #print(x[ind])
            #print(self.Y[i])
            return self.Y[i]

    def getCropName(self):
        a = self.name.split('.')
        return a[0]


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods=['post'])
def getvalue():
    state = request.form['state_name']
    state = state.lower()
    print(state)
    district = request.form['district_name']
    district = district.lower()
    print(district)
    crop = request.form['crop']
    crop = crop.lower()
    print(crop)
    season = request.form['season']
    season = season.lower()

    print(season)
    area = request.form['area']
    area_float = float(area)
    print(area)
    year = request.form['year']
    year_int = int(year)
    print(year_int)
    import pandas as pd
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    from sklearn.ensemble import RandomForestRegressor
    import os
    from sklearn.metrics import r2_score
    os.chdir(r"D:\6th SEM\TE ProJ\Final Project")
    crop_data = pd.read_csv("spice_moisture_final.csv")
    
    
    crop_data = crop_data.dropna()
    crop_data['State_Name'] = crop_data['State_Name'].str.rstrip()
    crop_data['Season'] = crop_data['Season'].str.rstrip()
    a = crop_data[crop_data['State_Name'] == state]
    b = a[a['District_Name'] == district]
    c = b[b['Season'] == season]
    f = c[c['Crop'] == crop]['Crop_Year']
    print(f)
    x = c[c['Crop'] == crop]['Area']
    print(x)
    y = c[c['Crop'] == crop]['Production']
    print(y)
    
    
    t = c[c['Crop'] == crop]['Temperature']
    h = c[c['Crop'] == crop]['humidity']
    sm = c[c['Crop'] == crop]['soil moisture']

    from pandas import DataFrame
    #for temperature
    for_temp = {'Year':f,'Area':x,'Temperature':t}
    temp = DataFrame(for_temp, columns=['Year', 'Area', 'Temperature'])

    x_temp = temp[['Year','Area']]
    y_temp = temp['Temperature']
    
    #for humidity
    for_humid =  {'Year':f,'Area':x,'Temperature':t,'humidity':h}
    humid = DataFrame(for_humid, columns=['Year', 'Area', 'Temperature','humidity'])
    
    x_hum = humid[['Area','Temperature']]
    y_hum = humid[['humidity']]
    
    #for soil mositure
    
    for_soilm =  {'Year':f,'Area':x,'humidity':h,'soil moisture':sm}
    soil_m = DataFrame(for_soilm, columns=['Year','Area', 'humidity','soil moisture'])
    
    x_soilm = soil_m[['Year','Area','humidity']]
    y_soilm = soil_m[['soil moisture']]
    
    #model
    model_temp1 = RandomForestRegressor()
    model_temp1.fit(x_temp,y_temp)
    
    model_humid = RandomForestRegressor()
    model_humid.fit(x_hum,y_hum)
    
    soilm = RandomForestRegressor()
    soilm.fit(x_soilm,y_soilm)
    
    #predictions
    pred_temp = model_temp1.predict([[year_int,area_float]])

    pred_humid = model_humid.predict([[area_float,pred_temp]])

    moist_pred = soilm.predict([[year_int,area_float,pred_humid]])
    
    class StackedAveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, models):
            self.models = models

        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            self.models_ = [clone(x) for x in self.models]

            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)

            return self

        # Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)
    
    variables = {'Year': f, 'Area': x,'Temperature':t,'humidity':h,'soil moisture':sm, 'Production': y}
    final = DataFrame(variables, columns=['Year', 'Area','Temperature','humidity', 'soil moisture','Production'])
    X = final[['Year', 'Area','Temperature','humidity','soil moisture']]
    Y = final['Production']
    
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='linear', degree=10, coef0=2.5)
    
    averaged_models = StackedAveragingModels(models=(KRR, ENet))

    model = averaged_models.fit(X, Y)
    #print(pred_rain,pred_temp,pred_humid)
    #prod2 = model.predict([[year_int,area_float, pred_rain ,pred_temp , pred_humid]])
    prod2 = model.predict([[year_int,area_float,pred_temp,pred_humid,moist_pred]])
    print('Reduced Training Dataset')
    print(X)
    print(Y)
    prod = model.predict(X)
    prod2 = abs(prod2)
    print('Predicted Temperature',pred_temp)
    print('Predicted Humididty',pred_humid)
    print('Predicted soil moisture',moist_pred)
    
    
    print("Prediction is: ", prod2)
    print(prod)
    print("R2 Score",abs(r2_score(Y,prod)))
    yld = prod2 / area_float
    return render_template("index.html", pr=prod2, yl=yld)

    


@app.route('/commodity/<name>')
def crop_profile(name):
    max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
    prev_crop_values = TwelveMonthPrevious(name)
    forecast_x = [i[0] for i in forecast_crop_values]
    forecast_y = [i[1] for i in forecast_crop_values]
    previous_x = [i[0] for i in prev_crop_values]
    previous_y = [i[1] for i in prev_crop_values]
    current_price = CurrentMonth(name)
    #print(max_crop)
    #print(min_crop)
    print(current_price)
    print(forecast_crop_values)
    #print(prev_crop_values)
    #print(str(forecast_x))
    crop_data = crops.crop(name)
    context = {
        "name":name,
        "max_crop": max_crop,
        "min_crop": min_crop,
        "forecast_values": forecast_crop_values,
        "forecast_x": str(forecast_x),
        "forecast_y":forecast_y,
        "previous_values": prev_crop_values,
        "previous_x":previous_x,
        "previous_y":previous_y,
        "current_price": current_price,
        "image_url":crop_data[0],
        "prime_loc":crop_data[1],
        "type_c":crop_data[2],
        "export":crop_data[3]
    }
    return render_template('commodity.html', context=context)



def CurrentMonth(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    #commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i.cropname):
            commodity = i
            break
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    print(commodity.name)
    current_price = (base[name.capitalize()]*current_wpi)/100
    return current_price

def TwelveMonthsForecast(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    #commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i.cropname):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
    max_index = 0
    min_index = 0
    max_value = 0
    min_value = 9999
    wpis = []
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    change = []

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), y, r])
        print(commodity.name)

        if current_predict > max_value:
            max_value = current_predict
            max_index = month_with_year.index((m, y, r))
        if current_predict < min_value:
            min_value = current_predict
            min_index = month_with_year.index((m, y, r))
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    max_month, max_year, r1 = month_with_year[max_index]
    min_month, min_year, r2 = month_with_year[min_index]
    min_value = min_value * base[name.capitalize()] / 100
    max_value = max_value * base[name.capitalize()] / 100
    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])
   # print("forecasr", wpis)
    x = datetime(max_year,max_month,1)
    x = x.strftime("%b %y")
    max_crop = [x, round(max_value,2)]
    x = datetime(min_year, min_month, 1)
    x = x.strftime("%b %y")
    min_crop = [x, round(min_value,2)]

    return max_crop, min_crop, crop_price


def TwelveMonthPrevious(name):
    name = name.lower()
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    #commodity = commodity_list[0]
    wpis = []
    crop_price = []
    for i in commodity_list:
        if name == str(i.cropname):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month - i >= 1:
            month_with_year.append((current_month - i, current_year, annual_rainfall[current_month - i - 1]))
        else:
            month_with_year.append((current_month - i + 12, current_year - 1, annual_rainfall[current_month - i + 11]))

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), 2021, r])
        print(commodity.name)
        wpis.append(current_predict)

    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y,m,1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2)])
   # print("previous ", wpis)
    new_crop_price =[]
    for i in range(len(crop_price)-1,-1,-1):
        new_crop_price.append(crop_price[i])
    return new_crop_price


if __name__ == "__main__":
    commodity_list = []
    black_pepper = Commodity(commodity_dict["black_pepper"][0],commodity_dict["black_pepper"][1],commodity_dict["black_pepper"][2])
    commodity_list.append(black_pepper)
    
    cardamom = Commodity(commodity_dict["cardamom"][0],commodity_dict["cardamom"][1],commodity_dict["cardamom"][2])
    commodity_list.append(cardamom)
    
    chilly = Commodity(commodity_dict["chilly"][0],commodity_dict["chilly"][1],commodity_dict["chilly"][2])
    commodity_list.append(chilly)
    
    coriander = Commodity(commodity_dict["coriander"][0],commodity_dict["coriander"][1],commodity_dict["coriander"][2])
    commodity_list.append(coriander)
    
    cumin = Commodity(commodity_dict["cumin"][0],commodity_dict["cumin"][1],commodity_dict["cumin"][2])
    commodity_list.append(cumin)
    
    garlic = Commodity(commodity_dict["garlic"][0],commodity_dict["garlic"][1],commodity_dict["garlic"][2])
    commodity_list.append(garlic)
    
    ginger = Commodity(commodity_dict["ginger"][0],commodity_dict["ginger"][1],commodity_dict["ginger"][2])
    commodity_list.append(ginger)
    
    rapeseed_mustard = Commodity(commodity_dict["rapeseed_mustard"][0],commodity_dict["rapeseed_mustard"][1],commodity_dict["rapeseed_mustard"][2])
    commodity_list.append(rapeseed_mustard)
    
    turmeric = Commodity(commodity_dict["turmeric"][0],commodity_dict["turmeric"][1],commodity_dict["turmeric"][2])
    commodity_list.append(turmeric)
    
    app.run()





