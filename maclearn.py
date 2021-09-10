class csv:
    def __init__(self, filename):
        import pandas as pd
        self.file = pd.read_csv(filename, sep=',', header = (0))

    def data(self):
        titles = list(self.file)
        label = titles.pop(-1)
        datas = []
        for i in titles:
            datas.append(list(self.file[i]))
        else:
            datas.append(list(self.file[label]))

        return datas


class xlsx:
    Features = []
    def __init__(self, filename):
        import pandas as pd
        self.file = pd.read_excel(filename, sheet_name='Sheet1')

    def data(self):
        self.title = list(self.file)
        datas = []
        for i in self.title:
            datas.append(list(self.file[i]))


        return datas

        


class GaussianNaiveBayes:
    Features = []
    data_fitted, predicted = False, False
    encoded = []
    meanings = []
    label_meanings = {}
    
    def fit(self, data):
        self.Features = data
        self.data_fitted = True
    
    def get_key(self, val):
        for key, value in self.label_meanings.items():
             if val == value:
                 return key
        return "key doesn't exist"
    
    def predict(self, *args):
        if self.data_fitted: # Check if fit method is called first, else, raise exception
            pass
        else:
            raise Exception("DataNotFittedError : Call fit method before calling predict method in order to fit the training data to the model")
            
        # import the necessary modules
        from sklearn import preprocessing
        from sklearn.naive_bayes import GaussianNB
        
        #declare the necessary variables
        args = list(args)
        a = []
        le = preprocessing.LabelEncoder()
        self.model = GaussianNB()
        for i in range(len(self.Features)):
                self.encoded.append(le.fit_transform(self.Features[i]))
                
        self.label = self.encoded.pop(-1)
        
        self.features=list(zip(*self.encoded))
        self.model.fit(self.features, self.label)
        
        for i in range(len(self.Features)-1):
                self.meanings.append({})
                for j in range(len(self.Features[i])):
                    self.meanings[i][(self.Features[i][j])] = self.encoded[i][j]
        for i in range(len(self.Features[-1])):
                self.label_meanings[self.Features[-1][i]] = self.label[i]
        for k in range(len(self.meanings)):
                a.append(self.meanings[k].get(args[k]))
        res = (self.model.predict([a]))
        self.predicted = True
        return self.get_key(res[0])
        
    
    def f1_measure(self):
        if self.predicted:
            pass
        else:
            raise Exception("NotPredictedError : Call predict method first to calculate the F1 Measure value.")
        original_label = list(self.label)
        predicted_label = []
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        f = [self.features]
        predicted = list(map(self.model.predict, f))
        #predicted_y = self.model.predict(self.features[0])
        for i in range(len(predicted)):  # Resizing predicted list
            predicted[i] = list(predicted[i])
            predicted_label = predicted[i]
        
        for i in range(len(original_label)):
            if predicted_label[i] == 1:
                if original_label[i] == 1:
                    true_positive += 1
                elif original_label[i] == 0:
                    false_positive += 1
                    
            if predicted_label[i] == 0:
                if original_label[i] == 1:
                    false_negative += 1
                elif original_label[i] == 0:
                    true_negative += 1
                    
        self.precision = (true_positive/(true_positive+false_positive))*100 # In percentage
        self.recall = (true_positive/(true_positive+false_negative))*100 # In percentage
        
        f1_measure = ((2*self.precision*self.recall)/(self.precision+self.recall)) / 100
        #print(f1_measure)
        return float("{:.4f}".format(f1_measure))
        
    def specificity(self):
        if self.predicted:
            pass
        else:
            raise Exception("NotPredictedError : Call predict method first to calculate the Specificity.")
        original_label = list(self.label)
        predicted_label = []
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        f = [self.features]
        predicted = list(map(self.model.predict, f))
        #predicted_y = self.model.predict(self.features[0])
        for i in range(len(predicted)):  # Resizing predicted list
            predicted[i] = list(predicted[i])
            predicted_label = predicted[i]
        
        for i in range(len(original_label)):
            if predicted_label[i] == 1:
                if original_label[i] == 1:
                    true_positive += 1
                elif original_label[i] == 0:
                    false_positive += 1
                    
            if predicted_label[i] == 0:
                if original_label[i] == 1:
                    false_negative += 1
                elif original_label[i] == 0:
                    true_negative += 1
                    
        return true_negative/(true_negative+false_positive)


class LinearRegression: 
    X, Y, slopes, c = [], [], [], 0   #  Initialize the variables
    data_fitted, predicted = False, False

    def fit(self, data):   #  Fit the data
        self.X = data[0]   #  Initialize X values
        self.Y = data[1]   #  Initialize Y values
        self.data_fitted = True

    def mul(self, x, y):   #  Multiply 2 lists
        multiplied = []
        if len(x)==len(y):
            for i in range(len(x)):
                multiplied.append(x[i]*y[i])
        return multiplied

    def square(self, x):   #  Square a list
        squared =[]
        for y in x:
            squared.append(y*y)
        return squared
    
    def mean(self, x):   #  Find mean of listed numbers
        # Mean of x, y, z is (x+y+z)/3
        mean_val = 0
        for i in x:
            mean_val += i
        return mean_val/len(x)
    
    def slope(self, x, y):   #  Find the slope
        #  m = [{mean(x)*mean(y) - mean(x*y)} / {mean(x^2) - mean(x^2)}]
        lob = (self.mean(x)*self.mean(y)) - (self.mean(self.mul(x, y)))
        hor = (self.mean(x)*self.mean(x)) - self.mean(self.square(x))
        return lob/hor
    
    def intercept(self, *args):   #  Find the Y-Intercept
        #  c = mean(y) - m*mean(x)
        args = list(args)
        y = args[-1]   #  Initialize y value
        intercept = self.mean(y)
        del args[-1]
        for i in range(len(args)):
            intercept -= self.slope(args[i], y)*self.mean(args[i])
        return intercept

    def slope_calc(self):   #  Calculate the needed slopes
        for i in range(len(self.X)):
            self.slopes.append(self.slope(self.X[i], self.Y))
        
    def predict(self, x):    #  Predict the Y values depending on X
                if self.data_fitted:
                        pass
                else:
                        raise Exception("DataNotFittedError : Call fit method before calling predict method in order to fit the training data to the model")
                m = self.slope(self.X, self.Y)   #  Calculate the slopes
                c = self.intercept(self.X, self.Y)   #  Calculate the intercepts
                self.predicted = True
                return (m*x)+c
        
    def r_squared(self):
                if self.predicted:
                        pass
                else:
                        raise Exception("NotPredictedError : Call predict method first to calculate the r-squared value.")
                original_y = self.Y
                predicted_y = list(map(self.predict, self.X))
                lob, hor = 0, 0
                for i in range(len(original_y)):
                        lob += (predicted_y[i]-self.mean(original_y))**2
                for i in range(len(original_y)):
                        hor += (original_y[i]-self.mean(original_y))**2
                        result = float("{:.4f}".format(lob/hor))
                return (result)
        
        

if __name__ == '__main__':
    Features = [["Sunny", "Sunny", "Cloudy", "Sunny", "Cloudy", "Cloudy", "Sunny"], ["Cold", "Warm", "Warm", "Warm", "Cold", "Cold", "Cold"], ["Indoor", "Outdoor", "Indoor", "Indoor", "Indoor", "Outdoor", "Outdoor", "Outdoor"], ["No", "No", "No", "No", "Yes", "Yes", "Yes"]]
    Features2 = [['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy'], ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'], ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']]

    model = LinearRegression()

    data = csv("lr.csv")
    model.fit(data.data())
    print(data.data())
    #print(model.predict(17))
