from ML_Algorithms import GaussianNaiveBayes
from ML_Algorithms import LinearRegression

# Datasets for classification
Features = [['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy'], ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'], ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']]  # Training Dataset - [Weather, Temperature, Play Golf]
Features2 = [["Sunny", "Sunny", "Cloudy", "Sunny", "Cloudy", "Cloudy", "Sunny"], ["Cold", "Warm", "Warm", "Warm", "Cold", "Cold", "Cold"], ["Indoor", "Outdoor", "Indoor", "Indoor", "Indoor", "Outdoor", "Outdoor", "Outdoor"], ["No", "No", "No", "No", "Yes", "Yes", "Yes"]]  # Training dataset - [Outlook, Temperature, Routine, Wear Coat]
TestSet = ["Overcast", "Mild"]
TestSet2 = ["Cloudy", "Warm", "Outdoor"]


# Datasets for Regression
#  (Pizza) 
Size = [6, 8, 12, 14, 18]
Price = [350, 775, 1150, 1395, 1675]

#  y[i] = x[i] + 1
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]



print("\nGaussian Naive Bayes Classification:")
nb = GaussianNaiveBayes()
nb.fit(Features)  # Fitting the training data to our model
print("Should we play, if our feature set is", TestSet, "?")
print("Answer: ", nb.predict(*TestSet))
print(nb.f1_measure()*100, "% performance")
print("Specificity - ", nb.specificity()*100, "%")

print("\n\nLinear Regression:")

lr = LinearRegression()
lr.fit(Size, Price)
print("The price of 17 inch pizza is {:.2f}".format(lr.predict(17)))  # Price of 17 inch pizza
print(lr.r_squared()*100, "% accurate")