<h2><a style="color:#FF0000;"><b>Installation - How to install</b></a></h2></br>
The easiest way to install  maclearn is using pip. Open your teminal and enter - <br/><code><i>pip install maclearn</i></code></br></br></br>
</br>
<h2><a style="color:#FF0000;"><b>Instruction - How to use maclearn</b></h2></a></br>
  Using <b>maclearn</b> in your project is simple enough. Currently, you'll find only 2 algorithms in this module. We are working to develop more.</br></br></br>

  <h3><b>Reading Excel Files (.csv/.xlsx)</b></h3>
  Reading excel files using <b>maclearn</b> is quite easy.</br>
  Let's say we want to read this file (PizzaPrice.csv) -
  <a href="lr.csv" name="MicrosoftExcelButton"></a>
<table>
  <thead><tr>
    <th>Size</th>
    <th>&nbsp;Price</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>&nbsp;6</td>
      <td>&nbsp;350</td>
    </tr>
    <tr>
      <td>&nbsp;8</td>
      <td>&nbsp;775</td>
    </tr>
    <tr>
      <td>&nbsp;12</td>
      <td>&nbsp;1150</td>
    </tr>
    <tr>
      <td>&nbsp;14</td>
      <td>&nbsp;1395</td>
    </tr>
    <tr>
      <td>&nbsp;17</td>
      <td>&nbsp;1648</td>
    </tr>
    <tr>
      <td>&nbsp;18</td>
      <td>&nbsp;1675</td>
    </tr>
  </tbody>
</table>
  </br>
  The file may be in .xlsx or .csv format but the extension doesn't matter. Here's how to read it -</br>
  For .csv format -
  <code>

            import maclearn # importing the module
            
            filename = "PizzaPrice.csv"  # Initialize the directory
            file = maclearn.csv(filename)  # Create a csv object
            data = file.data()  # Read the data and store it as a list
            print(data)  # Print the data
  </code></br>
  <pre>
            Result - [[6, 8, 12, 14, 17, 18], [350, 775, 1150, 1395, 1648, 1675]]
  </pre></br></br>

  For .xlsx format -
  <pre>

            import maclearn # importing the module
            
            filename = "PizzaPrice.xlsx"  # Initialize the directory
            file = maclearn.xlsx(filename)  # Create a xlsx object
            data = file.data()  # Read the data and store it as a list
            print(data)  # Print the data
  </pre></br>
  <pre>
            Result - [[6, 8, 12, 14, 17, 18], [350, 775, 1150, 1395, 1648, 1675]]
  </pre></br></br>
  If you want, you can put <code>file.data()</code> in the fit method of GaussianNaiveBayes or LinearRegression in order to fit the data to your model from an excel file


  </br></br></br>
  <h3><b>Gaussian Naive Bayes Classifier</b></h3>
  First create the object GaussianNaiveBayes()...</br>
  Then call the fit() method to fit the data to your model.</br>
  Note: Your data must be in this format - [Feature1, Feature2, Feature3, Label]...</br>
  Here, you can use as many features as you want...</br>
  After that, simply call the predict(*Features) method to predict the label....</br>
  You can calculate the <i>F1 Measure</i> for your model by simply calling the f1_measure() method.</br>
  You can also calculate the <i>specificity</i> for your model by simply calling the specificity() method.</br>
  <i>Example:</i>
  <code>

            import maclearn # Importing the module
            # Features
            Outlook = ["Sunny", "Sunny", "Cloudy", "Sunny", "Cloudy", "Cloudy", "Sunny"]
            Temperature = ["Cold", "Warm", "Warm", "Warm", "Cold", "Cold", "Cold"]
            Routine = ["Indoor", "Outdoor", "Indoor", "Indoor", "Indoor", "Outdoor", "Outdoor", "Outdoor"]
            WearCoat = ["No", "No", "No", "No", "Yes", "Yes", "Yes"] # Labels
            
            # You can also import the data from an excel file
            Features = [Outlook, Temperature, Routine, WearCoat] # Putting all features in a list
            
            model = maclearn.GaussianNaiveBayes()  # Creating the object
            model.fit(Features) # Fit the data into the model

            # Should we wear coat, if our feature set is cloudy, warm and outdoor?
            print(model.predict("Cloudy", "Warm", "Outdoor")) # Predict the label
            f1_value = model.f1_measure() # Calculate the F1 measure value for the model
            specificity = model.specificity() # Calculate the specificity for the model
            print(f1_value * 100, "%") # print F1 measure value in percentage
            print(specificity * 100, "%") # print specificity in percentage

  </code>
  </br></br></br>


  <h3><b>Linear Regression with numerical analysis</b></h3>
  First create the object - LinearRegression()...</br>
  Then call the fit(x, y) method to fit the data to your model.</br>
  This model is for <i>single variable linear regression</i>. So your data must be in this format - X=[2,3,4], Y=[5,6,7]</br>
  Then, just call the predict(x) method with the argument of x value to predict the y value...</br>
  You can also calculate the <i>R-Squared value</i> by simply calling r_squared() method.</br>
  <i>Example:</i>
  <code>

            import maclearn # Importing the module

            # You can also import the data from an excel file
            Size = [6, 8, 12, 14, 18] # Size of Pizza in inches (Independent Variable)
            Price = [350, 775, 1150, 1395, 1675] # Price of Pizza in Taka (Dependent Variable)
                        
            model = maclearn.LinearRegression()  # Creating the object
            model.fit(Size, Price) # Fit the data into the model

            predicted = model.predict(17) # Predicted the price of 17 inches pizza
            accuracy = model.r_squared() # Calculate the R-Squared value for the model

            print(predicted) # Print the predicted value
            print(accuracy * 100, "%") # Print the R-Squared value in percenntage
  </code>
  </br></br></br>

<h2><a style="color:#FF0000;"><b>Lisence</b></h2></a></br>
This module is completely free and open source. You can use and modify to improve the module if you want ;)
</br>Any suggestion will be highly appriciated. Gmail - <code>neural.gen.official@gmail.com</code>
</br></br>
Created by <b>Sajedur Rahman Fiad</b>