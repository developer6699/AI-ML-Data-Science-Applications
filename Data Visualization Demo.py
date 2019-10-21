#This program is developed in Jupyter notebook
#Run each part of the program in seperate cells to see the results efficiently.

#1st Part

import pycountry_convert
import pandas as pd
from pandas import DataFrame

#Reads the inputs dataset from its location
Input = pd.read_excel(r'indicator hiv estimated prevalence% 15-49.xlsx')


#Using the py_country module it assigns the names of the continent to which a country belongs to
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
continents = {
    'NA': 'North America',
    'SA': 'South America',
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'AN': 'Antartica',
    'EU': 'Europe',
}

#Adding a new column to the dataset and labeling it as Continent and assigning the continent names to the respective country/region
for column in Input[['Estimated HIV Prevalence% - (Ages 15-49)']]:
    column1 = Input[column]
    countries = column1.values
    cname=[]
    
#This part is basically assigning the names of the continents to respective country
for country in countries:
    try:
        x = continents[country_alpha2_to_continent_code(country_name_to_country_alpha2(country))]
        cname.append(x)
#If there is no certain name to a country/region we assign the value null to the respective country/region
    except KeyError:
        cname.append('null')
        continue

#Inserts the name of the new column to a certain location
Input.insert(loc=1, column='Continent', value=cname)
#print(Input) #Prints the new modified dataset

#Exporting the saved output file as both csv and xlsx file
from openpyxl.workbook import workbook
Output = Input.to_excel(r'Excel_Output.xlsx')
Output = Input.to_csv(r'Excel_Output.csv')
#-----------------------------------------------

#2nd Part

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

#Taking the newly saved file as an input
Input1 = pd.read_excel (r'Excel_Output.xlsx')
#Getting only columns from the input
Input1.columns

#In the col variable we are taking only the years 2000-2011 and slicing the rest of the years
col = Input1.iloc[: ,  24:36]
#print(col)

#Calculating the average value of the columns from the year 2000-2011
Input1['Average_value'] = col.mean(axis=1)
#print (Input1)

#Storing all the components of the respective continents to a seperate variables
Input1_NorthAmerica = Input1[Input1["Continent"] == "North America"]
#print(Input1_NorthAmerica)
Input1_SouthAmerica = Input1[Input1["Continent"] == "South America"]
Input1_Asia = Input1[Input1["Continent"] == "Asia"]
Input1_Australia = Input1[Input1["Continent"] == "Australia"]
Input1_Africa = Input1[Input1["Continent"] == "Africa"]
Input1_Europe = Input1[Input1["Continent"] == "Europe"]

#Creating an array, calculating the maximum value with respective continent and storing it in that array
Max_value=[]
Max_value.append(Input1_Asia['Average_value'].max())
Max_value.append(Input1_NorthAmerica['Average_value'].max())
Max_value.append(Input1_SouthAmerica['Average_value'].max())
Max_value.append(Input1_Australia['Average_value'].max())
Max_value.append(Input1_Africa['Average_value'].max())
Max_value.append(Input1_Europe['Average_value'].max())
#print(Max_value)

#Creating an array, calculating the maximum value with respective continent and storing it in that array
Min_value=[]
Min_value.append(Input1_Asia['Average_value'].min())
Min_value.append(Input1_NorthAmerica['Average_value'].min())
Min_value.append(Input1_SouthAmerica['Average_value'].min())
Min_value.append(Input1_Australia['Average_value'].min())
Min_value.append(Input1_Africa['Average_value'].min())
Min_value.append(Input1_Europe['Average_value'].min())
#print(Min_value)


objects = ('NorthAmerica', 'SouthAmerica', 'Asia', 'Australia', 'Africa', 'Europe')
Y_value = np.arange(len(objects))

#Plotting the bar chart for the highest of the average estimated values in the respective continent
plt.bar(Y_value,Max_value)
plt.xticks(Y_value, objects)
plt.xlabel('Continents')
plt.ylabel('Highest Average_value Estimated HIV in (2000-2011)')
plt.title('Highest Average_value Estimated HIV (Ages 15-49)')
plt.show()

#Plotting the bar chart for the lowest of the average estimated values in the respective continent
plt.bar(Y_value,Min_value)
plt.xticks(Y_value, objects)
plt.xlabel('Continents')
plt.ylabel('Lowest Average_value Estimated HIV in (2000-2011)')
plt.title('Lowest Average_value Estimated HIV (Ages 15-49)')
plt.show()


#Plotting a bar chart for the overlaid of the average estimated in the respective continent
indices = np.arange(len(Max_value))
width=0.5
plt.bar(indices, Max_value, label='Highest average Estimated HIV Prevalence')
plt.bar([i+0.25 * width for i in indices], Min_value,width = 0.75, color='r', alpha=0.75, label='Lowest average Estimated HIV Prevalence')
#Giving x and y values for plotting
plt.xticks(Y_value, objects)
plt.xlabel('Continents')
plt.ylabel('Average Estimated in (2000-2011)')
plt.title('Average Estimated - (Ages 15-49)')
plt.legend() #Tell us which color indicates what
plt.show()
#-----------------------------------------------------------------

#3rd Part

import matplotlib.pyplot as plt

#Storing only the year columns in the variable slicing the other columns
Y=Input1.columns.values[3:-1]
#Calculating the average of the values based on the continent with respective to the year
Averages = Input1.groupby(['Continent'])[Y].mean()
#print(Averages)

#Storing the average mean values of the respective continent over the years in their respective continent variable
#flatten function is used to flatten the nesting in an arbitrary list of values
NA=(Averages[Averages.index=='North America'].values).flatten()
SA=(Averages[Averages.index=='South America'].values).flatten()
OE=(Averages[Averages.index=='Australia'].values).flatten()
AN=(Averages[Averages.index=='Antartica'].values).flatten()
AS=(Averages[Averages.index=='Asia'].values).flatten()
EU=(Averages[Averages.index=='Europe'].values).flatten()
AF=(Averages[Averages.index=='Africa'].values).flatten()


#Plotting the averages of each continent respectively in seperate graphs
#Plotting a line chart North America
plt.plot(Y,NA)
plt.xlabel('Number of Years')
plt.ylabel('Averages of North America')
plt.show()

#Plotting a line chart South America
plt.plot(Y,SA)
plt.xlabel('Number of Years')
plt.ylabel('Averages of South America')
plt.show()

#Plotting a line chart Asia
plt.plot(Y,AS)
plt.xlabel('Number of Years')
plt.ylabel('Averages of Asia')
plt.show()

#Plotting a line chart Europe
plt.plot(Y,EU)
plt.xlabel('Number of Years')
plt.ylabel('Averages of Europe')
plt.show()

#Plotting a line chart Africa
plt.plot(Y,AF)
plt.xlabel('Number of Years')
plt.ylabel('Averages of Africa')
plt.show()

#Plotting a line chart Australia
plt.plot(Y,OE)
plt.xlabel('Number of Years')
plt.ylabel('Averages of Australia')
plt.show()

#Plotting a line chart Overlaid of all continents
plt.plot(Y,NA,label='North America')
plt.plot(Y,SA,label='South America')
plt.plot(Y,AS,label='Asia')
plt.plot(Y,EU,label='Europe ')
plt.plot(Y,AF,label='Africa')
plt.plot(Y,OE,label='Australia')

plt.xlabel('Number of Years')
plt.ylabel('Averages of all the continents')
plt.legend()
plt.show()
#-----------------------------------------------------------------------

#4th part
import pandas as pd

#Creating a variable to store the combined value the averages with respective to the continent
Combine = pd.merge(Input1, Averages, on='Continent')

#Purely integer-location based indexing for selection by position that is 1990
Year1 = list(Combine.iloc[:,14])
#print(Year1)
#Now combining it with the specific continent
Year1_continent = list(Combine.iloc[:,-22])

#Purely integer-location based indexing for selection by position that is 2010
Year2 = list(Combine.iloc[:,24])
#Now combining it with the specific continent
Year2_continent = list(Combine.iloc[:,-2])

#colors = ['red','green','orange','blue','brown','grey']

#The indexes indicate each of the continent, multiplied by the values to match the x with 220 and y with 220
colors = [0] * 48 + [1] * 47 + [2] * 53 + [3] * 21 + [4] * 38 + [5] * 13


#Plotting a scatter plot fo the year 1990
plt.scatter(Year1_continent, Year1, c=colors)
plt.xlabel('Average HIV estimated prevalence')
plt.ylabel('HIV estimated prevalence')
plt.title('Scatter plot for year 1990')
plt.show()

#Plotting a scatter plot fo the year 2010
plt.scatter(Year2_continent, Year2, c=colors)
plt.xlabel('Average HIV estimated prevalence')
plt.ylabel('HIV estimated prevalence')
plt.title('Scatter plot for year 2010')
plt.show()
