#Importing the pandas and assigning alias name to it
import pandas as pd


#Loading the students records from local directory by assigning a location path 
StudentRecord = pd.ExcelFile(r"C:\Users\Admin\Desktop\Record.xlsx")

#Analyzes the strings into logical syntactic components, typically in order to test conformability to a logical grammar
dataframe = StudentRecord.parse("Sheet1")


#Sorting the record alphabatically using the last name of the students
dataframe = dataframe.sort_values(by='Last Name')


#Assigning the location to the new excel file to store there
SortedStudentRecord = pd.ExcelWriter(r"C:\Users\Admin\Desktop\output.xlsx")

#Specifys on which sheet in the file to write to
dataframe.to_excel(SortedStudentRecord, sheet_name='Sheet1')

#Saving the sorted data to the new excel file
SortedStudentRecord.save()
