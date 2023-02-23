# Description

The MV-GUI is a user-friendly tool that has been designed 
to simplify the process of method verification. 
The user interface of MV-GUI is shown 
in figure below.

(![](/Figures/GUI_new.png)


and has a single page layout, 
making it easy to understand and use. 
The interface comprises of two tabs, 
the first being the "Select CSV file/s" tab, 
which allows the user to pick the input files. 
The input files are in .csv format and the user has 
the option to select either one or multiple files. 
These .csv files contain experiment values.

When the user selects only one file, 
the python script on the backend will automatically 
extract the information from the file, perform all 
the calculations needed for method verification, 
and display the final result on the GUI panel. 
On the other hand, if the user selects multiple files, 
the backend script will create a report for each file 
but not display it on the GUI panel. If the user wants
to view the report, they can choose the "Show report" 
option, which opens a window where they can select and 
display the previously prepared pdf report on the 
GUI panel.
