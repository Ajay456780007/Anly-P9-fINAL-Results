# from Sub_Functions.Analysis import \
#     Analysis  # This function is used for the Performance and Comparative Analysis of the data
# from Sub_Functions.Read_data import Read_data  # This is the function used for Reading the data
from Sub_Functions.Plot import ALL_GRAPH_PLOT  # This is the function used foe plotting the graphs
from Sub_Functions.Evaluate import \
    open_popup  # Thi is the function used for showing poup in the begining of the program

DB = ["DB2", "DB1"]  # These are the two DB which will be passed to the entire processing in the name of the datasets

Choose = open_popup("Do you need Complete Execution:?")  # POPUP Question

if Choose == "Yes":

    for i in range(len(DB)):
        # Calling This function READ THE DATA FROM THE MULTIPLE HOSPITALS AND Ensure the data wheteher it is verified or not
        # Read_data(DB[i])
        # # Instance for created for the Analysis to perform performance and Comparative Analysis
        # TP = Analysis(DB[i])
        # Performing the Comparative Analysis
        # TP.COMP_Analysis()

        # TP.PERF_Analysis()
        # Instance created for the Graph class for plotting
        PLOT = ALL_GRAPH_PLOT()
        # This graph is used for plotting the graphs
        PLOT.GRAPH_RESULT(DB[i])
else:

    for i in range(len(DB)):
        # Instance created for the Graph class for plotting
        PLOT = ALL_GRAPH_PLOT()
        # This graph is used for plotting the graphs
        PLOT.GRAPH_RESULT(DB[i])
