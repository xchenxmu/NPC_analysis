# NPC_analysis

## Python dependencies

## Requirements: 
1. matplotlib v3.9.0 
2. scipy v1.14.0
3. Python v3.12.9 
4. numpy v2.0.0 
5. sklearn v1.5.0 
6. pandas v2.2.2
7. pillow v10.3.0

## Installation: 
Require Python version 3.12.9 or higher.

## Parameter settings: 
In DBSCAN-based analyses, several general parameters must be predefined to achieve optimal performance:
(Please note that the following parameters are specifically tailored for the analysis of NPC images obtained using methods iU-ExM or NPC-ExM; for other imaging conditions, users are advised to adjust the parameters accordingly based on the specific dataset.)
threshold: depend on the overall signal intensity of the image, with the goal of achieving clear separation of individual NPC subunits.
eps/ min samples: for the first round of DBSCAN, the optimal parameters for identifying individual NPCs were set to eps = 15 and min samples = 25. In the second round of DBSCAN, the optimal parameters for identifying NUP96 were eps = 1 and min samples = 5, while for NUP62 and NUP153, the best-performing parameters were eps = 1 and min samples = 1.
len(v): to ensure data quality, NPCs containing a number of NUP localizations equal to or greater than a defined threshold were selected for analysis. Specifically, a threshold of 3 was applied for NUP96, while a threshold of 2 was used for NUP62 and NUP153.

## Code run: 
NPC segmentation: Code named NPC_segmentation.py
Radius measurement: Code named Radius_measurement.py
Heterogeneity index measurement: Code named HI_mesurement.py
