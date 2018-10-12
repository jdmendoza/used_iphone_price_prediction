# Used IPhone Valuation Tool

Link To Presentation: [Here](https://www.dropbox.com/s/0b8la41us2man16/iPhone_Valuation_Tool.pdf?dl=0)

**Problem**: Selling a used phone is annoying. It's hard to figure out marketprice. You don't want to overprice it and never have it be sold; nor do you want to price it low and lose out on potential profit for you next awesome phone.

**Solution**: I webscraped data for recently sold iPhones (Oct 3-6) and trained an ML algorithm to predict a suggested price.

**Results**: My algorithm predicts with an average error of $34, a great baseline!

## Steps To Rerun Project:

### Data Analysis + Modeling

1*) Run the 1_Scrape_Iphone_Data.ipynb to automatically scrape iPhone data.
2*) Run the data_files_merging/merg_dataframes.py to merge into one file (make sure to only have )
3) Run 2_Data_Visualization_Cleaning.ipynb to visualize the data and see the
4) Run 2.5_Data_Feature_Selection.ipynb to select the most import features
5) Run 3_Modeling_Data.ipynb for Model Experimentation and for final mode + predicting function.

/* Steps with an asterik can be skipped as the data is local.

### App Deployment

1) Run the app.py file
2) Navigate to http://127.0.0.1:5000 in your browser
