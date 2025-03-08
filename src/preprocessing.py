## Preprocessing

# Importing data and loading libraries
import pandas as pd
import numpy as np
kickstarter_df = pd.read_excel("/Users/jonahlee/Desktop/Personal/Repos/kickstarter-project/data/kickstarter.xlsx")

kickstarter_df.info()
# Looking for duplicates
duplicates = kickstarter_df[kickstarter_df.duplicated()]
print(duplicates)
# Looking for null values
print(kickstarter_df.isnull().sum())
# Addressing missing values in main_category
print(kickstarter_df['main_category'].unique())

# Addressing the nullness in category/main category overlap
category_to_main_category = {'Crafts': 'Crafts',
            'Dance': 'Dance',
            'Journalism': 'Journalism',
            'Photography': 'Photography',
            'Fashion': 'Fashion',
            'Music': 'Music',
            'Technology': 'Technology',
            'Film & Video': 'Film & Video'}
kickstarter_df['main_category'] = kickstarter_df['main_category'].fillna(kickstarter_df['category'].map(category_to_main_category))

## Addressing the target taking more than just successful or failed
kickstarter_df = kickstarter_df[kickstarter_df['state'].isin(['successful', 'failed'])]


## Feature Engineering
# Created a country to region map because there were way too many countries
country_region_map = {
    'US': 'North America',    'CA': 'North America', 
    'MX': 'North America',   'GT': 'North America', 
    'PA': 'North America',   'BZ': 'North America',   'SV': 'North America', 
    'CU': 'North America',   'PR': 'North America',   'JM': 'North America', 

    'GB': 'Europe',  'ES': 'Europe',   'IT': 'Europe', 
    'NL': 'Europe',   'DK': 'Europe',  'DE': 'Europe',    'SE': 'Europe', 
    'FR': 'Europe',   'IE': 'Europe',  'AT': 'Europe',   'CH': 'Europe', 
    'PL': 'Europe',   'BE': 'Europe',  'FI': 'Europe',   'EE': 'Europe', 
    'LV': 'Europe',   'LT': 'Europe',  'CZ': 'Europe',   'SK': 'Europe', 
    'BG': 'Europe',   'RO': 'Europe',  'HR': 'Europe',   'HU': 'Europe', 
    'SI': 'Europe',   'BA': 'Europe',  'CY': 'Europe',   'RS': 'Europe', 
    'UA': 'Europe',  'RU': 'Europe',  'IS': 'Europe',  'MT': 'Europe', 
    'MD': 'Europe',  'AL': 'Europe', 

    'CO': 'South America',  'VE': 'South America',  'AR': 'South America',   'BR': 'South America', 
    'PE': 'South America',   'CL': 'South America',   'BO': 'South America',  'EC': 'South America', 

    'LB': 'Asia',   'IR': 'Asia',  'SG': 'Asia',  'HK': 'Asia',  'IN': 'Asia', 
    'ID': 'Asia',   'KR': 'Asia',  'PH': 'Asia',  'CN': 'Asia', 
    'JP': 'Asia',   'TW': 'Asia',  'TH': 'Asia',  'VN': 'Asia', 
    'MY': 'Asia',  'PK': 'Asia',    'BD': 'Asia',   'AF': 'Asia', 
    'KG': 'Asia',    'MM': 'Asia',   'GE': 'Asia',    'AM': 'Asia', 
    'AZ': 'Asia',    'NP': 'Asia',   'IQ': 'Asia',    'ET': 'Asia', 

    'AU': 'Oceania',   'NZ': 'Oceania',   'FM': 'Oceania',  
    'TV': 'Oceania',   'WS': 'Oceania',  'GU': 'Oceania',   'CK': 'Oceania', 

    'MA': 'Africa',  'ZA': 'Africa', 'NG': 'Africa',   'KE': 'Africa', 
    'TZ': 'Africa',   'UG': 'Africa', 'GH': 'Africa',  'SN': 'Africa',  
    'RW': 'Africa',  'CD': 'Africa',  'ML': 'Africa',  'ET': 'Africa', 
    'SD': 'Africa',  'AO': 'Africa',  'ZM': 'Africa',   'MW': 'Africa', 'BW': 'Africa', 

    'IL': 'Middle East',  'AE': 'Middle East',  'SA': 'Middle East', 'OM': 'Middle East',  'KW': 'Middle East', 

    'TT': 'Caribbean',   'BS': 'Caribbean', 

    'SJ': 'Other', 'RE': 'Other',  'MO': 'Other', 'TV': 'Other', }
kickstarter_df['region'] = kickstarter_df['country'].map(country_region_map)
region_encoded = pd.get_dummies(kickstarter_df['region'], prefix = 'region', drop_first=True)

# Encoding category and main_category
category_encoded = pd.get_dummies(kickstarter_df['category'], prefix = 'category', drop_first=True)
main_category_encoded = pd.get_dummies(kickstarter_df['main_category'], prefix= 'main_category', drop_first=True)

# Encoding weekdays 
deadline_weekday_encoded = pd.get_dummies(kickstarter_df['deadline_weekday'], prefix = 'deadline', drop_first=True)
launched_at_weekday_encoded = pd.get_dummies(kickstarter_df['launched_at_weekday'], prefix = 'launched_at', drop_first=True)
created_at_weekday_encoded = pd.get_dummies(kickstarter_df['created_at_weekday'], prefix = 'created_at', drop_first=True)

# Creating time to deadline
kickstarter_df['project_duration'] = (kickstarter_df['deadline_yr'] - kickstarter_df['launched_at_yr']) * 365 + (kickstarter_df['deadline_month'] - kickstarter_df['launched_at_month']) * 30 + (kickstarter_df['deadline_day'] - kickstarter_df['launched_at_day'])

# Creating creation to launch
kickstarter_df['creation_to_launch'] = (kickstarter_df['launched_at_yr'] - kickstarter_df['created_at_yr']) * 365 + (kickstarter_df['launched_at_month'] - kickstarter_df['created_at_month']) * 30 + (kickstarter_df['launched_at_day'] - kickstarter_df['created_at_day'])

# Creating USD Goal because Goal amount is an important predictor but is not converted into a standardized money amount
kickstarter_df['usd_goal'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate']

# Save the preprocessed data for model building
kickstarter_df.to_csv('data/preprocessed_kickstarter.csv', index=False)

# Save the encoded features separately for easier access in model building
region_encoded.to_csv('data/region_encoded.csv', index=False)
category_encoded.to_csv('data/category_encoded.csv', index=False)
main_category_encoded.to_csv('data/main_category_encoded.csv', index=False)
deadline_weekday_encoded.to_csv('data/deadline_weekday_encoded.csv', index=False)
launched_at_weekday_encoded.to_csv('data/launched_at_weekday_encoded.csv', index=False)
created_at_weekday_encoded.to_csv('data/created_at_weekday_encoded.csv', index=False)