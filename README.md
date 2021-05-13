![Well Maasai Tanzania](./images/well-maasai-tanzania-1500x630.jpeg)

# Tanzania Waterpoint Functionality Classification using Random Forest

Author: **Chi Bui**

## Overview 

Tanzania is in the midst of a water crisis: 4 million people in the country do not have access to a source of safe water, and 30 million people lack access to improved sanitation. People living in such circumstances, particulary women and children, usually bear the responsibility of collecting water in their communities, spending a significant amount of time traveling long distances to collect water several trips each day.

The objective of this project is to use **Random Forest**, a powerful ensemble method to perform a **ternary classification** of the functionality of the water wells in Tanzania.

## Business Problems

Almost half the population of Tanzania is without basic access to safe water. Although there are many waterpoints already established in the country, a lot of them are in need of repair while others have failed altogether. 

The model built in this project could be utilized as one of the first steps in the waterpoint functionality diagnostic process. It can assist the Tanzanian Ministry of Water on identifying pumps that are in need of repair and/or no longer functional. A better understanding of which features might be contributing to waterpoints functionality could help improve maintenance operations, and ensure that safe, clean water becomes available to more people across the country.

## Dataset

The dataset provided on https://www.drivendata.org/ by **Taarifa** and the **Tanzanian Ministry of Water**. More details on the competition could be found [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/).

Please note that for the actual competition, the use of external data is not allowed. However, I did incorporate data from 2 external sources in my own analysis project. 



## Exploratory Data Analysis (EDA)

The master training set that I would be working with contains 59,364 entries and 40 columns, with the target variable being **`status_group`**.

```
functional                 0.54304
non functional             0.38429
functional needs repair    0.07267
```

A breakdown of the `status_group` class frequencies show that this dataset has some class imbalances that would have to be addressed during modeling. Although this is a **ternary** dataset, one class `"functional"` takes up 54.30% of the whole training dataset, while `"functional needs repair"` is only 7.27%.

### Numerical Features

#### Geospatial data vs `status_group`

![Waterpoint Location with Functionality & Altitude Encoded](./images/geospatial_scatter.jpg)

We can see that waterpoints are more densely distributed in some regions that in others. 
- There seems to be a high amount of `non functional` wells in the Southeast and Northwest regions of Tanzania. 
- Also, there are some large open spaces without any waterpoints being recorded. I'm not sure if this is just a lack of data, or lack of actual wells in these areas, or a combination of both. 

In addition, visually we can also see that more of the "larger" waterpoints (meaning they're higher in altitude `gps_height`) have been recorded as `functional`. 

#### `longitude`

![Longitude Distribution](./images/longitude.jpg)

The minimum `longitude` in this dataset is 0. Realistically, longitudes for Tanzania should be in the 29-40 range, and definitely should not be lower than 28. Therefore, these 0 values in `longitude` are most likely placeholders, and should be treated as missing values later on.

#### `gps_height`

![Altitude gps_height Distribution](./images/gps_height.jpg)

0 is also the mode of `gps_height` with approx. 34.3% the dataset. `gps_height` is explained as the "Altitude of the well" and since wells do go underground, it makes sense that 1496 of the datapoints have negative `gps_height`. 

One of the biggest challenges of this project is the large amount of 0s being used as placeholder for missing data in almost every numerical features. These 0 values also take up large portions of those each column.

### Categorical Features

#### `region`

![Well Functionality Status Grouped by Region](./images/stacked_region.jpg)

- **Iringa** is on the top in terms of both number of wells and functional wells proportion.
- Largest city and former capital **Dar es Salaam** has the least amount of wells, and also a very small portion of them are identified as `functional needs repair`
- **Lindi** has more wells than **Dar es Salaam**, but yet a very high portion of them are `non functional`. 

We can see there's some sort of relationship between the region/location of the well and its functionality, which does make sense. Wells in certain locations are perhaps receiving better maintenance from the local management, or somehow constructed better by certain installer than others. 

In addition to this, one of the few things that can caused well water contamination is the polution coming from nearby large residential areas, as well as industrial factories, plants, construction sites, etc. 

#### `quantity`

![Proportion of Well Functionality Statuses Grouped by Quantity](./images/stacked_quality.jpg)

Again, similar to `quantity`, the percentage of `non functional` in the **unknown** group is a lot higher than the others.

#### `extraction_type`

![Well Functionality Status by Extraction Type Class](./images/stacked_extract.jpg)

- **rose_pump** appears to have a higher than average proportion of `functional` wells of approx. **65%**
- while **other** has a very high percentage of `non functional` waterpoints of **80.7%**

#### `waterpoint_type`

```
| waterpoint_type             |   functional |   functional needs repair |   non functional |
|:----------------------------|-------------:|--------------------------:|-----------------:|
| dam                         |     0.857143 |                 0         |         0.142857 |
| cattle trough               |     0.724138 |                 0.0172414 |         0.258621 |
| improved spring             |     0.717752 |                 0.108557  |         0.173691 |
| communal standpipe          |     0.621476 |                 0.0792187 |         0.299306 |
| hand pump                   |     0.617714 |                 0.0588    |         0.323486 |
| communal standpipe multiple |     0.36629  |                 0.106247  |         0.527464 |
| other                       |     0.131723 |                 0.0459464 |         0.82233  |
```

![Well Functionality Status by Waterpoint Type](./images/stacked_wpt_type.jpg)

- In this case, looking at the normalized version of the contingency table alone could be misleading. **dam** and **cattle trough** have a very high percentage of `functional` wells; however, in the grand scheme of things, there're not really that many wells in these 2 groups altogether.
- What I find interesting here is the proportions of `non functionnal` in the 2 groups **other** (82%) and **communal standpipe multiple** (52%) which are a lot higher than the population's percentage of `non functional` (38%).

#### `source`

| source               |   functional |   functional needs repair |   non functional |
|:---------------------|-------------:|--------------------------:|-----------------:|
| spring               |     0.622268 |                0.0749706  |         0.302761 |
| rainwater harvesting |     0.604012 |                0.136502   |         0.259485 |
| other                |     0.597156 |                0.00473934 |         0.398104 |
| hand dtw             |     0.56865  |                0.0194508  |         0.411899 |
| river                |     0.56856  |                0.127029   |         0.304411 |
| shallow well         |     0.494554 |                0.0568419  |         0.448604 |
| machine dbh          |     0.489385 |                0.0443581  |         0.466257 |
| unknown              |     0.484848 |                0.0606061  |         0.454545 |
| dam                  |     0.38626  |                0.0366412  |         0.577099 |
| lake                 |     0.21232  |                0.0157274  |         0.771953 |

![Well Functionality Status by Water Source](./images/stacked_source.jpg)

- There are a significant amount of wells in the **shallow well**; and the percentage of `non functional` in this group is higher than average (44.8% compared to 38%).
- **rainwater harvesting** does not have too many wells, yet the percentage of `functional` well here is relatively high (60% compare to the average of 53%).

#### `date_recorded`

The proportions of well functionality in **rainwater harvesting** has led me to think that perhaps the functionality of the wells are also partially impacted by the time of the year the inspection/measurement was recorded. I'll extract the month & year from `date_recorded` to create new columns to see if there's any potential relationship between the record time and outcome.

| month_recorded   |   functional |   functional needs repair |   non functional |
|:-----------------|-------------:|--------------------------:|-----------------:|
| Jun              |     0.780347 |                 0.0260116 |         0.193642 |
| Sep              |     0.652439 |                 0.0304878 |         0.317073 |
| Mar              |     0.616525 |                 0.0501784 |         0.333296 |
| May              |     0.60119  |                 0.0297619 |         0.369048 |
| Dec              |     0.587762 |                 0.0338164 |         0.378422 |
| Feb              |     0.551997 |                 0.0752723 |         0.372731 |
| Apr              |     0.516373 |                 0.110327  |         0.3733   |
| Aug              |     0.511303 |                 0.0752528 |         0.413444 |
| Oct              |     0.509351 |                 0.0577558 |         0.432893 |
| Jul              |     0.501084 |                 0.0793697 |         0.419546 |
| Nov              |     0.494065 |                 0.0519288 |         0.454006 |
| Jan              |     0.410487 |                 0.126909  |         0.462604 |

![Proportion of Well Functionality Status Grouped by Month of Recording](./images/stacked_month.jpg)

| month_recorded   |        dry |   enough |   insufficient |   seasonal |    unknown |
|:-----------------|-----------:|---------:|---------------:|-----------:|-----------:|
| Sep              | 0.20122    | 0.554878 |       0.210366 |  0.0304878 | 0.00304878 |
| Dec              | 0.198068   | 0.58132  |       0.183575 |  0.0354267 | 0.00161031 |
| Jan              | 0.170997   | 0.534719 |       0.193985 |  0.0706975 | 0.0296016  |
| Jul              | 0.136186   | 0.555009 |       0.24794  |  0.0481423 | 0.0127223  |
| Nov              | 0.10905    | 0.547478 |       0.217359 |  0.10089   | 0.0252226  |
| Aug              | 0.106782   | 0.531826 |       0.262344 |  0.0832838 | 0.0157644  |
| Feb              | 0.10004    | 0.574264 |       0.263171 |  0.0523598 | 0.0101654  |
| Oct              | 0.0916758  | 0.566373 |       0.217088 |  0.112028  | 0.0128346  |
| Apr              | 0.0889169  | 0.551134 |       0.218136 |  0.137028  | 0.00478589 |
| Mar              | 0.0788359  | 0.556813 |       0.296833 |  0.0559768 | 0.011541   |
| May              | 0.0297619  | 0.675595 |       0.27381  |  0.0178571 | 0.00297619 |
| Jun              | 0.00867052 | 0.728324 |       0.248555 |  0.0115607 | 0.00289017 |

I think it's not coincident that **June** appears to record the highest amount of `functional` wells, and lowest amount of **dry** wells. Although Tanzania is a very big country, and the climate does vary considerably within it, generally, the main long rainy season lasts during **March**, **April**, and **May**.  

Overall, this EDA has shown that there are certainly values in the **`unknown`**s and **`other`**s of this dataset. Therefore, for all the categorical variables with missing values to be included in modeling, I will impute them with the string **`"NaN"`** instead of dropping them.



## Modeling

### Metrics

The main metric that I would be using to access my models' performance here is **Accuracy Score**. However, for this specific problem, we would also want to be able to identify non-operational waterpoints as well as those that are in need of repair early on, to help the Tanzanian Ministry of Water dispense resources and labors accordingly. Therefore, I would also be looking at **Recall Score** particularly of the 2 classes `non functional` and `functional needs repair` in evaluating my models.

### Train Test Split

The goal here is to have 3 sets of data:
1. **Training** `X_tt` (33,392 datapoints) - used for training 
2. **Validation** `X_val` (11,131 datapoints) - used for model selection/tuning/tweaking
3. **Testing** `X_test` (14,841 datapoints) - used for testing

### Building Preprocessing Pipeline

This dataset contains groups of features that indicate similar features of the wells, such as:
- `extraction_type`, `extraction_type_group`, and `extraction_type_class`
- `water_quality` and `quality_group`
- `quantity` and `quantity_group` (although further investigating on these 2 features shows that they actually store the exact same information)
- `source`, `source_type`, `source_type_class`
- `waterpoint_type` and `waterpoint_type_group`
- `payment` and `payment_type`

Focusing on the more specific categories would lead to the models containing too many features and slow down runtimes; while these more specific variables might not necessarily offer more values than the general ones. Therefore, for this model, I would only focus on the most generalized features in each of these group instead of including all of them. For example:
- Among `extraction_type`, `extraction_type_group`, and `extraction_type_class`, I would go with `extraction_type_class`
- Between `waterpoint_type` and `waterpoint_type_group`, I would choose `waterpoint_type_group`

### 1. Add Extra Features

#### 1.1. Extract Month/Year and Age Recorded Information from `date_recorded`

The purpose of this step is to extracts information from datetime column to create new features:

- `month_recorded`: first 3 character of name of the month recorded
- `year_recorded`: four-digit year
- `age_recorded`: difference between construction year and year recorded

#### 1.2. Adding Extra Features Regarding Big Cities

Although the competition itself does not allow the use of external data, I think there are meaningful ways we can incorporate external data into improving the overall predictive power and interpretability of the models here, especially when there is such a large amount of missing information in this dataset. 

Therefore, I would use external data my own study, and then remove this transformation from the pipeline when making predictions on testing data for the competition. 

The outcome of this step is to create 3 additional features:
- `nearest_big_city_name`: name of the nearest city with population over 100,000
- `nearest_big_city_distance`: distance to nearest city with population over 100,000
- `nearest_big_city_population`: population of nearest big city

### 2. Preprocessing

#### 2.1. Fill in missing values with string `"NaN"` in some Categorical & Boolean Variables

For 3 columns: `public_meeting`, `scheme_management`, `permit`

#### 2.2. Impute Longitude

The purpose of this step is to replace 0 values in longitudes with the aggregated means by region using `region_code`. The main reason I am using `region_code` as the groupby feature is purely because there are more unique values in `region_code` than in `region`.

#### 2.3. Binning `funder` & `installer` into 2 groups **major** and **minor**

Each of these 2 variables contains over 1,000 possible categories, which means that if I were to include them all as is into the model, I would end up with over 4,000 features after One Hot Encoding. One way to deal with high-cardinality in these 2 variables is by binning them into bigger groups:

- **major** - top 100 funders or installers - responsible for around 80% of the water wells in the dataset used for fitting
- **minor** - any entities that are not in the top 100 list

#### 2.4. Scaling Numerical Features

The numerical features I would be including in my models are:
`amount_tsh`, `gps_height`, `num_private`, `construction_year`, `year_recorded`, `age_recorded`, `population`, `nearest_big_city_population`, `longitude`, `latitude`, `nearest_big_city_distance`

- Scaling Longitude and Latitude are especially important here because they're on different scales.
- From experiments with different scalers, `MinMaxScaler()` was found to be the one that yields better results compared to other types of scaling (including `StandardScaler()`, `RobustScaler()`, and `PowerTransformer()`).

#### 2.5. One Hot Encoding Categorical Features

The categorical features I would be including in my models are:
`funder`, `installer`, `basin`, `region`, `public_meeting`, `scheme_management`, `permit`, `extraction_type_class`, `management_group`, `payment_type`, `quality_group`, `quantity`, `source_class`, `waterpoint_type_group`, `nearest_big_city_name`, `month_recorded`



