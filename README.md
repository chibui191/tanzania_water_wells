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
status_group	functional	functional needs repair	non functional
waterpoint_type			
dam	0.857143	0.000000	0.142857
cattle trough	0.724138	0.017241	0.258621
improved spring	0.717752	0.108557	0.173691
communal standpipe	0.621476	0.079219	0.299306
hand pump	0.617714	0.058800	0.323486
communal standpipe multiple	0.366290	0.106247	0.527464
other	0.131723	0.045946	0.822330
```

![Well Functionality Status by Waterpoint Type](./images/stacked_wpt_type.jpg)

- In this case, looking at the normalized version of the contingency table alone could be misleading. **dam** and **cattle trough** have a very high percentage of `functional` wells; however, in the grand scheme of things, there're not really that many wells in these 2 groups altogether.
- What I find interesting here is the proportions of `non functionnal` in the 2 groups **other** (82%) and **communal standpipe multiple** (52%) which are a lot higher than the population's percentage of `non functional` (38%).

