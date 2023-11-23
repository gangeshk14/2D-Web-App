# Design Thinking Project Task 3 Bonus Part

## Members
| Name                          | 
| ----------------------------- | 
| Gangesh Kumar (1007181)       |              
| Zhuang Yang Kun (1006933)     |             
| Christabel Lim (1007042)      |              
| Ernest Tan Wei Yan (1006883)  |              
| Ng Wan Qing (1007033)         |

## Overview
A user-friendly website for farmers to get predictions on crop yields based on factors such as Soil conditions(Nitrogen, Phosphorous, Potassium), Rainfall, Temperature, Season(Based on Date) and Location. This model assumes the farmers have easy access to technology that brings them the above information easily.

## To run:
### Create Virtual Environment 
```shell
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
flask run
```
## Using Web App
- Predict Section predicts yield for each crop. Crop sections contains all crops offered by the model
1. Scroll/Click Predict.
2. Type in Values.
- Example Values:
- Rainfall: 100 - 1000(mm)
- Nitrogen: 50-120
- Phosphorous:15-60
- Potassium:20-80
- pH: 4.8-7
- Date: (Select Any)
- Temperature: 25-34
- Location: (Select Any)

Afer predict is clicked, a bar graph containing information about yield to each crop will be shown.

## Libraries

```python
from flask import Flask, render_template, request
import datetime
import numpy as np
import pandas as pd
import json
```
<br/>

