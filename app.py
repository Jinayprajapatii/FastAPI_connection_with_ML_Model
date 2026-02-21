from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, computed_field, Field
from typing import Literal,Annotated
import pickle
import pandas as pd
import numpy as np


## import ML model

with open ('model.pkl','rb') as f:
    model=pickle.load(f)

app=FastAPI()


tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"]


## pydantic model to validate incoming data 
class UserInput(BaseModel):
    age:Annotated[int,Field(...,description='enter your age',gt=0,lt=120)]
    weight:Annotated[float,Field(...,gt=0,description='enter your weight')]
    height:Annotated[float,Field(...,gt=0,lt=2.5,description='enter your Height')]
    income_in_lpa:Annotated[float,Field(...,gt=0,description='enter your income')]
    somker: Annotated[bool,Field(...,description='are you a smoker or not')]
    city:Annotated[str,Field(...,description='enter your city where you live')]
    occupation:Annotated[Literal['retired',     'freelancer',        'student', 'government_job',
 'business_owner',     'unemployed',    'private_job'],Field(...,description='Enter your Occupation')]
    

    @computed_field
    @property
    def bmi(self) ->float:
        bmi=round(self.weight/(self.height**2))
        return bmi
    @ computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.somker and self.bmi >30:
            return 'High'
        elif self.somker and self.bmi > 27:
            return 'Medium'
        else:
            return 'Low'
        
    @computed_field
    @property
    def age_group(self) -> str:
        if self.age <25 :
            return 'Young'
        elif self.age < 45:
            return 'Adult'
        elif self.age <60:
            return 'Middle_age'
        
        return 'Senior'
    

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
@app.post('/predict')
def predict_premium(data:UserInput):
    input_df=pd.DataFrame([{
        'bmi' : data.bmi,
        'age_group' : data.age_group,
        'life_style' : data.lifestyle_risk,
        'city_tier' : data.city_tier,
        'income_lpa' : data.income_in_lpa,
        'occupation' : data.occupation
    }])
    

    prediction = model.predict(input_df)[0]
    return JSONResponse(status_code=200, content={'predicted_category' : prediction})