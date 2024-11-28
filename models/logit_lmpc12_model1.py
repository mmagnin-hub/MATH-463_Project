'''File: logit_lmpc12_model1.py

Using Model1 as the base model, include alternative-specifc parameters for the travel time attribute

Charlotte Bourgeois and Mathis Magnin (EPFL Master Students)
Tue Nov 28 20:00 2024

'''
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit

df = pd.read_csv('models/lpmc12.dat', sep='\t')
database = db.Database('lpmc12', df)

# Start with a model specification that includes alternative specific constants, 
# and cost and travel time of the different alternatives associated with generic parameters. 

# alternative specific constants
travel_mode = Variable('travel_mode')
driving_license = Variable('driving_license')
car_ownership = Variable('car_ownership')


# Travel time
dur_walking = Variable('dur_walking')

dur_cycling = Variable('dur_cycling')

dur_pt_access = Variable('dur_pt_access')
dur_pt_rail = Variable('dur_pt_rail')
dur_pt_bus = Variable('dur_pt_bus')
dur_pt_int = Variable('dur_pt_int')
pt_interchanges = Variable('pt_interchanges')

dur_driving = Variable('dur_driving')


# Costs
cost_transit = Variable('cost_transit')
bus_scale = Variable('bus_scale')

cost_driving_fuel = Variable('cost_driving_fuel')
cost_driving_ccharge = Variable('cost_driving_ccharge')
driving_traffic_percent = Variable('driving_traffic_percent')

# user travel mode
chosen_alternative = (travel_mode)

# public transport travel time
dur_pt = dur_pt_access + dur_pt_int*pt_interchanges + dur_pt_bus + dur_pt_rail

# car availability
has_a_car = car_ownership != 0


# Parameters
constant_2 = Beta('constant_2', 0, None, None, 0)
constant_3 = Beta('constant_3', 0, None, None, 0)
constant_4 = Beta('constant_4', 0, None, None, 0)

beta_cost = Beta('beta_fare', 0, None, None, 0)

# alternative specific parameter for travel time
beta_travel_time_1 = Beta('beta_travel_time_1', 0, None, None, 0)
beta_travel_time_2 = Beta('beta_travel_time_2', 0, None, None, 0)
beta_travel_time_3 = Beta('beta_travel_time_3', 0, None, None, 0)
beta_travel_time_4 = Beta('beta_travel_time_4', 0, None, None, 0)

# Utility function 
# walking
opt1_1 = (
    beta_travel_time_1 * dur_walking
)
# cycling
opt2_1 = (
    constant_2
    + beta_travel_time_2 * dur_cycling
)
# public transportation
opt3_1 = (
    constant_3
    + beta_cost * cost_transit
    + beta_travel_time_3 * dur_pt
)
# car 
opt4_1 = (
    has_a_car * (constant_4
    + beta_cost * (cost_driving_fuel + driving_traffic_percent*cost_driving_ccharge)
    + beta_travel_time_4 * dur_driving)
)
V_1 = {1: opt1_1, 2: opt2_1, 3: opt3_1, 4: opt4_1}

# logit model
logprob_1 = loglogit(V_1, None, chosen_alternative)
biogeme_1 = bio.BIOGEME(database, logprob_1)
biogeme_1.modelName = 'logit_lmpc12_model1'
results = biogeme_1.estimate()