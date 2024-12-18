'''File: logit_lmpc12_model0.py

Base model with generic attributes, except for travel time and cost, which is
alternative specific.

Mathis Magnin (EPFL Master Student)
Fri Nov 1 11:49 2024

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

# Parameters
constant_2 = Beta('constant_2', 0, None, None, 0)
constant_3 = Beta('constant_3', 0, None, None, 0)
constant_4 = Beta('constant_4', 0, None, None, 0)

beta_cost = Beta('beta_fare', 0, None, None, 0)
beta_travel_time = Beta('beta_travel_time', 0, None, None, 0)

# Utility function 
# walking
opt1_base = (
    beta_travel_time * dur_walking
)
# cycling
opt2_base = (
    constant_2
    + beta_travel_time * dur_cycling
)
# public transportation
opt3_base = (
    constant_3
    + beta_cost * cost_transit
    + beta_travel_time * dur_pt
)
# car 
opt4_base = (
    (constant_4
    + beta_cost * (cost_driving_fuel + driving_traffic_percent*cost_driving_ccharge)
    + beta_travel_time * dur_driving)
)
V_0 = {1: opt1_base, 2: opt2_base, 3: opt3_base, 4: opt4_base}

# logit model
logprob_0 = loglogit(V_0, None, chosen_alternative)
biogeme_0 = bio.BIOGEME(database, logprob_0)
biogeme_0.modelName = 'logit_lpmc12_base'
results = biogeme_0.estimate()

