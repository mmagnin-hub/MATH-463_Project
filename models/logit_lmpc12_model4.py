'''File: logit_lmpc12_model4.py

Using Model1 as the base model, add cross nested

Charlotte Bourgeois and Mathis Magnin (EPFL Master Students)
Tue Nov 28 20:00 2024

'''

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable
from biogeme.biogeme_logging import get_screen_logger, INFO
from biogeme.models import logcnl
from biogeme.nests import (
    OneNestForCrossNestedLogit,
    NestsForCrossNestedLogit,
)

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

# ID
trip_id = Variable('trip_id')

# user travel mode
chosen_alternative = (travel_mode)

# public transport travel time
dur_pt = dur_pt_access + dur_pt_int*pt_interchanges + dur_pt_bus + dur_pt_rail

# car availability
has_a_car = car_ownership != 0

# all people
all = trip_id != -1
# availability
av = {1:all,2:all,3:all,4:has_a_car}


logger = get_screen_logger(level=INFO)

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

mu_motorized = Beta('mu_motorized', 3, 1, None, 0)
mu_private = Beta('mu_private', 3, 1, None, 0)

alpha_car_motorized = Beta('alpha_car_motorized', 0.5, 0, 1, 0)
alpha_car_private = 1 - alpha_car_motorized

alpha_motorized = {1: alpha_car_motorized, 2: 0.0, 3: 1.0}
alpha_private = {1: alpha_car_private, 2: 1.0, 3: 0.0}

nest_motorized = OneNestForCrossNestedLogit(
    nest_param=mu_motorized,
    dict_of_alpha=alpha_motorized,
    name='motorized',
)
nest_private = OneNestForCrossNestedLogit(
    nest_param=mu_private,
    dict_of_alpha=alpha_private,
    name='private',
)

nests = NestsForCrossNestedLogit(
    choice_set=[1, 2, 3], tuple_of_nests=(nest_motorized, nest_private)
)


# logit model
logprob_cnl = logcnl(V_1, av, nests, chosen_alternative)
biogeme_cnl = bio.BIOGEME(database, logprob_cnl)
biogeme_cnl.modelName = 'logit_lmpc12_model4'