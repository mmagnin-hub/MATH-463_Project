'''File: logit_lmpc12_model2.py

Using Model1 as the base model, choose a socioeconomic characteristic : purpose of the trip

Orane Solim Koenga (EPFL Master Student)
Tue Nov 26 18:15 2024

'''
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit
from biogeme.segmentation import DiscreteSegmentationTuple, segmented_beta

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
beta_travel_time = Beta('beta_travel_time', 0, None, None, 0)

# alternative specific parameter for travel time
beta_travel_time_2 = Beta('beta_travel_time_2', 0, None, None, 0)
beta_travel_time_3 = Beta('beta_travel_time_3', 0, None, None, 0)
beta_travel_time_4 = Beta('beta_travel_time_4', 0, None, None, 0)

# socioeconomic characteristic: purpose 
purpose = Variable('purpose')

purpose_segmentation = DiscreteSegmentationTuple(variable=purpose, mapping={1: 'home-based work',2: 'home-based education', 3: 'home-based other',4: 'employers business', 5: 'non-homebased other'})

segmented_b_time = segmented_beta(beta_travel_time, [purpose_segmentation])
segmented_b_time_2 = segmented_beta(beta_travel_time_2, [purpose_segmentation])
segmented_b_time_3 = segmented_beta(beta_travel_time_3, [purpose_segmentation])
segmented_b_time_4 = segmented_beta(beta_travel_time_4, [purpose_segmentation])

# Utility function 
# walking
opt1_2 = (
    segmented_b_time * dur_walking
)
# cycling
opt2_2 = (
    constant_2
    + segmented_b_time_2 * dur_cycling
)
# public transportation
opt3_2 = (
    constant_3
    + beta_cost * cost_transit
    + segmented_b_time_3 * dur_pt
)
# car 
opt4_2 = (
    has_a_car * (constant_4
    + beta_cost * (cost_driving_fuel + driving_traffic_percent*cost_driving_ccharge)
    + segmented_b_time_4 * dur_driving)
)


V_2 = {1: opt1_2, 2: opt2_2, 3: opt3_2, 4: opt4_2}


# logit model
logprob_2 = loglogit(V_2, None, chosen_alternative)
biogeme_2 = bio.BIOGEME(database, logprob_2)
biogeme_2.modelName = 'logit_lmpc12_model2'
results = biogeme_2.estimate()
