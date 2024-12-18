import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit, boxcox
from biogeme.segmentation import DiscreteSegmentationTuple, segmented_beta
import numpy as np

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

new_purpose_segmentation = DiscreteSegmentationTuple(variable=purpose, mapping={1: 'work-education-related',2: 'work-education-related', 3: 'non-work-education-related',4: 'work-education-related', 5: 'non-work-education-related'})

segmented_b_time = segmented_beta(beta_travel_time, [new_purpose_segmentation])
segmented_b_time_2 = segmented_beta(beta_travel_time_2, [new_purpose_segmentation])
segmented_b_time_3 = segmented_beta(beta_travel_time_3, [new_purpose_segmentation])
segmented_b_time_4 = segmented_beta(beta_travel_time_4, [new_purpose_segmentation])

#Box-COx transformation of traveltime variables

lambda_boxcox = Beta('lambda_boxcox', 0.5 , -10, 10, 0)
boxcox_time_2 = boxcox(dur_cycling, lambda_boxcox)
boxcox_time_3 = boxcox(dur_pt, lambda_boxcox)
boxcox_time_4 = boxcox(dur_driving, lambda_boxcox)


# Utility function 
# walking
opt1_boxcox = (
    segmented_b_time * dur_walking
)
# cycling
opt2_boxcox = (
    constant_2
    + boxcox_time_2 * segmented_b_time_2
)
# public transportation
opt3_boxcox = (
    constant_3
    + beta_cost * cost_transit*0.2 ################### the only diff with model 3
    + boxcox_time_3 * segmented_b_time_3
)
# car 
opt4_boxcox = (
    (constant_4
    + beta_cost * (cost_driving_fuel + driving_traffic_percent*cost_driving_ccharge)
    + boxcox_time_4 * segmented_b_time_4)
)
V_boxcox = {1: opt1_boxcox, 2: opt2_boxcox, 3: opt3_boxcox, 4: opt4_boxcox}

logprob_boxcox = loglogit(V_boxcox, None, chosen_alternative)
biogeme_boxcox = bio.BIOGEME(database, logprob_boxcox)
biogeme_boxcox.modelName = 'logit_lmpc12_model3'
results = biogeme_boxcox.estimate()

