'''File: logit_lmpc12_model2bis.py

Using Model1 as the base model, choose a socioeconomic characteristic : gender

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

# socioeconomic characteristic bis : genre

genre = Variable('female')

genre_segmentation = DiscreteSegmentationTuple(
    variable=genre, mapping={1: 'female', 2:'other'})

segmented_constant_2 = segmented_beta(
    constant_2,
    [
        genre_segmentation,
    ],
)
segmented_constant_3 = segmented_beta(
    constant_3,
    [
        genre_segmentation,
    ],
)
segmented_constant_4 = segmented_beta(
    constant_4,
    [
        genre_segmentation,
    ],
)

# Utility function 
# walking
opt1_2bis = (
    beta_travel_time * dur_walking
)
# cycling
opt2_2bis = (
    segmented_constant_2
    + beta_travel_time_2 * dur_cycling
)
# public transportation
opt3_2bis = (
    segmented_constant_3
    + beta_cost * cost_transit
    + beta_travel_time_3 * dur_pt
)
# car 
opt4_2bis = (
    has_a_car * (segmented_constant_4
    + beta_cost * (cost_driving_fuel + driving_traffic_percent*cost_driving_ccharge)
    + beta_travel_time_4 * dur_driving)
)

V_2bis = {1: opt1_2bis, 2: opt2_2bis, 3: opt3_2bis, 4: opt4_2bis}


logprob_2bis = loglogit(V_2bis, None, chosen_alternative)
biogeme_2bis = bio.BIOGEME(database, logprob_2bis)
biogeme_2bis.modelName = 'logit_lmpc12_model2bis'
results = biogeme_2bis.estimate()