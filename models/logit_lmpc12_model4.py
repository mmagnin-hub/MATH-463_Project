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
from biogeme.nests import (
    OneNestForCrossNestedLogit,
    NestsForCrossNestedLogit,
)
from biogeme.models import lognested, logcnl
from biogeme.nests import (
    OneNestForNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    NestsForCrossNestedLogit,
)

mu_motorized = Beta('mu_motorized',3, 0, None, 0)
mu_unmotorized = Beta('mu_unmotorized', 1, 0, None, 1)

motorized = OneNestForNestedLogit(
    nest_param=mu_motorized, list_of_alternatives=[3,4], name='motorized'
)
unmotorized = OneNestForNestedLogit(nest_param=mu_unmotorized, list_of_alternatives=[1,2], name='unmotorized')
nests = NestsForNestedLogit(choice_set=[1, 2, 3, 4], tuple_of_nests=(motorized, unmotorized))


mu_private = Beta('mu_private', 3, 1, None, 0)

alpha_car_motorized = Beta('alpha_car_motorized', 0.5, 0, 1, 0)
alpha_car_private = 1 - alpha_car_motorized

alpha_motorized = {1:0.0 , 2: 0.0, 3: 1.0,4:alpha_car_motorized}
alpha_private = {1: 1.0 , 2: 1.0, 3: 0.0 ,4:alpha_car_private}

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

nests_cross = NestsForCrossNestedLogit(
    choice_set=[1, 2, 3, 4], tuple_of_nests=(nest_motorized, nest_private)
)


# logit model
#logprob_cnl = logcnl(V_1, av, nests, chosen_alternative)
#biogeme_cnl = bio.BIOGEME(database, logprob_cnl)
#biogeme_cnl.modelName = 'logit_lmpc12_model4'