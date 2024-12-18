import pandas as pd
from typing import NamedTuple

# Biogeme 
import biogeme.database as db
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Expression
from biogeme.expressions import Variable
from biogeme.models import loglogit, logit, nested


# models
from models.logit_lmpc12_model0 import V_0, logprob_0, biogeme_0, results as res_mod0, chosen_alternative
from models.logit_lmpc12_model1 import V_1, logprob_1, biogeme_1, results as res_mod1
from models.logit_lmpc12_model2 import V_2, logprob_2, biogeme_2, results as res_mod2
from models.logit_lmpc12_model3 import V_3, logprob_3, biogeme_3, results as res_mod3
from models.logit_lmpc12_model4 import results_nested, lognested, biogeme_nested, nests

# forecasting
from models_forecasting.logit_lmpc12_model_scenario1_boxcox import V_3 as V_scenario1, logprob_3 as logprob_scenario1,biogeme_3 as biogeme_scenario1, results as res_scenario1
from models_forecasting.logit_lmpc12_model_scenario1_nested import results_nested as results_scenario1, lognested as lognested_scenario1, biogeme_nested as biogeme_scenario1
from models_forecasting.logit_lmpc12_model_scenario2_boxcox import V_3 as V_scenario2, logprob_3 as logprob_scenario2,biogeme_3 as biogeme_scenario2, results as res_scenario2
from models_forecasting.logit_lmpc12_model_scenario2_nested import results_nested as results_scenario2, lognested as lognested_scenario2, biogeme_nested as biogeme_scenario2


# Load the dataset
df = pd.read_csv('models/lpmc12.dat', sep='\t')
database = db.Database('lpmc12', df)

# Define population sizes for each stratum
census = {
    'female_44_less': 2841376,
    'female_45_more': 1519948,
    'male_44_less': 2926408,
    'male_45_more': 1379198,
}

# Total population size
population_size = sum(census.values())

# Define filters for each stratum
filters = {
    'female_44_less': (df['female'] == 1) & (df['age'] <= 44),
    'female_45_more': (df['female'] == 1) & (df['age'] > 44),
    'male_44_less': (df['female'] == 0) & (df['age'] <= 44),
    'male_45_more': (df['female'] == 0) & (df['age'] > 44),
}

# Count the sample size in each stratum
sample_segments = {
    segment_name: segment_rows.sum() for segment_name, segment_rows in filters.items()
}
print(f'Sample segments: {sample_segments}')

# Total sample size
total_sample = sum(sample_segments.values())
print(f'Sample size: {total_sample}')

# Calculate weights
normalizedWeight = {
    segment_name: census[segment_name] * total_sample / (segment_size * population_size)
    for segment_name, segment_size in sample_segments.items()
}
print(f'Weights: {normalizedWeight}')

# Insert weights into the dataset
for segment_name, segment_rows in filters.items():
    df.loc[segment_rows, 'weight'] = normalizedWeight[segment_name]

# Verify sum of weights
sum_weights = df['weight'].sum()
print(f'Sum of the weights: {sum_weights}')

normalizedWeight = Variable('weight')


class IndicatorTuple(NamedTuple):
    """Tuple storing the value of an indicator, and the bounds on its confidence interval."""

    value: float
    lower: float
    upper: float

def market_share(utilities: dict[int, Expression], results, biogeme_model = None, is_nested = False) -> dict[str, IndicatorTuple]:
    """Calculate the market shares of all alternatives, given the
    specification of the utility functions.

    :param utilities: Specification of the utility functions. It is a
        dict where the keys are the IDs of the alternatives, and the
        values are the expressions of the utility functions.

    :return: A dictionary where each entry corresponds to an
        alternative, and associates its name with the IndicatorTuple
        containing the value of the market share, and the lower and
        upper bounds of the 90% confidence interval.
    """
    if is_nested:
        
        prob_walk = nested(utilities, None, nests, 1)
        prob_cycle = nested(utilities, None, nests, 2)
        prob_pt = nested(utilities, None, nests, 3)
        prob_car = nested(utilities, None, nests, 4)
    else:
        prob_walk = logit(utilities, None, 1)
        prob_cycle = logit(utilities, None, 2)
        prob_pt = logit(utilities, None, 3)
        prob_car = logit(utilities, None, 4)

    # Simulation setup
    simulate = {
        'weight': normalizedWeight,  # Assuming normalized weights are provided
        'Prob. walk': prob_walk,
        'Prob. cycle': prob_cycle,
        'Prob. PT': prob_pt,
        'Prob. car': prob_car,
    }
    
    # Creating Biogeme object
    biosim = BIOGEME(database, simulate)
    simulated_values = biosim.simulate(results.get_beta_values())
    print(simulated_values[['Prob. walk', 'Prob. cycle', 'Prob. PT', 'Prob. car']].describe())
    
    # Confidence intervals
    if is_nested:
        betas = biogeme_model.free_beta_names
    else:
        logprob = loglogit(utilities, None, chosen_alternative)
        biogeme = BIOGEME(database, logprob)
        betas = biogeme.free_beta_names
    sensitivity_betas = results.get_betas_for_sensitivity_analysis(
        betas, use_bootstrap=False
    )
    left, right = biosim.confidence_intervals(sensitivity_betas, 0.9)


    # Initialize market shares
    market_shares = {}

    # Iterate through alternatives
    for alt_name, prob_name in [
        ("Walking", "Prob. walk"),
        ("Cycling", "Prob. cycle"),
        ("Public transportation", "Prob. PT"),
        ("Car", "Prob. car"),
    ]:
        weighted_name = f"Weighted choice_prob. {alt_name.lower()}"

        # Calculate weighted probabilities
        simulated_values[weighted_name] = (
            simulated_values["weight"] * simulated_values[prob_name]
        )
        left[weighted_name] = left["weight"] * left[prob_name]
        right[weighted_name] = right["weight"] * right[prob_name]

        # Calculate mean values and bounds
        market_share_value = simulated_values[weighted_name].mean()
        market_share_lower = left[weighted_name].mean()
        market_share_upper = right[weighted_name].mean()

        # Store results in the market shares dictionary
        market_shares[alt_name] = IndicatorTuple(
            value=market_share_value,
            lower=market_share_lower,
            upper=market_share_upper,
        )

    return market_shares