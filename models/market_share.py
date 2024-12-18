
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Expression
from biogeme.models import loglogit, logit


from optima_variables import Choice, database, normalizedWeight


def market_share(utilities: dict[int, Expression]) -> dict[str, IndicatorTuple]:
    """Calculate the market shares of all alternatives, given the
    specification of the utility functions.

    :param utilities: specification of the utility functions. It is a
        dict where the keys are the IDs of the alternatives, and the
        values are the expressions of the utility functions.

    :return: a dictionary where each entry corresponds to an
        alternative, and associates its name with the IndicatorTuple
        containing the value of the market share, and the lower and
        upper bounds of the 90% confidence interval.


    """
    prob_pt = logit(utilities, None, 0)
    prob_car = logit(utilities, None, 1)
    prob_sm = logit(utilities, None, 2)

    simulate = {
        'weight': normalizedWeight,
        'Prob. PT': prob_pt,
        'Prob. car': prob_car,
        'Prob. SM': prob_sm,
    }
    biosim = BIOGEME(database, simulate)
    simulated_values = biosim.simulate(results.get_beta_values())

    # We also calculate confidence intervals for the calculated quantities
    betas = biogeme.free_beta_names
    b = results.get_betas_for_sensitivity_analysis(betas)
    left, right = biosim.confidence_intervals(b, 0.9)

    # Market shares are calculated using the weighted mean of the individual probabilities

    # Alternative car
    simulated_values['Weighted choice_prob. car'] = (
        simulated_values['weight'] * simulated_values['Prob. car']
    )
    left['Weighted choice_prob. car'] = left['weight'] * left['Prob. car']
    right['Weighted choice_prob. car'] = right['weight'] * right['Prob. car']

    market_share_car = simulated_values['Weighted choice_prob. car'].mean()
    market_share_car_left = left['Weighted choice_prob. car'].mean()
    market_share_car_right = right['Weighted choice_prob. car'].mean()

    # Alternative public transportation
    simulated_values['Weighted choice_prob. PT'] = (
        simulated_values['weight'] * simulated_values['Prob. PT']
    )
    left['Weighted choice_prob. PT'] = left['weight'] * left['Prob. PT']
    right['Weighted choice_prob. PT'] = right['weight'] * right['Prob. PT']

    market_share_pt = simulated_values['Weighted choice_prob. PT'].mean()
    market_share_pt_left = left['Weighted choice_prob. PT'].mean()
    market_share_pt_right = right['Weighted choice_prob. PT'].mean()

    # Alternative slow modes
    simulated_values['Weighted choice_prob. SM'] = (
        simulated_values['weight'] * simulated_values['Prob. SM']
    )
    left['Weighted choice_prob. SM'] = left['weight'] * left['Prob. SM']
    right['Weighted choice_prob. SM'] = right['weight'] * right['Prob. SM']

    market_share_sm = simulated_values['Weighted choice_prob. SM'].mean()
    market_share_sm_left = left['Weighted choice_prob. SM'].mean()
    market_share_sm_right = right['Weighted choice_prob. SM'].mean()

    return {
        'Car': IndicatorTuple(
            value=market_share_car,
            lower=market_share_car_left,
            upper=market_share_car_right,
        ),
        'Public transportation': IndicatorTuple(
            value=market_share_pt,
            lower=market_share_pt_left,
            upper=market_share_pt_right,
        ),
        'Slow modes': IndicatorTuple(
            value=market_share_sm,
            lower=market_share_sm_left,
            upper=market_share_sm_right,
        ),
    }
