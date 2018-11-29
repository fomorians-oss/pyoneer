def update_variables(source_vars, target_vars, rate=1.0):
    """
    Update the target variables with source variables by copy or linear
    interpolation.

    Args:
        source_vars: Variables to update from.
        target_vars: Variables to update to.
        rate (default: 1.0): The linear interpolation rate. A rate of 1.0 will
                             directly copy.
    """
    assert rate > 0.0 and rate <= 1.0, 'rate must be in [0, 1)'
    assert len(source_vars) != 0, 'source_vars must not be empty'
    assert len(target_vars) != 0, 'target_vars must not be empty'
    assert len(source_vars) == len(target_vars), \
        'source_vars length must equal target_vars length'

    for target_var, source_var in zip(target_vars, source_vars):
        if rate < 1.0:
            target_var.assign(rate * source_var + (1 - rate) * target_var)
        else:
            target_var.assign(source_var)
