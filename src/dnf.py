def format_vars(vars):
    terms = []
    for idx, val in enumerate(vars):
        terms.append(str(idx) if val.item() else "¬%s" % idx)
    return "( " + " ∧ ".join(terms) + " )"


def format_dnf(conj_signs, conj_weights):

    if conj_signs.shape != conj_weights.shape:
        raise Exception(
            "To format a DNF, signs and weights must have same shape. %s, %s"
            % (conj_signs.shape, conj_weights.shape)
        )

    if len(conj_signs.shape) != 2:
        raise Exception(
            "A DNF only has 2 dimensions. Input Tensors have %s dimensions."
            % len(conj_signs.shape)
        )

    conj_signs = conj_signs >= 0.5
    conj_weights = conj_weights >= 0.5

    no_vars, no_conjs = conj_signs.shape

    conj_strs = []
    for conj_idx in range(no_conjs):
        conj_terms = []
        for var_idx in range(no_vars):
            exists = conj_weights[var_idx, conj_idx].item()
            if exists:
                sign = not conj_signs[var_idx, conj_idx].item()
                conj_terms.append("¬%s" % var_idx if sign else str(var_idx))
        conj_strs.append("( " + " ∧ ".join(conj_terms) + " )")
    return "\n∨ ".join(conj_strs)
