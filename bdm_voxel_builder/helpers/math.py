def calculate_chance(x, output_y_domain, non_flat_x_segment = None, input_x_domain = [0,1]):
    """remaps x (from input_domain) to y on output_domain
    optionally flattens segments outside the flat_domain
    return y """
    i, j = input_x_domain
    a, b = output_y_domain

    if non_flat_x_segment is not None:
        u, v = non_flat_x_segment
        y = (b - a) / (v - u) * x
    else:
        y = (b - a) / (j - i) * x

    y = max(a, (min(b, y)))

def remap_trim(x, output_domain, cap_output_domain, input_domain = [0,1], ):
    """remaps x (from input domain) onto output domain
    trims output onto cap_output_domain"""
    i, j = input_domain
    a, b = output_domain
    y = (b - a)/(j - i) * x
    if cap_output_domain is not None:
        lo, m = cap_output_domain
        y = max(lo, (min(m, y)))
    return y

def remap(x, output_domain, input_domain = [0,1]):
    """remaps x (from input domain) onto output domain
    trims output onto cap_output_domain"""
    i, j = input_domain
    a, b = output_domain

    if i == j or a == b:
        raise ZeroDivisionError("Input or output domain is invalid.")

    y = (b - a)/(j - i) * x
    return y
