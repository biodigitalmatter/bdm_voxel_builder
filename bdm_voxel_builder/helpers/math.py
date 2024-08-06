
def remap_trim(
    x,
    output_domain,
    cap_output_domain,
    input_domain=(0, 1),
):
    """remaps x (from input domain) onto output domain
    trims output onto cap_output_domain"""
    i, j = input_domain
    a, b = output_domain
    y = (b - a) / (j - i) * x
    if cap_output_domain is not None:
        lo, m = cap_output_domain
        y = max(lo, (min(m, y)))
    return y


def remap(x, output_domain, input_domain=(0, 1)):
    """remaps x (from input domain) onto output domain
    trims output onto cap_output_domain"""
    i, j = input_domain
    a, b = output_domain

    if i == j or a == b:
        raise ValueError("Input or output domain is invalid.")

    y = (b - a) / (j - i) * x
    return y
