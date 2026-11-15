import re
from operator import itemgetter

def get_layers(model, filter_regexp):
    """
    Filters out the layers according to a regexp. Note that
    we omit biases.

    Args:
        - model: a nn.Module
        - filter_regexp: a regexp to filter the layers to keep
          according to their name in model.named_parameters().
          For instance, the regexp:

             down_layers\\.[123456]\\.(conv[12]|identity\\.conv))

          is keeping blocks down_layers from 1 to 6, and inside
          each block is keeping conv1, conv2 and identity.conv.

    Remarks:
        - We add (module\\.)? at the beginning of the regexp to
          account for the possible use of nn.parallel.DataParallel
    """

    all_layers = map(itemgetter(0), model.named_parameters())

    all_layers = filter(lambda x: "bias" not in x, all_layers)

    all_layers = map(lambda x: x.replace(".weight_orig", ""), all_layers)
    all_layers = map(lambda x: x.replace(".weight", ""), all_layers)

    filter_regexp = "(module\\.)?" + "(" + filter_regexp + ")"
    r = re.compile(filter_regexp)

    return list(filter(r.match, all_layers))

def attrsetter(*items):
    def resolve_attr(obj, attr):
        attrs = attr.split(".")
        head = attrs[:-1]
        tail = attrs[-1]

        for name in head:
            obj = getattr(obj, name)
        return obj, tail

    def g(obj, val):
        for attr in items:
            resolved_obj, resolved_attr = resolve_attr(obj, attr)
            setattr(resolved_obj, resolved_attr, val)

    return g
