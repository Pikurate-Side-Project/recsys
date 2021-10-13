def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        users, urls, labels = [], [], []
        for line in lines:
            if line.strip() != '':
                url, user, label = line.strip().split('\t')
                users += [user]
                urls += [url]
                labels += [label]

    return users, urls, labels


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm