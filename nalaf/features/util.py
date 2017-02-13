def is_number(string):
    try:
        float(string)
        return True
    except:
        return False


def masked_text(token, part, use_gold, used_pred, token_map=None, token_is_number_fun=None):
    if token_map is None:
        token_map = (lambda t: t.word)
    if token_is_number_fun is None:
        token_is_number_fun = token_map

    in_entity = token.get_entity(part, use_gold, used_pred)

    if in_entity is not None:
        return in_entity.class_id
    elif is_number(token.word):
        return token_is_number_fun(token)
    else:
        return token_map(token)
