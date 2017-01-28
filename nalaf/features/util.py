def masked_text(token, part):
    in_entity = token.get_entity(part)
    if in_entity is None:
        return token.word
    else:
        return in_entity.class_id
