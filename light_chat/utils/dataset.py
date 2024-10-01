def character_to_index_dictionary(characters):
    return {s: i for i, s in enumerate(sorted(set(characters)))}


def index_to_character_dictionary(characters):
    return {i: s for i, s in enumerate(sorted(set(characters)))}
