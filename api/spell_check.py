from hunspell import Hunspell
h = Hunspell('en_US')

def suggest(word):
    print("Word : " + word)
    if not h.spell(word):  # Check spelling with h.check instead of h.spell
        suggestions = h.suggest(word)
        return suggestions
    return []
    