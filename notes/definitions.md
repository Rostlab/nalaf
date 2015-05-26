## Definitions of Natural Language mutation mentions

#### Notes:
Combination of „filters“ in useful sense

### Inclusive
Is defined by the definition of a natural language mutation mention itself:
- Minimum Spaces 2-4 are applied
- Minimum Lettres 12-36 are applied

### Exclusive
Is defined by the definition of standard mutation mention:
- Maximum Spaces 2-4 are applied
- Maximum Lettres 12-36 are applied
- No standard convention names defined by regex array

### Ankit's Definition
Natural language mentions are defined by following set of rule: Length of the mention > 28 or mention with a minimum of 4 spaces separated word. Ex. "deletion of a conserved SLQ(Y/F)LA", "13-bp deletion at the beginning of exon 8".

#### Formal Definition
- Length of mention > 28
- Nr of spaces in mention > 4

----
> Obsolete
#### Dictionaries
- 2-3 dictionaries (minimum indicative word and connecting word)
- as pattern ([indicative]x-[connecting]?-[indicative/helping]x)x (immer connecting eingeschlossen)
- minimum 2-4 spaces
- minimum 12-36 character

> #### Simple high coverage
- substitution/deletion/… (high indicative word)
- and position as number or number-word
- anything in between
- minimum 2-4 spaces
- minimum 12-36 character
- maximum spaces 2-7
- maximum 12-100 lettres