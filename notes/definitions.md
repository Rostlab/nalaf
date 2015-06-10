## Definitions of Natural Language mutation mentions

#### Notes:
Combination of „filters“ in useful sense

### Inclusive
Is defined by the definition of a natural language mutation mention itself:
- Minimum Spaces 3 are applied
- Minimum Lettres 12-36 are applied
- Intersection with TmVar Regex until minimum lettres param of 35

```python
	nl_mentions_array = []
	for mention in each annotated mention:
		is_reported_mention = is_nl_mention(mention)
		if (is_reported_mention and not is_exclusive) or (not is_reported_mention and is_exclusive):
			nl_mentions_array.append(mention)


	def is_nl_mention(mention):
		# filter spaces
		spaces = len(mention.split(" "))
		if spaces < minimum_spaces:
			return False
		if spaces > maximum_spaces:
			return False

		# filter lettres
		lettres = len(mention)
		if lettres < minimum_lettres:
			return False
		if lettres > maximum_lettres:
			return False

		# filter conventions; just for exclusive
		for word in mention.split(" "):
			if word in conventions:  # regex check of conventions
				return False

	return nl_mentions_array  # contains all nl mentions according to parameters
```

### Exclusive
Is defined by the definition of standard mutation mention:
- Maximum Spaces 2 are applied
- Maximum Lettres 12-36 are applied
- Standard convention names defined by regex array

#### tmVar Regex
- Regex of [RegEx.NL](https://github.com/carstenuhlig/thesis-alex-carsten/blob/master/resources/RegEx.NL)

#### tmVar Full
- Full method of tmVar based on Web-based prediction

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
