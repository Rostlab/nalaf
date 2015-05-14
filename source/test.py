from bs4 import BeautifulSoup
filename = "test.html"

# simple method
minimum_spaces = 2
maximum_spaces = 5
minimum_lettres = 12
maximum_lettres = 100
indicatives = ["substitution", "deletion", "insertion", "mutation", "point mutation"]
positions = ["position", "[0-9]+"]

with open (filename, "r") as f:
    html_doc = f.read().replace("\n","")
    soup = BeautifulSoup(html_doc)
    sentences = soup.p.string.split(". ")

    print(sentences)
