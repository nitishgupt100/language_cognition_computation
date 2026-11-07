import stanza

print("Downloading Hindi Stanza model (one-time setup)...")
stanza.download('hi', processors='tokenize,pos,lemma,depparse')
print("Model downloaded and cached successfully!")