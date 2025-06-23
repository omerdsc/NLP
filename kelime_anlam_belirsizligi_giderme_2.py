from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk


sentences=[
    "I goto the bank flooded after the jeavy rain",
    "The revier bank was flooded after the heavy rain"
    ]


word="bank"


for s in sentences:
    print(f"Sentence: {s}")
    sense_simple_lesk=simple_lesk(s, word)
    print(f"Sense simple:{sense_simple_lesk.definition()}")
    
    sense_adapted_lesk=adapted_lesk(s, word)
    print(f"Sense adapted:{sense_adapted_lesk.definition()}")
    
    sense_cosine_lesk=cosine_lesk(s, word)
    print(f"Sense cosine:{sense_cosine_lesk.definition()}")

