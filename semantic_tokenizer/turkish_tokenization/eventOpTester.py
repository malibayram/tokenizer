from eventOperator import EventOperator

eventOp = EventOperator()

# Test the consonantHardening function
print(eventOp.consonantHardening("kitap", "cı"))
print(eventOp.revertConsonantHardening("kitapçı", "kitap"))
