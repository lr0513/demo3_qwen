from postprocess import postprocess


sentence = "Comparison with alkaline phosphatases and 5 - nucleotidase"

model_output = """
[
  {
    "name": "alkaline phosphatases",
    "type": "GENE",
    "start": 22,
    "end": 44
  }
]
"""

result = postprocess(sentence, model_output)

for entity in result:
    print(entity)