
from unstructured.partition.auto import partition

elements = partition("C:\Users\eee_j\docker\Personal_RAG\example_files\AIhgle.pdf")

print("\n\n".join([str(el) for el in elements]))