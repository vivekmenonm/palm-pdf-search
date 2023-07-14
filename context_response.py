import pandas as pd
from vertexai.preview.language_models import TextGenerationModel

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

context = """
Storage and content policy \n
How durable is my data in Cloud Storage? \n
Cloud Storage is designed for 99.999999999% (11 9's) annual durability, which is appropriate for even primary storage and
business-critical applications. This high durability level is achieved through erasure coding that stores data pieces redundantly
across multiple devices located in multiple availability zones.
Objects written to Cloud Storage must be redundantly stored in at least two different availability zones before the
write is acknowledged as successful. Checksums are stored and regularly revalidated to proactively verify that the data
integrity of all data at rest as well as to detect corruption of data in transit. If required, corrections are automatically
made using redundant data. Customers can optionally enable object versioning to add protection against accidental deletion.
"""



question = "in how many different zones it is available"

prompt = f"""Answer the question given in the contex below:
Context: {context}?\n
Question: {question} \n
Answer:
"""

print("Prompt is:", prompt)

print("Response is:",
    generation_model.predict(
        prompt,
    ).text
)