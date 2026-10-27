import chromadb


# Create a ChromaDB client
client = chromadb.Client()


# Create a collection
collection = client.create_collection(
    name="products"
)


# Add product descriptions
collection.add(
    documents=[
        "High-performance gaming laptop",
        "Desktop computer for office use",
        "Wireless Bluetooth mouse"
    ],
    ids=[
        "p1",
        "p2",
        "p3"
    ]
)


# Search for similar products
result = collection.query(
    query_texts=["mouse"],
    n_results=2
)


# Display the search results
print(result)